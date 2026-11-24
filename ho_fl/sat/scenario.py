"""Trace of a scenario: where every satellite is in every slot, and what follows from that (brief v2, section 5.1).

Model: circular orbits of a Walker-delta constellation, a sun fixed in the +x direction of the inertial frame for
the simulated day (which starts at the 2027 March equinox), a cylindrical Earth shadow, an Earth that rotates
about z with the inertial and the Earth-fixed frames coinciding at t = 0, ground stations visible above 10 degrees
of elevation, and +Grid links that are up when they are short enough and clear the atmosphere. No perturbations.

A trace is a directory with ``trace.npz`` (the arrays below), ``meta.json`` (parameters, software versions and
the sha256 of every array) and ``sanity.json`` (summary statistics, compared with appendix A of the brief).

    pos_km        T x N x 3  float32   inertial position
    sunlit        T x N      bool
    land          T x N      bool      sub-satellite point over land
    gs_visible    G x T x N  bool      one row per ground station of config.GROUND_STATIONS, in that order
    edges         E x 2      int16     satellite pairs, lower number first
    edge_kind     E          int8      0 in-plane RF, 1 inter-plane RF, 2 inter-plane laser
    edge_dist_km  T x E      float32
    edge_up       T x E      bool
"""
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import write_json
from ..randomness import stable_seed
from .config import GROUND_STATIONS, SCENARIOS
from .roles import Roles

R_EARTH_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
OMEGA_EARTH_RAD_S = 7.2921159e-5
EPOCH = "2027-03-20T00:00:00Z"
SUN_DIRECTION = (1.0, 0.0, 0.0)
MAX_LINK_KM = 5000.0          # a link is up when it is at most this long ...
MIN_GRAZING_KM = 80.0         # ... and its line of sight stays this far above the ground
MIN_ELEVATION_DEG = 10.0
LAND_DUTY_CYCLE = 0.3         # share of "land" when the land mask is not installed
LAND_BLOCK_SLOTS = 10         # the substitute changes every this many slots
EDGE_IN_PLANE, EDGE_INTER_PLANE, EDGE_LASER = 0, 1, 2
ARRAYS = ("pos_km", "sunlit", "land", "gs_visible", "edges", "edge_kind", "edge_dist_km", "edge_up")


def radius_km(altitude_km):
    return R_EARTH_KM + altitude_km


def mean_motion(altitude_km):
    """Angular rate of a circular orbit in rad/s."""
    return float(np.sqrt(MU_EARTH_KM3_S2 / radius_km(altitude_km) ** 3))


def period_s(altitude_km):
    return 2 * np.pi / mean_motion(altitude_km)


def positions_eci(constellation, times):
    """Inertial positions in km, shape ``(len(times), N, 3)``; satellite ``plane * S + slot``."""
    planes, per_plane = constellation["planes"], constellation["per_plane"]
    count = planes * per_plane
    plane, slot = np.repeat(np.arange(planes), per_plane), np.tile(np.arange(per_plane), planes)
    raan = 2 * np.pi * plane / planes
    phase = 2 * np.pi * slot / per_plane + 2 * np.pi * constellation["phasing"] * plane / count
    u = phase[None, :] + mean_motion(constellation["altitude_km"]) * np.asarray(times, dtype=np.float64)[:, None]
    inclination, r = np.deg2rad(constellation["inclination_deg"]), radius_km(constellation["altitude_km"])
    cos_u, sin_u, cos_raan, sin_raan = np.cos(u), np.sin(u), np.cos(raan)[None, :], np.sin(raan)[None, :]
    x = r * (cos_raan * cos_u - sin_raan * sin_u * np.cos(inclination))
    y = r * (sin_raan * cos_u + cos_raan * sin_u * np.cos(inclination))
    z = r * (sin_u * np.sin(inclination))
    return np.stack([x, y, z], axis=-1)


def sunlit_mask(positions):
    """False inside the cylindrical shadow behind the Earth."""
    sun = np.asarray(SUN_DIRECTION)
    along = positions @ sun
    across = np.linalg.norm(positions - along[..., None] * sun, axis=-1)
    return ~((along < 0) & (across < R_EARTH_KM))


def to_ecef(positions, times):
    """Rotate inertial positions into the Earth-fixed frame (the frames coincide at t = 0)."""
    angle = OMEGA_EARTH_RAD_S * np.asarray(times, dtype=np.float64)[:, None]
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    return np.stack([x * np.cos(angle) + y * np.sin(angle), -x * np.sin(angle) + y * np.cos(angle), z], axis=-1)


def sub_satellite_points(ecef):
    """Latitude and longitude in degrees of the points below the satellites."""
    radius = np.linalg.norm(ecef, axis=-1)
    return np.rad2deg(np.arcsin(ecef[..., 2] / radius)), np.rad2deg(np.arctan2(ecef[..., 1], ecef[..., 0]))


def land_mask(latitude, longitude, seed, source="auto"):
    """Whether each point is over land, and how that was decided.

    ``mask`` uses the ``global_land_mask`` package. ``duty_cycle`` is the substitute when the package is missing:
    a satellite is "over land" in 30% of the blocks of ten slots, decided by a hash of the seed, the satellite and
    the block. ``auto`` takes the mask when it can be imported. The two give different imaging shares (about 18.5%
    against 30% of the time for hybrid satellites), so the choice is recorded with the trace.
    """
    if source in ("auto", "mask"):
        try:
            from global_land_mask import globe
        except ImportError:
            if source == "mask":
                raise
        else:
            return np.asarray(globe.is_land(latitude, longitude), dtype=bool), "global_land_mask"
    if source not in ("auto", "duty_cycle"):
        raise ValueError(f"Unknown land source {source}")
    slots, count = latitude.shape
    blocks = (slots + LAND_BLOCK_SLOTS - 1) // LAND_BLOCK_SLOTS
    draw = np.array([[stable_seed(seed, "img", sid, block) for sid in range(count)] for block in range(blocks)]) / 2 ** 32
    return np.repeat(draw < LAND_DUTY_CYCLE, LAND_BLOCK_SLOTS, axis=0)[:slots], "duty_cycle"


def station_vector(name):
    latitude, longitude = np.deg2rad(GROUND_STATIONS[name])
    return np.array([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)])


def ground_visibility(ecef, sites):
    """``G x T x N``: satellite at least ``MIN_ELEVATION_DEG`` above the horizon of each station."""
    rows = []
    for name in sites:
        zenith = station_vector(name)
        offset = ecef - R_EARTH_KM * zenith
        sine = (offset @ zenith) / np.linalg.norm(offset, axis=-1)
        rows.append(sine >= np.sin(np.deg2rad(MIN_ELEVATION_DEG)))
    return np.stack(rows)


def slant_range_km(ecef, name):
    """Distance from a ground station to each satellite, same leading shape as ``ecef``."""
    return np.linalg.norm(ecef - R_EARTH_KM * station_vector(name), axis=-1)


def grid_edges(planes, per_plane):
    """+Grid: every satellite links to the next slot of its plane and to the same slot of the next plane, the
    seam between the last and the first plane included. Links between slot 0 of adjacent planes are optical."""
    roles = Roles(planes, per_plane)
    edges, kinds = [], []
    for plane in range(planes):
        for slot in range(per_plane):
            edges.append(sorted((roles.sid(plane, slot), roles.sid(plane, (slot + 1) % per_plane))))
            kinds.append(EDGE_IN_PLANE)
    for plane in range(planes):
        for slot in range(per_plane):
            edges.append(sorted((roles.sid(plane, slot), roles.sid((plane + 1) % planes, slot))))
            kinds.append(EDGE_LASER if slot == 0 else EDGE_INTER_PLANE)
    return np.asarray(edges, dtype=np.int16), np.asarray(kinds, dtype=np.int8)


def edge_geometry(positions, edges):
    """Length of every edge in every slot, and whether the link is up."""
    a, b = positions[:, edges[:, 0].astype(int)], positions[:, edges[:, 1].astype(int)]
    span = b - a
    distance = np.linalg.norm(span, axis=-1)
    # Closest approach of the segment a-b to the centre of the Earth.
    along = np.clip(-(a * span).sum(-1) / np.maximum(distance, 1e-9) ** 2, 0.0, 1.0)
    grazing = np.linalg.norm(a + along[..., None] * span, axis=-1) - R_EARTH_KM
    return distance, (distance <= MAX_LINK_KM) & (grazing >= MIN_GRAZING_KM)


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def trace_sha256(hashes):
    """One hash for the whole trace: the array hashes in the order of ``ARRAYS``."""
    return hashlib.sha256("".join(hashes[name] for name in ARRAYS).encode()).hexdigest()


def build_trace(scenario="s1", hours=24.0, slot_s=30, seed=42, land="auto"):
    """Arrays and metadata of a scenario. Deterministic: the seed only enters the substitute for the land mask."""
    constellation = SCENARIOS[scenario]
    slots = int(round(hours * 3600 / slot_s))
    times = np.arange(slots, dtype=np.float64) * slot_s
    positions = positions_eci(constellation, times)
    ecef = to_ecef(positions, times)
    latitude, longitude = sub_satellite_points(ecef)
    over_land, land_source = land_mask(latitude, longitude, seed, land)
    sites = list(GROUND_STATIONS)
    edges, kinds = grid_edges(constellation["planes"], constellation["per_plane"])
    distance, up = edge_geometry(positions, edges)
    arrays = {"pos_km": positions.astype(np.float32), "sunlit": sunlit_mask(positions), "land": over_land,
              "gs_visible": ground_visibility(ecef, sites), "edges": edges, "edge_kind": kinds,
              "edge_dist_km": distance.astype(np.float32), "edge_up": up}
    hashes = {name: array_sha256(arrays[name]) for name in ARRAYS}
    meta = {"scenario": scenario, "constellation": constellation, "hours": hours, "slot_s": slot_s, "slots": slots,
            "satellites": constellation["planes"] * constellation["per_plane"], "seed": seed, "epoch": EPOCH,
            "sun_direction_eci": list(SUN_DIRECTION), "land_source": land_source, "ground_stations": sites,
            "ground_station_coordinates": {name: list(GROUND_STATIONS[name]) for name in sites},
            "constants": {"earth_radius_km": R_EARTH_KM, "mu_km3_s2": MU_EARTH_KM3_S2, "earth_rate_rad_s": OMEGA_EARTH_RAD_S,
                          "max_link_km": MAX_LINK_KM, "min_grazing_km": MIN_GRAZING_KM, "min_elevation_deg": MIN_ELEVATION_DEG},
            "period_s": period_s(constellation["altitude_km"]), "arrays": {name: {"sha256": hashes[name], "shape": list(arrays[name].shape),
                                                                                  "dtype": str(arrays[name].dtype)} for name in ARRAYS},
            "trace_sha256": trace_sha256(hashes), "software": {"python": platform.python_version(), "numpy": np.__version__}}
    return arrays, meta


def write_trace(out, arrays, meta, sanity=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trace.npz", **arrays)
    write_json(out / "meta.json", meta)
    if sanity is not None:
        write_json(out / "sanity.json", sanity)
    return out


@dataclass
class RoundView:
    """What a round sees: the state in its first slot (decisions use it) and what holds over its slots."""
    round: int
    first_slot: int
    last_slot: int
    sunlit: np.ndarray         # N, in the first slot
    land: np.ndarray           # N, in the first slot
    sun_seconds: np.ndarray    # N, sunlit seconds within the round
    edge_dist_km: np.ndarray   # E, in the first slot
    edge_up: np.ndarray        # E, in the first slot
    gs_first: np.ndarray       # G x N, in the first slot
    gs_last: np.ndarray        # G x N, in the last slot


class Trace:
    def __init__(self, arrays, meta):
        self.meta = meta
        for name in ARRAYS:
            setattr(self, name, arrays[name])
        constellation = meta["constellation"]
        self.roles = Roles(constellation["planes"], constellation["per_plane"])
        self.slots, self.count = self.sunlit.shape
        self.slot_s = meta["slot_s"]
        self.sha256 = meta["trace_sha256"]
        self.edge_index = {(int(a), int(b)): index for index, (a, b) in enumerate(self.edges)}

    def edge(self, a, b):
        """Index of the edge between two neighbouring satellites."""
        return self.edge_index[(a, b) if a < b else (b, a)]

    def view(self, r, slots_per_round):
        first = r * slots_per_round
        last = first + slots_per_round - 1
        if last >= self.slots:
            raise ValueError(f"Round {r} lies beyond the {self.slots} slots of the trace")
        return RoundView(round=r, first_slot=first, last_slot=last, sunlit=self.sunlit[first], land=self.land[first],
                         sun_seconds=self.sunlit[first:last + 1].sum(0) * float(self.slot_s),
                         edge_dist_km=self.edge_dist_km[first].astype(np.float64), edge_up=self.edge_up[first],
                         gs_first=self.gs_visible[:, first], gs_last=self.gs_visible[:, last])

    def ecef(self, slot):
        """Earth-fixed positions in one slot (for slant ranges to ground stations)."""
        return to_ecef(self.pos_km[slot].astype(np.float64)[None], np.array([slot * self.slot_s], dtype=np.float64))[0]


def load_trace(path):
    """Read a trace and check every array against the hash in its metadata."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    with np.load(path / "trace.npz") as stored:
        arrays = {name: stored[name] for name in ARRAYS}
    for name in ARRAYS:
        if array_sha256(arrays[name]) != meta["arrays"][name]["sha256"]:
            raise ValueError(f"{path}: array {name} does not match the hash in meta.json")
    return Trace(arrays, meta)


def ensure_trace(path, scenario="s1", hours=24.0, slot_s=30, seed=42, land="auto"):
    """Load the trace at ``path``; build and write it first when it is missing."""
    path = Path(path)
    if not (path / "meta.json").exists():
        arrays, meta = build_trace(scenario, hours, slot_s, seed, land)
        write_trace(path, arrays, meta, sanity_statistics(Trace(arrays, meta)))
    return load_trace(path)


# --- summary statistics (appendix A of the brief) ----------------------------------------------------------------------

def reachable_planes(trace, sites, slots_per_round, rule="first_last"):
    """``R x P`` booleans: plane has a satellite in view of one of ``sites``.

    ``first_last``: in the first and in the last slot of the round (not necessarily the same satellite); this is
    the rule of the ground-server baseline. ``any``: in some slot of the round, which overstates reachability.
    """
    rows = [trace.meta["ground_stations"].index(name) for name in sites]
    planes, per_plane = trace.roles.planes, trace.roles.per_plane
    rounds = trace.slots // slots_per_round
    seen = trace.gs_visible[rows].any(0)[:rounds * slots_per_round].reshape(rounds, slots_per_round, planes, per_plane).any(-1)
    return seen[:, 0] & seen[:, -1] if rule == "first_last" else seen.any(1)


def sanity_statistics(trace, slots_per_round=20):
    """Numbers that characterise a trace; the tests compare those of S1 with appendix A of the brief."""
    roles, kinds = trace.roles, trace.edge_kind
    period = trace.meta["period_s"]
    rounds = trace.slots // slots_per_round
    starts = np.arange(rounds) * slots_per_round
    seam = np.array([{roles.plane(int(a)), roles.plane(int(b))} == {0, roles.planes - 1} and roles.planes > 2 for a, b in trace.edges])
    inter = kinds != EDGE_IN_PLANE
    hybrid, forward = roles.members("H"), roles.members("Z")
    dark = ~trace.sunlit
    per_plane_dark = dark.reshape(trace.slots, roles.planes, roles.per_plane).mean((0, 2))
    stats = {"period_min": period / 60, "slots": int(trace.slots), "rounds": int(rounds), "satellites": int(trace.count),
             "edges": {"in_plane": int((kinds == EDGE_IN_PLANE).sum()), "inter_plane_rf": int((kinds == EDGE_INTER_PLANE).sum()),
                       "laser": int((kinds == EDGE_LASER).sum())},
             "in_plane_km": [float(trace.edge_dist_km[:, ~inter].min()), float(trace.edge_dist_km[:, ~inter].max())],
             "inter_plane_km": [float(trace.edge_dist_km[:, inter & ~seam].min()), float(trace.edge_dist_km[:, inter & ~seam].max())],
             "seam_km": [float(trace.edge_dist_km[:, seam].min()), float(trace.edge_dist_km[:, seam].max())] if seam.any() else None,
             "all_edges_up": bool(trace.edge_up.all()), "edge_up_share": float(trace.edge_up.mean()),
             "eclipse_share": float(dark.mean()),
             "eclipse_min_per_orbit": [float(per_plane_dark.min() * period / 60), float(per_plane_dark.max() * period / 60)],
             "hybrid_imaging_share": float((trace.sunlit[:, hybrid] & trace.land[:, hybrid]).mean()),
             "forward_only_dark_at_round_start": float(dark[starts][:, forward].mean()),
             "land_source": trace.meta["land_source"], "ground": {}}
    for label, sites in (("1", trace.meta["ground_stations"][:1]), ("3", trace.meta["ground_stations"][:3])):
        rows = [trace.meta["ground_stations"].index(name) for name in sites]
        visible = trace.gs_visible[rows].any(0)
        first_last, any_time = (reachable_planes(trace, sites, slots_per_round, rule) for rule in ("first_last", "any"))
        in_round = visible[:rounds * slots_per_round].reshape(rounds, slots_per_round, trace.count).any(1)
        # Longest stretch without a visible slot, per satellite, within the trace.
        waits = [np.diff(np.concatenate([[0], np.flatnonzero(visible[:, sid]), [trace.slots]])).max() * trace.slot_s / 3600
                 for sid in range(trace.count)]
        stats["ground"][label] = {"sites": sites, "visible_share_of_slots": float(visible.mean()),
                                  "visible_share_of_rounds": float(in_round.mean()),
                                  "longest_wait_h_median": float(np.median(waits)), "longest_wait_h_max": float(max(waits)),
                                  "reachable_planes_first_last": float(first_last.sum(1).mean()),
                                  "reachable_planes_any_time": float(any_time.sum(1).mean()),
                                  "rounds_without_reachable_plane": int((first_last.sum(1) == 0).sum())}
    return stats


# Appendix A of the brief for S1 (24 h, 30 s slots): values computed independently on the research lead's side,
# each with the absolute tolerance this comparison allows. The imaging share only holds with the real land mask.
APPENDIX_A_S1 = {
    "period_min": (95.5, 0.05),
    "in_plane_km.0": (4277.4, 0.5), "in_plane_km.1": (4277.4, 0.5),
    "inter_plane_km.0": (2505.0, 2.0), "inter_plane_km.1": (3803.0, 2.0),
    "seam_km.0": (1730.0, 2.0), "seam_km.1": (3344.0, 2.0),
    "edge_up_share": (1.0, 0.0),
    "eclipse_share": (0.334, 0.002),
    "eclipse_min_per_orbit.0": (26.3, 0.2), "eclipse_min_per_orbit.1": (35.6, 0.2),
    "hybrid_imaging_share": (0.185, 0.003),
    "forward_only_dark_at_round_start": (0.333, 0.003),
    "ground.1.visible_share_of_rounds": (0.059, 0.002),
    "ground.1.reachable_planes_first_last": (2.85, 0.02),
    "ground.1.reachable_planes_any_time": (4.26, 0.02),
    "ground.3.visible_share_of_rounds": (0.159, 0.002),
    "ground.3.reachable_planes_first_last": (6.62, 0.02),
    "ground.3.rounds_without_reachable_plane": (0.0, 0.0),
}


def _lookup(stats, key):
    value = stats
    for part in key.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return float(value)


def check_against_appendix(stats):
    """Rows ``(name, value, reference, tolerance, ok)`` comparing the statistics of an S1 trace with appendix A.
    The imaging share is left out when the trace was built without the land mask."""
    rows = []
    for key, (reference, tolerance) in APPENDIX_A_S1.items():
        if key == "hybrid_imaging_share" and stats["land_source"] != "global_land_mask":
            continue
        value = _lookup(stats, key)
        rows.append((key, value, reference, tolerance, abs(value - reference) <= tolerance))
    return rows
