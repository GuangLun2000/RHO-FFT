"""The trace of scenario S1 against appendix A of the brief (values computed independently on the lead's side)."""
import json

import numpy as np
import pytest

from ho_fl.sat import scenario
from ho_fl.sat.cli import trace_main
from ho_fl.sat.config import SCENARIOS
from ho_fl.sat.scenario import (EDGE_IN_PLANE, EDGE_INTER_PLANE, EDGE_LASER, R_EARTH_KM, Trace, build_trace,
                                check_against_appendix, load_trace, reachable_planes, sanity_statistics, write_trace)


@pytest.fixture(scope="module")
def s1():
    """S1 for 24 h with the substitute for the land mask, so that it does not depend on the optional package."""
    arrays, meta = build_trace("s1", land="duty_cycle")
    return Trace(arrays, meta)


@pytest.fixture(scope="module")
def stats(s1):
    return sanity_statistics(s1)


def test_orbit_geometry(s1):
    constellation = SCENARIOS["s1"]
    assert scenario.period_s(550.0) / 60 == pytest.approx(95.5, abs=0.05)
    radius = np.linalg.norm(s1.pos_km.astype(np.float64), axis=-1)
    assert np.allclose(radius, R_EARTH_KM + 550.0, atol=0.01)
    # Highest latitude reached equals the inclination; the first satellite starts on the x axis at the equator.
    assert np.rad2deg(np.arcsin(np.abs(s1.pos_km[..., 2]).max() / (R_EARTH_KM + 550.0))) == pytest.approx(53.0, abs=0.05)
    assert np.allclose(s1.pos_km[0, 0], [R_EARTH_KM + 550.0, 0.0, 0.0], atol=1e-3)
    # Walker phasing: plane p is rotated by 360 p / P degrees and shifted by 360 F p / N degrees along the orbit.
    times = np.zeros(1)
    positions = scenario.positions_eci(constellation, times)[0]
    for plane in (1, 5, 11):
        raan, shift = 2 * np.pi * plane / 12, 2 * np.pi * plane / 120
        inclination = np.deg2rad(53.0)
        expected = (R_EARTH_KM + 550.0) * np.array([np.cos(raan) * np.cos(shift) - np.sin(raan) * np.sin(shift) * np.cos(inclination),
                                                    np.sin(raan) * np.cos(shift) + np.cos(raan) * np.sin(shift) * np.cos(inclination),
                                                    np.sin(shift) * np.sin(inclination)])
        assert np.allclose(positions[plane * 10], expected, atol=1e-6)


def test_grid_edges(s1):
    roles = s1.roles
    assert s1.edges.shape == (240, 2) and s1.edges.dtype == np.int16 and (s1.edges[:, 0] < s1.edges[:, 1]).all()
    assert [int((s1.edge_kind == kind).sum()) for kind in (EDGE_IN_PLANE, EDGE_INTER_PLANE, EDGE_LASER)] == [120, 108, 12]
    assert len({tuple(edge) for edge in s1.edges.tolist()}) == 240
    for index, (a, b) in enumerate(s1.edges.tolist()):
        kind = int(s1.edge_kind[index])
        if kind == EDGE_IN_PLANE:
            assert roles.plane(a) == roles.plane(b) and (roles.slot(a) - roles.slot(b)) % 10 in (1, 9)
        else:
            assert roles.slot(a) == roles.slot(b) and (roles.plane(a) - roles.plane(b)) % 12 in (1, 11)
            assert (kind == EDGE_LASER) == (roles.slot(a) == 0)
    # Every satellite has four links, and the roots form one ring that includes the seam between planes 11 and 0.
    degree = np.bincount(s1.edges.reshape(-1).astype(int), minlength=120)
    assert (degree == 4).all()
    laser = sorted(tuple(edge) for edge in s1.edges[s1.edge_kind == EDGE_LASER].tolist())
    assert laser == sorted(tuple(sorted((10 * p, 10 * ((p + 1) % 12)))) for p in range(12))
    assert s1.edge(0, 110) == s1.edge(110, 0)


def test_statistics_agree_with_appendix_a(stats):
    rows = check_against_appendix(stats)
    assert len(rows) == len(scenario.APPENDIX_A_S1) - 1  # the imaging share needs the real land mask
    assert all(ok for *_, ok in rows), [(name, value, reference) for name, value, reference, _, ok in rows if not ok]
    assert stats["all_edges_up"] and stats["edges"] == {"in_plane": 120, "inter_plane_rf": 108, "laser": 12}
    assert stats["ground"]["1"]["rounds_without_reachable_plane"] == 0
    # A satellite is in view of one station in about 6% of the rounds, which is about 2.6% of the time.
    assert stats["ground"]["1"]["visible_share_of_slots"] == pytest.approx(0.026, abs=0.002)
    assert stats["ground"]["1"]["longest_wait_h_median"] == pytest.approx(14.5, abs=0.3)


def test_imaging_share_with_the_land_mask():
    pytest.importorskip("global_land_mask")
    arrays, meta = build_trace("s1", land="mask")
    stats = sanity_statistics(Trace(arrays, meta))
    assert meta["land_source"] == "global_land_mask"
    assert stats["hybrid_imaging_share"] == pytest.approx(0.185, abs=0.003)
    assert all(ok for *_, ok in check_against_appendix(stats))


def test_substitute_for_the_land_mask(s1):
    assert s1.meta["land_source"] == "duty_cycle"
    assert s1.land.mean() == pytest.approx(0.3, abs=0.01)
    blocks = s1.land.reshape(288, 10, 120)
    assert (blocks == blocks[:, :1]).all()  # constant within a block of ten slots
    latitude = np.zeros((40, 6))
    first, source = scenario.land_mask(latitude, latitude, 42, "duty_cycle")
    again, _ = scenario.land_mask(latitude, latitude, 42, "duty_cycle")
    other, _ = scenario.land_mask(latitude, latitude, 43, "duty_cycle")
    assert source == "duty_cycle" and (first == again).all() and (first != other).any()
    with pytest.raises(ValueError):
        scenario.land_mask(latitude, latitude, 42, "coastline")


def test_eclipse_is_the_cylinder_behind_the_earth():
    positions = np.array([[[7000.0, 0.0, 0.0], [-7000.0, 0.0, 0.0], [-7000.0, 6000.0, 0.0], [-7000.0, 0.0, 6400.0],
                           [-1.0, 6372.0, 0.0], [0.0, 0.0, 7000.0]]])
    assert scenario.sunlit_mask(positions).tolist() == [[True, False, False, True, True, True]]


def test_ground_station_visibility_uses_the_elevation():
    zenith = scenario.station_vector("cambridge")
    overhead = (R_EARTH_KM + 550.0) * zenith
    # A satellite on the horizon plane of the station is below ten degrees of elevation.
    east = np.cross([0.0, 0.0, 1.0], zenith)
    east /= np.linalg.norm(east)
    low = R_EARTH_KM * zenith + 2000.0 * east
    visible = scenario.ground_visibility(np.array([[overhead, low, -overhead]]), ["cambridge"])
    assert visible.tolist() == [[[True, False, False]]]
    assert scenario.slant_range_km(overhead, "cambridge") == pytest.approx(550.0)


def test_reachable_planes_rules(s1):
    first_last = reachable_planes(s1, ["cambridge"], 20, "first_last")
    any_time = reachable_planes(s1, ["cambridge"], 20, "any")
    assert first_last.shape == any_time.shape == (144, 12)
    assert (any_time | ~first_last).all()  # a plane seen at both ends of a round is seen at some time
    three = reachable_planes(s1, ["cambridge", "punta_arenas", "hartebeesthoek"], 20)
    assert (three | ~first_last).all() and three.sum() > first_last.sum()


def test_round_view(s1):
    view = s1.view(3, 20)
    assert (view.first_slot, view.last_slot) == (60, 79)
    assert (view.sunlit == s1.sunlit[60]).all() and (view.land == s1.land[60]).all()
    assert np.allclose(view.sun_seconds, s1.sunlit[60:80].sum(0) * 30.0) and view.sun_seconds.max() <= 600.0
    assert view.edge_dist_km.shape == (240,) and view.gs_first.shape == (3, 120)
    assert (view.gs_last == s1.gs_visible[:, 79]).all()
    with pytest.raises(ValueError):
        s1.view(144, 20)


def test_trace_is_deterministic_and_checked_on_load(s1, tmp_path):
    arrays, meta = build_trace("s1", land="duty_cycle")
    assert meta["trace_sha256"] == s1.sha256
    assert build_trace("s1", land="duty_cycle", seed=43)[1]["trace_sha256"] != s1.sha256  # the substitute uses the seed
    write_trace(tmp_path / "s1", arrays, meta, sanity_statistics(s1))
    loaded = load_trace(tmp_path / "s1")
    assert loaded.sha256 == s1.sha256 and (loaded.edge_dist_km == s1.edge_dist_km).all()
    assert {"trace.npz", "meta.json", "sanity.json"} == {path.name for path in (tmp_path / "s1").iterdir()}
    stored = json.loads((tmp_path / "s1" / "meta.json").read_text())
    assert stored["land_source"] == "duty_cycle" and stored["epoch"] == "2027-03-20T00:00:00Z"
    assert set(stored["arrays"]) == set(scenario.ARRAYS) and stored["software"]["numpy"] == np.__version__
    arrays["sunlit"] = ~arrays["sunlit"]
    np.savez_compressed(tmp_path / "s1" / "trace.npz", **arrays)
    with pytest.raises(ValueError):
        load_trace(tmp_path / "s1")


def test_trace_command_writes_and_checks(tmp_path, capsys):
    out = tmp_path / "trace"
    assert trace_main(["--output", str(out), "--land", "duty_cycle"]) == 0
    printed = capsys.readouterr().out
    assert "statistics agree with appendix A" in printed and "| NO |" not in printed
    assert (out / "sanity.json").exists()
    assert trace_main(["--output", str(out)]) == 0 and "trace exists" in capsys.readouterr().out
    short = tmp_path / "short"
    assert trace_main(["--output", str(short), "--land", "duty_cycle", "--hours", "2"]) == 0
    assert load_trace(short).slots == 240


def test_model_tables_state_the_numbers_of_appendix_a(s1, tmp_path, capsys):
    from ho_fl.sat.cli import model_tables

    tables = model_tables(s1)
    assert "4277 km 19.75 Mbps" in tables and "550 km 3.035 Gbps" in tables
    assert "| t1 first-order update, cut 0 | 4358144 | 1.781 |" in tables and "| t2 first-order update, cut 12 | 30760 |" in tables
    rows = {(cells[1], cells[2]): cells for cells in ([cell.strip() for cell in line.split("|")] for line in tables.splitlines())
            if len(cells) == 10 and cells[2].isdigit()}
    assert float(rows["t1", "20"][6]) == 480.0 and float(rows["t1", "11"][6]) == pytest.approx(212.4, rel=0.02)
    # Qwen at 11 dBi: the idle cut is late from four hops, the next cut is in time, also in eclipse.
    assert rows["t1", "11"][7].count("LATE") == 2 and "LATE" not in rows["t1", "11"][8] and "LATE" not in rows["t1", "20"][7]
    # The ViT update is small: nothing is late at 13 or 11 dBi; it takes 8 dBi.
    assert all("LATE" not in rows["t2", gain][7] and float(rows["t2", gain][6]) == 480.0 for gain in ("20", "13", "11"))
    assert "LATE" in rows["t2", "8"][7]
    assert "| t1 | Z | 28 | on | 90.00 / no power | 0.2500 |" in tables and "| t2 | H | 3 | on | 4.50 / 18.00 |" in tables
    assert "without any training: hybrid satellites 0.850, forward-only satellites 0.875" in tables
    assert "has power for: hybrid satellites 0.844, forward-only satellites 0.869" in tables
    assert trace_main(["--output", str(tmp_path / "t"), "--land", "duty_cycle", "--tables"]) == 0
    assert "Link, timing and energy numbers of the model" in capsys.readouterr().out
