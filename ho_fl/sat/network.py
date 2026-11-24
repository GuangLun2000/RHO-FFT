"""What the network does with the reports of a round (brief v2, section 5.5). This is the ideal network: no link
or satellite fails; faults and the protection against them arrive in milestone P-S4.

Timeline of a round of ``T_r`` seconds:

- a participant finishes training after its compute time and floods its report (a header, plus the zeroth-order
  scalars if it has a zeroth-order part) to every satellite;
- its first-order update travels along the shorter side of its plane to the plane root, each satellite on the way
  adding its own; an update that reaches the root after the deadline ``D_up`` is not used this round (the
  participant still counts in the average);
- at ``D_up`` the roots exchange the plane sums over the laser ring, then every root sends the global sum down its
  plane. ``D_up = min(deadline, T_r - T_ring - T_broadcast)``, so the broadcast always ends within the round.

Arrival times use the size of a satellite's own update, not the growing partial sum on its path [brief: a stated
simplification]. Byte and energy counts use what is actually forwarded: a partial sum has the size of the lowest
cut merged into it, because it covers the blocks from that cut upwards.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .scenario import EDGE_IN_PLANE, EDGE_LASER


@dataclass
class Timeline:
    deadline_s: float     # D_up
    ring_s: float         # exchange of the plane sums between the roots
    broadcast_s: float    # the global sum from a root to the farthest satellite of its plane


def timeline(cfg, budget, trace, full_bytes):
    """Deadline of a round. The exchange and the broadcast are timed with the full update size."""
    roles = trace.roles
    ring = math.ceil(roles.planes / 2) * budget.worst_hop(np.flatnonzero(trace.edge_kind == EDGE_LASER), full_bytes)
    broadcast = roles.max_hops * budget.worst_hop(np.flatnonzero(trace.edge_kind == EDGE_IN_PLANE), full_bytes)
    return Timeline(min(cfg.deadline_s, cfg.round_s - ring - broadcast), ring, broadcast)


@dataclass
class Delivery:
    timeline: Timeline
    arrival_s: dict           # sid -> arrival of its first-order update at the plane root (only for those that send one)
    on_time: set              # sids whose first-order update is used
    late: list                # sids whose first-order update missed the deadline
    beacons: set              # sids every replica knows to have taken part: the size of this set divides the sum
    zo_received: set          # sids whose zeroth-order report every replica holds
    zo_dropped: list          # sids whose zeroth-order report no replica uses
    flood_s: float            # time from the start of the round until the last report has reached the last satellite
    bytes: dict               # bytes sent over "rf", "laser" and "ground" links
    tx_energy_j: np.ndarray   # energy every satellite spent sending, in joules
    components: list          # sets of planes whose roots exchanged their sums: one set of all planes when nothing fails


def plane_arms(roles, plane):
    """The two sides of a plane as paths from their far end to the root."""
    far = roles.per_plane // 2
    arms = [roles.path_to_root(roles.sid(plane, far))]
    if far + 1 < roles.per_plane:
        arms.append(roles.path_to_root(roles.sid(plane, far + 1)))
    return arms


def deliver(trace, budget, times, decisions, backend, method, round_s):
    """The outcome of a round's communication for the participants in ``decisions``."""
    roles = trace.roles
    participants = sorted(sid for sid, decision in decisions.items() if decision.participate)
    reporting = [sid for sid in participants if decisions[sid].zo]
    sent, energy = {"rf": 0, "laser": 0, "ground": 0}, np.zeros(roles.count)
    everyone = [list(range(roles.planes))]
    if method.exchange == "server":
        # A virtual server holds the model: nothing travels, nothing is late, nothing is spent on sending.
        senders = {sid for sid in participants if decisions[sid].fo_bytes}
        return Delivery(times, {}, senders, [], set(participants), set(reporting), [], 0.0, sent, energy, everyone)

    def account(sender, receiver, size):
        edge = trace.edge(sender, receiver)
        sent["laser" if budget.laser[edge] else "rf"] += size
        energy[sender] += budget.energy_per_byte[edge] * size

    # First-order updates towards the plane roots.
    cut = {sid: decisions[sid].cut for sid in participants if decisions[sid].fo_bytes}
    arrival = {sid: decisions[sid].tau_s + budget.path(roles.path_to_root(sid), decisions[sid].fo_bytes) for sid in cut}
    on_time = {sid for sid in cut if arrival[sid] <= times.deadline_s}
    for plane in range(roles.planes):
        for arm in plane_arms(roles, plane):
            lowest = None
            for sender, receiver in zip(arm, arm[1:]):
                if sender in cut:
                    lowest = cut[sender] if lowest is None else min(lowest, cut[sender])
                if lowest is not None:
                    account(sender, receiver, backend.fo_bytes(lowest))
    # Exchange between the roots: every plane sum goes both ways around the ring until each root has it.
    for plane in range(roles.planes):
        cuts = [cut[sid] for sid in on_time if roles.plane(sid) == plane]
        if not cuts:
            continue
        size = backend.fo_bytes(min(cuts))
        for direction, hops in ((1, roles.planes // 2), (-1, roles.planes - 1 - roles.planes // 2)):
            at = plane
            for _ in range(hops):
                following = (at + direction) % roles.planes
                account(roles.root(at), roles.root(following), size)
                at = following
    # The global sum from every root to the satellites of its plane.
    if on_time:
        size = backend.fo_bytes(min(cut[sid] for sid in on_time))
        for plane in range(roles.planes):
            for arm in plane_arms(roles, plane):
                for receiver, sender in zip(arm, arm[1:]):
                    account(sender, receiver, size)
    # Reports are flooded: a satellite passes a report on once over each of its links except the one it came by.
    received, dropped, flood = set(), [], 0.0
    if participants:
        ends = trace.edges.astype(int)
        delay = 8 * backend.zo_bytes() / budget.rate_bps + budget.distance_km * 1e3 / 299792458.0 + 1e-3
        graph = csr_matrix((delay, (ends[:, 0], ends[:, 1])), shape=(roles.count, roles.count))
        distance, previous = dijkstra(graph, directed=False, indices=participants, return_predecessors=True)
        nodes = np.arange(roles.count)
        links = np.bincount(ends.reshape(-1), minlength=roles.count)
        laser_links = np.bincount(ends[budget.laser].reshape(-1), minlength=roles.count)
        for row, sid in enumerate(participants):
            size = backend.zo_bytes() if decisions[sid].zo else backend.beacon_bytes
            others = nodes != sid
            came_by = trace.edge_matrix[previous[row, others], nodes[others]]
            energy[sid] += budget.node_energy_per_byte[sid] * size
            energy[others] += (budget.node_energy_per_byte[others] - budget.energy_per_byte[came_by]) * size
            over_laser = int(laser_links.sum() - budget.laser[came_by].sum())
            sent["laser"] += over_laser * size
            sent["rf"] += (int(links.sum()) - len(came_by) - over_laser) * size
            done = decisions[sid].tau_s + float(distance[row].max())
            flood = max(flood, done)
            if done > round_s:
                dropped.append(sid)   # cannot happen with the delays of this model; kept as a guard
            else:
                received.add(sid)
    return Delivery(times, arrival, on_time, sorted(set(cut) - on_time), received, {sid for sid in reporting if sid in received},
                    [sid for sid in reporting if sid not in received], flood, sent, energy, everyone)
