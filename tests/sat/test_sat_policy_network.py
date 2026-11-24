"""Participation decisions and the ideal network (brief sections 5.5, 5.6 and 6.1), on the sizes of the real tasks.

``Sizes`` stands in for the backend: decisions and deliveries only need the task and the size of an update per cut,
so the real Qwen and ViT sizes can be used without loading a model.
"""
import heapq
from types import SimpleNamespace

import numpy as np
import pytest

from ho_fl.sat.config import TASKS, SatConfig
from ho_fl.sat.energy import Batteries
from ho_fl.sat.links import LinkBudget, hop_delay_s
from ho_fl.sat.methods import METHODS, Method
from ho_fl.sat.network import deliver, plane_arms, timeline
from ho_fl.sat.policy import decide, lottery
from ho_fl.sat.scenario import Trace, build_trace

STATIC = Method("static", "table cuts without orbit-aware configuration", "table", True, "orbit", True, False, True)


class Sizes:
    beacon_bytes = 18

    def __init__(self, task, per_block, head=0):
        self.task, self.per_block, self.head = task, per_block, head

    def fo_bytes(self, cut):
        return 4 * ((self.task.depth - cut) * self.per_block + self.head)

    @property
    def full_bytes(self):
        return self.fo_bytes(0)

    def zo_bytes(self):
        return 58


QWEN = Sizes(TASKS["t1"], 38912)
VIT = Sizes(TASKS["t2"], 24576, head=7690)


@pytest.fixture(scope="module")
def s1():
    arrays, meta = build_trace("s1", land="duty_cycle")
    return Trace(arrays, meta)


def round_setup(s1, sizes, r=0, method="m7", gain=20.0, task="t1", batteries=None, **fields):
    cfg = SatConfig(task=task, rf_gain_dbi=gain, **fields).resolve()
    method = METHODS[method] if isinstance(method, str) else method
    view = s1.view(r, cfg.slots_per_round)
    budget = LinkBudget(s1, view, cfg.rf_gain_dbi, cfg.laser_gbps)
    times = timeline(cfg, budget, s1, sizes.full_bytes)
    decisions = decide(r, view, budget, times, batteries or Batteries(s1.roles), method, cfg, sizes, s1.roles)
    return SimpleNamespace(cfg=cfg, method=method, view=view, budget=budget, times=times, decisions=decisions)


# --- the lottery ---------------------------------------------------------------------------------------------------------

def test_lottery_is_a_shared_hash():
    draws = np.array([[lottery(42, r, sid) for sid in range(120)] for r in range(144)])
    assert draws.min() >= 0 and draws.max() < 1 and lottery(42, 3, 7) == draws[3, 7]
    assert lottery(43, 3, 7) != draws[3, 7] and lottery(42, 4, 7) != draws[3, 7] and lottery(42, 3, 8) != draws[3, 7]
    assert abs(draws.mean() - 0.5) < 0.01
    # About six invitations per class and round: 12 x 0.5, 48 x 0.125, 60 x 0.1.
    for slots, probability in ((range(0, 1), 0.5), (range(2, 10, 2), 0.125), (range(1, 10, 2), 0.1)):
        columns = [plane * 10 + slot for plane in range(12) for slot in slots]
        assert (draws[:, columns] < probability).sum(1).mean() == pytest.approx(6.0, abs=0.5)


def test_every_method_sees_the_same_invitations(s1):
    invited = {name: [sid for sid, d in round_setup(s1, QWEN, r=5, method=name).decisions.items() if d.invited] for name in METHODS}
    assert len(set(map(tuple, invited.values()))) == 1 and 6 <= len(invited["m7"]) <= 40


# --- decisions -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sizes, task", [(QWEN, "t1"), (VIT, "t2")])
def test_full_method_follows_the_cut_table(s1, sizes, task):
    spec, seen = TASKS[task], set()
    for r in range(0, 144, 3):
        setup = round_setup(s1, sizes, r=r, task=task)
        for sid, d in setup.decisions.items():
            if not d.invited:
                assert not d.participate and d.reason is None
                continue
            if d.kind == "Z" and not d.sunlit:
                assert not d.participate and d.reason == "eclipse"   # no power in eclipse
                seen.add("dark Z")
                continue
            assert d.participate and d.moved_up == 0 and d.on_time is not False
            expected = {"B": 0, "Z": spec.depth}.get(d.kind, spec.cut_imaging if d.imaging else spec.cut_idle)
            assert d.cut == expected and d.zo == (d.cut > 0) and d.fo_bytes == sizes.fo_bytes(d.cut)
            assert d.imaging == (d.kind == "H" and d.sunlit and bool(setup.view.land[sid]))
            assert d.hops == s1.roles.hops_to_root(sid)
            nominal = 5 * 3 * spec.t_fwd * {"B": 1, "H": 30, "Z": 150}[d.kind]
            slow = {"B": 1000 / 300, "H": 4.0}.get(d.kind, 1.0) if not d.sunlit else 1.0
            assert d.tau_s == pytest.approx(nominal * slow)
            seen.add(f"{d.kind} {'imaging' if d.imaging else 'sun' if d.sunlit else 'dark'}")
    assert {"dark Z", "B sun", "B dark", "H sun", "H dark", "H imaging", "Z sun"} <= seen


def test_methods_differ_in_who_takes_part_and_where_the_cut_is(s1):
    by_method = {name: round_setup(s1, QWEN, r=7, method=name).decisions for name in METHODS}
    invited = [sid for sid, d in by_method["m7"].items() if d.invited]
    # In-orbit first-order training needs the memory for full backpropagation: only backbone satellites have it.
    for sid in invited:
        d = by_method["m2"][sid]
        assert d.participate == (d.kind == "B") and (d.reason == "memory") == (d.kind != "B")
        assert not d.participate or (d.cut, d.zo) == (0, False)
    # Zeroth-order flooding: every block forward-only, nothing to send over the first-order channel for Qwen.
    assert all((d.cut, d.zo, d.fo_bytes) == (28, True, 0) for d in by_method["m4"].values() if d.participate)
    assert sorted(sid for sid, d in by_method["m4"].items() if d.participate) == sorted(sid for sid, d in by_method["m7"].items() if d.participate)
    # The ideal centralised method ignores memory, power and energy.
    assert all(by_method["m0"][sid].participate and (by_method["m0"][sid].cut, by_method["m0"][sid].zo) == (0, False) for sid in invited)
    head = round_setup(s1, VIT, r=7, method="m4", task="t2").decisions
    assert all((d.cut, d.fo_bytes) == (12, 4 * 7690) for d in head.values() if d.participate)


def test_low_battery_keeps_a_satellite_out(s1):
    batteries = Batteries(s1.roles)
    setup = round_setup(s1, QWEN, r=7)
    hybrid = next(sid for sid, d in setup.decisions.items() if d.participate and d.kind == "H")
    batteries.level[hybrid] = 30.0001  # just above the minimum: training would take it below
    again = round_setup(s1, QWEN, r=7, batteries=batteries).decisions[hybrid]
    assert not again.participate and again.reason == "energy"
    unconstrained = round_setup(s1, QWEN, r=7, method="m0", batteries=batteries).decisions[hybrid]
    assert unconstrained.participate


# --- deadline and lateness -----------------------------------------------------------------------------------------------

def test_deadline_follows_the_radio_rate(s1):
    """Appendix A for Qwen: 480 s down to 2.7 Mbps, then 441.4, 354.3 and 212.4 s at 13, 12 and 11 dBi."""
    for gain, expected in ((20, 480.0), (17, 480.0), (15, 480.0), (13, 441.4), (12, 354.3), (11, 212.4)):
        times = round_setup(s1, QWEN, gain=gain).times
        assert times.deadline_s == pytest.approx(expected, rel=0.02)
        assert times.ring_s == pytest.approx(0.1, abs=0.02)
    default = round_setup(s1, QWEN).times
    assert default.broadcast_s == pytest.approx(5 * 1.78, abs=0.05) and default.deadline_s == 480.0
    # The ViT update is 3.6 times smaller: its deadline stays at 480 s down to 11 dBi.
    assert all(round_setup(s1, VIT, gain=gain, task="t2").times.deadline_s == 480.0 for gain in (20, 13, 11))
    assert round_setup(s1, VIT, gain=8, task="t2").times.deadline_s < 200
    assert round_setup(s1, QWEN, deadline_s=300.0).times.deadline_s == 300.0


def test_only_distant_idle_hybrid_satellites_are_late_at_low_rate(s1):
    """Appendix A: at 0.45 Mbps the update of a hybrid satellite four hops from its root with cut 8 is late and
    everything else is on time. Orbit-aware configuration moves exactly those satellites to cut 16."""
    late, moved = [], []
    for r in range(144):
        static = round_setup(s1, QWEN, r=r, method=STATIC, gain=11.0).decisions
        adaptive = round_setup(s1, QWEN, r=r, gain=11.0).decisions
        for sid, d in static.items():
            if not d.participate:
                continue
            a = adaptive[sid]
            if d.on_time is False:
                late.append((d.kind, d.hops, d.cut))
                assert (a.cut, a.moved_up, a.on_time) == (16, 1, True) and a.arrival_s < d.arrival_s
                moved.append(sid)
            else:
                assert (a.cut, a.moved_up) == (d.cut, 0)
    assert late and set(late) == {("H", 4, 8)} and len(moved) == len(late)
    assert not [d for r in range(0, 144, 7) for d in round_setup(s1, QWEN, r=r, method=STATIC).decisions.values() if d.on_time is False]


# --- delivery ------------------------------------------------------------------------------------------------------------

def shortest_delays(trace, budget, origin, size):
    """Reference Dijkstra over the grid with the hop delays of a report."""
    best, previous, heap = {origin: 0.0}, {}, [(0.0, origin)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best[node]:
            continue
        for other in trace.roles.grid_neighbours(node):
            edge = trace.edge(node, other)
            candidate = distance + hop_delay_s(size, float(budget.rate_bps[edge]), float(budget.distance_km[edge]))
            if candidate < best.get(other, float("inf")) - 1e-15:
                best[other], previous[other] = candidate, node
                heapq.heappush(heap, (candidate, other))
    return best, previous


def test_bytes_energy_and_times_of_a_round(s1):
    setup = round_setup(s1, QWEN, r=7)
    roles, decisions = s1.roles, setup.decisions
    # Keep four participants of plane 0, at chosen cuts, so that every number can be written down.
    for d in decisions.values():
        d.participate = False
    chosen = {0: (0, False), 4: (8, True), 6: (16, True), 1: (28, True)}
    for sid, (cut, zo) in chosen.items():
        d = decisions[sid]
        d.participate, d.cut, d.zo, d.fo_bytes, d.tau_s = True, cut, zo, QWEN.fo_bytes(cut), {0: 0.6, 4: 18.0, 6: 18.0, 1: 90.0}[sid]
    result = deliver(s1, setup.budget, setup.times, decisions, QWEN, setup.method, 600)
    s0, s8, s16 = QWEN.fo_bytes(0), QWEN.fo_bytes(8), QWEN.fo_bytes(16)
    hop8, hop16 = setup.budget.hop(4, 3, s8), setup.budget.hop(6, 7, s16)
    assert result.arrival_s == pytest.approx({0: 0.6, 4: 18.0 + 4 * hop8, 6: 18.0 + 4 * hop16})  # the forward-only one sends none
    assert result.on_time == {0, 4, 6} and result.late == [] and result.beacons == {0, 1, 4, 6}
    assert result.zo_received == {1, 4, 6} and result.zo_dropped == [] and result.components == [list(range(12))]
    reports = 18 + 3 * 58
    # Up the two sides of plane 0 (four hops each), 11 laser hops for the one plane sum, the broadcast over nine
    # in-plane hops in each of the 12 planes; every report crosses 361 links (4 from its origin, 3 from the others).
    assert result.bytes["rf"] + result.bytes["laser"] == 4 * s8 + 4 * s16 + 11 * s0 + 108 * s0 + 361 * reports
    assert result.bytes["ground"] == 0 and 11 * s0 <= result.bytes["laser"] <= 11 * s0 + 24 * reports
    expected_energy, latest = np.zeros(120), 0.0
    for sender, receiver, size in ([(4, 3, s8), (3, 2, s8), (2, 1, s8), (1, 0, s8), (6, 7, s16), (7, 8, s16), (8, 9, s16), (9, 0, s16)]
                                   + [(10 * a, 10 * b, s0) for a, b in zip([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])]
                                   + [(10 * a, 10 * b, s0) for a, b in zip([0, 11, 10, 9, 8], [11, 10, 9, 8, 7])]):
        expected_energy[sender] += setup.budget.tx_energy_j(sender, receiver, size)
    for plane in range(12):
        for arm in plane_arms(roles, plane):
            for far, near in zip(arm, arm[1:]):
                expected_energy[near] += setup.budget.tx_energy_j(near, far, s0)
    for sid, (_, zo) in chosen.items():
        size = 58 if zo else 18
        delays, previous = shortest_delays(s1, setup.budget, sid, 58)
        latest = max(latest, decisions[sid].tau_s + max(delays.values()))
        for node in range(120):
            for other in roles.grid_neighbours(node):
                if previous.get(node) != other:
                    expected_energy[node] += setup.budget.tx_energy_j(node, other, size)
    assert np.allclose(result.tx_energy_j, expected_energy, rtol=1e-7, atol=1e-12)
    assert result.flood_s == pytest.approx(latest) and 90.0 < result.flood_s < 90.3   # a report crosses the grid in about 0.2 s


def test_late_updates_are_counted_but_not_used(s1):
    late_seen = False
    for r in range(144):
        setup = round_setup(s1, QWEN, r=r, method=STATIC, gain=11.0)
        result = deliver(s1, setup.budget, setup.times, setup.decisions, QWEN, setup.method, 600)
        participants = {sid for sid, d in setup.decisions.items() if d.participate}
        assert result.beacons == participants and result.on_time | set(result.late) == set(result.arrival_s)
        assert set(result.late) == {sid for sid in participants if setup.decisions[sid].on_time is False}
        assert all(result.arrival_s[sid] == pytest.approx(setup.decisions[sid].arrival_s) for sid in result.arrival_s)
        late_seen |= bool(result.late)
    assert late_seen


def test_a_virtual_server_needs_no_network(s1):
    setup = round_setup(s1, QWEN, r=7, method="m0")
    result = deliver(s1, setup.budget, setup.times, setup.decisions, QWEN, setup.method, 600)
    participants = {sid for sid, d in setup.decisions.items() if d.participate}
    assert result.on_time == result.beacons == participants and not result.late and not result.zo_received
    assert result.bytes == {"rf": 0, "laser": 0, "ground": 0} and not result.tx_energy_j.any() and result.arrival_s == {}
    empty = round_setup(s1, QWEN, r=7)
    for d in empty.decisions.values():
        d.participate = False
    nothing = deliver(s1, empty.budget, empty.times, empty.decisions, QWEN, empty.method, 600)
    assert not nothing.beacons and nothing.bytes == {"rf": 0, "laser": 0, "ground": 0} and nothing.flood_s == 0.0
