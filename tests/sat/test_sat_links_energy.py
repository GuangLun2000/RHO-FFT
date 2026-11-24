"""Link rates, hop delays, compute time and batteries against appendix A of the brief."""
import numpy as np
import pytest

from ho_fl.sat import energy
from ho_fl.sat.config import CLASS_SPECS, TASKS
from ho_fl.sat.links import GROUND_LINK, LASER_TX_POWER_W, RF_TX_POWER_W, LinkBudget, hop_delay_s, inter_satellite_radio
from ho_fl.sat.scenario import EDGE_LASER, Trace, build_trace

QWEN_LORA_BYTES = 4 * 1089536           # 28 blocks of 38,912 LoRA parameters
QWEN_BLOCK_BYTES = 4 * 38912
IN_PLANE_KM = 4277.4
SLOWDOWN = {name: spec.slowdown for name, spec in CLASS_SPECS.items()}


@pytest.fixture(scope="module")
def s1():
    arrays, meta = build_trace("s1", land="duty_cycle")
    return Trace(arrays, meta)


def test_radio_rates_between_satellites():
    radio = inter_satellite_radio(20.0)
    for distance, mbps in ((2505, 39.0), (3803, 23.3), (4277, 19.8), (5000, 15.6)):
        assert radio.rate_bps(distance) / 1e6 == pytest.approx(mbps, rel=0.005)
    for gain, mbps in ((17, 6.36), (15, 2.70), (13, 1.11), (12, 0.70), (11, 0.45), (10, 0.28)):
        assert inter_satellite_radio(gain).rate_bps(IN_PLANE_KM) / 1e6 == pytest.approx(mbps, rel=0.015)
    rates = radio.rate_bps(np.array([2505.0, 5000.0]))
    assert rates.shape == (2,) and rates[0] > rates[1]


def test_ground_link_rates():
    for distance, gbps in ((550, 3.04), (1000, 2.60), (2000, 2.11)):
        assert GROUND_LINK.rate_bps(distance) / 1e9 == pytest.approx(gbps, rel=0.005)


def test_hop_delays():
    rate = inter_satellite_radio(20.0).rate_bps(IN_PLANE_KM)
    for cut, seconds in ((0, 1.78), (8, 1.28), (16, 0.77)):
        assert hop_delay_s(QWEN_LORA_BYTES - cut * QWEN_BLOCK_BYTES, rate, IN_PLANE_KM) == pytest.approx(seconds, abs=0.01)
    flood = hop_delay_s(58, rate, IN_PLANE_KM)
    assert flood == pytest.approx(0.0153, abs=0.0002) and 11 * flood == pytest.approx(0.17, abs=0.005)
    # Six laser hops over the longest root-to-root link: the exchange between the plane roots.
    assert 6 * hop_delay_s(QWEN_LORA_BYTES, 10e9, 3803.0) == pytest.approx(0.1, abs=0.01)
    # Transmission, propagation and one millisecond of processing.
    assert hop_delay_s(1000, 8000.0, 299.792458) == pytest.approx(1.0 + 0.001 + 0.001)


def test_link_budget_of_a_round(s1):
    budget = LinkBudget(s1, s1.view(0, 20))
    laser = s1.edge_kind == EDGE_LASER
    assert (budget.rate_bps[laser] == 10e9).all() and (budget.tx_power_w[laser] == LASER_TX_POWER_W).all()
    assert (budget.tx_power_w[~laser] == RF_TX_POWER_W).all()
    assert budget.rate_bps[s1.edge(1, 2)] / 1e6 == pytest.approx(19.75, abs=0.02)
    hop = budget.hop(1, 2, QWEN_LORA_BYTES)
    assert hop == pytest.approx(1.78, abs=0.01) and budget.hop(2, 1, QWEN_LORA_BYTES) == hop
    assert budget.path([4, 3, 2, 1, 0], QWEN_LORA_BYTES) == pytest.approx(4 * hop)
    assert budget.path([0], QWEN_LORA_BYTES) == 0
    assert budget.tx_energy_j(1, 2, QWEN_LORA_BYTES) == pytest.approx(RF_TX_POWER_W * 8 * QWEN_LORA_BYTES / budget.rate_bps[s1.edge(1, 2)])
    assert budget.worst_hop(np.flatnonzero(laser), QWEN_LORA_BYTES) < 0.02
    slow = LinkBudget(s1, s1.view(0, 20), rf_gain_dbi=11.0, laser_gbps=1.0)
    assert slow.rate_bps[s1.edge(1, 2)] / 1e6 == pytest.approx(0.45, rel=0.015) and (slow.rate_bps[laser] == 1e9).all()


def test_compute_time_and_energy():
    t1, t2 = TASKS["t1"], TASKS["t2"]

    def seconds(task, kind, sunlit, cut, zo=True):
        return energy.compute_time_s(kind, sunlit, cut, task.depth, 5, 2, zo, task.t_fwd, SLOWDOWN)

    # With two directions the work is three forward passes per step whatever the cut.
    for cut in (0, 8, 16, 24, 28):
        assert seconds(t1, "B", True, cut) == pytest.approx(0.6)
    assert seconds(t1, "H", True, 8) == pytest.approx(18.0) and seconds(t1, "H", False, 8) == pytest.approx(72.0)
    assert seconds(t1, "Z", True, 28) == pytest.approx(90.0) and seconds(t1, "Z", False, 28) is None
    assert seconds(t1, "B", False, 0) == pytest.approx(2.0)
    assert seconds(t1, "H", True, 8, zo=False) == pytest.approx(14.6, abs=0.05)
    assert seconds(t1, "H", True, 16, zo=False) == pytest.approx(11.1, abs=0.05)
    assert [seconds(t2, kind, True, cut) for kind, cut in (("B", 0), ("H", 3), ("Z", 12))] == pytest.approx([0.15, 4.5, 22.5])
    for kind, cut, zo, wh in (("B", 0, True, 0.167), ("H", 8, True, 0.200), ("Z", 28, True, 0.250), ("H", 8, False, 0.162),
                              ("H", 16, False, 0.124)):
        assert energy.compute_energy_j(kind, True, seconds(t1, kind, True, cut, zo)) / 3600 == pytest.approx(wh, abs=0.001)
    # In eclipse a hybrid satellite takes four times as long at a quarter of the power: the same energy.
    assert energy.compute_energy_j("H", False, seconds(t1, "H", False, 8)) == pytest.approx(energy.compute_energy_j("H", True, 18.0))
    assert energy.power_cap_w("Z", False) == 0 and energy.power_cap_w("B", False) == 300


def test_battery_rules(s1):
    batteries = energy.Batteries(s1.roles)
    backbone, hybrid, forward = 0, 2, 1
    assert batteries.capacity[[backbone, hybrid, forward]].tolist() == [0.0, 100.0, 40.0]
    assert batteries.share(backbone) is None and batteries.share(hybrid) == 1.0
    assert batteries.feasible(backbone, 1e12)
    assert batteries.feasible(hybrid, 70 * 3600) and not batteries.feasible(hybrid, 70 * 3600 + 1)
    assert batteries.feasible(forward, 28 * 3600) and not batteries.feasible(forward, 28 * 3600 + 1)
    spent = np.zeros(120)
    spent[hybrid] = 3600.0
    batteries.update(np.zeros(120), 600, spent)  # a round in the dark: platform load and one watt-hour of training
    assert batteries.level[hybrid] == pytest.approx(100 - 30 * 600 / 3600 - 1) and batteries.level[forward] == pytest.approx(40 - 10 * 600 / 3600)
    assert batteries.level[backbone] == 0
    batteries.update(np.full(120, 600.0), 600, np.zeros(120))  # a round in the sun refills, never above the capacity
    assert batteries.level[hybrid] == 100 and batteries.level[forward] == 40
    batteries.update(np.zeros(120), 600, np.full(120, 1e9))
    assert (batteries.level == 0).all()


@pytest.mark.parametrize("training", [False, True])
def test_battery_floor_over_a_day(s1, training):
    """Appendix A: over 24 h the charge never drops below about 85% (hybrid) and 87% (forward-only). Training in
    every round a satellite has power for, which is more than the lottery asks, moves that by less than a point."""
    batteries, task = energy.Batteries(s1.roles), TASKS["t1"]
    lowest = np.ones(120)
    for r in range(144):
        view = s1.view(r, 20)
        spent = np.zeros(120)
        for sid in range(120):
            kind = s1.roles.kind(sid)
            seconds = energy.compute_time_s(kind, bool(view.sunlit[sid]), 0 if kind == "B" else task.depth, task.depth, 5, 2,
                                            True, task.t_fwd, SLOWDOWN)
            if training and kind != "B" and seconds is not None:
                cost = energy.compute_energy_j(kind, bool(view.sunlit[sid]), seconds)
                assert batteries.feasible(sid, cost)
                spent[sid] = cost
        batteries.update(view.sun_seconds, 600, spent)
        lowest = np.minimum(lowest, batteries.level / np.maximum(batteries.capacity, 1.0))
    assert lowest[s1.roles.members("H")].min() == pytest.approx(0.85, abs=0.01)
    assert lowest[s1.roles.members("Z")].min() == pytest.approx(0.87, abs=0.01)
