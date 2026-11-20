"""The default aggregation path must equal the upstream arithmetic bit for bit."""
from types import SimpleNamespace

import pytest
import torch

from ho_fl.aggregation import aggregate, apply
from ho_fl.config import Config


def cohort(k):
    """The default configuration with a cohort of ``k`` clients."""
    return Config(k=k).resolve()


def report(delta):
    return SimpleNamespace(delta=delta)


def upstream_update(state, deltas, k):
    """The aggregation of the upstream loop, written out as in HKU-WILL-Lab/ho-fl@ca74bed9."""
    delta_sum = {n: torch.zeros_like(p) for n, p in state.items()}
    for delta in deltas:
        for name in state:
            delta_sum[name].add_(delta[name])
    for name in state:
        state[name].add_(delta_sum[name], alpha=1 / k)
    return state


def random_case(k=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shapes = {"m.layers.0.lora_A": (8, 32), "m.layers.0.lora_B": (32, 8), "m.layers.1.lora_A": (8, 32)}
    state = {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}
    # Magnitudes spread over several orders, so the result depends on the order of the additions.
    deltas = [{name: torch.randn(shape, generator=generator) * 10.0 ** float(torch.randint(-6, 3, (1,), generator=generator))
               for name, shape in shapes.items()} for _ in range(k)]
    return state, deltas


@pytest.mark.parametrize("k", [1, 3, 6])
def test_default_path_is_bitwise_identical_to_the_upstream_loop(k):
    state, deltas = random_case(k)
    expected = upstream_update({name: tensor.clone() for name, tensor in state.items()}, deltas, k)
    update, log = aggregate([report(delta) for delta in deltas], None, cohort(k))
    apply(state, update)
    assert all(torch.equal(state[name], expected[name]) for name in state)
    assert log == {"electorate": "all", "stat": "mean", "weight": "cohort", "reports": k}
    assert update.divisor == {name: k for name in state}


def test_reports_are_accumulated_in_cohort_order():
    # In float32, 1e8 + 1 == 1e8, so the two orders give different sums.
    values = [1e8, 1.0, -1e8, 1.0]
    forward = [report({"w": torch.tensor([v])}) for v in values]
    update, _ = aggregate(forward, None, cohort(4))
    assert update.tensors["w"].item() == 1.0
    update, _ = aggregate(forward[::-1], None, cohort(4))
    assert update.tensors["w"].item() == 0.0


def test_aggregate_leaves_the_reports_untouched_and_divides_by_the_cohort_size():
    state, deltas = random_case(2)
    before = [{name: tensor.clone() for name, tensor in delta.items()} for delta in deltas]
    update, _ = aggregate([report(delta) for delta in deltas], None, cohort(6))
    assert all(torch.equal(delta[name], saved[name]) for delta, saved in zip(deltas, before) for name in delta)
    start = {name: tensor.clone() for name, tensor in state.items()}
    apply(state, update)
    for name in state:
        assert torch.allclose(state[name] - start[name], (deltas[0][name] + deltas[1][name]) / 6, rtol=1e-4, atol=1e-6)


def test_partial_arrivals_are_rejected_until_implemented():
    _, deltas = random_case(2)
    with pytest.raises(NotImplementedError):
        aggregate([report(delta) for delta in deltas], {0: True}, cohort(2))


# --- electorates and the planned (non-default) path ---------------------------------------------------------------

from ho_fl.aggregation import plan_units  # noqa: E402

LAYERS = 12


def cut_reports(cuts, seed=0, zo=True):
    """One report per cut with random components; with zo off the layers below the cut are exact zeros."""
    generator = torch.Generator().manual_seed(seed)
    reports = []
    for cut in cuts:
        delta = {f"m.layers.{layer}.lora_{ab}": torch.randn(4, 3, generator=generator) * (1.0 if zo or layer >= cut else 0.0)
                 for layer in range(LAYERS) for ab in "AB"}
        reports.append(SimpleNamespace(delta=delta, cut=cut))
    return reports


def setting(**fields):
    return Config(**fields).resolve()


def test_plan_orders_layers_numerically_and_finds_the_first_order_reports():
    plans = plan_units(cut_reports([3, 3, 6, 9, 9, 9]), None, setting(agg_electorate="fo", agg_weight="count"))
    assert [plan.layer for plan in plans] == list(range(LAYERS))
    assert [len(plan.first_order) for plan in plans] == [0] * 3 + [2] * 3 + [3] * 3 + [6] * 3
    assert plans[4].first_order == [0, 1] and plans[4].names == ["m.layers.4.lora_A", "m.layers.4.lora_B"]
    assert all(plan.arrived == list(range(6)) for plan in plans)


@pytest.mark.parametrize("fields, zo, expected", [
    # (rule, electorate size) on a layer with 0, 2 and 6 first-order reports
    ({}, "on", [("all", 6), ("all", 6), ("all", 6)]),
    ({"agg_electorate": "fo", "agg_weight": "count"}, "on", [("thin_all", 6), ("fo", 2), ("fo", 6)]),               # fo_mean
    ({"agg_electorate": "fo"}, "on", [("thin_all", 6), ("fo", 2), ("fo", 6)]),                                      # fo_diluted
    ({"agg_electorate": "fo", "fo_quorum": 3}, "on", [("thin_all", 6), ("thin_all", 6), ("fo", 6)]),
    ({"agg_electorate": "fo", "fo_quorum": 3, "thin_policy": "dilute"}, "on", [("hold", 0), ("dilute", 2), ("fo", 6)]),
    ({"agg_electorate": "fo", "fo_quorum": 3, "thin_policy": "hold"}, "on", [("hold", 0), ("hold", 0), ("fo", 6)]),
    # without zeroth-order updates only the first-order components exist
    ({}, "off", [("empty", 0), ("all", 2), ("all", 6)]),
    ({"agg_electorate": "fo", "agg_weight": "count"}, "off", [("empty", 0), ("fo", 2), ("fo", 6)]),
    ({"agg_electorate": "fo", "fo_quorum": 3}, "off", [("empty", 0), ("thin_all", 2), ("fo", 6)]),
])
def test_electorate_rules(fields, zo, expected):
    plans = plan_units(cut_reports([3, 3, 6, 9, 9, 9]), None, setting(zo_mode=zo, **fields))
    assert [(plans[layer].rule, len(plans[layer].members)) for layer in (0, 4, 10)] == expected


def summed(reports, name, members):
    total = torch.zeros_like(reports[0].delta[name])
    for index in members:
        total.add_(reports[index].delta[name])
    return total


def test_fo_mean_averages_the_first_order_reports_and_falls_back_to_all_reports():
    reports = cut_reports([3, 3, 6, 9, 9, 9])
    update, log = aggregate(reports, None, setting(agg_electorate="fo", agg_weight="count"))
    assert torch.equal(update.tensors["m.layers.4.lora_A"], summed(reports, "m.layers.4.lora_A", [0, 1])) and update.divisor["m.layers.4.lora_A"] == 2
    assert torch.equal(update.tensors["m.layers.7.lora_B"], summed(reports, "m.layers.7.lora_B", [0, 1, 2])) and update.divisor["m.layers.7.lora_B"] == 3
    # No first-order report below the lowest cut: the zeroth-order components are averaged as mean_all does.
    assert torch.equal(update.tensors["m.layers.0.lora_A"], summed(reports, "m.layers.0.lora_A", range(6))) and update.divisor["m.layers.0.lora_A"] == 6
    assert log["units"][4] == {"layer": 4, "arrived": 6, "first_order": 2, "rule": "fo", "members": 2, "divisor": 2}


def test_fo_diluted_keeps_the_cohort_weight_and_hold_skips_units():
    reports = cut_reports([3, 3, 6, 9, 9, 9])
    update, _ = aggregate(reports, None, setting(agg_electorate="fo"))
    assert torch.equal(update.tensors["m.layers.4.lora_A"], summed(reports, "m.layers.4.lora_A", [0, 1])) and update.divisor["m.layers.4.lora_A"] == 6
    held, log = aggregate(reports, None, setting(agg_electorate="fo", fo_quorum=3, thin_policy="hold"))
    assert sorted({int(name.split(".")[2]) for name in held.tensors}) == list(range(6, 12))
    state = {name: torch.zeros(4, 3) for name in reports[0].delta}
    apply(state, held)
    assert float(state["m.layers.4.lora_A"].abs().max()) == 0.0 and float(state["m.layers.6.lora_A"].abs().max()) > 0
    diluted, _ = aggregate(reports, None, setting(agg_electorate="fo", agg_weight="count", fo_quorum=3, thin_policy="dilute"))
    assert diluted.divisor["m.layers.4.lora_A"] == 6 and diluted.divisor["m.layers.10.lora_A"] == 6 and "m.layers.0.lora_A" not in diluted.tensors
    assert diluted.divisor["m.layers.7.lora_A"] == 3


def applied(reports, cfg):
    state = {name: torch.ones(4, 3) for name in reports[0].delta}
    update, _ = aggregate(reports, None, cfg)
    apply(state, update, cfg.server_lr)
    return state


def test_planned_path_equals_the_default_path_where_the_two_rules_coincide():
    # All reports first-order (FedAvg): fo_mean divides by K and accumulates in the same order.
    fedavg = cut_reports([0] * 6)
    expected = applied(fedavg, setting())
    assert all(torch.equal(applied(fedavg, setting(agg_electorate="fo", agg_weight="count"))[name], expected[name]) for name in expected)
    # Without zeroth-order updates fo_diluted is mean_all: the components it leaves out are exact zeros.
    mixed = cut_reports([3, 3, 6, 9, 9, 9], zo=False)
    expected = applied(mixed, setting(zo_mode="off"))
    actual = applied(mixed, setting(zo_mode="off", agg_electorate="fo"))
    assert all(torch.equal(actual[name], expected[name]) for name in expected)
    # With zeroth-order components present the two rules differ where first-order reports exist.
    noisy = cut_reports([3, 3, 6, 9, 9, 9])
    assert not torch.equal(applied(noisy, setting(agg_electorate="fo"))["m.layers.4.lora_A"], applied(noisy, setting())["m.layers.4.lora_A"])


def test_server_lr_multiplies_the_step_of_both_paths():
    reports = cut_reports([3, 3, 6, 9, 9, 9])
    for fields in ({}, {"agg_electorate": "fo", "agg_weight": "count"}):
        one, three = applied(reports, setting(**fields)), applied(reports, setting(server_lr=3.0, **fields))
        assert all(torch.allclose(three[name] - 1, 3 * (one[name] - 1), rtol=1e-5, atol=1e-6) for name in one)
