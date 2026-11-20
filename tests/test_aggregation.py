"""The default aggregation path must equal the upstream arithmetic bit for bit."""
from types import SimpleNamespace

import pytest
import torch

from ho_fl.aggregation import aggregate, apply


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
    update, log = aggregate([report(delta) for delta in deltas], None, SimpleNamespace(k=k))
    apply(state, update)
    assert all(torch.equal(state[name], expected[name]) for name in state)
    assert log == {"electorate": "all", "stat": "mean", "weight": "cohort", "reports": k}
    assert update.divisor == {name: k for name in state}


def test_reports_are_accumulated_in_cohort_order():
    # In float32, 1e8 + 1 == 1e8, so the two orders give different sums.
    values = [1e8, 1.0, -1e8, 1.0]
    forward = [report({"w": torch.tensor([v])}) for v in values]
    update, _ = aggregate(forward, None, SimpleNamespace(k=4))
    assert update.tensors["w"].item() == 1.0
    update, _ = aggregate(forward[::-1], None, SimpleNamespace(k=4))
    assert update.tensors["w"].item() == 0.0


def test_aggregate_leaves_the_reports_untouched_and_divides_by_the_cohort_size():
    state, deltas = random_case(2)
    before = [{name: tensor.clone() for name, tensor in delta.items()} for delta in deltas]
    update, _ = aggregate([report(delta) for delta in deltas], None, SimpleNamespace(k=6))
    assert all(torch.equal(delta[name], saved[name]) for delta, saved in zip(deltas, before) for name in delta)
    start = {name: tensor.clone() for name, tensor in state.items()}
    apply(state, update)
    for name in state:
        assert torch.allclose(state[name] - start[name], (deltas[0][name] + deltas[1][name]) / 6, rtol=1e-4, atol=1e-6)


def test_partial_arrivals_are_rejected_until_implemented():
    _, deltas = random_case(2)
    with pytest.raises(NotImplementedError):
        aggregate([report(delta) for delta in deltas], {0: True}, SimpleNamespace(k=2))
