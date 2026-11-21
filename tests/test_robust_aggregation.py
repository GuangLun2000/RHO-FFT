"""Robust statistics, aggregation units and the synthetic behaviour of appendix B of the brief."""
from types import SimpleNamespace

import pytest
import torch

from ho_fl.aggregation import aggregate, clip_scales, coordinate_median, geometric_median, plan_units, trimmed_mean, unit_groups
from ho_fl.config import NAMED_AGGREGATIONS, Config

NAME = "m.layers.8.w"   # one tensor of layer 8: first-order for a report with cut 8, zeroth-order for cut 16
DIMENSION, SIGNAL_SHARE, ZO_NORM = 20000, 0.4, 0.7


def named(name, **fields):
    return Config(**{"model": "qwen", **NAMED_AGGREGATIONS[name], **fields}).resolve()


# --- statistics -----------------------------------------------------------------------------------------------------

def test_coordinate_median_averages_the_two_middle_values_for_an_even_count():
    assert coordinate_median(torch.tensor([[1.0], [2.0], [3.0], [10.0]], dtype=torch.float64)).item() == 2.5
    assert coordinate_median(torch.tensor([[3.0, -1.0], [1.0, 5.0], [2.0, 0.0]], dtype=torch.float64)).tolist() == [2.0, 0.0]
    # Six zero-mean values: torch.median would return the lower middle one and bias the result downwards.
    values = torch.tensor([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    assert coordinate_median(values).item() == 0.0 and torch.median(values, dim=0).values.item() == -1.0


def test_trimmed_mean_drops_the_extremes_and_needs_three_rows():
    assert trimmed_mean(torch.tensor([[1.0], [2.0], [3.0], [100.0]], dtype=torch.float64)).item() == 2.5
    assert trimmed_mean(torch.tensor([[1.0], [100.0]], dtype=torch.float64)).item() == 50.5


def test_geometric_median_resists_an_outlier_and_handles_degenerate_inputs():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [100.0, 100.0]], dtype=torch.float64)
    assert geometric_median(points).norm() < 2.0 < points.mean(0).norm()
    assert torch.equal(geometric_median(points[:2]), points[:2].mean(0))            # two rows: the mean
    same = torch.ones(4, 3, dtype=torch.float64)
    assert torch.allclose(geometric_median(same), same[0]) and torch.isfinite(geometric_median(same)).all()


# --- units and clipping ---------------------------------------------------------------------------------------------

def lora_reports(cuts, depth=28):
    names = [f"m.layers.{layer}.{part}" for layer in range(depth) for part in ("lora_A", "lora_B")]
    generator = torch.Generator().manual_seed(0)
    return [SimpleNamespace(cut=cut, delta={name: torch.randn(3, generator=generator) for name in names}) for cut in cuts]


def test_units_are_layers_segments_or_the_whole_model():
    reports = lora_reports([8, 16, 24])
    names = list(reports[0].delta)
    assert [layer for layer, _ in unit_groups(names, named("mean_all"))] == list(range(28))
    segments = unit_groups(names, named("mean_all", agg_unit="segment"))
    assert [(layer, len(group)) for layer, group in segments] == [(0, 16), (8, 16), (16, 16), (24, 8)]
    assert [(layer, len(group)) for layer, group in unit_groups(names, named("robust_whole"))] == [(0, 56)]
    plans = plan_units(reports, None, named("fo_mean", agg_unit="segment"))
    assert [(plan.layer, plan.first_order, plan.rule) for plan in plans] == [(0, [], "thin_all"), (8, [0], "fo"), (16, [0, 1], "fo"), (24, [0, 1, 2], "fo")]
    (whole,) = plan_units(reports, None, named("robust_whole"))
    assert whole.members == [0, 1, 2] and whole.rule == "all" and whole.first_order == []
    # The probes always look layer by layer, whatever the aggregation unit.
    assert len(plan_units(reports, None, named("robust_whole"), unit="layer")) == 28


def test_whole_model_unit_needs_the_full_electorate_and_names_are_recognised():
    with pytest.raises(ValueError):
        Config(agg_unit="whole", agg_electorate="fo").resolve()
    assert Config(agg_electorate="fo", agg_stat="geomed", fo_quorum=3).resolve().aggregation_name == "quorum_robust"
    assert Config(agg_stat="geomed", agg_unit="whole").resolve().aggregation_name == "robust_whole"
    assert Config(agg_stat="cwmed", norm_clip=2.0).resolve().aggregation_name == "agg=all/cohort/quorum1/all/cwmed/clip2"
    assert Config(agg_stat="geomed").resolve().method_label == "HO-FL (β=0.75) [robust_all]"


def test_norm_clip_limits_a_component_to_a_multiple_of_the_median_norm():
    direction = torch.zeros(4)
    direction[0] = 1.0
    reports = [SimpleNamespace(cut=8, delta={NAME: scale * direction}) for scale in (1.0, 1.0, 1.0, 100.0)]
    cfg = named("mean_all", norm_clip=2.0, k=4)
    (plan,) = plan_units(reports, None, cfg)
    assert clip_scales(reports, plan, 2.0) == {0: 1.0, 1: 1.0, 2: 1.0, 3: pytest.approx(0.02)}
    update, _ = aggregate(reports, None, cfg)
    assert (update.tensors[NAME] / update.divisor[NAME])[0].item() == pytest.approx((1 + 1 + 1 + 2) / 4)


# --- appendix B: idealised model ------------------------------------------------------------------------------------

def unit_vector(generator):
    vector = torch.randn(DIMENSION, generator=generator, dtype=torch.float64)
    return vector / vector.norm()


def scenario(honest, zeroth, flip=None, seed=0):
    """Reports of one unit: an unreliable first-order one (if ``flip``), ``honest`` first-order ones, ``zeroth`` zeroth-order ones."""
    generator = torch.Generator().manual_seed(seed)
    signal = unit_vector(generator)
    component = lambda: SIGNAL_SHARE ** 0.5 * signal + (1 - SIGNAL_SHARE) ** 0.5 * unit_vector(generator)  # noqa: E731
    reports = [SimpleNamespace(cut=8, delta={NAME: -flip * component()})] if flip else []
    reports += [SimpleNamespace(cut=8, delta={NAME: component()}) for _ in range(honest)]
    reports += [SimpleNamespace(cut=16, delta={NAME: ZO_NORM * unit_vector(generator)}) for _ in range(zeroth)]
    return reports, signal


def applied(name, reports):
    update, _ = aggregate(reports, None, named(name))
    return update.tensors[NAME] / update.divisor[NAME]


def progress(name, honest, zeroth, flip=None, repeats=5):
    """Projection of the applied update on the signal direction, in units of one honest first-order report."""
    values = []
    for seed in range(repeats):
        reports, signal = scenario(honest, zeroth, flip, seed)
        values.append(float(applied(name, reports) @ signal) / SIGNAL_SHARE ** 0.5)
    return sum(values) / len(values)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_without_faults_the_mean_is_diluted_and_first_order_averaging_is_not(n):
    assert progress("mean_all", n, 6 - n) == pytest.approx(n / 6, abs=0.03)
    assert progress("fo_mean", n, 6 - n) == pytest.approx(1.0, abs=0.05)


def test_robust_statistics_match_the_reference_values_of_the_brief():
    assert progress("robust_all", 3, 3) == pytest.approx(0.47, abs=0.05)        # bounded but still diluted
    assert progress("quorum_robust", 3, 3) == pytest.approx(1.00, abs=0.05)
    assert progress("quorum_robust", 2, 4) == pytest.approx(0.28, abs=0.05)     # below the quorum: falls back to all reports
    assert progress("fo_robust", 2, 3, flip=1) == pytest.approx(0.60, abs=0.06)
    assert progress("robust_all", 2, 3, flip=20) == pytest.approx(0.18, abs=0.06)


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4])
def test_one_unreliable_first_order_report(m):
    assert progress("mean_all", m, 5 - m, flip=20) < -2
    assert abs(progress("robust_all", m, 5 - m, flip=20)) < 1
    if m <= 1:
        assert progress("fo_robust", m, 5 - m, flip=20) < -5       # no majority among the first-order reports
    # Quorum switching takes the branch that has a usable majority.
    reports, _ = scenario(m, 5 - m, flip=20)
    expected = "robust_all" if m <= 1 else "fo_robust"
    assert torch.equal(applied("quorum_robust", reports), applied(expected, reports))
