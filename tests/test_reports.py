"""Client reports and the layer and segment grouping helpers."""
import pytest
import torch

from ho_fl.reports import ClientReport, is_first_order, layer_groups, segment_bounds, segment_groups, segment_index


def lora_names(depth):
    return [f"base_model.model.model.layers.{i}.self_attn.{proj}.lora_{ab}.default.weight"
            for i in range(depth) for proj in ("q_proj", "v_proj") for ab in "AB"]


def test_layer_groups_collect_the_four_lora_tensors_of_each_layer():
    names = lora_names(12)
    groups = layer_groups(names)
    assert list(groups) == list(range(12))
    assert all(len(group) == 4 for group in groups.values())
    assert [name for group in groups.values() for name in group] == names


def test_segment_bounds_follow_the_tier_cuts():
    assert segment_bounds([8, 16, 24] * 10, 28) == [(0, 8), (8, 16), (16, 24), (24, 28)]
    assert segment_bounds([9, 3, 6], 12) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    # FedAvg (cut 0 everywhere) and DeComFL (cut = depth everywhere) have a single segment.
    assert segment_bounds([0] * 30, 12) == [(0, 12)]
    assert segment_bounds([12] * 30, 12) == [(0, 12)]


def test_segment_groups_and_index():
    names = lora_names(28)
    groups = segment_groups(names, [8, 16, 24], 28)
    assert [len(groups[i]) for i in range(4)] == [32, 32, 32, 16]
    bounds = segment_bounds([8, 16, 24], 28)
    assert [segment_index(layer, bounds) for layer in (0, 7, 8, 15, 16, 23, 24, 27)] == [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError):
        segment_index(28, bounds)


def test_first_order_suppliers_per_segment_match_the_tier_structure():
    cuts = [8] * 10 + [16] * 10 + [24] * 10
    suppliers = [sum(is_first_order(cut, low) for cut in cuts) for low, _ in segment_bounds(cuts, 28)]
    assert suppliers == [0, 10, 20, 30]
    assert all(is_first_order(0, layer) for layer in range(12))          # FedAvg: every component is first-order
    assert not any(is_first_order(12, layer) for layer in range(12))     # DeComFL: every component is zeroth-order


def test_client_report_train_loss_is_the_mean_of_the_step_losses():
    report = ClientReport(cid=3, round=1, cut=8, tier=0, delta={}, scalars=[], batches=[],
                          losses=[torch.tensor(1.0), torch.tensor(2.0), torch.tensor(6.0)], direction_master_seed=7)
    assert report.train_loss == 3.0 and report.fault is None
