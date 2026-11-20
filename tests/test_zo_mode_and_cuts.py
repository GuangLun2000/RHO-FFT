"""zo_mode, cut overrides, server_lr, the cost model and the new ledger keys."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ho_fl import algorithms
from ho_fl.batches import ClientStream
from ho_fl.config import Config
from ho_fl.costs import forward_equivalents
from ho_fl.model import partition_parameters
from ho_fl.randomness import RoundDirections
from ho_fl.train import Backend, logical_uplink, run, train_client
from legacy_loop import legacy_run
from test_hybrid_step import CUT, ToyWrapper, full_loss, make_batch
from test_tiny_training import read_json, read_jsonl


def test_zo_off_step_leaves_the_prefix_untouched_and_keeps_the_first_order_part():
    cfg = Config().resolve()
    wrapper, batch = ToyWrapper(), make_batch()
    before = {name: param.detach().clone() for name, param in wrapper.params.items()}
    prefix, suffix = partition_parameters(wrapper.params, CUT)
    directions = RoundDirections(7)
    loss, scalars = algorithms.hybrid_step(cfg, wrapper, wrapper.params, prefix, batch, None,
                                           algorithms.optimizer_for(wrapper.params, cfg), directions, 0, zo=False)
    reference = ToyWrapper()
    expected_loss = full_loss(reference, batch)
    expected = dict(zip(reference.params, torch.autograd.grad(expected_loss, list(reference.params.values()))))
    assert scalars == [] and directions.used_seeds == {} and directions.cache == {}
    assert torch.allclose(loss, expected_loss.detach(), rtol=1e-6, atol=0)
    for name in prefix:  # no gradient, hence no AdamW step and no weight decay
        assert wrapper.params[name].grad is None and torch.equal(wrapper.params[name], before[name])
    for name in suffix:
        assert torch.allclose(wrapper.params[name].grad, expected[name], rtol=1e-5, atol=1e-7)
        assert not torch.equal(wrapper.params[name], before[name])


def test_forward_equivalents_reproduce_appendix_a():
    for model_depth, cuts, expected_off in ((28, (8, 16, 24), (12.14, 9.29, 6.43)), (12, (3, 6, 9), (12.5, 10.0, 7.5))):
        for cut, off in zip(cuts, expected_off):
            assert forward_equivalents("hofl", cut=cut, depth=model_depth, E=5, q=2) == pytest.approx(15.0)
            assert forward_equivalents("hofl", cut=cut, depth=model_depth, E=5, q=2, zo=False) == pytest.approx(off, abs=0.005)
    assert forward_equivalents("fedavg", cut=0, depth=28, E=5, q=2) == 15.0
    assert forward_equivalents("decomfl", cut=28, depth=28, E=5, q=2) == 25.0  # one clean and 2q perturbed forwards per step
    with pytest.raises(ValueError):
        forward_equivalents("sgd", cut=0, depth=12, E=5, q=2)


def test_uplink_has_no_scalar_bytes_without_zeroth_order_updates():
    cfg = Config().resolve()
    with_zo, without = logical_uplink(cfg, 294912, 73728), logical_uplink(cfg, 294912, 73728, zo=False)
    assert (with_zo["zo_scalar_bytes"], without["zo_scalar_bytes"]) == (40, 0)
    assert without["fo_delta_bytes"] == with_zo["fo_delta_bytes"] and without["logical_uplink_bytes"] == with_zo["logical_uplink_bytes"] - 40


def test_default_run_writes_the_new_ledger_and_config_keys(tiny_config):
    cfg = tiny_config("sst2", rounds=1, evaluate_every=1)
    run(cfg)
    ledger = read_jsonl(Path(cfg.output) / "ledger.jsonl")
    config = read_json(Path(cfg.output) / "config.json")
    assert all(row["forward_equivalents"] == pytest.approx(15.0) and row["fault"] is None for row in ledger)
    assert all(row["tier"] == [3, 6, 9].index(row["resource_cut"]) and row["cut"] == row["resource_cut"] for row in ledger)
    assert config["aggregation"] == "mean_all" and config["cost_model"]["backward_ratio"] == 2.0
    assert config["method_label"] == "HO-FL (β=0.75)"


def test_zo_off_run_trains_only_the_layers_above_each_cut(tiny_config):
    cfg = tiny_config("sst2", rounds=1, evaluate_every=1, zo_mode="off", k=3)
    summary = run(cfg)
    ledger = read_jsonl(Path(cfg.output) / "ledger.jsonl")
    assert summary["method"] == "HO-FL (β=0.75) [zo=off]"
    assert all(row["zo_scalar_bytes"] == 0 and row["forward_equivalents"] < 15.0 for row in ledger)
    b = Backend(tiny_config("sst2", zo_mode="off"))
    state = {name: param.detach().clone() for name, param in b.params.items()}
    streams = [ClientStream(client, b.cfg.batch_size) for client in b.manifest["clients"]]
    cid = next(index for index, cut in enumerate(b.cuts) if cut == 6)
    report = train_client(b, state, streams, 1, cid, 6)
    below, above = partition_parameters(report.delta, 6)
    assert report.zo is False and report.scalars == [[]] * b.cfg.E
    assert all(float(delta.abs().max()) == 0.0 for delta in below.values())  # exact zeros, not small numbers
    assert any(float(delta.abs().max()) > 0 for delta in above.values())


def test_cut_override_makes_a_client_train_like_another_tier(tiny_config):
    b = Backend(tiny_config("sst2", zo_mode="off"))
    state = {name: param.detach().clone() for name, param in b.params.items()}
    streams = [ClientStream(client, b.cfg.batch_size) for client in b.manifest["clients"]]
    cid = next(index for index, cut in enumerate(b.cuts) if cut == 9)
    report = train_client(b, state, streams, 1, cid, 3)
    moved = {int(name.split(".layers.")[1].split(".")[0]) for name, delta in report.delta.items() if float(delta.abs().max()) > 0}
    assert report.cut == 3 and report.tier == 2 and moved == set(range(3, 12))
    with pytest.raises(ValueError):
        train_client(b, state, streams, 1, cid, 4)  # not a tier cut of the model
    fedavg = Backend(tiny_config("sst2", method="fedavg"))
    with pytest.raises(ValueError):
        train_client(fedavg, state, streams, 1, cid, 3)  # overrides exist for hofl only


def test_server_lr_scales_the_server_step(tiny_config):
    base = tiny_config("sst2", rounds=1, evaluate_every=1, k=2, E=1)
    summaries = {lr: run(replace(base, output=f"{base.output}-{lr}", server_lr=lr).resolve()) for lr in (1.0, 2.0)}
    reference = legacy_run(replace(base, output=base.output + "-legacy"))
    assert summaries[1.0]["final_parameter_sha256"] == reference["final_parameter_sha256"]
    assert summaries[2.0]["final_parameter_sha256"] != reference["final_parameter_sha256"]
    assert summaries[2.0]["method"].endswith("[server_lr=2]")
