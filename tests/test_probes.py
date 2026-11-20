"""Layer statistics and the counterfactual first-order probe, and the proof that probes are read-only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ho_fl.aggregation import aggregate
from ho_fl.batches import ClientStream
from ho_fl.config import Config
from ho_fl.probes import Probes, counterfactual_rows, first_order_delta, layer_stats
from ho_fl.train import Backend, run, train_client
from test_tiny_training import logs, read_jsonl


def cosine(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


def synthetic_reports():
    generator = torch.Generator().manual_seed(0)
    reports = []
    for cut in (3, 9, 9):
        delta = {f"m.layers.{layer}.lora_{ab}": torch.randn(5, generator=generator, dtype=torch.float64).float() for layer in range(12) for ab in "AB"}
        reports.append(SimpleNamespace(delta=delta, cut=cut))
    return reports


def layer_vector(report, layer):
    return torch.cat([report.delta[f"m.layers.{layer}.lora_A"], report.delta[f"m.layers.{layer}.lora_B"]]).double()


def test_layer_statistics_on_known_vectors():
    cfg = Config(k=3).resolve()
    reports = synthetic_reports()
    update, _ = aggregate(reports, None, cfg)
    rows = layer_stats(reports, None, update, cfg)
    assert [row["layer"] for row in rows] == list(range(12)) and [row["segment"] for row in rows] == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    v, w1, w2 = (layer_vector(report, 4) for report in reports)
    row = rows[4]
    assert (row["n_fo"], row["n_zo"], row["rule"], row["electorate"]) == (1, 2, "all", 3)
    assert row["fo_norm"] == pytest.approx(float(v.norm())) and row["fo_mean_norm"] == pytest.approx(float(v.norm()))
    assert row["zo_norm"] == pytest.approx(float((w1.norm() + w2.norm()) / 2)) and row["zo_mean_norm"] == pytest.approx(float(((w1 + w2) / 2).norm()))
    assert row["cos_fo_fo"] is None and row["cos_zo_zo"] == pytest.approx(cosine(w1, w2))
    assert row["cos_fo_zo"] == pytest.approx((cosine(v, w1) + cosine(v, w2)) / 2)
    assert row["update_norm"] == pytest.approx(float(((v + w1 + w2) / 3).norm()), rel=1e-5)
    assert (rows[0]["n_fo"], rows[0]["n_zo"], rows[0]["fo_norm"], rows[0]["cos_fo_zo"]) == (0, 3, None, None)
    assert (rows[10]["n_fo"], rows[10]["n_zo"], rows[10]["zo_mean_norm"]) == (3, 0, None)
    json.dumps(rows, allow_nan=False)


def test_layer_statistics_follow_the_aggregation_rule_and_skip_absent_components():
    reports = synthetic_reports()
    for report in reports:  # what zo_mode=off produces: exact zeros below the cut
        for name in report.delta:
            if int(name.split(".")[2]) < report.cut:
                report.delta[name] = torch.zeros(5)
    cfg = Config(k=3, zo_mode="off", agg_electorate="fo", agg_weight="count").resolve()
    update, _ = aggregate(reports, None, cfg)
    rows = layer_stats(reports, None, update, cfg)
    assert (rows[0]["rule"], rows[0]["n_fo"], rows[0]["n_zo"], rows[0]["update_norm"]) == ("empty", 0, 0, 0.0)
    assert (rows[4]["rule"], rows[4]["n_fo"], rows[4]["n_zo"], rows[4]["cos_fo_zo"]) == ("fo", 1, 0, None)
    assert rows[4]["update_norm"] == pytest.approx(float(layer_vector(reports[0], 4).norm()), rel=1e-5)  # the single first-order report, undiluted
    doubled = layer_stats(reports, None, update, Config(k=3, zo_mode="off", agg_electorate="fo", agg_weight="count", server_lr=2.0).resolve())
    assert doubled[4]["update_norm"] == pytest.approx(2 * rows[4]["update_norm"])
    json.dumps(rows, allow_nan=False)


def prepared(tiny_config, **fields):
    b = Backend(tiny_config("sst2", **fields))
    state = {name: param.detach().clone() for name, param in b.params.items()}
    streams = [ClientStream(client, b.cfg.batch_size) for client in b.manifest["clients"]]
    return b, state, streams


def test_counterfactual_of_a_first_order_client_is_its_own_update(tiny_config):
    b, state, streams = prepared(tiny_config, method="fedavg", E=3)
    report = train_client(b, state, streams, 1, 0, b.cuts[0])
    counterfactual = first_order_delta(b, state, report.batches)
    assert all(torch.equal(counterfactual[name], report.delta[name]) for name in state)
    rows = counterfactual_rows(report, counterfactual, b.cfg)
    pooled = [row for row in rows if row["scope"] == "pooled"]
    assert [row["group"] for row in pooled] == ["fo"] and pooled[0]["cosine"] == pytest.approx(1.0) and pooled[0]["norm_ratio"] == pytest.approx(1.0)
    assert len([row for row in rows if row["scope"] == "layer"]) == 12 and pooled[0]["dimension"] == sum(p.numel() for p in state.values())


def test_counterfactual_separates_first_order_and_zeroth_order_layers(tiny_config):
    b, state, streams = prepared(tiny_config, E=1)
    cid = next(index for index, cut in enumerate(b.cuts) if cut == 6)
    tokens_before, seen_before = b.processed_tokens, [stream.batches_seen for stream in streams]
    report = train_client(b, state, streams, 1, cid, 6)
    tokens_after, seen_after = b.processed_tokens, [stream.batches_seen for stream in streams]
    update, _ = aggregate([report], None, b.cfg)
    Path(b.cfg.output).mkdir(parents=True)
    probes = Probes(Config(**{**vars(b.cfg), "probe_rounds": "1", "layer_stats": True}).resolve(), b.cfg.output)
    probes.run(1, b, state, [report], None, update)
    # Read-only: parameters are back at the global state, no gradient is left, counters and streams did not move.
    assert all(torch.equal(b.params[name], state[name]) and b.params[name].grad is None for name in state)
    assert b.processed_tokens == tokens_after > tokens_before and [stream.batches_seen for stream in streams] == seen_after != seen_before
    rows = read_jsonl(Path(b.cfg.output) / "probe.jsonl")
    layer_rows = {row["layer"]: row for row in rows if row["scope"] == "layer"}
    pooled = {row["group"]: row for row in rows if row["scope"] == "pooled"}
    assert all(row["round"] == 1 and row["client"] == cid and row["cut"] == 6 for row in rows)
    # With one local step the layers above the cut see the exact gradient: same step as the counterfactual.
    assert all(layer_rows[layer]["group"] == "fo" and layer_rows[layer]["cosine"] > 0.9999 for layer in range(6, 12))
    assert pooled["fo"]["cosine"] > 0.9999 and pooled["fo"]["norm_ratio"] == pytest.approx(1.0, abs=1e-3)
    # Below the cut the update is a two-direction estimate: far from the first-order direction.
    assert all(layer_rows[layer]["group"] == "zo" for layer in range(6)) and abs(pooled["zo"]["cosine"]) < 0.5
    assert set(pooled) == {"fo", "zo", "zo_lora_A", "zo_lora_B"}
    assert pooled["zo"]["dimension"] == pooled["zo_lora_A"]["dimension"] + pooled["zo_lora_B"]["dimension"]
    # LoRA B starts at zero, so the first-order gradient of every lora_A is zero at the first step.
    assert pooled["zo_lora_A"]["cosine"] is None and pooled["zo_lora_A"]["counterfactual_norm"] == pytest.approx(0.0, abs=1e-9)
    assert len(read_jsonl(Path(b.cfg.output) / "layers.jsonl")) == 12


@pytest.mark.parametrize("fields", [{}, {"agg_electorate": "fo", "agg_weight": "count", "beta": 0.0}, {"method": "decomfl"}])
def test_probes_do_not_change_a_run(tiny_config, fields):
    plain = tiny_config("sst2", rounds=3, evaluate_every=1, k=3, E=2, **fields)
    probed = tiny_config("sst2", rounds=3, evaluate_every=1, k=3, E=2, layer_stats=True, probe_rounds="1,3", **fields)
    assert run(plain) == run(probed)
    assert logs(plain) == logs(probed)
    layers, probe = read_jsonl(Path(probed.output) / "layers.jsonl"), read_jsonl(Path(probed.output) / "probe.jsonl")
    assert len(layers) == 3 * 12 and sorted({row["round"] for row in probe}) == [1, 3]
    assert len([row for row in probe if row["scope"] == "layer"]) == 2 * 3 * 12
    assert not (Path(plain.output) / "layers.jsonl").exists() and not (Path(plain.output) / "probe.jsonl").exists()
    if fields.get("agg_electorate") == "fo":
        assert {row["rule"] for row in layers} <= {"fo", "thin_all"} and {row["rule"] for row in layers if row["layer"] < 3} == {"thin_all"}
