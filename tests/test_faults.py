"""Fault injection, clean twins, influence on the aggregate, explicit schedules and divergence handling."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ho_fl import harness
from ho_fl.config import Config
from ho_fl.faults import Faults, influence_rows, layer_filter, role_members
from ho_fl.aggregation import aggregate
from ho_fl.reports import ClientReport
from ho_fl.scheduling import ScheduleFileScheduler, preferred_clients, tiers_of
from ho_fl.train import run

CUTS = [8] * 10 + [16] * 10 + [24] * 10          # Qwen tiers: clients 0-9 are T1, 10-19 T2, 20-29 T3


def write_spec(path, *entries):
    path.write_text(json.dumps({"faults": list(entries)}))
    return str(path)


def spec_cfg(tmp_path, *entries, **fields):
    return Config(model="qwen", dataset="ag_news", fault_spec=write_spec(tmp_path / "faults.json", *entries), **fields).resolve()


def make_report(cid=0, cut=8, depth=28, seed=0):
    generator = torch.Generator().manual_seed(seed)
    delta = {f"m.layers.{layer}.{part}": torch.randn(4, generator=generator) for layer in range(depth) for part in ("lora_A", "lora_B")}
    return ClientReport(cid=cid, round=1, cut=cut, tier=0, delta=delta, scalars=[], batches=[], losses=[], direction_master_seed=0)


# --- selection --------------------------------------------------------------------------------------------------------

def test_roles_follow_the_ranking_of_the_default_sampler():
    preferred = preferred_clients(CUTS, 6, 42)
    assert role_members("preferred_t1", CUTS, 6, 42) == preferred                     # six T1 devices, best-ranked first
    ordinary = role_members("ordinary_t1", CUTS, 6, 42)
    assert sorted(preferred + ordinary) == list(range(10)) and len(ordinary) == 4
    assert sorted(role_members("t2", CUTS, 6, 42)) == list(range(10, 20)) and sorted(role_members("t3", CUTS, 6, 42)) == list(range(20, 30))
    assert role_members("preferred_t1", CUTS, 6, 43) != preferred                     # the device depends on the seed only


def test_specification_is_resolved_and_checked(tmp_path):
    entry = {"select": {"role": "preferred_t1", "count": 1}, "rounds": [41, 80], "kind": "lr_scale", "params": {"scale": 20}}
    faults = Faults(spec_cfg(tmp_path, entry, {"select": {"by_id": [25]}, "rounds": [1, 3], "kind": "stale", "params": {"lag": 2}}), CUTS)
    device = preferred_clients(CUTS, 6, 42)[0]
    assert faults.entries[0]["clients"] == [device] and faults.max_lag == 2
    assert faults.before_local(40, device) is None and faults.before_local(81, device) is None
    assert faults.before_local(41, device) == {"kinds": ["lr_scale"], "lr_scale": 20.0}
    assert faults.before_local(2, 25) == {"kinds": ["stale"], "lag": 2} and faults.before_local(2, 24) is None
    assert faults.metadata()["faults"][0]["kind"] == "lr_scale" and len(faults.metadata()["sha256"]) == 64
    for bad in ({**entry, "kind": "collude"}, {**entry, "select": {"by_id": [30]}}, {**entry, "kind": "sign_flip", "params": {"layers": "S7"}}):
        with pytest.raises((ValueError, IndexError)):
            Faults(spec_cfg(tmp_path, bad), CUTS)
    with pytest.raises(NotImplementedError):
        Faults(spec_cfg(tmp_path, {**entry, "kind": "late"}), CUTS)
    with pytest.raises(ValueError):
        Config(dataset="squad", fault_spec="x.json").resolve()


def test_label_noise_maps_labels_reproducibly(tmp_path):
    entry = lambda **params: {"select": {"by_id": [3]}, "rounds": [1, 9], "kind": "label_noise", "params": params}  # noqa: E731
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    shift = Faults(spec_cfg(tmp_path, entry(rate=1.0, map="shift")), CUTS).before_local(1, 3)["label_map"]
    assert shift(labels, 0).tolist() == [1, 2, 3, 0, 1, 2, 3, 0]
    none = Faults(spec_cfg(tmp_path, entry(rate=0.0)), CUTS).before_local(1, 3)["label_map"]
    assert none(labels, 0).tolist() == labels.tolist()
    half = Faults(spec_cfg(tmp_path, entry(rate=0.5, map="random")), CUTS)
    first, again, other_step = half.before_local(1, 3)["label_map"](labels, 0), half.before_local(1, 3)["label_map"](labels, 0), half.before_local(1, 3)["label_map"](labels, 1)
    assert first.tolist() == again.tolist() and first.tolist() != other_step.tolist() and set(first.tolist()) <= {0, 1, 2, 3}


# --- report-level faults ----------------------------------------------------------------------------------------------

def test_layer_filter():
    cfg = Config(model="qwen").resolve()
    assert [layer for layer in range(28) if layer_filter("S1", cfg)(layer)] == list(range(8, 16))
    assert [layer for layer in range(28) if layer_filter([2, 5], cfg)(layer)] == [2, 3, 4]
    assert all(layer_filter(None, cfg)(layer) for layer in range(28))


def test_report_level_faults_change_the_selected_layers_and_keep_the_clean_twin(tmp_path):
    entry = lambda kind, **params: {"select": {"by_id": [0]}, "rounds": [1, 1], "kind": kind, "params": params}  # noqa: E731
    report = make_report()
    in_s1 = lambda name: 8 <= int(name.split(".")[2]) < 16  # noqa: E731
    flipped = Faults(spec_cfg(tmp_path, entry("sign_flip", scale=20, layers="S1")), CUTS).after_local(1, report)
    assert flipped.fault == "sign_flip" and flipped.clean is report.delta and flipped is not report
    assert all(torch.equal(flipped.delta[name], -20 * report.delta[name] if in_s1(name) else report.delta[name]) for name in report.delta)
    zero = Faults(spec_cfg(tmp_path, entry("noop")), CUTS).after_local(1, report)
    assert all(not tensor.any() for tensor in zero.delta.values()) and zero.clean is report.delta
    faults = Faults(spec_cfg(tmp_path, entry("gauss", scale=3, layers="S1")), CUTS)
    noisy, again = faults.after_local(1, report), faults.after_local(1, report)
    norm = lambda delta: sum(float(delta[name].double().square().sum()) for name in delta if in_s1(name)) ** 0.5  # noqa: E731
    assert norm(noisy.delta) == pytest.approx(3 * norm(report.delta), rel=1e-5)
    assert all(torch.equal(noisy.delta[name], again.delta[name]) for name in report.delta)
    assert all(torch.equal(noisy.delta[name], report.delta[name]) for name in report.delta if not in_s1(name))
    assert faults.after_local(2, report) is report and faults.after_local(1, replace(report, cid=1)).fault is None   # other round, other client


def test_influence_is_one_under_the_mean_and_larger_when_fewer_reports_vote(tmp_path):
    entry = {"select": {"by_id": [0]}, "rounds": [1, 1], "kind": "sign_flip", "params": {"scale": 1, "layers": "S1"}}
    faults = Faults(spec_cfg(tmp_path, entry), CUTS)
    reports = [faults.after_local(1, make_report(cid, cut, seed=cid)) for cid, cut in ((0, 8), (1, 8), (10, 16), (11, 16), (20, 24), (21, 24))]
    for name, expected in (("mean_all", 1.0), ("fo_mean", 3.0)):        # two of six reports are first-order on S1: 1/2 against 1/6
        fields = {"mean_all": {}, "fo_mean": {"agg_electorate": "fo", "agg_weight": "count"}}[name]
        cfg = Config(model="qwen", **fields).resolve()
        update, _ = aggregate(reports, None, cfg)
        rows = influence_rows(reports, update, cfg)
        assert [row["layer"] for row in rows] == list(range(8, 16)) and all(row["segment"] == 1 and row["clients"] == [0] for row in rows)
        assert all(row["relative"] == pytest.approx(expected, rel=1e-5) for row in rows), name
    assert influence_rows([make_report()], None, Config(model="qwen").resolve()) == []


# --- explicit schedules -------------------------------------------------------------------------------------------------

def test_schedule_file_scheduler(tmp_path):
    path = tmp_path / "schedule.jsonl"
    path.write_text('{"round": 1, "clients": [0, 5, 12], "cut_override": {"5": 16}}\n{"round": 2, "clients": [0, 20, 21]}\n')
    scheduler = ScheduleFileScheduler(path, CUTS, 3)
    plan = scheduler.sample(1)
    assert plan.clients == [0, 5, 12] and (plan.cut(0), plan.cut(5), plan.cut(12)) == (8, 16, 16)
    assert scheduler.sample(2).cut_override == {} and scheduler.inclusion()[0] == 1.0 and scheduler.probability(5) == pytest.approx(0.5 / 3)
    assert scheduler.metadata()["rounds"] == 2 and scheduler.metadata()["overridden_rounds"] == 1
    with pytest.raises(ValueError):
        scheduler.sample(3)
    for bad in ('{"round": 1, "clients": [0, 0, 1]}', '{"round": 1, "clients": [0, 1]}', '{"round": 1, "clients": [0, 1, 2], "cut_override": {"9": 8}}'):
        path.write_text(bad + "\n")
        with pytest.raises(ValueError):
            ScheduleFileScheduler(path, CUTS, 3)
    with pytest.raises(ValueError):
        Config(sampler="schedule_file").resolve()
    assert Config(sampler="schedule_file", schedule_file="x.jsonl").resolve().beta == 0.0


# --- end to end on the tiny model --------------------------------------------------------------------------------------

def rows(cfg, name):
    return harness.read_jsonl(Path(cfg.output) / name)


def tiny_schedule(tiny_config, tmp_path, rounds=2):
    """A fixed schedule for the tiny model: client ``f`` (a T1 device) takes part in every round."""
    probe = tiny_config("sst2", rounds=1, k=3, E=2)
    run(probe)
    cuts = json.loads((Path(probe.output) / "config.json").read_text())["resource_cuts"]
    tiers = tiers_of(cuts)
    f, strong, weak = tiers.index(0), [cid for cid, tier in enumerate(tiers) if tier == 0][1], tiers.index(2)
    path = tmp_path / "schedule.jsonl"
    path.write_text("".join(json.dumps({"round": r, "clients": [f, strong, weak], "cut_override": {str(strong): sorted(set(cuts))[1]}}) + "\n"
                            for r in range(1, rounds + 1)))
    return f, strong, sorted(set(cuts)), {"sampler": "schedule_file", "schedule_file": str(path), "k": 3, "E": 2, "rounds": rounds}


def test_runs_with_schedule_faults_and_twins(tiny_config, tmp_path):
    f, strong, cuts, schedule = tiny_schedule(tiny_config, tmp_path)
    clean = tiny_config("sst2", **schedule)
    clean_summary = run(clean)
    ledger = rows(clean, "ledger.jsonl")
    assert [row["client"] for row in ledger[:3]] == json.loads(Path(schedule["schedule_file"]).read_text().splitlines()[0])["clients"]
    assert {row["client"]: row["cut"] for row in ledger[:3]}[strong] == cuts[1] and ledger[1]["resource_cut"] == cuts[0]   # the override is logged
    assert all(row["fault"] is None for row in ledger) and not (Path(clean.output) / "influence.jsonl").exists()

    # A specification that is never active changes nothing.
    idle = tiny_config("sst2", **schedule, fault_spec=write_spec(tmp_path / "idle.json", {"select": {"by_id": [f]}, "rounds": [50, 60], "kind": "lr_scale", "params": {"scale": 20}}))
    idle_summary = run(idle)
    assert idle_summary["final_parameter_sha256"] == clean_summary["final_parameter_sha256"]
    assert all((Path(idle.output) / name).read_bytes() == (Path(clean.output) / name).read_bytes() for name in ("rounds.jsonl", "curves.jsonl", "ledger.jsonl"))

    flip = tiny_config("sst2", **schedule, fault_spec=write_spec(tmp_path / "flip.json", {"select": {"by_id": [f]}, "rounds": [1, 2], "kind": "sign_flip", "params": {"scale": 1, "layers": "S1"}}))
    flip_summary = run(flip)
    assert flip_summary["final_parameter_sha256"] != clean_summary["final_parameter_sha256"]
    assert [row["fault"] for row in rows(flip, "ledger.jsonl")] == ["sign_flip", None, None] * 2
    influence = rows(flip, "influence.jsonl")
    assert {row["round"] for row in influence} == {1, 2} and {row["segment"] for row in influence} == {1}
    assert all(row["relative"] == pytest.approx(1.0, rel=1e-4) and row["clients"] == [f] for row in influence)   # equal-weight mean
    assert json.loads((Path(flip.output) / "config.json").read_text())["faults"]["faults"][0]["clients"] == [f]

    # Training-level fault: the twin is retrained on the same batches without touching streams or counters.
    scaled = tiny_config("sst2", **schedule, fault_spec=write_spec(tmp_path / "lr.json", {"select": {"by_id": [f]}, "rounds": [1, 2], "kind": "lr_scale", "params": {"scale": 20}},
                                                                        {"select": {"by_id": [strong]}, "rounds": [2, 2], "kind": "stale", "params": {"lag": 1}}))
    run(scaled)
    assert [row["fault"] for row in rows(scaled, "ledger.jsonl")] == ["lr_scale", None, None, "lr_scale", "stale", None]
    assert [row["processed_input_tokens"] for row in rows(scaled, "rounds.jsonl")] == [row["processed_input_tokens"] for row in rows(clean, "rounds.jsonl")]
    scaled_influence = rows(scaled, "influence.jsonl")
    assert {row["round"] for row in scaled_influence} == {1, 2} and all(row["relative"] == pytest.approx(1.0, rel=1e-4) for row in scaled_influence)
    assert any(len(row["clients"]) == 2 for row in scaled_influence if row["round"] == 2)

    # Robust aggregation runs end to end and gives a different model.
    robust = tiny_config("sst2", **schedule, agg_electorate="fo", agg_stat="geomed", fo_quorum=2)
    assert run(robust)["final_parameter_sha256"] != clean_summary["final_parameter_sha256"]


def test_divergence_stops_the_run_or_raises(tiny_config, tmp_path):
    f, _, _, schedule = tiny_schedule(tiny_config, tmp_path, rounds=3)
    spec = write_spec(tmp_path / "huge.json", {"select": {"by_id": [f]}, "rounds": [2, 3], "kind": "lr_scale", "params": {"scale": 1e30}})
    stopping = tiny_config("sst2", **schedule, evaluate_every=1, fault_spec=spec, on_nonfinite="stop")
    summary = run(stopping)
    status = json.loads((Path(stopping.output) / "status.json").read_text())
    assert summary["status"] == "DIVERGED" and summary["diverged_round"] == 2 and status == {"status": "DIVERGED", "last_completed_round": 1, "diverged_round": 2}
    assert summary["round"] == 1 and 0 <= summary["accuracy"] <= 1                    # the last finite evaluation is kept
    assert harness.run_status(stopping.output) in harness.FINISHED
    raising = tiny_config("sst2", **schedule, fault_spec=spec)
    with pytest.raises((AssertionError, ValueError)):
        run(raising)
    assert harness.run_status(raising.output) == "FAILED"


def test_per_class_accuracy_is_logged_on_request(tiny_config):
    cfg = tiny_config("sst2", rounds=1, k=2, E=1, per_class_eval=True)
    run(cfg)
    curve = rows(cfg, "curves.jsonl")
    assert all(len(row["per_class_accuracy"]) == 2 and all(0 <= value <= 1 for value in row["per_class_accuracy"]) for row in curve)
    plain = tiny_config("sst2", rounds=1, k=2, E=1)
    run(plain)
    assert all("per_class_accuracy" not in row for row in rows(plain, "curves.jsonl"))
    assert [row["accuracy"] for row in rows(plain, "curves.jsonl")] == [row["accuracy"] for row in curve]
