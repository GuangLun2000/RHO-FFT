"""Replica groups and whole runs of the satellite engine on the tiny assets (CPU, a few seconds each)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ho_fl.sat.engine as engine
from ho_fl.aggregation import Update, aggregate, apply
from ho_fl.io import tensor_hash
from ho_fl.sat.aggregate import Groups, replica_difference
from ho_fl.sat.backend import SatBackend
from ho_fl.sat.cli import main as cli_main
from ho_fl.sat.client import train_participation
from ho_fl.sat.config import SatConfig
from ho_fl.sat.energy import Batteries
from ho_fl.sat.engine import run
from ho_fl.sat.links import LinkBudget
from ho_fl.sat.methods import METHODS
from ho_fl.sat.network import timeline
from ho_fl.sat.policy import decide
from ho_fl.sat.scenario import load_trace
from ho_fl.sat.testing import build_tiny_assets

METHOD_NAMES = ("m0", "m2", "m4", "m7")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return build_tiny_assets(tmp_path_factory.mktemp("sat-engine"))


@pytest.fixture(scope="module")
def make(assets, tmp_path_factory):
    """Factory for resolved tiny configurations with their own output directory."""
    root, counter = tmp_path_factory.mktemp("sat-runs"), iter(range(1000))

    def build(task="t2", hours=1.0, **fields):
        return SatConfig(**{**assets[task], "hours": hours, "output": str(root / f"run{next(counter)}"), **fields}).resolve()

    return build


@pytest.fixture(scope="module")
def runs(make):
    """One simulated hour (six rounds) of every method on the vision task, same seed."""
    result = {}
    for name in METHOD_NAMES:
        cfg = make(method=name)
        result[name] = (cfg, run(cfg))
    return result


# --- replica groups ------------------------------------------------------------------------------------------------------

def test_groups_are_copy_on_write():
    state = {"a.layers.0.w": torch.zeros(3), "a.layers.1.w": torch.ones(2)}
    groups = Groups(state, planes=4)
    assert groups.ids == [0] and groups.members(0) == [0, 1, 2, 3] and groups.state(groups.of_plane(2)) is state
    assert replica_difference(groups, state) == 0.0
    first, second = groups.split(0, [[0, 1, 2], [3]])
    assert (first, second) == (0, 1) and groups.state(second) is state and groups.members(second) == [3]
    update = Update({"a.layers.0.w": torch.full((3,), 4.0), "a.layers.1.w": torch.zeros(2)}, {"a.layers.0.w": 2, "a.layers.1.w": 2})
    groups.apply({first: update, second: None}, server_lr=3.0)
    # The updated group has new tensors; the old ones, still held by the other group, did not change.
    assert groups.state(first)["a.layers.0.w"].tolist() == [6.0, 6.0, 6.0] and groups.state(second) is state
    assert state["a.layers.0.w"].tolist() == [0.0, 0.0, 0.0]
    # Three planes are 6 away from the start in three coordinates, one plane did not move: mean 4.5.
    expected = ((0.75 * 3 * 1.5 ** 2 + 0.25 * 3 * 4.5 ** 2) / (3 * 4.5 ** 2)) ** 0.5
    assert replica_difference(groups, state) == pytest.approx(expected)
    groups.merge_identical()
    assert groups.ids == [0, 1]
    groups.apply({second: update}, server_lr=3.0)
    groups.merge_identical()
    assert groups.ids == [0] and groups.group_of == [0, 0, 0, 0] and replica_difference(groups, state) == 0.0
    with pytest.raises(AssertionError):
        groups.split(0, [[0, 1], [2]])


# --- whole runs ----------------------------------------------------------------------------------------------------------

def test_run_writes_its_files(runs):
    cfg, summary = runs["m7"]
    out = Path(cfg.output)
    assert sorted(path.name for path in out.iterdir()) == ["config.json", "eval.jsonl", "ledger.jsonl", "rounds.jsonl", "status.json", "summary.json"]
    assert read_json(out / "status.json") == {"status": "COMPLETED", "last_completed_round": 6} and summary == read_json(out / "summary.json")
    rounds, ledger, evals, config = read_jsonl(out / "rounds.jsonl"), read_jsonl(out / "ledger.jsonl"), read_jsonl(out / "eval.jsonl"), read_json(out / "config.json")
    assert [row["round"] for row in rounds] == list(range(6)) and [row["sim_hours"] for row in rounds][-1] == 1.0
    # Evaluations at whole simulated hours: before the first round and after the sixth.
    assert [(row["round"], row["sim_hours"]) for row in evals] == [(0, 0.0), (6, 1.0)]
    assert all(row["replica_groups"] == 1 and row["replica_difference"] == 0.0 and row["count"] == 360 for row in evals)
    assert {"accuracy", "macro_f1", "loss", "groups", "split", "scope", "wall_s"} <= set(evals[0])
    assert len(ledger) == sum(row["participant_count"] for row in rounds) == sum(summary["participations"].values())
    for row in rounds:
        assert row["counted"] == row["participant_count"] == sum(row["participants"].values()) == sum(row["cut_histogram"].values())
        assert all(row["participants"][kind] <= row["invited"][kind] for kind in "BHZ")
        assert sum(row["invited"].values()) == row["participant_count"] + sum(row["skipped"].values())
        assert row["deadline_s"] == 480.0 and row["first_order_late"] == 0 and row["zo_reports"]["dropped"] == 0
        assert row["zo_reports"]["sent"] == row["zo_reports"]["used"] and row["replica_groups"] == 1
        assert row["bytes"]["ground"] == 0 and row["bytes"]["rf"] > 0 and row["bytes"]["laser"] > 0
    keys = {"round", "sid", "kind", "plane", "slot", "cut", "zo", "imaging", "sunlit", "tau_s", "arrival_s", "first_order_used", "late",
            "fo_bytes", "report_bytes", "energy_compute_wh", "energy_transmit_wh", "battery_before", "battery_after", "train_loss", "fault"}
    assert all(keys <= set(row) for row in ledger)
    for row in ledger:
        assert row["cut"] == {"B": 0, "Z": 12}.get(row["kind"], 6 if row["imaging"] else 3) and row["zo"] == (row["cut"] > 0)
        assert row["first_order_used"] and not row["late"] and row["report_bytes"] == (58 if row["zo"] else 18)
        assert (row["kind"], row["plane"], row["slot"]) == ("BZHZHZHZHZ"[row["sid"] % 10], row["sid"] // 10, row["sid"] % 10)
        assert row["sunlit"] or row["kind"] != "Z"
    for key in ("trace_sha256", "partition_sha256", "manifest_sha256", "data_sha256", "git", "initial_parameter_sha256"):
        assert summary[key] == config[key] and summary[key]
    assert config["method_switches"]["cut_rule"] == "table" and config["first_order_bytes_by_cut"]["12"] == 4 * 330
    assert summary["status"] == "COMPLETED" and summary["last_evaluation"]["sim_hours"] == 1.0 and summary["final_evaluation"] is None
    assert summary["final_parameter_sha256"]["0"] != summary["initial_parameter_sha256"]
    assert set(summary["energy_wh_per_satellite_day"]) == {"compute", "transmit"} and summary["bytes_per_day"]["laser"] > 0
    # Wall times are logged next to the results: per participation, per round with its parts, and their sums.
    assert all(row["wall_s"] > 0 for row in ledger) and all(set(row["wall_parts_s"]) == {"train", "deliver", "aggregate", "log"} for row in rounds)
    assert all(sum(row["wall_parts_s"].values()) <= row["wall_s"] + 1e-6 for row in rounds)
    parts = summary["wall_breakdown_s"]
    assert set(parts) == {"startup", "train", "deliver", "aggregate", "log", "evaluation"} and sum(parts.values()) <= summary["wall_s"] + 1e-6
    assert parts["train"] == pytest.approx(sum(row["wall_parts_s"]["train"] for row in rounds))
    assert parts["train"] >= sum(row["wall_s"] for row in ledger) - 1e-6 and config["prompt_tokens"] is None
    assert load_trace(cfg.trace_dir).sha256 == summary["trace_sha256"]
    with pytest.raises(RuntimeError):
        run(cfg)  # a run never writes into a directory that holds something


def test_methods_are_paired(runs):
    """Same seed: same trace, partition, initial model, invitations and zeroth-order directions for every method."""
    summaries = {name: summary for name, (_, summary) in runs.items()}
    for key in ("trace_sha256", "partition_sha256", "data_sha256", "initial_parameter_sha256"):
        assert len({summary[key] for summary in summaries.values()}) == 1
    rounds = {name: read_jsonl(Path(cfg.output) / "rounds.jsonl") for name, (cfg, _) in runs.items()}
    assert len({json.dumps([row["invited"] for row in rows]) for rows in rounds.values()}) == 1
    ledgers = {name: {(row["round"], row["sid"]): row for row in read_jsonl(Path(cfg.output) / "ledger.jsonl")} for name, (cfg, _) in runs.items()}
    for name in ("m0", "m2", "m4"):
        common = set(ledgers[name]) & set(ledgers["m7"])
        assert common and all(ledgers[name][key]["lottery"] == ledgers["m7"][key]["lottery"]
                              and ledgers[name][key]["direction_master_seed"] == ledgers["m7"][key]["direction_master_seed"] for key in common)
    assert len({summary["final_parameter_sha256"]["0"] for summary in summaries.values()}) == 4  # the methods do differ


def test_methods_behave_as_specified(runs):
    ledgers = {name: read_jsonl(Path(cfg.output) / "ledger.jsonl") for name, (cfg, _) in runs.items()}
    rounds = {name: read_jsonl(Path(cfg.output) / "rounds.jsonl") for name, (cfg, _) in runs.items()}
    # Ideal centralised: everyone invited trains with full backpropagation; nothing travels and nothing is skipped.
    assert all((row["cut"], row["zo"], row["report_bytes"]) == (0, False, 18) for row in ledgers["m0"])
    assert all(row["bytes"] == {"rf": 0, "laser": 0, "ground": 0} and not any(row["skipped"].values()) for row in rounds["m0"])
    assert all(row["participants"] == row["invited"] for row in rounds["m0"])
    # In-orbit first-order: backbone satellites only, the others lack the memory.
    assert {row["kind"] for row in ledgers["m2"]} == {"B"} and all(row["cut"] == 0 for row in ledgers["m2"])
    assert all(row["skipped"]["memory"] == row["invited"]["H"] + row["invited"]["Z"] for row in rounds["m2"])
    assert all(row["zo_reports"]["sent"] == 0 for row in rounds["m2"])
    # Zeroth-order flooding: every block forward-only; the head is the only first-order payload.
    assert all((row["cut"], row["zo"], row["fo_bytes"]) == (12, True, 4 * 330) for row in ledgers["m4"])
    assert {row["kind"] for row in ledgers["m4"]} == {"B", "H", "Z"}
    # The same satellites take part in the full method and in zeroth-order flooding (same constraints).
    assert [(row["round"], row["sid"]) for row in ledgers["m4"]] == [(row["round"], row["sid"]) for row in ledgers["m7"]]
    assert sum(row["skipped"]["eclipse"] for row in rounds["m7"]) > 0 and all(row["skipped"]["memory"] == 0 for row in rounds["m7"])


def test_runs_are_reproducible_and_depend_on_seed_and_step(make, runs):
    cfg, summary = runs["m7"]
    repeat = make(method="m7")
    again = run(repeat)
    assert again["final_parameter_sha256"] == summary["final_parameter_sha256"] and again["last_evaluation"]["accuracy"] == summary["last_evaluation"]["accuracy"]
    strip = lambda rows: [{key: value for key, value in row.items() if not key.startswith("wall")} for row in rows]  # noqa: E731
    for name in ("rounds.jsonl", "ledger.jsonl", "eval.jsonl"):  # identical up to the measured wall time
        assert strip(read_jsonl(Path(cfg.output) / name)) == strip(read_jsonl(Path(repeat.output) / name))
    other_seed, larger_step = run(make(method="m7", seed=43)), run(make(method="m7", server_lr=3.0))
    assert other_seed["initial_parameter_sha256"] != summary["initial_parameter_sha256"]
    assert other_seed["partition_sha256"] != summary["partition_sha256"] and other_seed["trace_sha256"] == summary["trace_sha256"]
    assert larger_step["final_parameter_sha256"] != summary["final_parameter_sha256"] and larger_step["initial_parameter_sha256"] == summary["initial_parameter_sha256"]


@pytest.mark.parametrize("task, method", [("t1", "m7"), ("t2", "m7"), ("t2", "m0"), ("t1", "m4")])
def test_first_round_equals_central_aggregation(task, method, make):
    """E0-a item 4 through the whole engine: one round on the ideal network ends in bitwise the state that
    ``ho_fl.aggregation.aggregate`` + ``apply`` give for the same participants (the replay is exact on the CPU)."""
    cfg = make(task=task, hours=1 / 6, method=method, server_lr=3.0)
    summary = run(cfg)
    b, trace = SatBackend(cfg), load_trace(cfg.trace_dir)
    view = trace.view(0, cfg.slots_per_round)
    budget = LinkBudget(trace, view, cfg.rf_gain_dbi, cfg.laser_gbps)
    decisions = decide(0, view, budget, timeline(cfg, budget, trace, b.full_bytes), Batteries(trace.roles), METHODS[method], cfg, b, trace.roles)
    state = {name: param.detach().clone() for name, param in b.params.items()}
    reports = [train_participation(b, state, 0, sid, d.cut, zo=d.zo) for sid, d in sorted(decisions.items()) if d.participate]
    assert len(reports) >= 5 and tensor_hash(state) == summary["initial_parameter_sha256"]
    update, _ = aggregate(reports, None, replace(b.cfg, k=len(reports)))
    apply(state, update, cfg.server_lr)
    assert tensor_hash(state) == summary["final_parameter_sha256"]["0"]
    assert read_jsonl(Path(cfg.output) / "rounds.jsonl")[0]["participant_count"] == len(reports)


def test_forward_only_data_variant_reports_the_held_out_part(make):
    cfg = make(task="t1", variant="zonly")
    summary = run(cfg)
    evals = read_jsonl(Path(cfg.output) / "eval.jsonl")
    assert all({"held_out_accuracy", "held_out_macro_f1"} <= set(row) and row["groups"][0]["held_out_count"] > 0 for row in evals)
    assert summary["held_out"]["held_out_accuracy"] == evals[-1]["held_out_accuracy"] and summary["variant"] == "zonly"
    assert read_json(Path(cfg.output) / "config.json")["evaluation"]["held_out_examples"] == evals[0]["groups"][0]["held_out_count"]


def test_validation_split_is_used_for_tuning(make):
    summary = run(make(eval_split="validation"))
    assert summary["eval_split"] == "validation" and summary["last_evaluation"]["scope"] == "whole split"


def test_non_finite_values_end_a_run_as_diverged(make, monkeypatch):
    original = engine.train_participation

    def broken(b, state, r, sid, cut, zo=True):
        report = original(b, state, r, sid, cut, zo=zo)
        if r == 2:
            next(iter(report.delta.values())).fill_(float("nan"))
        return report

    monkeypatch.setattr(engine, "train_participation", broken)
    cfg = make(hours=1.0)
    summary = run(cfg)
    out = Path(cfg.output)
    assert summary["status"] == "DIVERGED" and summary["diverged_round"] == 2
    assert read_json(out / "status.json") == {"status": "DIVERGED", "last_completed_round": 2, "diverged_round": 2}
    # Nothing of the broken round is logged; the run keeps its last finite evaluation.
    assert [row["round"] for row in read_jsonl(out / "rounds.jsonl")] == [0, 1] and summary["last_evaluation"]["round"] == 0


def test_train_command(assets, tmp_path, capsys):
    arguments = [part for key, value in assets["t2"].items() if key != "tiny" for part in (f"--{key.replace('_', '-')}", str(value))]
    out = tmp_path / "cli-run"
    assert cli_main(["train", *arguments, "--tiny", "--method", "m2", "--hours", "0.5", "--eval-every", "3", "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "COMPLETED: t2 m2 seed 42" in printed and printed.count("EVAL") == 2
    assert read_json(out / "summary.json")["rounds"] == 3
    with pytest.raises(SystemExit):
        cli_main(["tune"])
    with pytest.raises(SystemExit):
        cli_main(["nothing"])
