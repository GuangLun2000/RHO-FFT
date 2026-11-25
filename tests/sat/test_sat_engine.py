"""Replica groups and whole runs of the satellite engine on the tiny assets (CPU, a few seconds each)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ho_fl.sat.engine as engine
from ho_fl.aggregation import Update, aggregate, apply
from ho_fl.io import tensor_hash
from ho_fl.model import block_index, register_head
from ho_fl.sat.aggregate import Groups, Term, channel_statistics, extension, first_order_turn, replica_difference, step_length, sum_terms, weigh_terms
from ho_fl.sat.backend import SatBackend
from ho_fl.sat.cli import main as cli_main
from ho_fl.sat.client import train_participation
from ho_fl.sat.config import SatConfig
from ho_fl.sat.energy import Batteries
from ho_fl.sat.engine import run
from ho_fl.sat.links import LinkBudget
from ho_fl.sat.methods import METHODS, resolve
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


# --- the two channels of an update ----------------------------------------------------------------------------------------

def test_channel_statistics_split_the_update_by_channel():
    """Three participants on a model of two blocks and a head: a backbone satellite (first-order everywhere), a
    hybrid one with cut 1 and a forward-only one with cut 2 (zeroth-order below the cut, the head first-order)."""
    register_head(["head.w"], 2)
    low, high, head = "m.layers.0.w", "m.layers.1.w", "head.w"
    terms = [Term(0, {low: torch.tensor([3.0, 4.0]), high: torch.tensor([1.0, 0.0]), head: torch.tensor([2.0])}, {}),
             Term(1, {high: torch.tensor([1.0, 0.0]), head: torch.tensor([2.0])}, {low: torch.tensor([3.0, 4.0])}),
             Term(2, {head: torch.tensor([2.0])}, {low: torch.tensor([-3.0, 4.0]), high: torch.tensor([0.0, 5.0])})]
    before = [tensor.clone() for term in terms for part in (term.first_order, term.zeroth_order) for tensor in part.values()]
    stats = channel_statistics(terms)
    assert (stats["counted"], stats["blocks"], stats["fo_n"], stats["zo_n"]) == (3, [0, 1, 2], [1, 2, 3], [2, 1, 0])
    # Block 0: the first-order sum is (3, 4), the zeroth-order sum (0, 8); both are divided by the three participants.
    assert stats["fo_norm"] == pytest.approx([5 / 3, 2 / 3, 6 / 3]) and stats["zo_norm"] == pytest.approx([8 / 3, 5 / 3, 0.0])
    assert stats["cos"][0] == pytest.approx(0.8) and stats["cos"][1] == pytest.approx(0.0) and stats["cos"][2] is None
    # A diagnostic: the terms, which are the participants' own tensors, are left as they were.
    after = [tensor for term in terms for part in (term.first_order, term.zeroth_order) for tensor in part.values()]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert channel_statistics([]) is None and channel_statistics([Term(0, {}, {})]) is None


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
    # Who sends what to which block (``channels`` of a round): blocks 0 to 11 and the head as block 12.
    for name in METHOD_NAMES:
        for row in rounds[name]:
            stats, people, n = row["channels"], row["participants"], row["counted"]
            assert stats["counted"] == n and stats["blocks"] == list(range(13)) and stats["fo_n"][12] == n and stats["zo_n"][12] == 0
            assert all(value >= 0 for value in stats["fo_norm"] + stats["zo_norm"]) and stats["cos"][12] is None
            if name in ("m0", "m2"):      # everybody backpropagates through every block
                assert stats["fo_n"] == [n] * 13 and stats["zo_n"] == [0] * 13 and stats["cos"] == [None] * 13
                assert all(value > 0 for value in stats["fo_norm"]) and stats["zo_norm"] == [0.0] * 13
            elif name == "m4":            # every block forward-only, the head by backpropagation
                assert stats["fo_n"][:12] == [0] * 12 and stats["zo_n"][:12] == [n] * 12 and stats["cos"] == [None] * 13
                assert all(value > 0 for value in stats["zo_norm"][:12]) and stats["fo_norm"][:12] == [0.0] * 12
            else:                         # the full method: the first-order share of a block grows with its height
                idle = row["cut_histogram"].get("3", 0)
                assert stats["fo_n"][:3] == [people["B"]] * 3 and stats["zo_n"][:3] == [people["H"] + people["Z"]] * 3
                assert stats["fo_n"][3:6] == [people["B"] + idle] * 3 and stats["fo_n"][6:12] == [people["B"] + people["H"]] * 6
                assert stats["zo_n"][6:12] == [people["Z"]] * 6
                assert all(-1.0 <= value <= 1.0 for value in stats["cos"][:12] if value is not None)
    assert any(value is not None for row in rounds["m7"] for value in row["channels"]["cos"])


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
    assert "COMPLETED: t2 m2 seed 42" in printed and printed.count("EVAL ") == 2
    # Each evaluation is announced before it starts, with the number of examples it will go through.
    assert printed.count("EVALUATING at round") == 2 and "EVALUATING at round 0/" in printed and "examples of the" in printed
    assert read_json(out / "summary.json")["rounds"] == 3
    with pytest.raises(SystemExit):
        cli_main(["tune"])
    with pytest.raises(SystemExit):
        cli_main(["nothing"])


# --- variants of a method: who divides by what, and the step-length search ---------------------------------------------------

VARIANT_RUNS = {"m7 fo_mean": ("m7", "fo_mean"), "m7 zo_zeroed": ("m7", "zo_zeroed"), "m7 fo_mean,zo_zeroed": ("m7", "fo_mean,zo_zeroed"),
                "m7 line": ("m7", "fo_mean,line"), "m7 line,zo_zeroed": ("m7", "fo_mean,line,zo_zeroed"), "m2 fo_mean": ("m2", "fo_mean")}


@pytest.fixture(scope="module")
def variant_runs(make):
    """One simulated hour of the full method under every variant, and of the backbone-only method under ``fo_mean``."""
    result = {}
    for name, (method, variant) in VARIANT_RUNS.items():
        cfg = make(method=method, variant=variant)
        result[name] = (cfg, run(cfg))
    return result


def test_variants_are_switches_of_a_method(make):
    assert resolve("m7") is METHODS["m7"] and resolve("m7", ["zonly"]) is METHODS["m7"] and METHODS["m7"].upstream_rule
    changed = resolve("m7", ["fo_mean", "line", "zonly"])
    assert (changed.fo_average, changed.zo_weight, changed.zo_search, changed.upstream_rule) == ("senders", 1.0, True, False)
    assert (changed.key, changed.cut_rule, changed.zo, changed.label) == ("m7", "table", True, "RHO-FFT [fo_mean, line]")
    assert resolve("m4", ["zo_zeroed"]).zo_weight == 0.0 and not resolve("m2", ["fo_mean"]).upstream_rule
    for method, variant in (("m2", "line"), ("m0", "zo_zeroed")):   # nothing zeroth-order to act on
        with pytest.raises(ValueError):
            resolve(method, [variant])
        with pytest.raises(ValueError):
            make(method=method, variant=variant)
    with pytest.raises(ValueError):
        make(method="m7", variant="fo_mean,nothing")
    with pytest.raises(ValueError):
        make(method="m7", variant="line", q=3)   # the two probes per batch are the forward passes of q = 2 directions
    assert make(method="m7", variant="zonly,fo_mean,line").data_variant == "zonly"


def test_weigh_terms_divides_each_channel_by_its_own_number():
    """The three participants of the channel test. Block 0: first-order (3, 4) from one, zeroth-order (0, 8) from
    two. Block 1: first-order (2, 0) from two, zeroth-order (0, 5) from one. Head: first-order 6 from all three."""
    register_head(["head.w"], 2)
    low, high, head, idle = "m.layers.0.w", "m.layers.1.w", "head.w", "m.layers.1.v"
    state = {low: torch.zeros(2), high: torch.zeros(2), head: torch.zeros(1), idle: torch.zeros(2)}

    def terms():
        return [Term(0, {low: torch.tensor([3.0, 4.0]), high: torch.tensor([1.0, 0.0]), head: torch.tensor([2.0])}, {}),
                Term(1, {high: torch.tensor([1.0, 0.0]), head: torch.tensor([2.0])}, {low: torch.tensor([3.0, 4.0])}),
                Term(2, {head: torch.tensor([2.0])}, {low: torch.tensor([-3.0, 4.0]), high: torch.tensor([0.0, 5.0])})]

    def applied(method):
        update = weigh_terms(state, terms(), method)
        assert idle not in update.tensors   # nobody sent anything for it
        return {name: (tensor / update.divisor[name]).tolist() for name, tensor in update.tensors.items()}

    plain = applied(METHODS["m7"])
    reference = sum_terms(state, terms())
    assert plain == {name: (reference.tensors[name] / 3).tolist() for name in (low, high, head)}   # the rule of HO-FL
    assert applied(resolve("m7", ["zo_zeroed"])) == {low: [1.0, pytest.approx(4 / 3)], high: [pytest.approx(2 / 3), 0.0], head: [2.0]}
    own = applied(resolve("m7", ["fo_mean"]))
    assert own[low] == pytest.approx([3.0, 4.0 + 8 / 3]) and own[high] == pytest.approx([1.0, 5 / 3]) and own[head] == [2.0]
    assert applied(resolve("m7", ["fo_mean", "zo_zeroed"])) == {low: [3.0, 4.0], high: [1.0, 0.0], head: [2.0]}
    # A block without any first-order update keeps the zeroth-order mean over everyone.
    only_zo = weigh_terms(state, [Term(0, {head: torch.tensor([2.0])}, {low: torch.tensor([3.0, 6.0])})] * 3, resolve("m7", ["fo_mean"]))
    assert (only_zo.tensors[low] / only_zo.divisor[low]).tolist() == [3.0, 6.0] and only_zo.divisor[low] == 3
    assert weigh_terms(state, [], METHODS["m7"]) is None


def test_step_length_reads_three_losses():
    assert step_length([3.25, 1.25, 1.25]) == (pytest.approx(1.5), "minimum between the probes")    # (a - 1.5)^2 + 1
    assert step_length([3.0, 2.0, 1.0]) == (2.0, "minimum beyond the last probe")                    # still falling at the last probe
    assert step_length([25.0, 16.0, 9.0]) == (2.0, "minimum beyond the last probe")                  # (a - 5)^2: never past a measured point
    assert step_length([3.0, 2.4, 1.0]) == (2.0, "minimum beyond the last probe")                    # falling ever faster
    assert step_length([1.0, 2.0, 3.0]) == (0.0, "uphill") and step_length([1.0, 1.0, 1.0]) == (0.0, "uphill")
    assert step_length([1.0, float("nan"), 0.5]) == (0.0, "no valid probes") and step_length([1.0, 0.5, float("inf")])[0] == 0.0
    assert step_length(None) == (0.0, "no valid probes")


def test_extension_tops_up_the_fresh_step_by_at_most_one_previous_step():
    previous = {"a": torch.tensor([2.0, 0.0]), "b": torch.tensor([0.0])}
    along, across, back = {"a": torch.tensor([1.0, 0.0])}, {"a": torch.tensor([0.0, 7.0])}, {"a": torch.tensor([-1.0, 3.0])}
    assert extension(along, previous, 2.0) == (1.0, 0.5)                       # 2 - 0.5 would be 1.5: one previous step at most
    assert extension(along, previous, 1.2) == (pytest.approx(0.7), 0.5)        # the fresh step already covers half a previous step
    assert extension(along, previous, 0.3) == (0.0, 0.5) and extension(along, previous, 0.0) == (0.0, 0.5)   # never negative
    assert extension(across, previous, 0.8) == (pytest.approx(0.8), 0.0)
    assert extension(back, previous, 0.6) == (pytest.approx(0.6), -0.5)        # a fresh step pointing back is not made up for
    assert extension(along, None, 2.0) == (0.0, 0.0) and extension(along, {"a": torch.zeros(2)}, 2.0) == (0.0, 0.0)


def test_groups_keep_their_previous_step():
    state = {"a.layers.0.w": torch.zeros(2)}
    groups = Groups(state, planes=2)
    assert groups.step(0) is None
    step = {"a.layers.0.w": torch.ones(2)}
    groups.place(0, {"a.layers.0.w": torch.ones(2)}, step)
    first, second = groups.split(0, [[0], [1]])
    assert groups.step(second) is step and groups.state(second) is groups.state(first)
    groups.place(second, {"a.layers.0.w": torch.ones(2)}, {"a.layers.0.w": torch.full((2,), 2.0)})
    groups.merge_identical()
    assert groups.ids == [0, 1]          # the same state reached by different steps: the next search would differ
    groups.place(second, {"a.layers.0.w": torch.ones(2)}, {"a.layers.0.w": torch.ones(2)})
    groups.merge_identical()
    assert groups.ids == [0] and groups.step(0) is step


def test_first_order_turn_compares_two_rounds():
    register_head(["head.w"], 2)
    low, head = "m.layers.0.w", "head.w"
    first = [Term(0, {low: torch.tensor([1.0, 0.0]), head: torch.tensor([2.0])}, {}), Term(1, {head: torch.tensor([1.0])}, {low: torch.tensor([9.0, 9.0])})]
    turn, sums = first_order_turn(first)
    assert turn == {} and sums[low].tolist() == [1.0, 0.0] and sums[head].tolist() == [3.0]   # zeroth-order parts are not read
    second = [Term(0, {low: torch.tensor([1.0, 1.0]), head: torch.tensor([-4.0])}, {})]
    turn, _ = first_order_turn(second, sums)
    assert turn == {0: pytest.approx(0.5 ** 0.5), 2: pytest.approx(-1.0)}
    assert first_order_turn(None, sums) == ({}, {}) and first_order_turn([Term(0, {}, {})], sums)[0] == {}


def test_a_variant_that_changes_nothing_for_a_method_changes_no_bit(runs, variant_runs):
    """Backbone satellites only: every participant sends a first-order update for every tensor, so dividing each
    sum by its senders is dividing by everyone. The variant's code path ends in bitwise the default one's state."""
    assert variant_runs["m2 fo_mean"][1]["final_parameter_sha256"] == runs["m2"][1]["final_parameter_sha256"]
    assert variant_runs["m2 fo_mean"][1]["last_evaluation"] == runs["m2"][1]["last_evaluation"]


def test_variants_keep_the_runs_paired_and_do_what_they_say(runs, variant_runs):
    rounds = {name: read_jsonl(Path(cfg.output) / "rounds.jsonl") for name, (cfg, _) in variant_runs.items()}
    ledgers = {name: read_jsonl(Path(cfg.output) / "ledger.jsonl") for name, (cfg, _) in variant_runs.items()}
    full = {(row["round"], row["sid"]): row for row in read_jsonl(Path(runs["m7"][0].output) / "ledger.jsonl")}
    names = [name for name in VARIANT_RUNS if name.startswith("m7")]
    for name in names:   # the same satellites in the same rounds with the same cuts and the same lottery
        assert {(row["round"], row["sid"]): (row["cut"], row["lottery"], row["kind"]) for row in ledgers[name]} == \
            {key: (row["cut"], row["lottery"], row["kind"]) for key, row in full.items()}
        assert variant_runs[name][1]["initial_parameter_sha256"] == runs["m7"][1]["initial_parameter_sha256"]
    hashes = [variant_runs[name][1]["final_parameter_sha256"]["0"] for name in names] + [runs["m7"][1]["final_parameter_sha256"]["0"]]
    assert len(set(hashes)) == len(hashes)   # every variant changes the model
    for name in names:
        config = read_json(Path(variant_runs[name][0].output) / "config.json")
        switches = config["method_switches"]
        assert switches["fo_average"] == ("senders" if "fo_mean" in name or "line" in name else "all")
        assert switches["zo_weight"] == (0.0 if "zo_zeroed" in name else 1.0) and switches["zo_search"] == ("line" in name)
        assert "mean_all" not in config["aggregation"] and variant_runs[name][1]["variant"] == VARIANT_RUNS[name][1]
    # Without the search the participants compute what they always compute; a weight of zero only skips the replay.
    for name in ("m7 fo_mean", "m7 zo_zeroed", "m7 fo_mean,zo_zeroed"):
        assert all(row["zo"] == (row["cut"] > 0) and row["tau_s"] == full[row["round"], row["sid"]]["tau_s"] for row in ledgers[name])
        assert all(row["search"] is None for row in rounds[name])
        replayed = [sum(row["channels"]["zo_n"]) > 0 for row in rounds[name]]
        assert all(replayed) if "zo_zeroed" not in name else not any(replayed)
    # With the search only the satellites without backpropagation run a zeroth-order part: three losses per batch
    # instead of two random directions per batch, at the same compute time. The others leave the lower blocks alone.
    for name in ("m7 line", "m7 line,zo_zeroed"):
        for row in ledgers[name]:
            before = full[row["round"], row["sid"]]
            assert row["zo"] == (row["cut"] == 12) and row["report_bytes"] == (58 if row["zo"] else 18)
            assert row["tau_s"] == before["tau_s"] if row["kind"] in "BZ" else row["tau_s"] < before["tau_s"]
        assert all(sum(row["channels"]["zo_n"]) == 0 for row in rounds[name])   # no update along random directions any more
        for row in rounds[name]:
            search, forward_only = row["search"], row["participants"]["Z"]
            assert row["zo_reports"]["sent"] == row["zo_reports"]["used"] == forward_only
            assert 0.0 <= search["alpha"] <= 2.0 and 0.0 <= search["beta"] <= 1.0
            if row["round"] == 0:
                assert (search["reports"], search["examples"], search["loss"], search["beta"], search["case"]) == (0, 0, None, 0.0, "no previous step")
                continue
            assert (search["reports"], search["examples"]) == (forward_only, forward_only * 5 * 16) and len(search["loss"]) == 3
            assert search["alpha"] == step_length(search["loss"])[0] and search["case"] == step_length(search["loss"])[1]
            expected = min(1.0, max(0.0, search["alpha"] - max(search["kappa"], 0.0))) if name == "m7 line" else 0.0
            assert search["beta"] == pytest.approx(expected)
    assert any(row["search"]["beta"] > 0 for row in rounds["m7 line"])
    # The control takes the same probes and leaves them unused: until the first extension the two runs are the same run.
    first = next(row["round"] for row in rounds["m7 line"] if row["search"]["beta"] > 0)
    for a, b in zip(rounds["m7 line"][:first + 1], rounds["m7 line,zo_zeroed"][:first + 1]):
        assert a["task_loss"] == b["task_loss"] and a["search"]["loss"] == b["search"]["loss"] and a["search"]["kappa"] == b["search"]["kappa"]
    # How far the first-order sum turned since the previous round is logged for every block that has one in both rounds.
    for name, rows in rounds.items():
        assert rows[0]["channels"]["fo_cos_prev"] == [None] * 13
        assert all(-1.0 <= value <= 1.0 for row in rows[1:] for value in row["channels"]["fo_cos_prev"] if value is not None)
        assert all(row["channels"]["fo_cos_prev"][12] is not None for row in rows[1:])


def test_a_run_with_the_search_is_reproducible(make, variant_runs):
    cfg, summary = variant_runs["m7 line"]
    again = run(make(method="m7", variant="fo_mean,line"))
    assert again["final_parameter_sha256"] == summary["final_parameter_sha256"] and again["last_evaluation"]["accuracy"] == summary["last_evaluation"]["accuracy"]


def test_first_round_without_the_zeroth_order_part_is_the_first_order_mean(make):
    """One round of the full method with each tensor's first-order sum divided by its senders and nothing else
    added: bitwise the state computed here from the participants' own updates."""
    cfg = make(hours=1 / 6, method="m7", variant="fo_mean,zo_zeroed", server_lr=3.0)
    summary = run(cfg)
    b, trace = SatBackend(cfg), load_trace(cfg.trace_dir)
    view = trace.view(0, cfg.slots_per_round)
    budget = LinkBudget(trace, view, cfg.rf_gain_dbi, cfg.laser_gbps)
    decisions = decide(0, view, budget, timeline(cfg, budget, trace, b.full_bytes), Batteries(trace.roles), resolve("m7", cfg.variants), cfg, b, trace.roles)
    state = {name: param.detach().clone() for name, param in b.params.items()}
    reports = [train_participation(b, state, 0, sid, d.cut, zo=d.zo) for sid, d in sorted(decisions.items()) if d.participate]
    cuts = sorted({report.cut for report in reports})
    assert cuts == [0, 3, 12] or cuts == [0, 3, 6, 12]
    for name, tensor in state.items():
        senders = [report for report in reports if block_index(name) >= report.cut]
        total = torch.zeros_like(tensor)
        for report in senders:
            total.add_(report.delta[name])
        tensor.add_(total, alpha=cfg.server_lr / len(senders))
    assert tensor_hash(state) == summary["final_parameter_sha256"]["0"]


def test_the_search_runs_on_the_text_task(make):
    """No head on the text task: a forward-only satellite sends no first-order update at all, only its three losses."""
    cfg = make(task="t1", hours=0.5, method="m7", variant="fo_mean,line", eval_every=3)
    summary = run(cfg)
    rounds, ledger = read_jsonl(Path(cfg.output) / "rounds.jsonl"), read_jsonl(Path(cfg.output) / "ledger.jsonl")
    assert summary["status"] == "COMPLETED" and all(row["fo_bytes"] == 0 and row["zo"] for row in ledger if row["kind"] == "Z")
    assert all(row["search"]["reports"] == row["participants"]["Z"] for row in rounds[1:]) and rounds[1]["search"]["loss"] is not None
