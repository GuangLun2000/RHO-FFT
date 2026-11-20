"""Logic of the P0 pipeline that does not need a run: plans, verdicts, budget decisions, report rendering."""
import importlib.util
import json
from pathlib import Path

import pytest

from ho_fl import harness

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p0", REPO / "experiments" / "p0" / "p0.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)

SST2 = ("opt", "sst2", "iid", "hofl", 0.75)
FEDAVG = ("opt", "ag_news", "dirichlet_05", "fedavg", 0.0)
UNIFORM = ("opt", "ag_news", "dirichlet_05", "hofl", 0.0)
BIASED = ("opt", "ag_news", "dirichlet_05", "hofl", 0.75)


def test_recorded_revisions_are_full_commit_hashes():
    assert set(p0.REVISIONS) == {"baseline", "instrumented"}
    assert all(len(sha) == 40 and set(sha) <= set("0123456789abcdef") for sha in p0.REVISIONS.values())


def test_smoke_configurations_are_those_of_the_brief():
    assert {name: fields["method"] for name, fields in p0.SMOKE.items()} == {"hofl": "hofl", "fedavg": "fedavg", "decomfl": "decomfl"}
    assert all((f["model"], f["dataset"], f["rounds"], f["evaluate_every"], f["seed"]) == ("opt", "sst2", 3, 3, 42) for f in p0.SMOKE.values())
    assert p0.SMOKE["hofl"]["beta"] == 0.75 and "beta" not in p0.SMOKE["fedavg"]
    assert harness.cli(p0.SMOKE["hofl"]) == ["--model", "opt", "--dataset", "sst2", "--method", "hofl", "--beta", "0.75",
                                             "--rounds", "3", "--evaluate-every", "3", "--seed", "42"]


def test_e0_plans():
    lite, full, opt = p0.e0_specs("lite", 160), p0.e0_specs("full", 160), p0.e0_specs("opt", 160)
    assert (len(opt), len(lite), len(full)) == (5, 9, 37)
    assert [s["name"] for s in p0.e0_specs("qwen", 160)] == [s["name"] for s in lite[5:]]
    assert len({s["name"] for s in full}) == 37
    assert [s["name"] for s in full[:9]] == [s["name"] for s in lite]
    first = lite[0]
    assert first["name"] == "opt-sst2-iid-hofl-b0.75-s42" and first["timing_key"] == "opt/sst2/iid/hofl"
    assert first["fields"] == {"model": "opt", "dataset": "sst2", "partition": "iid", "method": "hofl", "seed": 42,
                               "rounds": 160, "beta": 0.75, "save_state_rounds": "40"}
    fedavg = next(s for s in lite if s["fields"]["method"] == "fedavg" and s["fields"]["model"] == "qwen")
    assert "beta" not in fedavg["fields"] and "save_state_rounds" not in fedavg["fields"]
    assert {tuple(s["group"]) for s in full} == set(p0.REFERENCE)
    short = p0.e0_specs("opt", 3)
    assert short[0]["name"].endswith("-s42-r3") and "save_state_rounds" not in short[0]["fields"]
    seeds = {}
    for s in full:
        seeds.setdefault(s["fields"]["model"], set()).add(s["seed"])
    assert seeds == {model: set(values) for model, values in p0.PLANNED_SEEDS.items()}


def comparison(bitwise, difference, **overrides):
    return {"bitwise": bitwise, "hash_equal": bitwise, "max_abs_state_diff": difference, "shape_equal": True,
            "selection_equal": True, "same_start": True, **overrides}


BITWISE, NOISY, VERY_NOISY = comparison(True, 0.0), comparison(False, 3e-9), comparison(False, 1e-5)


@pytest.mark.parametrize("repeat, candidate, status, expected", [
    (BITWISE, comparison(True, 0.0), "COMPLETED", "PASS_BITWISE"),
    (BITWISE, comparison(False, 1e-9), "COMPLETED", "FAIL"),
    (NOISY, comparison(False, 5e-8), "COMPLETED", "PASS_TOLERANCE"),
    (NOISY, comparison(False, 5e-8, selection_equal=False), "COMPLETED", "FAIL"),
    (NOISY, comparison(False, 5e-8, shape_equal=False), "COMPLETED", "FAIL"),
    (NOISY, comparison(False, 1e-4), "COMPLETED", "FAIL"),
    (VERY_NOISY, comparison(False, 1e-4), "COMPLETED", "INCONCLUSIVE"),
    (NOISY, comparison(False, None), "COMPLETED", "INCONCLUSIVE"),
    (NOISY, comparison(True, 0.0), "COMPLETED", "PASS_BITWISE"),
    (BITWISE, comparison(True, 0.0), "FAILED", "RUN_FAILED"),
])
def test_i1_verdict(repeat, candidate, status, expected):
    assert p0.i1_verdict(repeat, candidate, status) == expected


class FakeContext:
    def __init__(self, root, rounds=160, budget=float("inf")):
        self.root, self.e0_root, self.tiny, self.head, self.hardware = root, root / "e0", False, "0" * 40, "gpu-x-torch0"
        self.golden_root, self.regress_root = root / "golden" / "x", root / "regress" / "y"
        self.timings_path = self.e0_root / "timings-gpu-x-torch0.json"
        self.plan, self.rounds, self.budget = "opt", rounds, budget

    def remaining(self):
        return self.budget

    def sync(self):
        pass


def run_name(group, seed, rounds=160):
    return next(s["name"] for s in p0.e0_specs("full", rounds) if tuple(s["group"]) == group and s["seed"] == seed)


def fake_run(root, name, group, seed, accuracy, rounds=160, ledger=()):
    model, dataset, partition, method, beta = group
    path = root / "e0" / "runs" / name
    harness.write_json(path / "status.json", {"status": "COMPLETED", "last_completed_round": rounds})
    harness.write_json(path / "config.json", {"model": model, "dataset": dataset, "partition": partition, "method": method, "beta": beta,
                                               "seed": seed, "rounds": rounds, "resource_cuts": [3] * 10 + [6] * 10 + [9] * 10,
                                               "bottom_dimensions": {"3": 73728, "6": 147456, "9": 221184}})
    harness.write_json(path / "summary.json", {"accuracy": accuracy / 100, "task_loss": 0.4, "final_parameter_sha256": "f" * 64})
    harness.write_json(path / "run_meta.json", {"status": "COMPLETED", "wall_s": 600.0, "revision": "abc", "hardware": "gpu-x-torch0"})
    rows = [{"round": 0, "accuracy": 0.5, "task_loss": 0.7, "training_task_loss": None, "cumulative_logical_uplink_bytes": 0},
            {"round": rounds, "accuracy": accuracy / 100, "task_loss": 0.4, "training_task_loss": 0.5, "cumulative_logical_uplink_bytes": 2_000_000_000}]
    (path / "curves.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))


def test_supply_from_ledger_counts_first_order_reports_per_segment(tmp_path):
    # Round 1: cuts 3, 3, 6, 9 -> widths 2 / 3 / 4. Round 2: cuts 6, 9, 9, 9 -> widths 0 / 1 / 4.
    ledger = [{"round": 1, "client": c, "cut": cut} for c, cut in ((0, 3), (1, 3), (10, 6), (20, 9))] \
        + [{"round": 2, "client": c, "cut": cut} for c, cut in ((10, 6), (20, 9), (21, 9), (22, 9))]
    fake_run(tmp_path, "run", UNIFORM, 42, 60.0, ledger=ledger)
    supply = p0.supply_from_ledger(tmp_path / "e0" / "runs" / "run", harness.read_json(tmp_path / "e0" / "runs" / "run" / "config.json"))
    assert supply["rounds"] == 2 and supply["mean_width"] == [1.0, 2.0, 4.0]
    assert (supply["s1_width_zero"], supply["s1_width_le_2"]) == (0.5, 1.0)
    assert supply["s1_top6_share"] == 1.0 and supply["s1_effective_sources"] == pytest.approx(2.0)
    assert supply["max_presence"] == 1.0  # clients 10 and 20 were selected in both rounds
    assert p0.supply_from_ledger(tmp_path / "missing", {"resource_cuts": [3, 6, 9]}) is None


def test_e0_groups_checklist_and_report(tmp_path):
    for seed, accuracy in zip(range(42, 47), (86.2, 86.4, 86.6, 86.8, 87.0)):
        fake_run(tmp_path, run_name(SST2, seed), SST2, seed, accuracy)
    fake_run(tmp_path, run_name(FEDAVG, 42), FEDAVG, 42, 60.0)
    fake_run(tmp_path, run_name(UNIFORM, 42), UNIFORM, 42, 58.0, ledger=[{"round": 1, "client": 0, "cut": 3}])
    fake_run(tmp_path, run_name(BIASED, 42), BIASED, 42, 61.0)
    # A renamed copy and a set-aside attempt must not be counted.
    fake_run(tmp_path, run_name(FEDAVG, 42) + "-old", FEDAVG, 42, 99.0)
    (tmp_path / "e0" / "runs" / (run_name(FEDAVG, 43) + ".stale-1")).mkdir()
    ctx = FakeContext(tmp_path)
    runs, groups = p0.e0_results(ctx)
    by_group = {tuple(g["group"]): g for g in groups}
    assert len(runs) == 8 and len(groups) == len(p0.REFERENCE)
    assert by_group[SST2]["verdict"] == "WITHIN" and by_group[SST2]["mean"] == pytest.approx(86.6)
    assert by_group[SST2]["tolerance"] == 1.0 and by_group[FEDAVG]["tolerance"] == pytest.approx(5.94)
    assert by_group[FEDAVG]["accuracies"] == [60.0] and by_group[FEDAVG]["verdict"] == "OUTSIDE_PRELIMINARY"
    assert by_group[UNIFORM]["verdict"] == "WITHIN_PRELIMINARY" and by_group[UNIFORM]["uplink_gb"] == pytest.approx(2.0)
    assert by_group[("qwen", "ag_news", "dirichlet_05", "fedavg", 0.0)]["verdict"] == "NOT_RUN"

    data = p0.collect(ctx)
    # One seed outside the tolerance is not a verdict; nothing has failed, but nothing can pass either.
    assert data["checklist"] == {"tests": "NOT_RUN", "I1": "NOT_RUN", "I2": "NOT_RUN", "E0": "INCOMPLETE", "G0": "INCOMPLETE"}
    assert data["used_gpu_hours"] == pytest.approx(9 * 600 / 3600)
    assert data["bottom_dimensions"]["opt"]["match"] is True
    text = p0.render(data)
    assert "## 1. Gate G0 checklist" in text and "OUTSIDE_PRELIMINARY" in text and "First-order supply" in text
    assert "HO-FL beta 0.75 minus beta 0 is +3.00 points" in text and "FedAvg minus HO-FL beta 0 is +2.00 points" in text

    for seed in range(43, 47):
        fake_run(tmp_path, run_name(FEDAVG, seed), FEDAVG, seed, 60.0)
    data = p0.collect(ctx)
    assert {tuple(g["group"]): g["verdict"] for g in data["e0"]["groups"]}[FEDAVG] == "OUTSIDE"
    assert data["checklist"]["E0"] == "CHECK" and data["checklist"]["G0"] == "BLOCKED"
    assert "Gate G0 is blocked" in p0.render(data)


def test_runs_with_another_number_of_rounds_are_not_compared(tmp_path):
    fake_run(tmp_path, run_name(SST2, 42, rounds=3), SST2, 42, 50.0, rounds=3)
    _, groups = p0.e0_results(FakeContext(tmp_path, rounds=3))
    assert {tuple(g["group"]): g["verdict"] for g in groups}[SST2] == "NOT_COMPARABLE"
    assert all(g["verdict"] == "NOT_RUN" for g in p0.e0_results(FakeContext(tmp_path))[1])


def test_checklist_states():
    def data(tests, verdicts, replay, groups, tiny=False, golden=None):
        return {"tiny": tiny, "tests": tests, "golden": golden,
                "regress": {"i1": {n: {"verdict": v} for n, v in verdicts.items()}, "i2": {"verdict": replay}},
                "e0": {"groups": [{"verdict": v} for v in groups]}}
    passing = {"tests": 88, "failures": 0, "errors": 0, "skipped": 0}
    good = {"hofl": "PASS_BITWISE", "fedavg": "PASS_TOLERANCE", "decomfl": "PASS_BITWISE"}
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN"] * 9))["G0"] == "PASS"
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN_PRELIMINARY"] * 5 + ["NOT_RUN"] * 4)) == \
        {"tests": "PASS", "I1": "PASS", "I2": "PASS", "E0": "INCOMPLETE", "G0": "INCOMPLETE"}
    assert p0.checklist(data(passing, {**good, "hofl": "FAIL"}, "PASS", ["WITHIN"] * 9))["G0"] == "BLOCKED"
    inconclusive = p0.checklist(data(passing, {**good, "hofl": "INCONCLUSIVE"}, "PASS", ["WITHIN"] * 9))
    assert (inconclusive["I1"], inconclusive["G0"]) == ("INCONCLUSIVE", "NEEDS_DECISION")
    assert p0.checklist(data({**passing, "failures": 1}, good, "PASS", ["WITHIN"] * 9))["G0"] == "BLOCKED"
    assert p0.checklist(data({**passing, "skipped": 88}, good, "PASS", ["WITHIN"] * 9))["tests"] == "FAIL"
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN"] * 9, tiny=True))["G0"] == "DRY_RUN"
    # On a reproducible device the upstream logs must equal the instrumented ones, or I1 does not hold.
    drift = {"mode": "bitwise", "configs": {"hofl": {"upstream": {"files_identical": {"rounds.jsonl": True, "curves.jsonl": False}}}}}
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN"] * 9, golden=drift))["I1"] == "FAIL"
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN"] * 9, golden={**drift, "mode": "tolerance"}))["I1"] == "PASS"


def test_e0_queue_respects_the_budget_and_probes_lazily(tmp_path, monkeypatch):
    ctx = FakeContext(tmp_path, budget=5000.0)
    calls = []

    def run(context, fields, output, tree, module="ho_fl.train", revision=None, label=None, cache_tag=None):
        calls.append(label or Path(output).name)
        if label and label.startswith("preflight"):
            if "fedavg" in label:
                return {"status": "FAILED", "error": "CUDA out of memory", "timing": None}
            return {"status": "COMPLETED", "timing": {"round_s": 10.0, "eval_s": 5.0, "startup_s": 20.0}}
        if harness.run_status(output) == "COMPLETED":
            return {"status": "COMPLETED", "reused": True, "wall_s": 2000.0}
        harness.write_json(Path(output) / "status.json", {"status": "COMPLETED"})
        context.budget -= 2000.0
        return {"status": "COMPLETED", "wall_s": 2000.0, "timing": {"round_s": 11.0, "eval_s": 5.0, "startup_s": 20.0}}

    monkeypatch.setattr(p0, "run", run)
    # A 160-round run is estimated at 1.25 * (20 + 17 * 5 + 160 * 10) = 2131 s; two of them fit into 5000 s.
    assert p0.stage_e0(ctx) is False
    queue = harness.read_json(tmp_path / "e0" / "queue.json")
    assert [entry["status"] for entry in queue["queue"]] == ["COMPLETED", "COMPLETED", "SKIPPED_BUDGET", "SKIPPED_PREFLIGHT", "SKIPPED_BUDGET"]
    assert queue["queue"][3]["reason"] == "CUDA out of memory" and queue["hardware"] == "gpu-x-torch0"
    # The second configuration is probed only after the first one has run.
    assert calls[:4] == ["preflight opt/sst2/iid/hofl", "opt-sst2-iid-hofl-b0.75-s42", "preflight opt/ag_news/dirichlet_05/hofl",
                         "opt-ag_news-dirichlet_05-hofl-b0.75-s42"]
    timings = harness.read_json(ctx.timings_path)
    assert timings["opt/sst2/iid/hofl"]["source"] == "opt-sst2-iid-hofl-b0.75-s42" and "opt/ag_news/dirichlet_05/fedavg" not in timings

    ctx.budget = float("inf")
    calls.clear()
    p0.stage_e0(ctx)
    statuses = [entry["status"] for entry in harness.read_json(tmp_path / "e0" / "queue.json")["queue"]]
    assert statuses == ["COMPLETED", "COMPLETED", "COMPLETED", "SKIPPED_PREFLIGHT", "COMPLETED"]
    assert "preflight opt/ag_news/dirichlet_05/hofl" not in calls  # its timing is already known


def test_hardware_slug_is_a_safe_directory_name():
    slug = p0.hardware_slug("cpu")
    assert slug.startswith("cpu-torch") and "/" not in slug and " " not in slug and "+" not in slug


def test_parse_junit(tmp_path):
    (tmp_path / "pytest.xml").write_text(
        '<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="2" tests="5" time="1.5">'
        '<testcase classname="tests.test_a" name="test_ok" time="0.1"/>'
        '<testcase classname="tests.test_a" name="test_bad" time="0.1"><failure message="assert 1 == 2">trace</failure></testcase>'
        '</testsuite></testsuites>')
    result = p0.parse_junit(tmp_path / "pytest.xml")
    assert (result["tests"], result["failures"], result["errors"], result["skipped"]) == (5, 1, 0, 2)
    assert result["failed"] == [{"test": "tests.test_a::test_bad", "kind": "failure", "message": "assert 1 == 2"}]
    assert p0.parse_junit(tmp_path / "missing.xml") is None
