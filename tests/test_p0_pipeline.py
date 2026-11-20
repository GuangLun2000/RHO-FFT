"""Logic of the P0 pipeline that does not need a run: plans, verdicts, report rendering."""
import importlib.util
from pathlib import Path

import pytest

from ho_fl import harness

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p0", REPO / "experiments" / "p0" / "p0.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


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


BITWISE = {"bitwise": True, "hash_equal": True, "max_abs_state_diff": 0.0}
NOISY = {"bitwise": False, "hash_equal": False, "max_abs_state_diff": 3e-9}
VERY_NOISY = {"bitwise": False, "hash_equal": False, "max_abs_state_diff": 1e-5}


@pytest.mark.parametrize("repeat, comparison, status, expected", [
    (BITWISE, {"bitwise": True, "hash_equal": True, "max_abs_state_diff": 0.0}, "COMPLETED", "PASS_BITWISE"),
    (BITWISE, {"bitwise": False, "hash_equal": False, "max_abs_state_diff": 1e-9}, "COMPLETED", "FAIL"),
    (NOISY, {"bitwise": False, "hash_equal": False, "max_abs_state_diff": 5e-8}, "COMPLETED", "PASS_TOLERANCE"),
    (NOISY, {"bitwise": False, "hash_equal": False, "max_abs_state_diff": 1e-4}, "COMPLETED", "FAIL"),
    (VERY_NOISY, {"bitwise": False, "hash_equal": False, "max_abs_state_diff": 1e-4}, "COMPLETED", "INCONCLUSIVE"),
    (NOISY, {"bitwise": False, "hash_equal": False, "max_abs_state_diff": None}, "COMPLETED", "INCONCLUSIVE"),
    (NOISY, {"bitwise": True, "hash_equal": True, "max_abs_state_diff": 0.0}, "COMPLETED", "PASS_BITWISE"),
    (BITWISE, {"bitwise": True, "hash_equal": True, "max_abs_state_diff": 0.0}, "FAILED", "RUN_FAILED"),
])
def test_i1_verdict(repeat, comparison, status, expected):
    assert p0.i1_verdict(repeat, comparison, status) == expected


def fake_run(root, name, group, seed, accuracy, rounds=160):
    model, dataset, partition, method, beta = group
    path = root / "e0" / "runs" / name
    harness.write_json(path / "status.json", {"status": "COMPLETED", "last_completed_round": rounds})
    harness.write_json(path / "config.json", {"model": model, "dataset": dataset, "partition": partition, "method": method,
                                               "beta": beta, "seed": seed, "rounds": rounds})
    harness.write_json(path / "summary.json", {"accuracy": accuracy / 100, "task_loss": 0.4, "final_parameter_sha256": "f" * 64})
    harness.write_json(path / "run_meta.json", {"status": "COMPLETED", "wall_s": 600.0, "revision": "abc"})
    (path / "curves.jsonl").write_text('{"round": 0, "accuracy": 0.5, "task_loss": 0.7, "training_task_loss": null}\n'
                                      f'{{"round": {rounds}, "accuracy": {accuracy / 100}, "task_loss": 0.4, "training_task_loss": 0.5}}\n')


class FakeContext:
    def __init__(self, root):
        self.root, self.e0_root, self.tiny, self.head, self.hardware = root, root / "e0", False, "0" * 40, "cpu-torch0"
        self.golden_root, self.regress_root = root / "golden" / "x", root / "regress" / "y"


def test_e0_groups_checklist_and_report(tmp_path):
    sst2 = ("opt", "sst2", "iid", "hofl", 0.75)
    fedavg = ("opt", "ag_news", "dirichlet_05", "fedavg", 0.0)
    short = ("opt", "ag_news", "dirichlet_05", "decomfl", 0.0)
    for seed, accuracy in zip(range(42, 47), (86.2, 86.4, 86.6, 86.8, 87.0)):
        fake_run(tmp_path, f"sst2-s{seed}", sst2, seed, accuracy)
    fake_run(tmp_path, "fedavg-s42", fedavg, 42, 60.0)
    fake_run(tmp_path, "decomfl-s42-r3", short, 42, 25.0, rounds=3)
    (tmp_path / "e0" / "runs" / "fedavg-s43.stale-1").mkdir()
    ctx = FakeContext(tmp_path)
    runs, groups = p0.e0_results(ctx)
    by_group = {tuple(g["group"]): g for g in groups}
    assert len(runs) == 7 and len(groups) == len(p0.REFERENCE)
    assert by_group[sst2]["verdict"] == "WITHIN" and abs(by_group[sst2]["mean"] - 86.6) < 1e-9
    assert by_group[sst2]["tolerance"] == 1.0 and by_group[fedavg]["tolerance"] == pytest.approx(5.94)
    assert by_group[fedavg]["verdict"] == "OUTSIDE" and by_group[short]["verdict"] == "NOT_COMPARABLE"
    assert by_group[("qwen", "ag_news", "dirichlet_05", "fedavg", 0.0)]["verdict"] == "NOT_RUN"

    data = p0.collect(ctx)
    assert data["checklist"] == {"tests": "NOT_RUN", "I1": "NOT_RUN", "I2": "NOT_RUN", "E0": "CHECK", "G0": "BLOCKED"}
    assert data["used_gpu_hours"] == pytest.approx(7 * 600 / 3600)
    text = p0.render(data)
    assert "## 1. Gate G0 checklist" in text and "OUTSIDE" in text and "Gate G0 is blocked" in text


def test_checklist_states():
    def data(tests, verdicts, replay, groups, tiny=False):
        return {"tiny": tiny, "tests": tests, "regress": {"i1": {n: {"verdict": v} for n, v in verdicts.items()}, "i2": {"verdict": replay}},
                "e0": {"groups": [{"verdict": v} for v in groups]}}
    passing = {"tests": 88, "failures": 0, "errors": 0}
    good = {"hofl": "PASS_BITWISE", "fedavg": "PASS_TOLERANCE", "decomfl": "PASS_BITWISE"}
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN"] * 9))["G0"] == "PASS"
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN_PRELIMINARY"] * 5 + ["NOT_RUN"] * 4)) == \
        {"tests": "PASS", "I1": "PASS", "I2": "PASS", "E0": "INCOMPLETE", "G0": "INCOMPLETE"}
    assert p0.checklist(data(passing, {**good, "hofl": "FAIL"}, "PASS", ["WITHIN"] * 9))["G0"] == "BLOCKED"
    assert p0.checklist(data(passing, {**good, "hofl": "INCONCLUSIVE"}, "PASS", ["WITHIN"] * 9))["I1"] == "INCONCLUSIVE"
    assert p0.checklist(data({"tests": 88, "failures": 1, "errors": 0}, good, "PASS", ["WITHIN"] * 9))["G0"] == "BLOCKED"
    assert p0.checklist(data(passing, good, "PASS", ["WITHIN"] * 9, tiny=True))["G0"] == "DRY_RUN"


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
