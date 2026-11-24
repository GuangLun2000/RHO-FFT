"""The reading of results in the E0 pipeline (``experiments/sat/e0/e0.py``): the checklist of gate G0 and the report.
The pipeline itself is exercised by its ``--tiny`` dry run."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sat_e0", REPO / "experiments" / "sat" / "e0" / "e0.py")
e0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0)


def rows(item, passed):
    return {"status": "PASSED" if passed else "FAILED", "model": "vit", "device": "cuda", "gpu": "G4",
            "rows": [{"item": item, "check": "a check", "passed": passed, "detail": "detail"}]}


def feedback(**changes):
    data = {"generated_at": "now", "tiny": False, "env": {"gpu_name": "G4", "torch": "2.7.1", "git": {"head": "abc"}},
            "tests": {"returncode": 0, "tests": 400, "failures": 0, "errors": 0},
            "trace": {"returncode": 0, "land_source": "global_land_mask", "trace_sha256": "8a0e", "path": "p", "matches_report": True},
            "partitions": [{"name": "t1-base-s42", "task": "t1", "examples_per_satellite": 446, "unused": 11, "partition_sha256": "47",
                            "data_sha256": "3f", "matches_report": True}],
            "selfcheck": {"t2": rows(2, True), "t1": rows(4, True), "upstream": rows(3, True)},
            "memory": {"t1": None, "t2": None},
            "smoke": [{"name": "t2-m7", "task": "t2", "method": "m7", "summary": {"status": "COMPLETED", "wall_s_per_round": 1.5, "wall_s": 60.0},
                       "verdict": {"status": "COMPLETED", "loss_first_third": 2.3, "loss_last_third": 1.9, "loss_decreased": True, "passed": True},
                       "curve": [{"sim_hours": 0.0, "accuracy": 0.1, "macro_f1": 0.1, "loss": 2.3}], "eval_wall_s": [4.0], "participants_per_round": 17.0}],
            "t0": [{"name": "qwen-zoon-s43", "seed": 43, "mode": "on", "status": "COMPLETED", "accuracy": 0.86, "task_loss": 0.4, "wall_s": 600.0, "timing": None},
                   {"name": "qwen-zooff-s43", "seed": 43, "mode": "off", "status": "COMPLETED", "accuracy": 0.858, "task_loss": 0.4, "wall_s": 400.0, "timing": None}],
            "t0_reference": {"on": 85.99, "off": 85.95}}
    data.update(changes)
    return data


def test_checklist_reports_pass_fail_and_not_run():
    verdicts = {item: verdict for item, _, verdict, _ in e0.checklist(feedback())}
    assert verdicts == {1: "PASS", 2: "PASS", 3: "PASS", 4: "PASS", 5: "NOT RUN", 6: "PASS", 7: "PASS"}
    failed = e0.checklist(feedback(selfcheck={"t2": rows(2, True), "t1": rows(2, False), "upstream": None}, tests=None))
    assert {item: verdict for item, _, verdict, _ in failed} == {1: "NOT RUN", 2: "FAIL", 3: "NOT RUN", 4: "NOT RUN", 5: "NOT RUN", 6: "PASS", 7: "PASS"}
    assert "1 of 2 checks on t1, t2" in failed[1][3]
    nothing = e0.checklist(feedback(tests=None, trace=None, selfcheck={"t2": None, "t1": None, "upstream": None}, smoke=[]))
    assert {verdict for _, _, verdict, _ in nothing} == {"NOT RUN"}


def test_report_states_the_gate_and_the_three_seed_difference():
    text = e0.render(feedback())
    assert "not decided yet, items 5 were not run" in text and "| 43 | this session | 86.00 | 85.80 | 0.20 |" in text
    assert "| 42 | session of 2026-11-21 (reference file) | 85.99 | 85.95 | 0.04 |" in text
    assert "Difference over 2 seeds: mean +0.12 points" in text and "3.32 points" in text
    complete = feedback(selfcheck={"t2": rows(5, True), "t1": rows(2, True), "upstream": {**rows(3, True), "rows": rows(3, True)["rows"] + rows(4, True)["rows"]}})
    assert "**all items pass.**" in e0.render(complete)
    assert "a check FAILED (items 2)" in e0.render(feedback(selfcheck={"t2": rows(2, False), "t1": None, "upstream": None}))
    assert "the numbers below mean nothing" in e0.render(feedback(tiny=True))


def test_smoke_verdict_reads_a_run_directory(tmp_path):
    assert e0.smoke_verdict(tmp_path) == {"status": "MISSING", "passed": False}
    (tmp_path / "summary.json").write_text(json.dumps({"status": "COMPLETED"}))
    (tmp_path / "rounds.jsonl").write_text("".join(json.dumps({"round": r, "task_loss": loss}) + "\n" for r, loss in enumerate([2.0, 1.9, None, 1.5, 1.4, 1.3])))
    verdict = e0.smoke_verdict(tmp_path)
    assert verdict["passed"] and verdict["loss_first_third"] == 2.0 and verdict["loss_last_third"] == 1.3
    (tmp_path / "summary.json").write_text(json.dumps({"status": "DIVERGED"}))
    assert not e0.smoke_verdict(tmp_path)["passed"]


def test_expected_hashes_and_run_lists():
    assert set(e0.EXPECTED["partition_sha256"]) == {f"{task}-{variant}-s{seed}" for task in ("t1", "t2") for variant in ("base", "zonly") for seed in (42, 43, 44)}
    assert set(e0.EXPECTED["trace_sha256"]) == {"global_land_mask", "duty_cycle"}
    assert [name for name, _, _ in e0.SMOKE][0] == "t2-m7" and e0.T0_SEEDS == (43, 44)
    assert e0.T0_BASE == {"model": "qwen", "dataset": "ag_news", "partition": "dirichlet_05", "method": "hofl", "beta": 0.75}
    reference = json.loads((REPO / e0.T0_REFERENCE["file"]).read_text())["runs"]
    assert (reference[e0.T0_REFERENCE["on"]]["final_accuracy_pct"], reference[e0.T0_REFERENCE["off"]]["final_accuracy_pct"]) == (85.99, 85.95)
