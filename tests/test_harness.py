"""Run management helpers: command lines, timing estimates, run comparison and the subprocess launcher."""
import json
from pathlib import Path

import pytest
import torch

from ho_fl import harness
from ho_fl.states import save_state
from ho_fl.testing import build_tiny_assets

REPO = Path(__file__).resolve().parents[1]


def test_cli_turns_fields_into_entry_point_arguments():
    assert harness.cli({"model": "opt", "beta": 0.75, "save_state_rounds": "40", "local_files_only": True}) == \
        ["--model", "opt", "--beta", "0.75", "--save-state-rounds", "40", "--local-files-only"]
    assert harness.cli({"local_files_only": False, "lr": 1e-05}) == ["--no-local-files-only", "--lr", "1e-05"]


def test_timing_separates_startup_rounds_and_evaluations():
    # Start-up 10 s, evaluation 2 s, round 3 s, evaluations at rounds 0 and 3.
    progress = {"round_times": {"1": 15.0, "2": 18.0, "3": 21.0}, "eval_times": {"0": 12.0, "3": 23.0}}
    record = harness.timing(progress)
    assert (record["round_s"], record["eval_s"], record["startup_s"]) == (3.0, 2.0, 10.0)
    assert harness.evaluation_count(160, 10) == 17 and harness.evaluation_count(3, 3) == 2 and harness.evaluation_count(5, 2) == 4
    assert harness.estimate_seconds(record, 160, 10) == 10.0 + 2.0 * 17 + 3.0 * 160
    assert harness.timing({"round_times": {}, "eval_times": {"0": 1.0}}) is None
    assert harness.timing({"round_times": {"1": 2.0}, "eval_times": {}}) is None


def make_fingerprint(**changes):
    base = {"status": "COMPLETED", "task_loss": [0.7, 0.6, 0.5], "selected_clients": [[1, 2], [3, 4], [5, 6]],
            "evaluation": [{"round": 0, "accuracy": 0.5, "task_loss": 0.69}, {"round": 3, "accuracy": 0.6, "task_loss": 0.6}],
            "final_parameter_sha256": "f" * 64, "initial_parameter_sha256": "0" * 64, "partition_id": "p",
            "file_sha256": {"rounds.jsonl": "a", "curves.jsonl": "b", "ledger.jsonl": "c"}}
    return {**json.loads(json.dumps(base)), **changes}


def test_identical_fingerprints_are_bitwise_equal():
    result = harness.compare_fingerprints(make_fingerprint(), make_fingerprint())
    assert result["bitwise"] and result["hash_equal"] and all(result["files_identical"].values())
    assert result["differences"] == [] and result["max_abs_task_loss_diff"] == 0.0


def test_candidate_with_additional_log_keys_still_matches():
    candidate = make_fingerprint()
    for row in candidate["evaluation"]:
        row["simulated_time_s"] = 1.0
    candidate["file_sha256"]["curves.jsonl"] = "changed"
    result = harness.compare_fingerprints(make_fingerprint(), candidate)
    assert result["bitwise"] and result["evaluation_equal"] and not result["files_identical"]["curves.jsonl"]


@pytest.mark.parametrize("changes, field", [
    ({"task_loss": [0.7, 0.6, 0.5001]}, "rounds_equal"),
    ({"task_loss": [0.7, 0.6]}, "rounds_equal"),
    ({"selected_clients": [[1, 2], [3, 4], [6, 5]]}, "selection_equal"),
    ({"evaluation": [{"round": 0, "accuracy": 0.5, "task_loss": 0.69}, {"round": 3, "accuracy": 0.61, "task_loss": 0.6}]}, "evaluation_equal"),
    ({"final_parameter_sha256": "e" * 64}, "hash_equal"),
    ({"initial_parameter_sha256": "1" * 64}, "same_start"),
])
def test_any_difference_is_reported(changes, field):
    result = harness.compare_fingerprints(make_fingerprint(), make_fingerprint(**changes))
    assert not result["bitwise"] and not result[field] and result["differences"]


def test_reference_without_hash_or_without_rounds():
    upstream = make_fingerprint(final_parameter_sha256=None)
    result = harness.compare_fingerprints(upstream, make_fingerprint())
    assert result["hash_equal"] is None and result["bitwise"]
    empty = make_fingerprint(task_loss=[], selected_clients=[], evaluation=[])
    assert not harness.compare_fingerprints(empty, empty)["bitwise"]


def test_state_difference(tmp_path):
    state = {"m.layers.0.a": torch.zeros(4), "m.layers.1.a": torch.ones(4)}
    save_state(tmp_path / "a.safetensors", state)
    save_state(tmp_path / "b.safetensors", {**state, "m.layers.1.a": torch.ones(4) + 0.25})
    assert harness.state_difference(tmp_path / "a.safetensors", tmp_path / "a.safetensors") == 0.0
    assert harness.state_difference(tmp_path / "a.safetensors", tmp_path / "b.safetensors") == 0.25
    assert harness.state_difference(tmp_path / "a.safetensors", tmp_path / "missing.safetensors") is None
    save_state(tmp_path / "c.safetensors", {"other": torch.zeros(1)})
    with pytest.raises(ValueError):
        harness.state_difference(tmp_path / "a.safetensors", tmp_path / "c.safetensors")


def test_run_status_of_replay_directories(tmp_path):
    assert harness.run_status(tmp_path / "none", "ho_fl.replay") == "MISSING"
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "replay_errors.csv").write_text("x")
    assert harness.run_status(tmp_path / "partial", "ho_fl.replay") == "FAILED"
    harness.write_json(tmp_path / "done" / "summary.json", {"status": "PASSED"})
    assert harness.run_status(tmp_path / "done", "ho_fl.replay") == "COMPLETED"
    assert harness.run_status(tmp_path / "none") == "MISSING"


def test_run_training_runs_reuses_sets_aside_and_reports_failures(tiny_root, tmp_path):
    fields = {**build_tiny_assets(tiny_root)["sst2"], "rounds": 2, "evaluate_every": 2, "k": 2, "E": 1, "save_state_rounds": "2"}
    messages = []
    output = tmp_path / "work" / "run"
    record = harness.run_training(fields, output, tree=REPO, mirror=tmp_path / "mirror" / "run", revision="abc", echo=messages.append)
    assert record["status"] == "COMPLETED" and not record["reused"] and record["timing"]["rounds_measured"] == 2
    meta = harness.read_json(output / "run_meta.json")
    assert meta["status"] == "COMPLETED" and meta["revision"] == "abc" and set(meta["round_times"]) == {"1", "2"}
    assert (tmp_path / "mirror" / "run" / "summary.json").exists() and (tmp_path / "mirror" / "run.log").exists()
    fingerprint = harness.fingerprint(output)
    assert len(fingerprint["task_loss"]) == 2 and fingerprint["final_parameter_sha256"]
    assert harness.compare_fingerprints(fingerprint, harness.fingerprint(tmp_path / "mirror" / "run"))["bitwise"]

    again = harness.run_training(fields, output, tree=REPO, echo=messages.append)
    assert again["reused"] and again["status"] == "COMPLETED" and again["timing"] == record["timing"]

    interrupted = tmp_path / "work" / "interrupted"
    harness.write_json(interrupted / "status.json", {"status": "RUNNING", "last_completed_round": 1})
    failed = harness.run_training({**fields, "method": "no_such_method"}, interrupted, tree=REPO, echo=messages.append)
    assert failed["status"] == "FAILED" and "Unsupported method" in failed["log_tail"]
    assert (tmp_path / "work" / "interrupted.stale-1" / "status.json").exists()
    assert harness.read_json(interrupted / "run_meta.json")["status"] == "FAILED"
    retried = harness.run_training(fields, interrupted, tree=REPO, echo=messages.append)
    assert retried["status"] == "COMPLETED" and (tmp_path / "work" / "interrupted.stale-2" / "run_meta.json").exists()
    # Same configuration, same machine: the two completed runs agree bit for bit.
    assert harness.compare_fingerprints(fingerprint, harness.fingerprint(interrupted))["bitwise"]
    assert any("FAILED" in message for message in messages)
