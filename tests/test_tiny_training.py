"""End-to-end runs of the training loop on the synthetic tiny model (CPU, a few seconds each).

``test_run_matches_the_upstream_loop`` is the executable form of invariant I1: with every new option at its default,
``ho_fl.train.run`` must write byte-identical logs and reach the same final parameters as the frozen upstream loop
in ``legacy_loop.py``. Being a comparison on the same machine, it does not depend on hardware or library versions.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import ho_fl.train as train
from ho_fl.aggregation import aggregate
from ho_fl.batches import ClientStream
from ho_fl.io import tensor_hash
from ho_fl.randomness import stable_seed
from ho_fl.replay import run as replay_run
from ho_fl.train import Backend, run, train_client
from legacy_loop import legacy_run

LOGS = ("rounds.jsonl", "curves.jsonl", "ledger.jsonl", "manifest.json.gz")
# Keys of the upstream logs. Later phases may add keys but must keep these (brief section 6.12).
LEDGER_KEYS = {"round", "client", "cut", "resource_cut", "d_ZO", "direction_master_seed", "p", "inclusion_probability",
               "header_bytes", "zo_scalar_bytes", "fo_delta_bytes", "logical_uplink_bytes", "accounting_scope"}
ROUND_KEYS = {"round", "processed_samples", "selected_clients", "task_loss", "processed_input_tokens",
              "processed_answer_tokens"}
CURVE_KEYS = {"accuracy", "task_loss", "count", "evaluation_scope", "round", "processed_samples",
              "cumulative_logical_uplink_bytes", "training_task_loss"}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def logs(cfg):
    return {name: (Path(cfg.output) / name).read_bytes() for name in LOGS}


def config_without_output(cfg):
    config = read_json(Path(cfg.output) / "config.json")
    config.pop("output")
    return config


def assert_logs_match_upstream(candidate, reference):
    """Every key the upstream loop writes has the same value; the manifest is byte-identical.

    Later milestones add keys to the logs (brief section 6.12), so rows are compared on the upstream keys.
    """
    for name in ("rounds.jsonl", "curves.jsonl", "ledger.jsonl"):
        expected = read_jsonl(Path(reference.output) / name)
        actual = read_jsonl(Path(candidate.output) / name)
        assert len(actual) == len(expected) > 0, name
        assert [{key: row.get(key, "<missing>") for key in old} for row, old in zip(actual, expected)] == expected, name
    assert (Path(candidate.output) / "manifest.json.gz").read_bytes() == (Path(reference.output) / "manifest.json.gz").read_bytes()
    expected = config_without_output(reference)
    actual = config_without_output(candidate)
    assert {key: actual.get(key, "<missing>") for key in expected} == expected


def test_run_writes_the_upstream_files_and_keys(tiny_config, tmp_path):
    cfg = tiny_config("sst2", rounds=3, evaluate_every=2)
    summary = run(cfg)
    out = tmp_path / "run0"
    assert read_json(out / "status.json") == {"status": "COMPLETED", "last_completed_round": 3}
    assert summary == read_json(out / "summary.json")
    assert summary["final_parameter_sha256"] != summary["initial_parameter_sha256"]
    rounds, curves, ledger = (read_jsonl(out / name) for name in ("rounds.jsonl", "curves.jsonl", "ledger.jsonl"))
    assert [row["round"] for row in rounds] == [1, 2, 3] and [row["round"] for row in curves] == [0, 2, 3]
    assert len(ledger) == 3 * cfg.k
    assert all(ROUND_KEYS <= set(row) for row in rounds)
    assert all(CURVE_KEYS <= set(row) for row in curves)
    assert all(LEDGER_KEYS <= set(row) for row in ledger)
    assert [row["client"] for row in ledger[:cfg.k]] == rounds[0]["selected_clients"]
    config = read_json(out / "config.json")
    # ho-fl-memory reads these three entries of config.json.
    assert len(config["actual_cuts"]) == 30 and set(config["bottom_dimensions"]) == {"3", "6", "9"}
    assert {"fixed_preferred_clients", "tie_seed"} <= set(config["sampler_metadata"])


def test_saved_state_matches_the_reported_final_hash_and_saving_changes_nothing(tiny_config, tmp_path):
    plain = tiny_config("sst2", rounds=2)
    saving = tiny_config("sst2", rounds=2, save_state_rounds="1,2")
    summary_plain, summary_saving = run(plain), run(saving)
    assert logs(plain) == logs(saving)
    assert summary_plain == summary_saving
    out = tmp_path / "run1" / "states"
    assert sorted(p.name for p in out.iterdir()) == ["round_0001.safetensors", "round_0002.safetensors"]
    final = load_file(str(out / "round_0002.safetensors"))
    assert tensor_hash({name: final[name] for name in sorted(final)}) == summary_saving["final_parameter_sha256"]
    with safe_open(str(out / "round_0002.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["parameter_sha256"] == summary_saving["final_parameter_sha256"]
    assert (metadata["round"], metadata["seed"], metadata["model"]) == ("2", "42", "opt")
    first = load_file(str(out / "round_0001.safetensors"))
    assert tensor_hash({name: first[name] for name in sorted(first)}) != summary_saving["final_parameter_sha256"]


def test_init_state_starts_from_the_saved_state(tiny_config):
    first = tiny_config("sst2", rounds=2, save_state_rounds="2")
    summary = run(first)
    assert not (Path(first.output) / "init_state.json").exists()
    saved = Path(first.output) / "states" / "round_0002.safetensors"
    second = tiny_config("sst2", rounds=1, init_state=str(saved))
    resumed = run(second)
    record = read_json(Path(second.output) / "init_state.json")
    assert record["parameter_sha256"] == summary["final_parameter_sha256"] == resumed["init_state_parameter_sha256"]
    assert record["matches_run"] == {"seed": True, "partition_id": True} and "init_state_parameter_sha256" not in summary
    assert record["metadata"]["round"] == "2" and record["path"] == str(saved)
    # Round 0 of the second run evaluates the loaded state: same numbers as the last evaluation of the first run.
    before, after = read_jsonl(Path(first.output) / "curves.jsonl")[-1], read_jsonl(Path(second.output) / "curves.jsonl")[0]
    assert (after["accuracy"], after["task_loss"]) == (before["accuracy"], before["task_loss"])
    assert resumed["initial_parameter_sha256"] == summary["initial_parameter_sha256"]
    assert resumed["final_parameter_sha256"] != summary["final_parameter_sha256"]


def test_repeated_runs_are_bitwise_identical_on_cpu(tiny_config):
    first, second = tiny_config("sst2"), tiny_config("sst2")
    assert run(first) == run(second)
    assert logs(first) == logs(second)


CASES = {
    "hofl_default": ("sst2", {}),
    "hofl_uniform_ag_news": ("ag_news", {"partition": "dirichlet_05", "beta": 0.0, "k": 4, "E": 2, "eval_batch_size": 512}),
    "fedavg": ("sst2", {"method": "fedavg", "k": 3, "E": 2}),
    "fedprox": ("sst2", {"method": "fedprox", "k": 3, "E": 2}),
    "decomfl": ("sst2", {"method": "decomfl", "k": 3, "E": 2}),
    "historical_torch": ("sst2", {"sampler": "historical_torch", "beta": 0.0, "k": 3, "E": 2}),
    "historical_numpy_uniform": ("sst2", {"sampler": "historical_numpy_uniform", "beta": 0.0, "k": 3, "E": 2}),
    "squad_training_path": ("squad", {"k": 2, "E": 2, "batch_size": 4}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_run_matches_the_upstream_loop(case, tiny_config, monkeypatch):
    dataset, fields = CASES[case]
    if dataset == "squad":
        # Generation-based SQuAD evaluation is out of scope here; the training path is what the loop touches.
        monkeypatch.setattr(Backend, "evaluate", lambda self, round_id, final: [{"task_loss": 0.0, "evaluation_scope": "stub"}])
    reference = tiny_config(dataset, rounds=3, evaluate_every=2, **fields)
    candidate = replace(reference, output=reference.output + "-candidate")
    expected, actual = legacy_run(reference), run(candidate)
    assert actual == expected
    assert_logs_match_upstream(candidate, reference)
    assert expected["final_parameter_sha256"] != expected["initial_parameter_sha256"]


def test_equivalence_check_notices_a_changed_accumulation_order(tiny_config, monkeypatch):
    """The final parameter hash is what exposes a changed aggregation; with a BF16 backbone the logs may not move."""
    reference = tiny_config("sst2", rounds=3, evaluate_every=3)
    expected = legacy_run(reference)
    monkeypatch.setattr(train, "aggregate", lambda reports, arrivals, cfg: aggregate(reports[::-1], arrivals, cfg))
    changed = run(replace(reference, output=reference.output + "-reversed"))
    assert changed["final_parameter_sha256"] != expected["final_parameter_sha256"]


def test_train_client_report_records_what_the_client_did(tiny_config):
    cfg = tiny_config("sst2", k=2, E=3)
    b = Backend(cfg)
    state = {name: param.detach().clone() for name, param in b.params.items()}
    streams = [ClientStream(client, cfg.batch_size) for client in b.manifest["clients"]]
    cid = 7
    report = train_client(b, state, streams, 1, cid, b.cuts[cid])
    fresh = ClientStream(b.manifest["clients"][cid], cfg.batch_size)
    assert report.batches == [fresh.next() for _ in range(cfg.E)]
    assert (report.cid, report.round, report.cut, report.fault) == (cid, 1, b.cuts[cid], None)
    assert report.tier == [3, 6, 9].index(b.resource_cuts[cid])
    assert len(report.losses) == cfg.E and [len(step) for step in report.scalars] == [cfg.q] * cfg.E
    assert report.direction_master_seed == stable_seed(cfg.seed, cfg.model, 1, cid, "zo_master")
    assert all(torch.equal(report.delta[name], b.params[name].detach() - state[name]) for name in state)
    assert any(report.delta[name].abs().max() > 0 for name in state)
    # Only this client's data stream advanced, and a cut other than the client's own is refused before anything runs.
    assert [stream.batches_seen for stream in streams] == [cfg.E if index == cid else 0 for index in range(cfg.n)]
    with pytest.raises(ValueError):
        train_client(b, state, streams, 1, cid, 99)
    assert streams[cid].batches_seen == cfg.E


def test_replay_passes_on_the_tiny_model(tiny_config):
    summary = replay_run(tiny_config("sst2"))
    assert summary["status"] == "PASSED" and summary["comparisons"] > 0
