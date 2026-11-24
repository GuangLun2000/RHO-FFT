"""E0-a of the brief (section 10.1), items 2 to 5, on the tiny models. The checks themselves live in
``ho_fl.sat.selfcheck`` and also run on the GPU with the real models (stage S of the Colab notebook)."""
import json

import pytest
import torch

from ho_fl.sat import selfcheck
from ho_fl.sat.backend import SatBackend
from ho_fl.sat.config import SatConfig
from ho_fl.sat.testing import build_tiny_assets
from ho_fl.train import Backend


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return build_tiny_assets(tmp_path_factory.mktemp("sat-e0a"))


def backend(assets, task):
    return SatBackend(SatConfig(**assets[task]).resolve())


def assert_all_passed(rows, items, count):
    assert [row for row in rows if not row["passed"]] == []
    assert {row["item"] for row in rows} == items and len(rows) == count


@pytest.mark.parametrize("task", ["t1", "t2"])
def test_item_2_hybrid_step_with_explicit_cuts(task, assets):
    # Three hybrid cuts, cut 0 and the cut at the model depth.
    assert_all_passed(selfcheck.check_hybrid_step(backend(assets, task)), {2}, 5)


def test_item_3_participation_is_bitwise_the_upstream_client(tiny_config):
    rows = selfcheck.check_upstream_participation(Backend(tiny_config("sst2")))
    assert_all_passed(rows, {3}, 3)
    assert [row["check"].split(":")[0] for row in rows] == ["cut 3", "cut 6", "cut 9"]


@pytest.mark.parametrize("task", ["t1", "t2"])
def test_item_4_aggregation_equals_the_central_one(task, assets):
    rows = selfcheck.check_aggregation(backend(assets, task))
    assert_all_passed(rows, {4}, 8)
    # On the CPU the replay is exact, so the new state is even bitwise the centrally aggregated one.
    assert all("bitwise True" in row["detail"] for row in rows if "within" in row["check"])


@pytest.mark.parametrize("task", ["t1", "t2"])
def test_item_5_replay_is_the_same_on_every_replica(task, assets):
    assert_all_passed(selfcheck.check_replay(backend(assets, task)), {5}, 2)


@pytest.mark.parametrize("task", ["t1", "t2"])
def test_estimator_diagnostic_compares_the_training_scalars_with_exact_derivatives(task, assets):
    b = backend(assets, task)
    before, cursor = {name: param.detach().clone() for name, param in b.params.items()}, b.streams[0].cursor
    rows = selfcheck.check_estimator(b, directions=16, batches=2)
    cuts = [cut for cut in b.cuts if cut > 0]
    assert [(row["cut"], row["mu"]) for row in rows] == [(cut, mu) for cut in cuts for mu in selfcheck.ESTIMATOR_RADII]
    assert all(row["finite"] and row["directions"] == 32 for row in rows) and [row["configured"] for row in rows] == [row["mu"] == b.cfg.mu for row in rows]
    assert {row["dimension"] for row in rows if row["cut"] == cuts[-1]} == {b.dimensions[cuts[-1]]}
    # At the configured radius the finite difference follows the exact directional derivative on the tiny models;
    # at ten times that radius it has left the linear regime, which is what the diagnostic is there to show.
    configured = [row for row in rows if row["configured"]]
    assert len(configured) == len(cuts) and all(row["correlation"] > 0.8 and 0.5 < row["slope"] < 1.5 for row in configured)
    assert min(row["correlation"] for row in configured) > max(row["correlation"] for row in rows if row["mu"] == 0.1)
    # Nothing is trained, no gradient is left behind and no data stream is advanced.
    assert all(torch.equal(param, before[name]) and param.grad is None for name, param in b.params.items()) and b.streams[0].cursor == cursor
    table = selfcheck.render_estimator(rows, b.cfg.mu)
    assert "0.01 (configured)" in table and table.count("\n| ") == len(rows) + 2


def test_selfcheck_command(assets, tmp_path, capsys):
    arguments = [part for key, value in assets["t2"].items() if key != "tiny" for part in (f"--{key.replace('_', '-')}", str(value))]
    assert selfcheck.main([*arguments, "--tiny", "--output", str(tmp_path / "check")]) == 0
    printed = capsys.readouterr().out
    assert "15 of 15 checks passed" in printed and "FAIL" not in printed and "Zeroth-order scalars against exact directional derivatives" in printed
    assert (tmp_path / "check" / "selfcheck.json").exists() and (tmp_path / "check" / "selfcheck.md").exists()
    record = json.loads((tmp_path / "check" / "selfcheck.json").read_text())
    assert record["status"] == "PASSED" and record["mu"] == 0.01 and record["prompt_tokens"] is None and len(record["estimator"]) == 12
    assert all({"worst", "bitwise"} <= set(row) for row in record["rows"]) and all(row["bitwise"] is not False for row in record["rows"])


def test_a_failing_check_is_reported(assets, monkeypatch, tmp_path, capsys):
    b = backend(assets, "t1")
    # A replay that returns something else than the local update must fail items 4 and 5 visibly.
    original = selfcheck.replay_zeroth_order
    monkeypatch.setattr(selfcheck, "replay_zeroth_order", lambda state, report, cfg: {n: t + 1e-3 for n, t in original(state, report, cfg).items()})
    rows = selfcheck.check_replay(b)
    assert all(row["passed"] for row in rows)  # still the same on both replicas ...
    import ho_fl.sat.aggregate as aggregate
    monkeypatch.setattr(aggregate, "replay_zeroth_order", lambda state, report, cfg: {n: t + 1e-3 for n, t in original(state, report, cfg).items()})
    rows = selfcheck.check_aggregation(b)  # ... but no longer what the participants computed
    assert any(not row["passed"] for row in rows)
    assert selfcheck.finish(rows, tmp_path / "failed", "title", {}) == 1 and "FAILED" in capsys.readouterr().out
