"""The backend of a satellite run: sizes, the local step, evaluation, and the memory calibration dry run."""
import json

import pytest
import torch

from ho_fl import classification
from ho_fl.model import block_index
from ho_fl.sat import calibrate
from ho_fl.sat.backend import SatBackend, classification_metrics
from ho_fl.sat.client import train_participation, zo_packet
from ho_fl.sat.config import SatConfig
from ho_fl.sat.datasets import main as data_main
from ho_fl.sat.testing import build_tiny_assets


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return build_tiny_assets(tmp_path_factory.mktemp("sat-backend"))


@pytest.fixture(scope="module")
def backends(assets):
    return {task: SatBackend(SatConfig(**assets[task]).resolve()) for task in ("t1", "t2")}


def test_metrics_from_a_confusion_matrix():
    # Rows are true classes. Class 2 never occurs but is predicted once; class 3 neither occurs nor is predicted.
    confusion = torch.tensor([[8, 2, 0, 0], [1, 3, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    metrics = classification_metrics(confusion, loss_sum=30.0)
    assert metrics["accuracy"] == pytest.approx(11 / 15) and metrics["loss"] == pytest.approx(2.0) and metrics["count"] == 15
    assert metrics["per_class_accuracy"] == [pytest.approx(0.8), pytest.approx(0.6), None, None]
    # F1 of class 0: 2*8 / (10 + 9); of class 1: 2*3 / (5 + 5); the average is over the two classes that occur.
    assert metrics["macro_f1"] == pytest.approx((16 / 19 + 6 / 10) / 2)


def test_sizes_follow_the_cut(backends):
    text, vision = backends["t1"], backends["t2"]
    assert text.cuts == vision.cuts == [0, 3, 6, 9, 12]
    assert [text.fo_bytes(cut) for cut in text.cuts] == [4 * 1024 * (12 - cut) for cut in text.cuts]
    head = 4 * (32 * 10 + 10)
    assert [vision.fo_bytes(cut) for cut in vision.cuts] == [4 * 1024 * (12 - cut) + head for cut in vision.cuts]
    assert text.full_bytes == text.fo_bytes(0) and vision.fo_bytes(12) == head and text.fo_bytes(12) == 0
    assert text.zo_bytes() == vision.zo_bytes() == 58
    assert text.kinds[:4] == ["B", "Z", "H", "Z"] and len(text.streams) == 120
    assert text.classes == 4 and vision.classes == 10 and text.image_size is None and vision.image_size == 16
    assert text.cfg.verbalizers == [" A", " B", " C", " D"] and text.tokens.tolist() == sorted(set(text.tokens.tolist()))


def test_text_evaluation_agrees_with_the_upstream_evaluation(backends):
    b = backends["t1"]
    ours = b.evaluate("test")
    upstream = classification.evaluate(b.wrapper, b.data["test"], b.tokens, b.tokenizer.pad_token_id, b.cfg.eval_batch_size,
                                       b.cfg.device, per_class=True)
    assert ours["accuracy"] == upstream["accuracy"] and ours["count"] == upstream["validation_count"] == 360
    assert ours["loss"] == pytest.approx(upstream["validation_loss"], rel=1e-12)
    assert ours["per_class_accuracy"] == pytest.approx(upstream["per_class_accuracy"])
    subset = b.evaluate("test", indices=list(range(0, 360, 3)))
    assert subset["count"] == 120 and 0 <= subset["macro_f1"] <= 1
    assert b.curve_indices is None and b.final_indices is None and b.held_out_indices is None  # small split: no subset


def test_image_evaluation_counts_every_example_once(backends):
    b = backends["t2"]
    metrics = b.evaluate("validation")
    assert metrics["count"] == 240 and 0 <= metrics["accuracy"] <= 1 and metrics["loss"] > 0
    parts = [b.evaluate("validation", indices=list(range(start, 240, 2))) for start in (0, 1)]
    assert sum(part["accuracy"] * part["count"] for part in parts) == pytest.approx(metrics["accuracy"] * 240)
    assert sum(part["loss"] * part["count"] for part in parts) == pytest.approx(metrics["loss"] * 240, rel=1e-5)


@pytest.mark.parametrize("task", ["t1", "t2"])
def test_participation_report(task, backends):
    b = backends[task]
    state = {name: param.detach().clone() for name, param in b.params.items()}
    seen = b.streams[4].batches_seen
    report = train_participation(b, state, 7, 4, 6)
    assert (report.cid, report.round, report.cut, report.tier, report.zo, report.fault) == (4, 7, 6, 1, True, None)
    assert len(report.losses) == b.cfg.E and [len(step) for step in report.scalars] == [b.cfg.q] * b.cfg.E
    assert b.streams[4].batches_seen == seen + b.cfg.E and all(set(batch) <= set(b.streams[4].indices) for batch in report.batches)
    assert all(torch.equal(report.delta[name], b.params[name].detach() - state[name]) for name in state)
    packet = zo_packet(report, b.cfg)
    assert [step["step"] for step in packet] == list(range(b.cfg.E)) and all(len(step["directions"]) == b.cfg.q for step in packet)
    assert packet[2]["directions"][1]["scalar"] == float(report.scalars[2][1])
    # Forward-only blocks frozen: they do not move at all, and nothing is drawn or reported for them.
    frozen = train_participation(b, state, 7, 4, 6, zo=False)
    assert frozen.zo is False and frozen.scalars == [[]] * b.cfg.E and zo_packet(frozen, b.cfg) == [{"step": e, "directions": []} for e in range(b.cfg.E)]
    assert all(float(delta.abs().max()) == 0 for name, delta in frozen.delta.items() if block_index(name) < 6)
    assert any(float(delta.abs().max()) > 0 for name, delta in frozen.delta.items() if block_index(name) >= 6)
    # A backbone satellite backpropagates through everything and has no zeroth-order part.
    full = train_participation(b, state, 7, 0, 0)
    assert full.zo is False and full.tier == 0 and full.scalars == [[]] * b.cfg.E
    with pytest.raises(ValueError):
        train_participation(b, state, 7, 4, 5)


def test_memory_calibration_dry_run(assets, tmp_path, capsys):
    for task in ("t1", "t2"):
        out = tmp_path / task
        result = calibrate.run(SatConfig(**assets[task]).resolve(), out)
        rows = result["rows"]
        assert [(row["cut"], row["mode"]) for row in rows] == [(0, "inference"), (0, "hybrid"), (3, "hybrid"), (6, "hybrid"), (9, "hybrid"),
                                                              (12, "hybrid"), (3, "frozen"), (6, "frozen"), (9, "frozen")]
        assert all(row["peak_allocated_bytes"] is None and len(row["measurements"]) == 3 for row in rows)  # no CUDA here
        assert [row["forward_only_parameters"] for row in rows[1:6]] == [1024 * cut for cut in (0, 3, 6, 9, 12)]
        assert json.loads((out / "memory.json").read_text())["status"] == "COMPLETED" and "| 12 |" in (out / "memory.md").read_text()
        with pytest.raises(RuntimeError):
            calibrate.run(SatConfig(**assets[task]).resolve(), out)
    assert "Memory states of the scenario" in capsys.readouterr().out


def test_data_command_prints_the_partition(assets, capsys):
    arguments = [part for key, value in assets["t1"].items() if key != "tiny" for part in (f"--{key.replace('_', '-')}", str(value))]
    assert data_main([*arguments, "--tiny", "--variant", "zonly"]) == 0
    printed = capsys.readouterr().out
    assert "variant zonly" in printed and "partition sha256" in printed and "held out for the forward-only satellites" in printed
