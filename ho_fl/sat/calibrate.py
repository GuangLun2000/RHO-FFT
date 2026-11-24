"""Peak GPU memory of one participation at every cut of a task (brief v2, section 7.6).

    python -m ho_fl.sat.calibrate --task t1 --output outputs/sat/memory/t1 [--device cuda] [--cache-dir cache]

The table this writes is what ties the cut table of the scenario to memory: a hybrid satellite that is idle has
the memory of the lowest hybrid cut (8 for Qwen, 3 for ViT) and one that is imaging has the memory of the next cut
(16, 6). For each cut the tool runs the five local steps of a participation, optimizer included, on a synthetic
batch of full length (16 prompts of the task's maximum length, or 16 images) and records the peak of
``torch.cuda.max_memory_allocated``. The frozen backbone is part of that number; datasets are not loaded.

``ho-fl-profile`` of upstream HO-FL measures the same quantity for Qwen at prompt lengths 512 and 1024 and for the
upstream cuts only; it is left as it is. On a device without CUDA the steps run and the memory columns stay empty,
which is enough for a dry run.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from ..algorithms import hybrid_step, optimizer_for, restore
from ..io import write_json
from ..model import load_model, partition_parameters
from ..randomness import RoundDirections
from .config import parse_sat_config
from .datasets import verbalizers
from .testing import TINY_TEXT_CLASSES
from .vision import load_vit

REPETITIONS, WARMUPS, BATCH = 3, 1, 16
CLASS_COUNT = 10   # HumAID and EuroSAT both have ten classes


def load_parts(sat):
    """The model of the task without its data: ``(cfg, wrapper, params, tokens, batch)``."""
    task = sat.spec
    if task.kind == "text":
        classes = TINY_TEXT_CLASSES if sat.tiny else CLASS_COUNT
        cfg = sat.train_config(verbalizers=verbalizers(classes))
        _, wrapper, params, tokenizer, tokens = load_model(cfg)
        token = tokenizer.encode("good", add_special_tokens=False)[0]
        batch = {"input_ids": torch.full((BATCH, task.max_length), token, device=cfg.device, dtype=torch.long),
                 "attention_mask": torch.ones((BATCH, task.max_length), device=cfg.device, dtype=torch.long),
                 "labels": torch.arange(BATCH, device=cfg.device) % classes}
        shape = [BATCH, task.max_length]
    else:
        cfg = sat.train_config()
        _, wrapper, params = load_vit(cfg, CLASS_COUNT, task.depth)
        tokens, size = None, wrapper.lm.config.image_size
        generator = torch.Generator().manual_seed(0)
        batch = {"input_ids": torch.rand((BATCH, 3, size, size), generator=generator).to(cfg.device) * 2 - 1,
                 "attention_mask": None, "labels": torch.arange(BATCH, device=cfg.device) % CLASS_COUNT}
        shape = [BATCH, 3, size, size]
    return cfg, wrapper, params, tokens, batch, shape


def measure(cfg, wrapper, params, tokens, batch, cut, mode):
    """Peak memory of a participation with this cut. ``mode``: ``hybrid`` (zeroth-order below the cut),
    ``frozen`` (the blocks below the cut untouched) or ``inference`` (one forward pass, no training)."""
    cuda = torch.device(cfg.device).type == "cuda"
    prefix, _ = partition_parameters(params, cut)
    original = {name: param.detach().cpu().clone() for name, param in params.items()}
    rows = []
    for repeat in range(-WARMUPS, REPETITIONS):
        restore(params, original)
        for param in params.values():
            param.grad = None
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(cfg.device)
            torch.cuda.reset_peak_memory_stats(cfg.device)
        if mode == "inference":
            with torch.no_grad():
                value = wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=tokens)
            assert torch.isfinite(value.float()).all()
            del value
        else:
            optimizer, directions = optimizer_for(params, cfg), RoundDirections(41242)
            for e in range(cfg.E):
                loss, scalars = hybrid_step(cfg, wrapper, params, prefix, batch, tokens, optimizer, directions, e,
                                            zo=mode == "hybrid", cut=cut)
                assert torch.isfinite(loss)
                del loss, scalars
            del optimizer, directions
        if cuda:
            torch.cuda.synchronize(cfg.device)
        if repeat >= 0:
            rows.append({"repeat": repeat,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(cfg.device) if cuda else None,
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(cfg.device) if cuda else None})
    restore(params, original)
    peaks = [row["peak_allocated_bytes"] for row in rows if row["peak_allocated_bytes"] is not None]
    reserved = [row["peak_reserved_bytes"] for row in rows if row["peak_reserved_bytes"] is not None]
    trainable = sum(param.numel() for param in params.values())
    below = sum(param.numel() for param in prefix.values())
    return {"cut": cut, "mode": mode, "peak_allocated_bytes": max(peaks) if peaks else None,
            "peak_reserved_bytes": max(reserved) if reserved else None, "peak_allocated_gb": max(peaks) / 1e9 if peaks else None,
            "forward_only_parameters": below, "first_order_parameters": trainable - below,
            "first_order_bytes": 4 * (trainable - below), "measurements": rows}


def run(sat, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    task = sat.spec
    cfg, wrapper, params, tokens, batch, shape = load_parts(sat)
    rows = [measure(cfg, wrapper, params, tokens, batch, 0, "inference")]
    for cut in sorted({0, *task.ladder}):
        rows.append(measure(cfg, wrapper, params, tokens, batch, cut, "hybrid"))
    for cut in task.ladder[:-1]:
        rows.append(measure(cfg, wrapper, params, tokens, batch, cut, "frozen"))
    cuda = torch.device(cfg.device).type == "cuda"
    by_cut = {row["cut"]: row["peak_allocated_gb"] for row in rows if row["mode"] == "hybrid"}
    result = {"task": task.key, "model": task.model, "depth": task.depth, "batch_shape": shape, "local_steps": cfg.E,
              "directions": cfg.q, "tiny": sat.tiny, "repetitions": REPETITIONS, "warmups": WARMUPS,
              "scope": "one participation: backbone, adapters, five local steps and the optimizer; no dataset on the device",
              "device": torch.cuda.get_device_name(0) if cuda else "cpu", "torch_version": torch.__version__,
              "memory_states_gb": {"hybrid_idle": by_cut[task.cut_idle], "hybrid_imaging": by_cut[task.cut_imaging],
                                   "backbone": by_cut[0], "forward_only": by_cut[task.depth]},
              "rows": rows, "status": "COMPLETED"}
    write_json(out / "memory.json", result)
    (out / "memory.md").write_text(render(result))
    print(render(result), flush=True)
    return result


def render(result):
    show = lambda value: "n/a" if value is None else f"{value:.2f}"  # noqa: E731
    names = {"inference": "inference only (one forward pass)", "hybrid": "hybrid: forward-only below the cut",
             "frozen": "blocks below the cut frozen (no zeroth-order part)"}
    lines = [f"Peak GPU memory of one participation, task {result['task']} ({result['model']}, {result['depth']} blocks), "
             f"batch {'x'.join(str(n) for n in result['batch_shape'])}, on {result['device']} (torch {result['torch_version']}).", "",
             "| cut | training below the cut | first-order parameters | first-order update (MB) | peak allocated (GB) | peak reserved (GB) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in result["rows"]:
        reserved = None if row["peak_reserved_bytes"] is None else row["peak_reserved_bytes"] / 1e9
        lines.append(f"| {row['cut']} | {names[row['mode']]} | {row['first_order_parameters'] if row['mode'] != 'inference' else 0} | "
                     f"{row['first_order_bytes'] / 1e6 if row['mode'] != 'inference' else 0:.3f} | {show(row['peak_allocated_gb'])} | {show(reserved)} |")
    states = result["memory_states_gb"]
    lines += ["", f"Memory states of the scenario: backbone satellite {show(states['backbone'])} GB, hybrid satellite idle "
                  f"{show(states['hybrid_idle'])} GB and imaging {show(states['hybrid_imaging'])} GB, forward-only satellite "
                  f"{show(states['forward_only'])} GB."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    sat = parse_sat_config(argv)
    result = run(sat, sat.output)
    print(json.dumps({"status": result["status"], "memory_states_gb": result["memory_states_gb"]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
