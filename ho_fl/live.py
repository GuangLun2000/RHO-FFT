"""Readable progress lines for the notebook output, built from the logs a run writes while it trains.

``harness.launch`` follows ``rounds.jsonl`` and ``curves.jsonl`` of a running training process; the functions here
format those rows, and summarise the probe logs (``attribution.jsonl``, ``dose_probe.jsonl``) one measured round at a
time. Everything is read-only and only affects what is printed.
"""
import math


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _minutes(seconds):
    return f"{seconds / 60:.1f} min"


def round_line(row, total, elapsed):
    """One training round: ``round 40/160 | train loss 0.4123 | 2.1 min, about 6 min left``."""
    r = row["round"]
    text = f"round {r}/{total}" if total else f"round {r}"
    if _number(row.get("task_loss")):
        text += f" | train loss {row['task_loss']:.4f}"
    text += f" | {_minutes(elapsed)}"
    if total and r:
        text += f", about {elapsed / r * (total - r) / 60:.0f} min left"
    return text


def eval_line(row, total):
    """One evaluation: ``EVAL round 40/160 | test accuracy 84.12% | test loss 0.4521 | train loss 0.4123``."""
    r = row["round"]
    parts = [f"EVAL round {r}/{total}" if total else f"EVAL round {r}"]
    if _number(row.get("accuracy")):
        parts.append(f"test accuracy {100 * row['accuracy']:.2f}%")
    for key, name in (("exact_match", "exact match"), ("f1", "F1")):
        if _number(row.get(key)):
            parts.append(f"{name} {row[key]:.4f}")
    if _number(row.get("task_loss")):
        parts.append(f"test loss {row['task_loss']:.4f}")
    if _number(row.get("training_task_loss")):
        parts.append(f"train loss {row['training_task_loss']:.4f}")
    if row.get("per_class_accuracy"):
        parts.append("per class " + " / ".join("n/a" if value is None else f"{100 * value:.0f}" for value in row["per_class_accuracy"]))
    return " | ".join(parts)


def attribution_lines(rows):
    """The segment rows of one measured round of ``attribution.jsonl``: how much the applied update lowers the loss on
    the fixed evaluation examples (to first order), and which part of it comes from zeroth-order components."""
    rows = sorted(rows, key=lambda row: row["segment"])
    r, total = rows[0]["round"], sum(row["pred_total"] for row in rows if _number(row.get("pred_total")))
    head = f"attribution round {r}: eval loss {rows[0]['val_loss']:.4f}, predicted decrease from this round's update {total:+.3g}"
    if not all(row.get("decomposed") for row in rows):
        return [head + " (robust aggregation: not split by order)"]
    fo, zo = sum(row["pred_fo"] for row in rows), sum(row["pred_zo"] for row in rows)
    share = f"{zo / total:+.1%}" if total else "n/a"
    segments = ", ".join(f"S{row['segment']} {row['pred_zo']:+.2g}" for row in rows)
    return [f"{head} = first-order {fo:+.3g} + zeroth-order {zo:+.3g} (zeroth-order share {share}; by segment: {segments})"]


def dose_lines(rows):
    """One round of ``dose_probe.jsonl``: progress on the scarce segment S1 per dose m = 0..5 under the heaviest fault,
    in units of what one honest first-order report achieves, for the rules that decide H2."""
    reference = next((row for row in rows if row.get("kind") == "reference"), None)
    cells = {(row["m"], row["rule"], row["fault"]): row for row in rows if row.get("kind") == "candidate"}
    if reference is None:
        return []
    r, unit = reference["round"], reference["pred_fo_report"]
    if not _number(unit) or unit <= 0:
        return [f"dose round {r}: the first-order reports did not lower the loss this round (unit {unit:+.3g}); no normalised line"]
    lines = [f"dose round {r}: one zeroth-order report moves S1 {reference['pred_zo_report'] / unit:+.2f}x as far as a first-order report"]
    fault = "flip20" if any(key[2] == "flip20" for key in cells) else None
    doses = sorted({key[0] for key in cells})
    for rule in ("fo_robust", "robust_all", "quorum_robust", "mean_all"):
        values = [cells.get((m, rule, fault)) for m in doses]
        if fault and all(values):
            lines.append(f"  {fault}, progress for m = {doses[0]}..{doses[-1]} | {rule:13s} " + " ".join(f"{cell['pred'] / unit:+6.2f}" for cell in values))
    return lines


# Probe logs followed while a run trains, with the function that summarises one round.
FOLLOWED = {"attribution.jsonl": attribution_lines, "dose_probe.jsonl": dose_lines}
