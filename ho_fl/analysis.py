"""Tables and decision rules computed from run directories (brief section 6.11, the part needed so far).

Every function reads the files a run wrote (``config.json``, ``summary.json``, ``curves.jsonl``, ``ledger.jsonl``,
``layers.jsonl``, ``probe.jsonl``) and returns plain rows, so the same code serves a live session and bundles that
were downloaded from several sessions. Accuracies are in percent; differences are percentage points.
Imports neither torch nor transformers.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import harness


def find_runs(roots):
    """Directories of completed training runs below ``roots`` (attempts that were set aside are left out)."""
    found = []
    for root in roots:
        for status in sorted(Path(root).rglob("status.json")):
            run_dir = status.parent
            if ".stale-" in str(run_dir) or "preflight" in run_dir.parts:
                continue
            if (harness.read_json(status) or {}).get("status") == "COMPLETED" and (run_dir / "summary.json").exists():
                found.append(run_dir)
    return found


def summary_row(run_dir):
    """One row per run: what was run, how it ended and what it cost."""
    run_dir = Path(run_dir)
    run = harness.load_run(run_dir)
    config, summary, meta = run["config"], run["summary"], run["meta"]
    ledger = harness.read_jsonl(run_dir / "ledger.jsonl")
    curve = [row for row in run["curves"] if "accuracy" in row]
    rounds = max((row["round"] for row in ledger), default=0)
    tiers = defaultdict(int)
    for row in ledger:
        tiers[row.get("tier")] += 1
    return {"run": run_dir.name, "model": config.get("model"), "dataset": config.get("dataset"), "partition": config.get("partition"),
            "seed": config.get("seed"), "method": config.get("method"), "beta": config.get("beta"), "label": config.get("method_label"),
            "sampler": config.get("sampler"), "aggregation": config.get("aggregation"), "zo_mode": config.get("zo_mode"),
            "server_lr": config.get("server_lr"), "rounds": config.get("rounds"), "status": run["status"],
            "final_accuracy_pct": 100 * summary["accuracy"] if "accuracy" in summary else None,
            "final_f1": summary.get("f1"), "final_exact_match": summary.get("exact_match"),
            "final_eval_loss": summary.get("task_loss"), "final_train_loss": curve[-1].get("training_task_loss") if curve else None,
            "uplink_bytes_per_round": curve[-1]["cumulative_logical_uplink_bytes"] / rounds if curve and rounds else None,
            "forward_equivalents_per_round": sum(row.get("forward_equivalents") or 0 for row in ledger) / rounds if rounds else None,
            "participations_t1": tiers[0], "participations_t2": tiers[1], "participations_t3": tiers[2],
            "wall_s": meta.get("wall_s"), "hardware": meta.get("hardware"), "revision": meta.get("revision"),
            "partition_id": summary.get("partition_id"), "initial_parameter_sha256": summary.get("initial_parameter_sha256"),
            "final_parameter_sha256": summary.get("final_parameter_sha256")}


def curve_rows(run_dir):
    run_dir = Path(run_dir)
    return [{"run": run_dir.name, "round": row["round"], "accuracy_pct": 100 * row["accuracy"], "eval_loss": row["task_loss"],
             "train_loss": row.get("training_task_loss"), "cumulative_uplink_bytes": row.get("cumulative_logical_uplink_bytes")}
            for row in harness.read_jsonl(run_dir / "curves.jsonl") if "accuracy" in row]


def log_rows(run_dir, name):
    """Rows of ``layers.jsonl`` or ``probe.jsonl`` with the run's name in front."""
    run_dir = Path(run_dir)
    return [{"run": run_dir.name, **row} for row in harness.read_jsonl(run_dir / name)]


def write_csv(path, rows):
    """Write rows (dicts) as CSV with the union of their keys, in order of first appearance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _standard_error(values):
    values = [value for value in values if value is not None]
    return statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None


def probe_summary(rows):
    """Pooled comparison with the first-order counterfactual, per probe round, tier and group of layers.

    ``null_cosine`` is the standard deviation of the cosine between a random direction and a fixed one in that
    dimension (1 / sqrt(d)): a mean cosine has to be compared with it, and with its own standard error.
    """
    groups = defaultdict(list)
    for row in rows:
        if row.get("scope") == "pooled":
            groups[(row["round"], row["tier"], row["group"])].append(row)
    result = []
    for (round_id, tier, group), members in sorted(groups.items()):
        cosines = [row["cosine"] for row in members]
        ratios = sorted(row["norm_ratio"] for row in members if row["norm_ratio"] is not None)
        dimension = _mean([row["dimension"] for row in members])
        result.append({"round": round_id, "tier": tier, "group": group, "clients": len(members), "cosine_mean": _mean(cosines),
                       "cosine_se": _standard_error(cosines), "null_cosine": dimension ** -0.5 if dimension else None,
                       "norm_ratio_median": statistics.median(ratios) if ratios else None, "dimension": dimension})
    return result


LAYER_METRICS = ("n_fo", "n_zo", "fo_norm", "zo_norm", "fo_mean_norm", "zo_mean_norm", "cos_fo_fo", "cos_zo_zo", "cos_fo_zo", "update_norm")


def layer_summary(rows):
    """Layer statistics averaged over the rounds and layers of each segment."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["segment"]].append(row)
    result = []
    for segment, members in sorted(groups.items()):
        entry = {"segment": segment, "layers": len({row["layer"] for row in members}), "rounds": len({row["round"] for row in members})}
        entry.update({metric: _mean([row.get(metric) for row in members]) for metric in LAYER_METRICS})
        rules = defaultdict(int)
        for row in members:
            rules[row["rule"]] += 1
        entry["rules"] = {rule: count / len(members) for rule, count in sorted(rules.items())}
        result.append(entry)
    return result


def _paired(accuracy, arms):
    """Seeds for which every listed arm has a result."""
    return sorted(set.intersection(*(set(accuracy.get(arm, {})) for arm in arms))) if arms else []


def decide_h1a(accuracy, planned_seeds=3):
    """Hypothesis H1a (dilution), from final accuracies ``{arm: {seed: percent}}`` of one model.

    Per seed: G_C = C(beta 0) - A(beta 0), the gain of averaging among first-order reports only;
    G_beta = A(beta 0.75) - A(beta 0), the gain of biased sampling under equal-weight averaging;
    G_lr = max(A x2, A x3) - A(beta 0), the gain of a larger global step.
    Supported when mean G_C >= 0.7 * mean G_beta and mean G_lr < 0.5 * mean G_C (brief section 2.2).
    """
    seeds = _paired(accuracy, ["A-b0", "C-b0", "A-b0.75"])
    result = {"seeds": seeds, "per_seed": [], "verdict": "NOT_RUN", "preliminary": len(seeds) < planned_seeds}
    if not seeds:
        return result
    for seed in seeds:
        base = accuracy["A-b0"][seed]
        controls = [accuracy[arm][seed] - base for arm in ("Ax2-b0", "Ax3-b0") if seed in accuracy.get(arm, {})]
        result["per_seed"].append({"seed": seed, "g_c": accuracy["C-b0"][seed] - base, "g_beta": accuracy["A-b0.75"][seed] - base,
                                   "g_lr": max(controls) if len(controls) == 2 else None})
    result["g_c"], result["g_beta"] = _mean([row["g_c"] for row in result["per_seed"]]), _mean([row["g_beta"] for row in result["per_seed"]])
    with_control = [row for row in result["per_seed"] if row["g_lr"] is not None]
    result["g_lr"] = _mean([row["g_lr"] for row in with_control])
    result["g_c_sd"] = statistics.stdev([row["g_c"] for row in result["per_seed"]]) if len(seeds) > 1 else None
    result["g_beta_sd"] = statistics.stdev([row["g_beta"] for row in result["per_seed"]]) if len(seeds) > 1 else None
    if result["g_beta"] <= 0:
        result["verdict"], result["reason"] = "CANNOT_JUDGE", "biased sampling gave no gain over uniform sampling, so there is nothing to recover"
    elif result["g_lr"] is None or len(with_control) < len(seeds):
        result["verdict"], result["reason"] = "CANNOT_JUDGE", "the step-size controls (A x2, A x3) are not complete"
    else:
        recovered = result["g_c"] >= 0.7 * result["g_beta"]
        not_step_size = result["g_lr"] < 0.5 * result["g_c"]
        result["recovered_share"] = result["g_c"] / result["g_beta"]
        result["verdict"] = "SUPPORTED" if recovered and not_step_size else "NOT_SUPPORTED"
        result["reason"] = (f"first-order-only averaging recovers {100 * result['recovered_share']:.0f}% of the gain of biased sampling (needs 70%); "
                            f"the best step-size control gains {result['g_lr']:+.2f} points against {result['g_c']:+.2f} (needs less than half)")
    return result


def decide_h1(accuracy, planned_seeds=3):
    """Model-level part of hypothesis H1: switching the zeroth-order updates off changes the final accuracy by
    less than the seed-to-seed standard deviation of the baseline. One entry per beta; arms A (zo on, mean_all)
    and B (zo off, same weight 1/K)."""
    result = {}
    for beta in ("b0", "b0.75"):
        seeds = _paired(accuracy, [f"A-{beta}", f"B-{beta}"])
        entry = {"seeds": seeds, "verdict": "NOT_RUN", "preliminary": len(seeds) < planned_seeds}
        if seeds:
            differences = [accuracy[f"A-{beta}"][seed] - accuracy[f"B-{beta}"][seed] for seed in seeds]
            entry["per_seed"] = [{"seed": seed, "a_minus_b": difference} for seed, difference in zip(seeds, differences)]
            entry["a_minus_b"] = statistics.mean(differences)
            if len(seeds) > 1:
                entry["seed_sd_of_a"] = statistics.stdev(accuracy[f"A-{beta}"][seed] for seed in seeds)
                entry["verdict"] = "SUPPORTED" if abs(entry["a_minus_b"]) < entry["seed_sd_of_a"] else "NOT_SUPPORTED"
            else:
                entry["verdict"] = "CANNOT_JUDGE"
                entry["reason"] = "one seed gives no seed-to-seed standard deviation"
        result[beta] = entry
    return result


# --- speed of training and the controls added after the first session -------------------------------------------------
#
# After 160 rounds no arm has converged, so the final accuracy mostly measures how fast an arm learns. The helpers
# below read a curve (rows of ``curve_rows``, ordered by round) and make that speed explicit.

def accuracy_at(curve, round_id):
    """Accuracy (%) at ``round_id``, linearly interpolated between evaluations; ``None`` outside the curve."""
    for left, right in zip(curve, curve[1:]):
        if left["round"] <= round_id <= right["round"]:
            share = (round_id - left["round"]) / (right["round"] - left["round"])
            return left["accuracy_pct"] + share * (right["accuracy_pct"] - left["accuracy_pct"])
    return curve[0]["accuracy_pct"] if len(curve) == 1 and curve[0]["round"] == round_id else None


def rounds_to_reach(curve, target):
    """First round at which the accuracy reaches ``target`` (%), linearly interpolated; ``None`` if it never does."""
    if curve and curve[0]["accuracy_pct"] >= target:
        return float(curve[0]["round"])
    for left, right in zip(curve, curve[1:]):
        if left["accuracy_pct"] < target <= right["accuracy_pct"]:
            share = (target - left["accuracy_pct"]) / (right["accuracy_pct"] - left["accuracy_pct"])
            return left["round"] + share * (right["round"] - left["round"])
    return None


def mean_accuracy(curve, upto):
    """Average of the accuracy over rounds 0 to ``upto`` (trapezoid rule); ``None`` if the curve ends earlier."""
    end = accuracy_at(curve, upto)
    if end is None or upto <= 0:
        return None
    points = [(row["round"], row["accuracy_pct"]) for row in curve if row["round"] < upto] + [(upto, end)]
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:])) / (upto - points[0][0])


def step_size_control(accuracy):
    """Does a larger server step alone reproduce the gain of first-order-only averaging (uniform sampling)?

    Per seed: g_c = C - A; g_x2 = A x2 - A; g_x3 = A x3 - A; matched_x2 = C x2 - A x2, the two electorates at the
    same server step. ``explained`` is true when the best step-size control reaches at least half of g_c, the
    threshold of the brief (section 2.2); ``None`` when it cannot be computed or first-order-only averaging gained nothing.
    """
    seeds = _paired(accuracy, ["A-b0", "C-b0"])
    rows = []
    for seed in seeds:
        value = lambda arm: accuracy.get(arm, {}).get(seed)  # noqa: E731
        difference = lambda a, b: value(a) - value(b) if value(a) is not None and value(b) is not None else None  # noqa: E731
        rows.append({"seed": seed, "g_c": difference("C-b0", "A-b0"), "g_x2": difference("Ax2-b0", "A-b0"),
                     "g_x3": difference("Ax3-b0", "A-b0"), "matched_x2": difference("Cx2-b0", "Ax2-b0")})
    result = {"seeds": seeds, "per_seed": rows}
    result.update({key: _mean([row[key] for row in rows]) for key in ("g_c", "g_x2", "g_x3", "matched_x2")})
    controls = [result[key] for key in ("g_x2", "g_x3") if result[key] is not None]
    result["explained"] = max(controls) >= 0.5 * result["g_c"] if controls and result["g_c"] is not None and result["g_c"] > 0 else None
    return result


def sampling_speedup(curves, round_id):
    """Gain of biased over uniform sampling at an early round, with the zeroth-order updates on (A) and off (B).

    ``curves`` is ``{arm: {seed: curve}}``. If the gain is the same with the zeroth-order updates switched off, it
    cannot come from lower zeroth-order variance. ``max_gap_on_off`` is the largest distance between the two
    biased-sampling curves (A and B at beta 0.75) over the rounds both were evaluated at.
    """
    arms = ["A-b0", "A-b0.75", "B-b0", "B-b0.75"]
    rows = []
    for seed in _paired(curves, arms):
        at = {arm: accuracy_at(curves[arm][seed], round_id) for arm in arms}
        if any(value is None for value in at.values()):
            continue
        off = {row["round"]: row["accuracy_pct"] for row in curves["B-b0.75"][seed]}
        gaps = [abs(row["accuracy_pct"] - off[row["round"]]) for row in curves["A-b0.75"][seed] if row["round"] in off]
        rows.append({"seed": seed, "gain_zo_on": at["A-b0.75"] - at["A-b0"], "gain_zo_off": at["B-b0.75"] - at["B-b0"],
                     "max_gap_on_off": max(gaps) if gaps else None})
    return {"round": round_id, "per_seed": rows, "gain_zo_on": _mean([row["gain_zo_on"] for row in rows]),
            "gain_zo_off": _mean([row["gain_zo_off"] for row in rows]), "max_gap_on_off": _mean([row["max_gap_on_off"] for row in rows])}


def attribution_summary(rows, every=1):
    """First-order decrease of the evaluation loss attributed to first-order and zeroth-order components, per segment.

    Rows come from ``attribution.jsonl`` (one per measured round and segment). ``pred_*`` are summed over the measured
    rounds; the zeroth-order share is ``pred_zo / (pred_fo + pred_zo)`` with a standard error over rounds. The check
    of the first-order approximation compares the change of the measured evaluation loss between consecutive
    measurements with ``every`` times the predicted decrease of the earlier one (the rounds in between are not
    measured, so this is a rough check).
    """
    by_segment = defaultdict(list)
    for row in rows:
        if row.get("kind") != "error" and "segment" in row:
            by_segment[row["segment"]].append(row)
    result = []
    for segment, members in sorted(by_segment.items()):
        decomposed = all(row["decomposed"] for row in members)
        entry = {"segment": segment, "rounds": len(members), "decomposed": decomposed,
                 "pred_total": sum(row["pred_total"] for row in members)}
        if decomposed:
            fo, zo = [row["pred_fo"] or 0.0 for row in members], [row["pred_zo"] or 0.0 for row in members]
            entry.update(pred_fo=sum(fo), pred_zo=sum(zo), n_fo=_mean([row.get("n_fo") for row in members]), n_zo=_mean([row.get("n_zo") for row in members]),
                         cos_fo=_mean([row["cos_fo"] for row in members]), cos_zo=_mean([row["cos_zo"] for row in members]),
                         norm_fo=_mean([row["norm_fo"] for row in members]), norm_zo=_mean([row["norm_zo"] for row in members]))
            total = sum(fo) + sum(zo)
            entry["zo_share"] = sum(zo) / total if total else None
            entry["zo_share_se"] = (statistics.stdev(zo) * len(zo) ** 0.5 / abs(total)) if len(zo) > 1 and total else None
        result.append(entry)
    rounds = sorted({row["round"]: row["val_loss"] for rows_ in by_segment.values() for row in rows_}.items())
    predicted = defaultdict(float)
    for row in rows:
        if row.get("kind") != "error" and "segment" in row:
            predicted[row["round"]] += row["pred_total"]
    pairs = [(rounds[index + 1][1] - loss, -every * predicted[round_id]) for index, (round_id, loss) in enumerate(rounds[:-1])]
    check = None
    if len(pairs) > 2:
        actual, expected = np.array([pair[0] for pair in pairs]), np.array([pair[1] for pair in pairs])
        check = {"intervals": len(pairs), "actual_change": float(actual.sum()), "predicted_change": float(expected.sum()),
                 "correlation": float(np.corrcoef(actual, expected)[0, 1]) if actual.std() > 0 and expected.std() > 0 else None}
    return {"segments": result, "first_order_check": check}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Write summary.csv, curves.csv, layers.csv and probe.csv for the completed runs below the given directories.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = find_runs(args.roots)
    write_csv(args.output / "summary.csv", [summary_row(run) for run in runs])
    write_csv(args.output / "curves.csv", [row for run in runs for row in curve_rows(run)])
    write_csv(args.output / "layers.csv", [row for run in runs for row in log_rows(run, "layers.jsonl")])
    write_csv(args.output / "probe.csv", [row for run in runs for row in log_rows(run, "probe.jsonl")])
    print(f"{len(runs)} runs -> {args.output}")


if __name__ == "__main__":
    main()
