"""Layer-wise first-order supply (brief sections 1.2 and 6.8).

A client reports first-order updates for the layers at or above its cut, so the tier cuts split the model into
segments S0 (below every cut: no first-order source), S1, S2, S3. The first-order width of a segment in a round is
the number of selected clients whose cut is at or below the segment's first layer.

Three views of the same quantities:
- ``analytic``: expectations from the cuts and a scheduler's inclusion probabilities;
- ``simulate``: the distribution of the width over rounds, by running a scheduler without any model;
- ``empirical``: what a finished run did, from its ``ledger.jsonl``.

``python -m ho_fl.supply plan`` prints the supply map of the schedulers of experiment E1a next to the reference
values of appendix A of the brief. It needs neither a GPU nor a model and loads no deep-learning library.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import SPECS, Config
from .scheduling import make_scheduler

# Trainable LoRA parameters per layer (rank 8 on q_proj and v_proj); config.json's bottom_dimensions is authoritative.
LAYER_PARAMETERS = {"opt": 24576, "qwen": 38912}

# The scheduler settings of experiment E1a.
E1A_SETTINGS = {
    "dep_round_v1 beta=0": {"sampler": "dep_round_v1", "beta": 0.0},
    "dep_round_v1 beta=0.75": {"sampler": "dep_round_v1", "beta": 0.75},
    "tier_equalised beta=0.75": {"sampler": "tier_equalised", "beta": 0.75},
    "tier_quota 3,1,2": {"sampler": "tier_quota", "tier_quota": "3,1,2"},
    "tier_quota 4,1,1": {"sampler": "tier_quota", "tier_quota": "4,1,1"},
    "tier_quota 2,2,2": {"sampler": "tier_quota", "tier_quota": "2,2,2"},
}

# Appendix A of the brief (three tiers of 10, K=6, q=2). The brief has no column for tier_quota 2,2,2.
APPENDIX_A = {
    "dep_round_v1 beta=0": {"width": [2.0, 4.0, 6.0], "s1_max_share": 0.10, "s1_top6_share": 0.60, "n_eff": [10, 20, 30],
                            "max_presence": 0.20, "p_s1_0": 0.064, "p_s1_1": 0.265, "p_s1_le2": 0.692, "bias": 0.0,
                            "dzo_over_q": {"qwen": 311296, "opt": 73728}},
    "dep_round_v1 beta=0.75": {"width": [5.0, 5.5, 6.0], "s1_max_share": 0.16, "s1_top6_share": 0.96, "n_eff": [6.49, 7.81, 9.23],
                               "max_presence": 0.80, "p_s1_0": 0.0, "p_s1_1": 0.0001, "p_s1_le2": 0.004, "bias": 2.25,
                               "dzo_over_q": {"qwen": 194560, "opt": 46080}},
    "tier_equalised beta=0.75": {"width": [5.0, 5.5, 6.0], "s1_max_share": 0.10, "s1_top6_share": 0.60, "n_eff": [10, 11.98, 14.12],
                                 "max_presence": 0.50, "p_s1_0": 0.0, "p_s1_1": 0.0002, "p_s1_le2": 0.004, "bias": 1.125,
                                 "dzo_over_q": {"qwen": 194560, "opt": 46080}},
    "tier_quota 3,1,2": {"width": [3.0, 4.0, 6.0], "s1_max_share": 0.10, "s1_top6_share": 0.60, "n_eff": [10, 16, 25.71],
                         "max_presence": 0.30, "p_s1_0": 0.0, "p_s1_1": 0.0, "p_s1_le2": 0.0, "bias": 0.167,
                         "dzo_over_q": {"qwen": 285355, "opt": 67584}},
    "tier_quota 4,1,1": {"width": [4.0, 5.0, 6.0], "s1_max_share": 0.10, "s1_top6_share": 0.60, "n_eff": [10, 14.71, 20],
                         "max_presence": 0.40, "p_s1_0": 0.0, "p_s1_1": 0.0, "p_s1_le2": 0.0, "bias": 0.5,
                         "dzo_over_q": {"qwen": 233472, "opt": 55296}},
}


def segments(cuts, depth):
    """Half-open layer ranges delimited by the distinct cuts: S0 (if any cut is above 0), then one per cut."""
    edges = [0, *sorted(set(int(cut) for cut in cuts)), depth]
    return [(low, high) for low, high in zip(edges, edges[1:]) if low < high]


def reference_cuts(model):
    """Ten clients per tier. Which clients hold which cut does not change any quantity computed here."""
    return [cut for cut in SPECS[model][3] for _ in range(10)]


def _shares(mass, top):
    total = mass.sum()
    if total <= 0:
        return {"max_share": None, "top_share": None, "n_eff": None}
    share = mass / total
    return {"max_share": float(share.max()), "top_share": float(np.sort(share)[::-1][:top].sum()), "n_eff": float(1 / np.square(share).sum())}


def analytic(cuts, inclusion, depth, k, dimensions=None, q=None):
    """Expected supply per segment and the scheduler-level summaries, from cuts and inclusion probabilities.

    ``share`` of a device in a segment is its fraction of all first-order reports the segment receives; ``n_eff``
    is the inverse of the sum of squared shares.
    """
    cuts, inclusion = np.asarray(cuts), np.asarray(inclusion, dtype=float)
    rows = []
    for index, (low, high) in enumerate(segments(cuts, depth)):
        capable = cuts <= low
        rows.append({"segment": index, "first_layer": low, "last_layer": high - 1, "fo_capable": int(capable.sum()),
                     "width_mean": float(inclusion[capable].sum()), **_shares(inclusion * capable, k)})
    result = {"segments": rows, "max_presence": float(inclusion.max()),
              "bias": float(len(cuts) * np.square(inclusion / k - 1 / len(cuts)).sum())}
    if dimensions is not None:
        result["dzo_over_q"] = float((inclusion / k * np.asarray(dimensions, dtype=float) / q).sum())
    return result


def width_statistics(cohort_cuts, bounds, k, quorum=3):
    """Per segment: mean first-order width and its distribution over the given rounds (lists of the cohort's cuts)."""
    rows = []
    for index, (low, high) in enumerate(bounds):
        widths = np.asarray([sum(1 for cut in cohort if cut <= low) for cohort in cohort_cuts])
        rows.append({"segment": index, "first_layer": low, "last_layer": high - 1, "rounds": len(widths),
                     "width_mean": float(widths.mean()), "p_width_0": float((widths == 0).mean()),
                     "p_width_1": float((widths == 1).mean()), "p_width_le2": float((widths <= 2).mean()),
                     "p_below_quorum": float((widths < quorum).mean()),
                     "histogram": [int((widths == value).sum()) for value in range(k + 1)]})
    return rows


def simulate(make, cuts, depth, seeds, rounds, k, quorum=3):
    """Run ``make(seed)`` schedulers for ``rounds`` rounds each and pool the width distribution over all rounds."""
    cohort_cuts = []
    for seed in seeds:
        scheduler = make(seed)
        for round_id in range(1, rounds + 1):
            plan = scheduler.sample(round_id)
            scheduler.observe(round_id, None)
            cohort_cuts.append([plan.cut(cid) for cid in plan.clients])
    return width_statistics(cohort_cuts, segments(cuts, depth), k, quorum)


def empirical(run_dir, quorum=3):
    """Supply of a finished run from its ledger: widths per segment and the concentration of the sources.

    The realised shares of one 160-round run scatter around the analytic expectations (the largest share over
    devices is biased upwards), so they are reported next to them, not tested against them.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        return None
    depth, k = SPECS[config["model"]][2], config["k"]
    bounds = segments(SPECS[config["model"]][3], depth)
    cohorts = {}
    for row in rows:
        cohorts.setdefault(row["round"], []).append(row)
    result = width_statistics([[row["cut"] for row in cohort] for cohort in cohorts.values()], bounds, k, quorum)
    for entry in result:
        sources = Counter(row["client"] for row in rows if row["cut"] <= entry["first_layer"])
        entry.update(_shares(np.asarray(list(sources.values()), dtype=float), k))
        entry["sources"] = len(sources)
    presence = Counter(row["client"] for row in rows)
    return {"segments": result, "rounds": len(cohorts), "max_presence": max(presence.values()) / len(cohorts)}


def plan(model, seeds=200, rounds=160, k=6, q=2, quorum=3):
    """Analytic and simulated supply of every E1a scheduler setting for ``model``."""
    depth = SPECS[model][2]
    cuts = reference_cuts(model)
    dimensions = [LAYER_PARAMETERS[model] * cut for cut in cuts] if model in LAYER_PARAMETERS else None
    result = {}
    for name, fields in E1A_SETTINGS.items():
        def make(seed, fields=fields):
            return make_scheduler(Config(model=model, k=k, q=q, seed=seed, **fields).resolve(), cuts, cuts)
        expected = analytic(cuts, make(0).inclusion(), depth, k, dimensions, q)
        result[name] = {"analytic": expected, "simulated": simulate(make, cuts, depth, range(seeds), rounds, k, quorum),
                        "simulation": {"seeds": seeds, "rounds": rounds}}
    return result


def plan_rows(model, result):
    """Long-format rows (one value per row) of ``plan`` for supply.csv."""
    rows = []
    for name, entry in result.items():
        for kind in ("analytic", "simulated"):
            segment_rows = entry[kind]["segments"] if kind == "analytic" else entry[kind]
            for segment in segment_rows:
                for metric, value in segment.items():
                    if metric not in ("segment", "histogram"):
                        rows.append({"source": kind, "scheduler": name, "model": model, "segment": segment["segment"], "metric": metric, "value": value})
        for metric in ("max_presence", "bias", "dzo_over_q"):
            if metric in entry["analytic"]:
                rows.append({"source": "analytic", "scheduler": name, "model": model, "segment": "", "metric": metric, "value": entry["analytic"][metric]})
    return rows


def plan_table(model, result):
    """The supply map as text, one block per scheduler setting, with the appendix A value where the brief has one."""
    def cell(value, reference=None, digits=2, percent=False):
        scale = 100 if percent else 1
        text = "n/a" if value is None else f"{scale * value:.{digits}f}" + ("%" if percent else "")
        if reference is None:
            return text
        return text + f" (brief {scale * reference:.{digits}f}" + ("%" if percent else "") + ")"

    lines = [f"Supply map for {model}: three tiers of 10 clients, cuts {SPECS[model][3]}, depth {SPECS[model][2]}.",
             "S1, S2, S3 are the segments that start at the three cuts; width = first-order reports per round."]
    for name, entry in result.items():
        reference = APPENDIX_A.get(name, {})
        analytic_rows, simulated = entry["analytic"]["segments"][-3:], entry["simulated"][-3:]
        lines.append(f"\n{name}")
        lines.append("  expected width S1 / S2 / S3: " + " / ".join(
            cell(row["width_mean"], reference.get("width", [None] * 3)[i]) for i, row in enumerate(analytic_rows)))
        lines.append("  effective sources (N_eff) S1 / S2 / S3: " + " / ".join(
            cell(row["n_eff"], reference.get("n_eff", [None] * 3)[i]) for i, row in enumerate(analytic_rows)))
        lines.append("  S1 largest single share: " + cell(analytic_rows[0]["max_share"], reference.get("s1_max_share"), 1, True)
                     + "; S1 share of the 6 largest sources: " + cell(analytic_rows[0]["top_share"], reference.get("s1_top6_share"), 1, True)
                     + "; largest presence of a client: " + cell(entry["analytic"]["max_presence"], reference.get("max_presence"), 1, True))
        lines.append("  simulated rounds with S1 width 0 / 1 / <=2 / <3: "
                     + cell(simulated[0]["p_width_0"], reference.get("p_s1_0"), 2, True) + " / "
                     + cell(simulated[0]["p_width_1"], reference.get("p_s1_1"), 2, True) + " / "
                     + cell(simulated[0]["p_width_le2"], reference.get("p_s1_le2"), 2, True) + " / "
                     + cell(simulated[0]["p_below_quorum"], None, 2, True))
        tail = "  participation bias term N*sum((p_i-1/N)^2): " + cell(entry["analytic"]["bias"], reference.get("bias"), 3)
        if "dzo_over_q" in entry["analytic"]:
            tail += "; sampling-weighted d_ZO/q: " + cell(entry["analytic"]["dzo_over_q"], reference.get("dzo_over_q", {}).get(model), 0)
        lines.append(tail)
    return "\n".join(lines)


def empirical_table(run_dirs, quorum=3):
    lines = []
    for run_dir in run_dirs:
        result = empirical(run_dir, quorum)
        if result is None:
            lines.append(f"{run_dir}: empty ledger")
            continue
        last = result["segments"][-3:]
        lines.append(f"{Path(run_dir).name}: {result['rounds']} rounds; mean width S1/S2/S3 "
                     + " / ".join(f"{row['width_mean']:.2f}" for row in last)
                     + f"; S1 width 0 in {100 * last[0]['p_width_0']:.1f}% of rounds, <=2 in {100 * last[0]['p_width_le2']:.1f}%, "
                     + f"below {quorum} in {100 * last[0]['p_below_quorum']:.1f}%; S1 sources {last[0]['sources']}"
                     + (f", N_eff {last[0]['n_eff']:.2f}, top-K share {100 * last[0]['top_share']:.1f}%" if last[0]["n_eff"] else "")
                     + f"; largest presence {100 * result['max_presence']:.1f}%")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="supply map of the E1a schedulers; no training, no model")
    planner.add_argument("--model", choices=sorted(SPECS), default="qwen")
    planner.add_argument("--seeds", type=int, default=200)
    planner.add_argument("--rounds", type=int, default=160)
    planner.add_argument("--output", type=Path, default=None, help="directory for supply.csv and supply.json")
    runs = commands.add_parser("run", help="supply of finished runs, from their ledgers")
    runs.add_argument("run_dirs", nargs="+", type=Path)
    runs.add_argument("--quorum", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(empirical_table(args.run_dirs, args.quorum))
        return
    result = plan(args.model, args.seeds, args.rounds)
    print(plan_table(args.model, result))
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"supply_{args.model}.json").write_text(json.dumps(result, indent=1) + "\n")
        rows = plan_rows(args.model, result)
        with (args.output / f"supply_{args.model}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwritten: {args.output / f'supply_{args.model}.csv'}")


if __name__ == "__main__":
    main()
