#!/usr/bin/env python
"""P1 pipeline (measurement): supply map, the ablation ladder E1c with the probes of E1b, and a feedback report.

Stages (finished runs under the same root are reused; nothing persists between sessions):

  supply     E1a: supply map of the schedulers, CPU only, seconds          -> <root>/supply/
  queue      E1b + E1c arms (and the FedAvg / DeComFL bounds) in priority
             order under a time budget, optionally several runs at a time  -> <root>/runs/, <root>/queue.json
  feedback   tables, the H1 and H1a computations, CSV files and a bundle   -> <root>/feedback/

The arms are defined in arms.json. Block 1 holds the five runs that decide H1a, block 2 adds the runs for H1 and
the two bounds, block 3 completes the ladder, block 4 holds the controls added after the first session (larger
server step for other arms, more zeroth-order directions, longer training; first seed of the session only).
`--tiny` replaces model and data by the synthetic tiny assets and runs on CPU: a plumbing check whose numbers
mean nothing.

The baseline regression (invariants I1 and I2) is not repeated here: run `experiments/p0/p0.py env golden regress`
in the same session before the queue.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ho_fl import analysis, grid, harness, supply  # noqa: E402
from ho_fl.grid import fmt, say, table  # noqa: E402

ARMS = json.loads((Path(__file__).parent / "arms.json").read_text())
DEFAULT_ROUNDS = ARMS["base"]["rounds"]
# Final accuracy (%) and standard deviation that the brief (section 7) quotes from the HO-FL paper for AG News,
# Dirichlet 0.5. Reference values, not results of this repository.
REFERENCE = {
    "qwen": {"A-b0": (85.51, 0.43), "A-b0.75": (86.69, 0.56), "fedavg": (88.15, 0.21), "decomfl": (55.83, 0.08)},
    "opt": {"A-b0": (59.58, 2.24), "A-b0.75": (62.85, 6.52), "fedavg": (78.24, 2.97), "decomfl": (23.38, 0.08)},
}
PLANNED_SEEDS = {"qwen": 3, "opt": 5}
# Accuracy levels (%) used to express how fast an arm learns ("rounds to reach").
TARGETS = {"qwen": (80.0, 85.0, 87.0), "opt": (40.0, 50.0, 60.0)}


def hardware_slug(device):
    import re

    import torch

    name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + "-torch" + torch.__version__.split("+")[0]


class Context:
    def __init__(self, args):
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tiny = args.tiny
        self.model = "opt" if args.tiny else args.model
        self.seeds = [int(part) for part in str(args.seeds).split(",")]
        self.rounds = args.rounds or (3 if args.tiny else DEFAULT_ROUNDS)
        self.device = "cpu" if args.tiny else args.device
        self.cache_dir = str(Path(args.cache_dir).resolve())
        self.blocks = sorted({int(part) for part in str(args.blocks).split(",")})
        self.jobs = max(1, args.jobs)
        self.deadline = args.deadline or (time.time() + 3600 * args.budget_hours if args.budget_hours else None)
        self.quiet = args.quiet
        self.assets = {}
        if self.tiny:
            from ho_fl.testing import build_tiny_assets
            self.assets = build_tiny_assets(self.root / "assets", datasets=("ag_news",))
        self.hardware = hardware_slug(self.device)
        self.head = grid.git_head(REPO)

    def common(self, fields):
        """Fields added to every run: where model and data come from and which device to use."""
        if self.tiny:
            return {key: value for key, value in self.assets["ag_news"].items() if key not in ("model", "dataset")}
        return {"device": self.device, "cache_dir": self.cache_dir}


def specs(ctx, blocks=None):
    """The runs of the plan in queue order: the ladder seed by seed (arms in the order of arms.json), then the
    arms marked ``first_seed_only`` for the first seed. An arm with its own ``rounds`` keeps its ratio to the base."""
    result = []
    for controls in (False, True):
        for seed in ctx.seeds[:1] if controls else ctx.seeds:
            for arm in ARMS["arms"]:
                if bool(arm.get("first_seed_only")) != controls or (blocks is not None and arm["block"] not in blocks):
                    continue
                rounds = ctx.rounds * arm.get("rounds", DEFAULT_ROUNDS) // DEFAULT_ROUNDS
                fields = {"model": ctx.model, **ARMS["base"], **arm["fields"], "seed": seed, "rounds": rounds}
                if arm.get("probes"):
                    probe_rounds = [r for r in (int(part) for part in ARMS["probe_rounds"].split(",")) if r <= rounds]
                    fields["probe_rounds"] = ",".join(str(r) for r in sorted(set(probe_rounds) | {rounds}))
                name = f"{ctx.model}-s{seed}-{arm['name']}" + ("" if ctx.rounds == DEFAULT_ROUNDS else f"-r{ctx.rounds}")
                result.append({"name": name, "arm": arm["name"], "seed": seed, "block": arm["block"],
                               "timing_key": f"{ctx.model}/{arm['timing_key']}", "fields": fields})
    return result


# --- supply (E1a) ---------------------------------------------------------------------------------------------------

def stage_supply(ctx):
    say(f"== supply (E1a, {ctx.model}) ==")
    result = supply.plan(ctx.model, seeds=10 if ctx.tiny else 200, rounds=160)
    text = supply.plan_table(ctx.model, result)
    out = ctx.root / "supply"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"supply_{ctx.model}.txt").write_text(text + "\n")
    harness.write_json(out / f"supply_{ctx.model}.json", result)
    analysis.write_csv(out / f"supply_{ctx.model}.csv", supply.plan_rows(ctx.model, result))
    say(text)
    return True


# --- queue (E1b, E1c) -----------------------------------------------------------------------------------------------

def stage_queue(ctx):
    plan = specs(ctx, ctx.blocks)
    say(f"== queue ({ctx.model}, seeds {ctx.seeds}, blocks {ctx.blocks}, {len(plan)} runs of {ctx.rounds} rounds, "
        f"up to {ctx.jobs} at a time, {ctx.hardware}) ==")
    from ho_fl.config import Config
    for spec in plan:  # every configuration is validated before anything is launched
        Config(**{**spec["fields"], **ctx.common(spec["fields"])}).resolve()
    record = grid.run_queue(plan, root=ctx.root, tree=REPO, hardware=ctx.hardware, revision=ctx.head, common=ctx.common,
                            deadline=ctx.deadline, jobs=ctx.jobs)
    # The record of a queue covers the blocks of this call; keep the entries of the other blocks as they were.
    history = harness.read_json(ctx.root / "queue_all.json", {"entries": []})
    fresh = {entry["name"]: entry for entry in record["queue"]}
    history["entries"] = [fresh.pop(entry["name"], entry) for entry in history["entries"]] + list(fresh.values())
    history.update({key: record[key] for key in ("recorded_at", "hardware", "head", "jobs", "jobs_requested", "preflight_failures")})
    harness.write_json(ctx.root / "queue_all.json", history)
    say()
    say(table(["run", "status", "wall min", "estimate min", "note"],
              [[e["name"], e["status"] + (" (reused)" if e.get("reused") else ""), fmt(e["wall_s"] / 60, 1) if e.get("wall_s") else "",
                fmt(e["estimated_s"] / 60, 1) if e.get("estimated_s") else "", e.get("reason") or e.get("error") or ""] for e in record["queue"]]))
    return all(entry["status"] == "COMPLETED" for entry in record["queue"])


# --- feedback -------------------------------------------------------------------------------------------------------

def collect(ctx):
    plan = specs(ctx)
    arms = {arm["name"]: arm for arm in ARMS["arms"]}
    runs, accuracy, curves = [], {}, {}
    for spec in plan:
        run_dir = ctx.root / "runs" / spec["name"]
        if harness.run_status(run_dir) != "COMPLETED":
            continue
        row = analysis.summary_row(run_dir)
        row.update(arm=spec["arm"], block=spec["block"], what=arms[spec["arm"]]["what"])
        layers, probe = harness.read_jsonl(run_dir / "layers.jsonl"), harness.read_jsonl(run_dir / "probe.jsonl")
        runs.append({"summary": row, "dir": str(run_dir), "curve": analysis.curve_rows(run_dir),
                     "layers": analysis.layer_summary(layers) if layers else None,
                     "probe": analysis.probe_summary(probe) if probe else None,
                     "probe_errors": [r for r in probe if r.get("scope") == "error"],
                     "supply": supply.empirical(run_dir) if row["method"] == "hofl" else None})
        accuracy.setdefault(spec["arm"], {})[spec["seed"]] = row["final_accuracy_pct"]
        curves.setdefault(spec["arm"], {})[spec["seed"]] = runs[-1]["curve"]
    comparable = ctx.rounds == DEFAULT_ROUNDS and not ctx.tiny
    planned = PLANNED_SEEDS.get(ctx.model, 3)
    timings = harness.read_json(ctx.root / f"timings-{ctx.hardware}.json", {})
    metas = [harness.read_json(path) or {} for path in sorted(ctx.root.rglob("run_meta.json"))]
    return {"generated_at": grid.now(), "tiny": ctx.tiny, "model": ctx.model, "seeds": ctx.seeds, "rounds": ctx.rounds,
            "comparable": comparable, "head": ctx.head, "hardware": ctx.hardware, "runs": runs, "accuracy": accuracy,
            "h1a": analysis.decide_h1a(accuracy, planned), "h1": analysis.decide_h1(accuracy, planned),
            "control": analysis.step_size_control(accuracy),
            "speedup": analysis.sampling_speedup(curves, max(1, ctx.rounds // 4)),
            "targets": TARGETS.get(ctx.model, TARGETS["qwen"]),
            "queue": harness.read_json(ctx.root / "queue_all.json"), "timings": timings,
            "supply_text": (ctx.root / "supply" / f"supply_{ctx.model}.txt").read_text() if (ctx.root / "supply" / f"supply_{ctx.model}.txt").exists() else None,
            "wall_hours": sum(meta.get("wall_s") or 0 for meta in metas) / 3600,
            "planned_runs": len(plan), "planned_seeds": planned,
            "regression": regression_verdict(ctx)}


def regression_verdict(ctx):
    """Verdicts of the P0 regression stage for this revision and hardware, if it was run under the usual root."""
    for candidate in sorted(ctx.root.parent.glob("p0*/regress/*/verdicts.json")):
        verdicts = harness.read_json(candidate) or {}
        if verdicts.get("head") == ctx.head and ctx.hardware in str(candidate):
            return {"i1": {name: entry["verdict"] for name, entry in verdicts["i1"].items()}, "i2": verdicts["i2"]["verdict"],
                    "golden_mode": verdicts.get("golden_mode"), "path": str(candidate)}
    return None


def accuracy_table(data):
    reference = REFERENCE.get(data["model"], {}) if data["comparable"] else {}
    rows = []
    for arm in ARMS["arms"]:
        members = sorted((run["summary"] for run in data["runs"] if run["summary"]["arm"] == arm["name"]), key=lambda row: row["seed"])
        if not members:
            rows.append([arm["name"], arm["what"], "not run", "", "", "", "", "", ""])
            continue
        values = [row["final_accuracy_pct"] for row in members]
        paper = reference.get(arm["name"])
        rows.append([arm["name"], arm["what"], ", ".join(str(row["seed"]) for row in members), ", ".join(f"{v:.2f}" for v in values),
                     f"{statistics.mean(values):.2f}", fmt(statistics.mean(row["final_eval_loss"] for row in members), 4),
                     fmt(statistics.mean(row["forward_equivalents_per_round"] for row in members), 1),
                     fmt(statistics.mean(row["uplink_bytes_per_round"] for row in members) / 1e6, 2),
                     f"{paper[0]:.2f} ({paper[1]:.2f})" if paper else ""])
    return table(["arm", "what it is", "seeds", "final accuracy % per seed", "mean", "eval loss", "forward-eq. per round",
                  "uplink MB per round", "paper (sd)"], rows)


def speed_table(data):
    """How fast each arm learns: accuracy after the base number of rounds, rounds needed for fixed accuracy levels."""
    window, targets, rows = data["rounds"], data["targets"], []

    def average(values, digits):
        values = [value for value in values if value is not None]
        return fmt(statistics.mean(values), digits) if values else "n/a"

    for arm in ARMS["arms"]:
        members = [run for run in data["runs"] if run["summary"]["arm"] == arm["name"]]
        if not members:
            continue
        reached = []
        for target in targets:
            hits = [analysis.rounds_to_reach(run["curve"], target) for run in members]
            reached.append(average(hits, 0) + (f" ({sum(hit is not None for hit in hits)} of {len(hits)})" if None in hits and any(hit is not None for hit in hits) else "")
                           if any(hit is not None for hit in hits) else "not reached")
        rows.append([arm["name"], len(members), members[0]["summary"]["rounds"], average([analysis.accuracy_at(run["curve"], window) for run in members], 2),
                     average([run["summary"]["final_accuracy_pct"] for run in members], 2), *reached,
                     average([analysis.mean_accuracy(run["curve"], window) for run in members], 2)])
    return table(["arm", "seeds", "rounds", f"accuracy % at round {window}", "final accuracy %",
                  *[f"rounds to reach {target:g}%" for target in targets], f"mean accuracy % over rounds 0-{window}"], rows)


def step_ratio_table(data):
    """Norm of the applied update per segment relative to the baseline A-b0 of the same seed."""
    base = {run["summary"]["seed"]: {entry["segment"]: entry["update_norm"] for entry in run["layers"]}
            for run in data["runs"] if run["layers"] and run["summary"]["arm"] == "A-b0"}
    rows = []
    for run in data["runs"]:
        reference = base.get(run["summary"]["seed"])
        if not run["layers"] or not reference or run["summary"]["arm"] == "A-b0":
            continue
        ratios = [fmt(entry["update_norm"] / reference[entry["segment"]], 2) if reference.get(entry["segment"]) else "n/a" for entry in run["layers"]]
        rows.append([run["summary"]["arm"], run["summary"]["seed"], *ratios])
    segments = max((len(run["layers"]) for run in data["runs"] if run["layers"]), default=0)
    return table(["arm", "seed", *[f"S{index}" for index in range(segments)]], rows) if rows else None


def render(data):
    out = []
    add = out.append
    add("# P1 feedback" + (" (TINY DRY RUN: plumbing check on CPU, the numbers mean nothing)" if data["tiny"] else ""))
    add(f"\nGenerated {data['generated_at']} for revision `{data['head'][:12]}` on `{data['hardware']}`. Model `{data['model']}`, "
        f"AG News Dirichlet 0.5, seeds {data['seeds']}, {data['rounds']} rounds per run; {len(data['runs'])} of {data['planned_runs']} planned runs completed.\n")
    if not data["comparable"] and not data["tiny"]:
        add("The runs do not have the default number of rounds, so nothing below is compared with the paper.\n")

    add("## 1. Baseline regression in this session\n")
    regression = data["regression"]
    if regression is None:
        add("No regression verdict for this revision and GPU was found next to this root. Without it the runs below "
            "are not known to start from an intact baseline (invariants I1, I2).")
    else:
        add(f"I1: {regression['i1']} (golden mode: {regression['golden_mode']}); I2 replay: {regression['i2']}.")

    add("\n## 2. E1c: ablation ladder\n")
    add(accuracy_table(data))
    add("\nForward-equivalents: one forward pass through all blocks for one batch; a backward pass costs two (accounting model, not a measurement).")
    add("\nSpeed of learning. No arm has converged after the base number of rounds, so the final accuracy largely measures speed:\n")
    add(speed_table(data))

    add("\n## 3. H1a: is the gain of biased sampling a matter of dilution?\n")
    h1a = data["h1a"]
    if h1a["verdict"] == "NOT_RUN":
        add("Needs arms A-b0, C-b0 and A-b0.75 of one seed; not available yet.")
    else:
        add(table(["seed", "G_C = C(b0) - A(b0)", "G_beta = A(b0.75) - A(b0)", "G_lr = max(Ax2, Ax3) - A(b0)"],
                  [[row["seed"], f"{row['g_c']:+.2f}", f"{row['g_beta']:+.2f}", f"{row['g_lr']:+.2f}" if row["g_lr"] is not None else "n/a"] for row in h1a["per_seed"]]
                  + [["mean", f"{h1a['g_c']:+.2f}", f"{h1a['g_beta']:+.2f}", f"{h1a['g_lr']:+.2f}" if h1a["g_lr"] is not None else "n/a"]]))
        add(f"\nRule of the brief: supported when mean G_C >= 0.7 x mean G_beta and mean G_lr < 0.5 x mean G_C. "
            f"**{h1a['verdict']}**" + (" (preliminary: " + f"{len(h1a['seeds'])} of {data['planned_seeds']} seeds)" if h1a["preliminary"] else "")
            + f": {h1a.get('reason', '')}.")

    control = data["control"]
    if control["seeds"]:
        show = lambda value: f"{value:+.2f}" if value is not None else "n/a"  # noqa: E731
        add("\nStep-size control (uniform sampling): does a larger server step alone give what first-order-only averaging gives?\n")
        add(table(["seed", "C - A", "A x2 - A", "A x3 - A", "C x2 - A x2 (same step)"],
                  [[row["seed"], show(row["g_c"]), show(row["g_x2"]), show(row["g_x3"]), show(row["matched_x2"])] for row in control["per_seed"]]
                  + [["mean", show(control["g_c"]), show(control["g_x2"]), show(control["g_x3"]), show(control["matched_x2"])]]))
        add("\n" + {True: "The larger step alone reaches at least half of the gain of first-order-only averaging, so that gain cannot be "
                            "attributed to the electorate by this comparison; the same-step column is the cleaner one.",
                     False: "The larger step alone stays below half of the gain of first-order-only averaging.",
                     None: "Not enough runs to compare, or first-order-only averaging gained nothing."}[control["explained"]])

    add("\n## 4. H1: how much do the zeroth-order updates contribute?\n")
    add("Model level, A (zeroth-order on) minus B (off, same weight 1/K):\n")
    add(table(["sampling", "seeds", "A - B per seed", "mean", "seed sd of A", "verdict"],
              [[beta, ", ".join(str(s) for s in entry["seeds"]) or "none",
                ", ".join(f"{row['a_minus_b']:+.2f}" for row in entry.get("per_seed", [])), f"{entry['a_minus_b']:+.2f}" if "a_minus_b" in entry else "",
                fmt(entry.get("seed_sd_of_a"), 2), entry["verdict"] + (f" ({entry['reason']})" if entry.get("reason") else "")]
               for beta, entry in data["h1"].items()]))
    probed = [run for run in data["runs"] if run["probe"]]
    if probed:
        add("\nReport level (probe): each client's actual update against its first-order counterfactual from the same state "
            "and batches. `zo`: all layers below the client's cut pooled; `fo`: all layers at or above it (self-check, close to 1). "
            "The null value is 1/sqrt(dimension), the scatter of the cosine of a random direction.\n")
        for run in probed:
            add(f"`{run['summary']['run']}`:\n")
            add(table(["round", "tier", "layers", "clients", "mean cosine", "standard error", "null 1/sqrt(d)", "median norm ratio actual/first-order"],
                      [[row["round"], f"T{row['tier'] + 1}", row["group"], row["clients"], fmt(row["cosine_mean"], 4), fmt(row["cosine_se"], 4),
                        fmt(row["null_cosine"], 4), fmt(row["norm_ratio_median"], 3)] for row in run["probe"]]))
            for error in run["probe_errors"]:
                add(f"\nProbe failed at round {error['round']}: {error['error']}")
            add("")
        add("Reading: H1 expects the `zo` cosine in the 1e-3 to 1e-2 range and the norm ratio between 0.3 and 3. With AdamW "
            "every coordinate moves by about the learning rate per step whatever the gradient, so the norm ratio says little; "
            "the cosine and A minus B carry the test. `zo_lora_A` at round 1 has no first-order direction to compare with "
            "(LoRA B starts at zero, so the gradient of A is zero at the first step).")

    with_layers = [run for run in data["runs"] if run["layers"]]
    if with_layers:
        add("\n## 5. Layer statistics (averages over rounds and over the layers of a segment)\n")
        add("S0 lies below every cut (no first-order source), S1 to S3 start at the three cuts. Norms are of the per-layer update vectors.\n")
        rows = []
        for run in with_layers:
            for entry in run["layers"]:
                rows.append([run["summary"]["arm"], run["summary"]["seed"], f"S{entry['segment']}", fmt(entry["n_fo"], 2), fmt(entry["n_zo"], 2),
                             fmt(entry["fo_norm"], 5), fmt(entry["zo_norm"], 5), fmt(entry["fo_mean_norm"], 5), fmt(entry["zo_mean_norm"], 5),
                             fmt(entry["cos_fo_fo"], 3), fmt(entry["cos_zo_zo"], 4), fmt(entry["cos_fo_zo"], 4), fmt(entry["update_norm"], 5),
                             ", ".join(f"{rule} {100 * share:.0f}%" for rule, share in entry["rules"].items())])
        add(table(["arm", "seed", "segment", "first-order reports", "zeroth-order reports", "norm of a first-order component",
                   "norm of a zeroth-order component", "norm of the first-order mean", "norm of the zeroth-order mean", "cosine FO-FO",
                   "cosine ZO-ZO", "cosine FO-ZO", "norm of the applied update", "electorate rule"], rows))

        ratios = step_ratio_table(data)
        if ratios:
            add("\nNorm of the applied update relative to the baseline A-b0 of the same seed (an arm that averages first-order "
                "reports only, or scales the server step, acts as a larger step on some or all segments):\n")
            add(ratios)

    supplied = [run for run in data["runs"] if run["supply"]]
    add("\n## 6. First-order supply\n")
    if data["supply_text"]:
        add("Supply map (E1a, no training):\n\n```\n" + data["supply_text"].rstrip() + "\n```")
    if supplied:
        add("\nRealised in the runs (one run scatters around the expectations above; the largest share is biased upwards):\n")
        add(table(["run", "mean width S1 / S2 / S3", "rounds with S1 width 0", "S1 width <= 2", "S1 sources", "S1 N_eff", "largest presence"],
                  [[run["summary"]["run"], " / ".join(f"{s['width_mean']:.2f}" for s in run["supply"]["segments"][-3:]),
                    f"{100 * run['supply']['segments'][-3]['p_width_0']:.1f}%", f"{100 * run['supply']['segments'][-3]['p_width_le2']:.1f}%",
                    run["supply"]["segments"][-3]["sources"], fmt(run["supply"]["segments"][-3]["n_eff"], 2),
                    f"{100 * run['supply']['max_presence']:.1f}%"] for run in supplied]))

    add("\n## 7. Queue, timing and GPU hours\n")
    queue = data["queue"]
    if queue:
        add(f"Runs at a time: {queue.get('jobs')} (requested {queue.get('jobs_requested')}).\n")
        add(table(["run", "block", "status", "wall min", "estimated min", "note"],
                  [[e["name"], e["block"], e["status"] + (" (reused)" if e.get("reused") else ""), fmt(e["wall_s"] / 60, 1) if e.get("wall_s") else "",
                    fmt(e["estimated_s"] / 60, 1) if e.get("estimated_s") else "", e.get("reason") or e.get("error") or ""]
                   for e in queue["entries"]]))
        if queue.get("preflight_failures"):
            add("\nPreflight failures: " + json.dumps(queue["preflight_failures"]))
    if data["timings"]:
        add("\nMeasured alone on the GPU (three-round preflight, or a full run when one run at a time is used):\n")
        add(table(["cost profile", "s per round", "s per evaluation", "start-up s", "peak GPU memory GiB", "estimated min per 160-round run", "measured on"],
                  [[key, fmt(t["round_s"], 2), fmt(t["eval_s"], 2), fmt(t["startup_s"], 1), fmt(t["peak_mb"] / 1024, 1) if t.get("peak_mb") is not None else "n/a",
                    fmt(harness.estimate_seconds(t, DEFAULT_ROUNDS, 10) / 60, 1), t.get("source")] for key, t in data["timings"].items()]))
        shared = {key: t["shared"] for key, t in data["timings"].items() if "shared" in t}
        if shared:
            add("\nWhile sharing the GPU: " + "; ".join(f"{key}: {t['round_s']:.2f} s per round with {t['jobs']} runs at a time" for key, t in shared.items()) + ".")
    add(f"\nWall time of all runs recorded under this root: {data['wall_hours']:.2f} h (sum over runs; with several runs at a time the session is shorter).")

    add("\n## 8. What this means and what to do next\n")
    for line in findings(data):
        add(f"- {line}")
    return "\n".join(out) + "\n"


def findings(data):
    lines = []
    if data["tiny"]:
        lines.append("This was the tiny dry run. It only shows that every stage executes; switch it off for real results.")
    if data["regression"] is None and not data["tiny"]:
        lines.append("Run the baseline regression (P0 stages env, golden, regress) in the same session: these results build on I1.")
    elif data["regression"] and (not all(v.startswith("PASS") for v in data["regression"]["i1"].values()) or data["regression"]["i2"] != "PASS"):
        lines.append("The baseline regression did not pass in this session. Treat every number in this report as unverified until it does.")
    reference = REFERENCE.get(data["model"], {}) if data["comparable"] else {}
    for arm, (paper, sd) in reference.items():
        values = list(data["accuracy"].get(arm, {}).values())
        if values:
            difference = statistics.mean(values) - paper
            lines.append(f"E0 check, {arm}: {statistics.mean(values):.2f} against the paper's {paper:.2f} ({difference:+.2f} points, "
                         f"{abs(difference) / sd:.1f} paper standard deviations, {len(values)} of {data['planned_seeds']} seeds; "
                         f"the acceptance rule max(1.0, 2 sd) applies to the mean over all seeds).")
    h1a = data["h1a"]
    if h1a["verdict"] in ("SUPPORTED", "NOT_SUPPORTED"):
        lines.append(f"H1a {h1a['verdict']}" + (" on a preliminary basis" if h1a["preliminary"] else "") + f": {h1a['reason']}."
                     + (" A single seed cannot separate this from seed noise: the paper's seed standard deviation for these runs is of the same order as the gains." if len(h1a["seeds"]) == 1 else ""))
    elif h1a["verdict"] == "CANNOT_JUDGE":
        lines.append(f"H1a cannot be judged: {h1a['reason']}.")
    control, speedup = data["control"], data["speedup"]
    if control["explained"] is not None:
        lines.append(f"Step-size control: first-order-only averaging gains {control['g_c']:+.2f} points over the baseline; doubling the server step gains "
                     f"{fmt(control['g_x2'], 2)}, tripling it {fmt(control['g_x3'], 2)}. "
                     + ("A larger step alone explains the gain by the rule of the brief. " if control["explained"] else "A larger step alone does not explain the gain. ")
                     + (f"At the same doubled step, first-order-only averaging is {control['matched_x2']:+.2f} points from the baseline." if control["matched_x2"] is not None
                        else "The same-step comparison (C x2 against A x2) has not been run."))
    if speedup["per_seed"]:
        lines.append(f"Biased sampling at round {speedup['round']}: {speedup['gain_zo_on']:+.2f} points over uniform sampling with the zeroth-order updates on, "
                     f"{speedup['gain_zo_off']:+.2f} with them off; the two biased-sampling curves are never more than {speedup['max_gap_on_off']:.2f} points apart. "
                     "A gain that is the same without zeroth-order updates cannot come from lower zeroth-order variance.")
    for beta, entry in data["h1"].items():
        if "a_minus_b" in entry:
            lines.append(f"H1 ({beta}): switching the zeroth-order updates off changes the final accuracy by {-entry['a_minus_b']:+.2f} points "
                         f"({entry['verdict']}" + (f", {entry['reason']}" if entry.get("reason") else "") + ").")
    for run in data["runs"]:
        if run["probe"]:
            zo = [row for row in run["probe"] if row["group"] == "zo" and row["cosine_mean"] is not None]
            fo = [row for row in run["probe"] if row["group"] == "fo" and row["cosine_mean"] is not None]
            if zo:
                lines.append(f"Probe, `{run['summary']['arm']}` seed {run['summary']['seed']}: pooled zeroth-order cosine between "
                             f"{min(row['cosine_mean'] for row in zo):+.4f} and {max(row['cosine_mean'] for row in zo):+.4f} over rounds and tiers "
                             f"(null scatter about {statistics.mean(row['null_cosine'] for row in zo):.4f}); first-order self-check cosine at least "
                             f"{min(row['cosine_mean'] for row in fo):.4f}." if fo else "")
    queue = data["queue"]
    if queue:
        counts = {}
        for entry in queue["entries"]:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        pending = [entry["name"] for entry in queue["entries"] if entry["status"] != "COMPLETED"]
        if pending:
            lines.append(f"Runs that did not complete: {counts}. A later session has to repeat them ({', '.join(pending)}); nothing is carried over between sessions.")
    missing = data["planned_runs"] - len(data["runs"])
    if missing and not queue:
        lines.append(f"{missing} planned runs have not been run.")
    lines.append("Gate G1 needs the H1 and H1a results for the research lead's decision; bring this report back.")
    return [line for line in lines if line]


def stage_feedback(ctx):
    say("== feedback ==")
    data = collect(ctx)
    out = ctx.root / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    harness.write_json(out / "feedback_P1.json", data)
    text = render(data)
    (out / "feedback_P1.md").write_text(text)
    run_dirs = [run["dir"] for run in data["runs"]]
    analysis.write_csv(out / "summary.csv", [run["summary"] for run in data["runs"]])
    analysis.write_csv(out / "curves.csv", [row for run_dir in run_dirs for row in analysis.curve_rows(run_dir)])
    analysis.write_csv(out / "layers.csv", [row for run_dir in run_dirs for row in analysis.log_rows(run_dir, "layers.jsonl")])
    analysis.write_csv(out / "probe.csv", [row for run_dir in run_dirs for row in analysis.log_rows(run_dir, "probe.jsonl")])
    grid.write_bundle(ctx.root, out / "feedback_P1_bundle.zip")
    say(f"H1a: {data['h1a']['verdict']}; H1: " + ", ".join(f"{beta} {entry['verdict']}" for beta, entry in data["h1"].items())
        + f"; {len(data['runs'])} of {data['planned_runs']} runs completed" if ctx.quiet else text)
    say(f"written: {out / 'feedback_P1.md'}, CSV files and {out / 'feedback_P1_bundle.zip'}")
    return True


STAGES = {"supply": stage_supply, "queue": stage_queue, "feedback": stage_feedback}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], help="one or more stages, run in the order given")
    parser.add_argument("--root", default="outputs/p1", help="directory that receives every result of the pipeline")
    parser.add_argument("--model", choices=["qwen", "opt"], default="qwen")
    parser.add_argument("--seeds", default="42", help="comma-separated seeds; the queue goes seed by seed")
    parser.add_argument("--blocks", default="1,2,3,4", help="blocks of arms.json to queue")
    parser.add_argument("--rounds", type=int, default=0, help="rounds per run (default 160; other values are not comparable with the paper)")
    parser.add_argument("--jobs", type=int, default=1, help="runs that may share the GPU; limited further by the measured memory per run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="cache", help="download and cache directory shared by all runs")
    parser.add_argument("--budget-hours", type=float, default=0.0, help="time budget of the queue, counted from now")
    parser.add_argument("--deadline", type=float, default=0.0, help="alternative to --budget-hours: absolute deadline as a Unix timestamp")
    parser.add_argument("--tiny", action="store_true", help="dry run on CPU with synthetic tiny assets")
    parser.add_argument("--quiet", action="store_true", help="feedback stage: print one line instead of the whole report")
    args = parser.parse_args(argv)
    ctx = Context(args)
    names = list(STAGES) if "all" in args.stages else args.stages
    results = {}
    for name in names:
        try:
            results[name] = bool(STAGES[name](ctx))
        except Exception as error:  # a broken stage must not hide the results of the others
            import traceback
            traceback.print_exc()
            say(f"stage {name} raised {error!r}")
            results[name] = False
        say()
    say("stage results: " + ", ".join(f"{name}={'ok' if ok else 'ATTENTION'}" for name, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
