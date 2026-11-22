#!/usr/bin/env python
"""P1 pipeline (measurement): supply map, the ablation ladder E1c with the probes of E1b, and a feedback report.

Stages (finished runs under the same root are reused; nothing persists between sessions):

  supply     E1a: supply map of the schedulers, CPU only, seconds          -> <root>/supply/
  queue      E1b + E1c arms (and the FedAvg / DeComFL bounds) in priority
             order under a time budget, optionally several runs at a time  -> <root>/runs/, <root>/queue.json
  feedback   tables, the H1 and H1a computations, CSV files and a bundle   -> <root>/feedback/

The arms are defined in arms.json. Block 1 holds the five runs that decide H1a, block 2 adds the runs for H1 and
the two bounds, block 3 completes the ladder; these ran in session 1 and are read from references/ instead of being
trained again (runs are deterministic on a given GPU and library version). Block 4 holds controls added after
session 1 (larger server step, longer training). Blocks 5 and 6 check the HO-FL authors' statements on the
zeroth-order updates under the authors' settings (AG News, uniform highest boundary; SQuAD, Top-FO-Only); the
numerical check of the zeroth-order estimator (`python -m ho_fl.zo_check`) writes into <root>/zo_check/ and is
reported here too. Block 7 is a pilot of E3: one faulty device under HO-FL's default, against its fault-free twin
A-b0.75-attr, with the resilience metrics of brief section 6.11. `--tiny` replaces model and data by the synthetic tiny assets and runs on CPU (SQuAD arms are
left out): a plumbing check whose numbers mean nothing.

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
# What the HO-FL authors state about the zeroth-order updates, with their numbers (Qwen2.5-1.5B), and the arms of
# this repository that reproduce each setting. "on" is HO-FL with zeroth-order updates, "off" the same with the
# bottom frozen (zeroth-order updates off, direct averaging). Single runs of ours against the authors' 3-seed means.
AUTHOR_STATEMENTS = [
    {"key": "top_fo_only_ag_news", "source": "rebuttal to reviewer 76MC (Top-FO-Only, freeze bottom, beta=0.75)",
     "task": "AG News Dirichlet 0.5, accuracy %", "metric": "accuracy", "authors_on": 86.69, "authors_off": 83.37, "on": "A-b0.75", "off": "B-b0.75"},
    {"key": "top_fo_only_squad", "source": "rebuttal to reviewer 76MC (Top-FO-Only, freeze bottom, beta=0.75)",
     "task": "SQuAD, full-validation F1", "metric": "f1", "authors_on": 86.92, "authors_off": 83.65, "on": "A-b0.75-squad", "off": "B-b0.75-squad"},
    {"key": "uniform_boundary", "source": "paper Table 9 (HO-FL, uniform highest ZO boundary, beta=0)",
     "task": "AG News Dirichlet 0.5, accuracy %", "metric": "accuracy", "authors_on": 63.22, "authors_off": None, "on": "Au24-b0", "off": "Bu24-b0"},
    {"key": "pure_zo", "source": "paper Tables 2 and 9 (DeComFL, every layer zeroth-order)",
     "task": "AG News Dirichlet 0.5, accuracy %", "metric": "accuracy", "authors_on": 55.83, "authors_off": None, "on": "decomfl", "off": None},
]
# Pre-registered reading of a Top-FO-Only comparison: the authors' drop is reproduced if ours is at least half of it.
REPRODUCED_SHARE = 0.5
# A drop of at least this many points when the zeroth-order updates are switched off counts as a contribution.
CONTRIBUTION_POINTS = 1.0
# Fault window of the E3 pilot (block 7; the rounds of experiments/faults/e3_*.json). A pilot fault whose largest drop
# stays below PILOT_VISIBLE points cannot separate configurations in E3 (the paper's seed sd of HO-FL is 0.56).
PILOT_WINDOW = (41, 80)
PILOT_VISIBLE = 1.0
# Accuracy levels (%) used to express how fast an arm learns ("rounds to reach").
TARGETS = {"qwen": (80.0, 85.0, 87.0), "opt": (40.0, 50.0, 60.0)}


def load_references(model, directory=Path(__file__).parent / "references"):
    """Runs of earlier sessions (see import_reference.py): ``[{"arm", "seed", "curve", ...}]`` for ``model``."""
    found = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("model") != model:
            continue
        for arm, run in record["runs"].items():
            found.append({**run, "arm": arm, "source": path.name, "revision": record.get("revision"), "hardware": record.get("hardware")})
    return found


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
        self.hardware = grid.hardware_slug(self.device)
        self.head = grid.git_head(REPO)
        self.references = [] if self.tiny else load_references(self.model)

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
                if ctx.tiny and arm.get("dataset") == "squad":
                    continue  # the tiny assets have no generation-ready SQuAD
                rounds = ctx.rounds * arm.get("rounds", DEFAULT_ROUNDS) // DEFAULT_ROUNDS
                fields = {"model": ctx.model, **ARMS["base"], **arm["fields"], "seed": seed, "rounds": rounds}
                if fields.get("uniform_cut") == "highest":
                    from ho_fl.config import SPECS
                    fields["uniform_cut"] = max(SPECS[ctx.model][3])
                if fields.get("fault_spec"):
                    fields["fault_spec"] = str(REPO / fields["fault_spec"])  # relative to the repository root
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
        status = harness.run_status(run_dir)
        # A run with a fault that diverged is a result too (it keeps its last finite evaluation).
        if status != "COMPLETED" and not (status == "DIVERGED" and spec["fields"].get("fault_spec")):
            continue
        row = analysis.summary_row(run_dir)
        row.update(arm=spec["arm"], block=spec["block"], what=arms[spec["arm"]]["what"])
        layers, probe = harness.read_jsonl(run_dir / "layers.jsonl"), harness.read_jsonl(run_dir / "probe.jsonl")
        runs.append({"summary": row, "dir": str(run_dir), "curve": analysis.curve_rows(run_dir),
                     "layers": analysis.layer_summary(layers) if layers else None,
                     "probe": analysis.probe_summary(probe) if probe else None,
                     "probe_errors": [r for r in probe if r.get("scope") == "error"],
                     "supply": supply.empirical(run_dir) if row["method"] == "hofl" else None})
        runs[-1]["attribution"] = attribution(run_dir, spec["fields"])
        accuracy.setdefault(spec["arm"], {})[spec["seed"]] = row["final_accuracy_pct"]
        curves.setdefault(spec["arm"], {})[spec["seed"]] = runs[-1]["curve"]
    # Arms of earlier sessions that this session did not train again.
    for reference in ctx.references if ctx.rounds == DEFAULT_ROUNDS else []:
        if reference["arm"] not in arms or reference["seed"] in accuracy.get(reference["arm"], {}):
            continue
        arm = arms[reference["arm"]]
        row = {"run": f"{ctx.model}-s{reference['seed']}-{reference['arm']}", "arm": reference["arm"], "block": arm["block"], "what": arm["what"],
               "seed": reference["seed"], "method": arm["fields"].get("method"), "rounds": reference["final_round"], "final_accuracy_pct": reference["final_accuracy_pct"],
               "final_eval_loss": reference["final_eval_loss"], "final_f1": None, "forward_equivalents_per_round": None, "uplink_bytes_per_round": None,
               "source": f"{reference['source']} (revision {str(reference['revision'])[:7]})", "hardware": reference["hardware"]}
        curve = [{"run": row["run"], **point, "train_loss": None, "cumulative_uplink_bytes": None} for point in reference["curve"]]
        runs.append({"summary": row, "dir": None, "curve": curve, "layers": None, "probe": None, "probe_errors": [], "supply": None, "attribution": None})
        accuracy.setdefault(reference["arm"], {})[reference["seed"]] = row["final_accuracy_pct"]
        curves.setdefault(reference["arm"], {})[reference["seed"]] = curve
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
            "regression": grid.regression_verdict(ctx.root.parent, ctx.head, ctx.hardware),
            "zo_check": harness.read_json(ctx.root / "zo_check" / "zo_check.json"),
            "canary": canary(runs, ctx), "statements": statements(runs), "pilot": fault_pilot(runs)}


def fault_pilot(runs):
    """E3 pilot (block 7): every faulty run against the same training without the fault, with the metrics of brief
    section 6.11. The twin is A-b0.75-attr of this session (evaluated every 4 rounds like the pilot), else A-b0.75 of
    session 1 (every 10 rounds, so only every 20th round is common)."""
    by_arm = {run["summary"]["arm"]: run for run in runs}
    twin = by_arm.get("A-b0.75-attr") or by_arm.get("A-b0.75")
    pilots = [run for run in runs if run["summary"].get("block") == 7 and run["dir"]]
    if twin is None or not pilots:
        return None
    clean_at = {int(point["round"]): point["accuracy_pct"] for point in twin["curve"]}
    entries, drops = [], {}
    for run in pilots:
        metrics = analysis.resilience_metrics(run["curve"], twin["curve"], PILOT_WINDOW)
        summary = harness.read_json(Path(run["dir"]) / "summary.json", {})
        drops[run["summary"]["arm"]] = dict((r, drop) for r, drop in (metrics or {}).get("drops", []))
        # Before the window both runs train identically, so these drops must be exactly zero.
        before = [abs(drop) for r, drop in drops[run["summary"]["arm"]].items() if r < PILOT_WINDOW[0]]
        entries.append({"arm": run["summary"]["arm"], "what": run["summary"]["what"], "status": run["summary"]["status"],
                        "diverged_round": summary.get("diverged_round"), "metrics": metrics, "max_gap_before_window": max(before, default=None)})
    marks = sorted({r for values in drops.values() for r in values if r % 8 == 0 and r >= PILOT_WINDOW[0] - 9})
    curve = [{"round": r, "clean": clean_at[r], **{arm: values.get(r) for arm, values in drops.items()}} for r in marks]
    return {"twin": twin["summary"]["arm"], "twin_source": twin["summary"].get("source", "this session"), "window": list(PILOT_WINDOW),
            "arms": entries, "curve": curve}


def attribution(run_dir, fields):
    rows = harness.read_jsonl(Path(run_dir) / "attribution.jsonl")
    return analysis.attribution_summary(rows, every=fields.get("attribution_every") or 1) if rows else None


def final_metric(summary, metric):
    return summary.get("final_f1") if metric == "f1" else summary.get("final_accuracy_pct")


def statements(runs):
    """The authors' statements on the zeroth-order updates next to ours (single runs; see AUTHOR_STATEMENTS)."""
    by_arm = {}
    for run in runs:
        by_arm.setdefault(run["summary"]["arm"], run["summary"])
    rows = []
    for statement in AUTHOR_STATEMENTS:
        on, off = by_arm.get(statement["on"]), by_arm.get(statement["off"]) if statement["off"] else None
        ours_on = final_metric(on, statement["metric"]) if on else None
        ours_off = final_metric(off, statement["metric"]) if off else None
        entry = {**statement, "ours_on": ours_on, "ours_off": ours_off,
                 "sources": [run.get("source", "this session") for run in (on, off) if run]}
        authors_drop = statement["authors_on"] - statement["authors_off"] if statement["authors_off"] is not None else None
        ours_drop = ours_on - ours_off if ours_on is not None and ours_off is not None else None
        entry.update(authors_drop=authors_drop, ours_drop=ours_drop)
        if ours_on is None or (statement["off"] and ours_off is None):
            entry["verdict"] = "NOT_RUN"
        elif authors_drop is not None:
            entry["verdict"] = "REPRODUCED" if ours_drop >= REPRODUCED_SHARE * authors_drop else "NOT_REPRODUCED"
        elif ours_drop is not None:
            entry["verdict"] = "ZO_CONTRIBUTES" if ours_drop >= CONTRIBUTION_POINTS else "NO_MEASURABLE_CONTRIBUTION"
        else:
            entry["verdict"] = "REPORTED"
        rows.append(entry)
    return rows


def canary(runs, ctx):
    """A-b0.75-attr trains exactly like A-b0.75 of session 1; identical curves show that pairing across sessions holds."""
    current = next((run for run in runs if run["summary"]["arm"] == "A-b0.75-attr" and run["dir"]), None)
    reference = next((ref for ref in ctx.references if ref["arm"] == "A-b0.75" and ref["seed"] == (current or {}).get("summary", {}).get("seed")), None)
    if current is None or reference is None:
        return None
    ours = {round(point["round"]): (round(point["accuracy_pct"], 2), round(point["eval_loss"], 4)) for point in current["curve"]}
    theirs = {point["round"]: (point["accuracy_pct"], point["eval_loss"]) for point in reference["curve"]}
    common = sorted(set(ours) & set(theirs))
    differing = [r for r in common if ours[r] != theirs[r]]
    return {"rounds_compared": len(common), "differing_rounds": differing, "identical": bool(common) and not differing,
            "same_hardware": reference["hardware"] == ctx.hardware, "reference_revision": reference["revision"]}


def accuracy_table(data):
    reference = REFERENCE.get(data["model"], {}) if data["comparable"] else {}
    rows = []
    for arm in ARMS["arms"]:
        members = sorted((run["summary"] for run in data["runs"] if run["summary"]["arm"] == arm["name"]), key=lambda row: row["seed"])
        if not members:
            rows.append([arm["name"], arm["what"], "not run", "", "", "", "", "", ""])
            continue
        values = [row["final_accuracy_pct"] if row["final_accuracy_pct"] is not None else row.get("final_f1") for row in members]
        paper = reference.get(arm["name"])

        def mean_of(key, scale=1.0, digits=2):
            present = [row[key] for row in members if row.get(key) is not None]
            return fmt(statistics.mean(present) / scale, digits) if present else "n/a"

        label = arm["name"] + (" (F1)" if members[0].get("final_accuracy_pct") is None else "")
        sources = sorted({row.get("source", "this session") for row in members})
        rows.append([label, arm["what"], ", ".join(str(row["seed"]) for row in members), ", ".join(f"{v:.2f}" if v is not None else "n/a" for v in values),
                     f"{statistics.mean(v for v in values if v is not None):.2f}" if any(v is not None for v in values) else "n/a",
                     mean_of("final_eval_loss", digits=4), mean_of("forward_equivalents_per_round", digits=1), mean_of("uplink_bytes_per_round", 1e6, 2),
                     f"{paper[0]:.2f} ({paper[1]:.2f})" if paper else "", "; ".join(sources)])
    return table(["arm", "what it is", "seeds", "final accuracy % per seed", "mean", "eval loss", "forward-eq. per round",
                  "uplink MB per round", "paper (sd)", "result from"], rows)


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

    add("\n## 2. Do the zeroth-order updates help? The HO-FL authors' statements next to ours\n")
    add(zo_section(data))

    add("\n## 2b. E1c: ablation ladder\n")
    add(accuracy_table(data))
    add("\nForward-equivalents: one forward pass through all blocks for one batch; a backward pass costs two (accounting model, not a measurement).")
    add("\nSpeed of learning. No arm has converged after the base number of rounds, so the final accuracy largely measures speed:\n")
    add(speed_table(data))

    if data.get("pilot"):
        add("\n## 2c. E3 pilot: how much does one faulty device hurt HO-FL's default?\n")
        add(pilot_section(data["pilot"]))

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


def zo_section(data):
    out = []
    add = out.append
    show = lambda value, digits=2: "n/a" if value is None else f"{value:.{digits}f}"  # noqa: E731
    signed = lambda value: "n/a" if value is None else f"{value:+.2f}"  # noqa: E731
    add("Authors' numbers are 3-seed means (Qwen2.5-1.5B); ours are single runs of seed 42 under the same settings. "
        f"A Top-FO-Only drop counts as reproduced if ours is at least {REPRODUCED_SHARE:.0%} of the authors'; without an authors' "
        f"number, switching the zeroth-order updates off has to cost at least {CONTRIBUTION_POINTS:g} point to count as a contribution.\n")
    add(table(["statement (source)", "task and metric", "authors: ZO on / off (drop)", "ours: ZO on / off (drop)", "verdict", "our results from"],
              [[s["source"], s["task"], f"{show(s['authors_on'])} / {show(s['authors_off'])} ({signed(s['authors_drop'])})",
                f"{show(s['ours_on'])} / {show(s['ours_off'])} ({signed(s['ours_drop'])})", s["verdict"], "; ".join(s["sources"]) or ""]
               for s in data["statements"]]))
    canary_record = data.get("canary")
    if canary_record:
        add(f"\nCross-session check: A-b0.75-attr (this session) against A-b0.75 (session 1, revision {str(canary_record['reference_revision'])[:7]}): "
            + (f"identical at all {canary_record['rounds_compared']} evaluations, so results of the two sessions can be paired."
               if canary_record["identical"] else f"DIFFERENT at rounds {canary_record['differing_rounds'][:8]}"
               + (" (same GPU model and torch version)" if canary_record["same_hardware"] else " (different GPU or torch version)")
               + "; pair only arms trained in this session."))
    attributed = [run for run in data["runs"] if run.get("attribution")]
    if attributed:
        add("\nAttribution: first-order decrease of the loss on 1,024 fixed evaluation examples, summed over the measured rounds (every "
            "4th), split into the part due to first-order and to zeroth-order components of the applied update. A zeroth-order share "
            "near 0 within its standard error means the zeroth-order components did not lower the loss.\n")
        rows = []
        for run in attributed:
            for entry in run["attribution"]["segments"]:
                rows.append([run["summary"]["arm"], f"S{entry['segment']}", show(entry.get("n_fo"), 1), show(entry.get("n_zo"), 1),
                             f"{entry.get('pred_fo', 0):.4g}" if entry.get("pred_fo") is not None else "n/a",
                             f"{entry.get('pred_zo', 0):.4g}" if entry.get("pred_zo") is not None else "n/a",
                             (f"{entry['zo_share']:+.3f} ± {entry['zo_share_se']:.3f}" if entry.get("zo_share_se") is not None else show(entry.get("zo_share"), 3)),
                             fmt(entry.get("cos_fo"), 3), fmt(entry.get("cos_zo"), 3)])
        add(table(["arm", "segment", "first-order reports", "zeroth-order reports", "predicted decrease, first-order part",
                   "predicted decrease, zeroth-order part", "zeroth-order share (± se)", "cosine first-order part", "cosine zeroth-order part"], rows))
        for run in attributed:
            check = run["attribution"]["first_order_check"]
            if check:
                add(f"\nFirst-order check for {run['summary']['arm']}: measured change of the evaluation loss {check['actual_change']:+.4f}, "
                    f"predicted {check['predicted_change']:+.4f} over {check['intervals']} intervals (correlation {fmt(check['correlation'], 2)}).")
    check = data.get("zo_check")
    if check:
        add("\nNumerical check of the zeroth-order estimator (no training): HO-FL's scalar v / mu against the exact directional derivative "
            "from autograd, on the directions training draws. Correlation and slope 1 mean the finite difference is accurate. The last "
            "column is the cosine of the q-direction estimate with the gradient: HO-FL's / built from exact derivatives / theory sqrt(q/d).\n")
        rows = []
        for row in check["rows"]:
            information = {entry["q"]: entry for entry in row["information"]}
            def cell(q):
                entry = information.get(q)
                return "n/a" if entry is None else f"{entry['cosine_hofl']:.4f} / {entry['cosine_exact_derivatives']:.4f} / {entry['theory']:.4f}"
            rows.append([row["state"], row["precision"], f"{row['mu']:g}", row["cut"], row["directions"], fmt(row["correlation"], 3), fmt(row["slope"], 3),
                         fmt(row["sign_agreement"], 3), cell(2), cell(max(information))])
        add(table(["state", "backbone", "mu", "cut", "directions", "correlation", "slope", "sign agreement", "cosine q=2", f"cosine q={check['q']}"], rows))
    return "\n".join(out)


def pilot_section(pilot):
    out = []
    add = out.append
    signed = lambda value: "n/a" if value is None else f"{value:+.2f}"  # noqa: E731
    first, last = pilot["window"]
    add(f"One preferred T1 device (in the cohort in about 80% of rounds under beta = 0.75) is faulty in rounds {first} to {last}. "
        f"Each faulty run is compared with the same training without the fault, `{pilot['twin']}` ({pilot['twin_source']}): same seed, "
        f"data stream and first {first - 1} rounds, so every gap is the effect of the fault. Metrics of brief section 6.11, in accuracy "
        "points (drop = without the fault minus with it); recovery counts the rounds after the window until the drop stays within "
        "0.5 for two evaluations.\n")
    rows = []
    for entry in pilot["arms"]:
        metrics = entry["metrics"]
        status = entry["status"] + (f" at round {entry['diverged_round']}" if entry.get("diverged_round") else "")
        if metrics is None:
            rows.append([entry["arm"], entry["what"], status, "no common evaluation in the window", "", "", "", ""])
            continue
        rows.append([entry["arm"], entry["what"], status, signed(metrics["mean_drop"]), f"{signed(metrics['max_drop'])} (round {metrics['max_drop_round']})",
                     "censored" if metrics["recovery_censored"] else str(metrics["recovery_rounds"]),
                     f"{signed(metrics['final_gap'])} (round {metrics['final_round']})", fmt(entry["max_gap_before_window"], 2)])
    add(table(["arm", "fault", "status", "mean drop in window", "largest drop", "recovery rounds", "final gap",
               f"largest gap before round {first} (must be 0)"], rows))
    if pilot["curve"]:
        arms = [entry["arm"] for entry in pilot["arms"]]
        add("\nAccuracy without the fault, and the drop each fault causes:\n")
        add(table(["round", "no fault %", *[f"drop {arm}" for arm in arms]],
                  [[f"{point['round']} (fault)" if first <= point["round"] <= last else point["round"], f"{point['clean']:.2f}",
                    *[signed(point.get(arm)) for arm in arms]] for point in pilot["curve"]]))
    return "\n".join(out)


def findings(data):
    lines = []
    if data["tiny"]:
        lines.append("This was the tiny dry run. It only shows that every stage executes; switch it off for real results.")
    if data["regression"] is None and not data["tiny"]:
        lines.append("Run the baseline regression (P0 stages env, golden, regress) in the same session: these results build on I1.")
    for statement in data["statements"]:
        if statement["verdict"] in ("REPRODUCED", "NOT_REPRODUCED", "ZO_CONTRIBUTES", "NO_MEASURABLE_CONTRIBUTION"):
            lines.append(f"Zeroth-order updates, {statement['source']}, {statement['task']}: {statement['verdict']} "
                         f"(ours {statement['ours_drop']:+.2f} when switched off" + (f", authors {statement['authors_drop']:+.2f}" if statement["authors_drop"] is not None else "") + ").")
    canary_record = data.get("canary")
    if canary_record and not canary_record["identical"]:
        lines.append("A-b0.75-attr did not reproduce session 1: results of the two sessions must not be paired; compare only arms of this session.")
    for entry in (data.get("pilot") or {}).get("arms", []):
        metrics = entry["metrics"]
        if entry["status"] == "DIVERGED":
            lines.append(f"E3 pilot, {entry['arm']}: diverged at round {entry['diverged_round']}. Under the equal-weight mean this fault breaks "
                         "HO-FL's default, so every faulty E3 arm needs on_nonfinite=stop and divergence counts as not recovered.")
        elif metrics:
            lines.append(f"E3 pilot, {entry['arm']}: mean drop {metrics['mean_drop']:+.2f} points in the window, largest {metrics['max_drop']:+.2f} at round "
                         f"{metrics['max_drop_round']}, " + ("not recovered" if metrics["recovery_censored"] else f"recovered {metrics['recovery_rounds']} rounds after the window")
                         + f", final gap {metrics['final_gap']:+.2f}. "
                         + ("Measurable: E3 can separate configurations with this fault." if metrics["max_drop"] >= PILOT_VISIBLE else
                            f"Below {PILOT_VISIBLE:g} point: with this fault E3 could not separate configurations; strengthen it or change the window before coding the E3 grid."))
        if entry.get("max_gap_before_window"):
            lines.append(f"E3 pilot, {entry['arm']}: differs from its twin before the fault starts (up to {entry['max_gap_before_window']:.2f} points), so the pairing does not hold.")
    check = data.get("zo_check")
    if check:
        bf16 = [row for row in check["rows"] if row["precision"] == "bfloat16" and row["mu"] == 1e-3 and row["correlation"] is not None]
        fp32 = [row for row in check["rows"] if row["precision"] == "float32" and row["mu"] == 1e-3 and row["correlation"] is not None]
        if bf16:
            lines.append(f"Zeroth-order estimator at the training setting (bfloat16, mu=1e-3): correlation with the exact derivative "
                         f"{min(row['correlation'] for row in bf16):.3f} to {max(row['correlation'] for row in bf16):.3f} over cuts and states"
                         + (f"; float32 reference {min(row['correlation'] for row in fp32):.3f} to {max(row['correlation'] for row in fp32):.3f}" if fp32 else "")
                         + ". Close to 1: the finite difference is accurate and any weakness of the zeroth-order part is its information per step; "
                           "well below 1 in bfloat16 only: precision limits it.")
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
    run_dirs = [run["dir"] for run in data["runs"] if run["dir"]]
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
    parser.add_argument("--blocks", default="5,7,6,4", help="blocks of arms.json to queue (blocks 1-3 ran in session 1 and come from references/)")
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
