#!/usr/bin/env python
"""P2 pipeline, first part: the dose-response experiment E2 (hypothesis H2, the quorum crossover).

Stages (finished runs under the same root are reused; nothing persists between sessions):

  base       one HO-FL run with the default settings whose last state is the common starting point  -> <root>/runs/
  dose       schedule and fault files, then the arms in priority order under a time budget:
             dose m (honest first-order reports next to the unreliable one) x aggregation rule x
             {no fault, flip20, flip1}                                                               -> <root>/runs/
  feedback   loss, accuracy and influence tables, the H2 computation and a bundle                   -> <root>/feedback/

The arms are defined in ho_fl/dose.py. `--tiny` runs a reduced version on CPU with the synthetic tiny assets: a
plumbing check whose numbers mean nothing. Run `experiments/p0/p0.py env golden regress` in the same session first.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ho_fl import analysis, dose, grid, harness  # noqa: E402
from ho_fl.grid import fmt, say, table  # noqa: E402

BASE = {"dataset": "ag_news", "partition": "dirichlet_05"}
BASE_ROUNDS, DOSE_ROUNDS = 40, 20


class Context:
    def __init__(self, args):
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tiny = args.tiny
        self.model = "opt" if args.tiny else args.model
        self.seed = args.seed
        self.device = "cpu" if args.tiny else args.device
        self.cache_dir = str(Path(args.cache_dir).resolve())
        self.jobs = max(1, args.jobs)
        self.deadline = args.deadline or (time.time() + 3600 * args.budget_hours if args.budget_hours else None)
        self.quiet = args.quiet
        self.base_rounds, self.dose_rounds = (4, 3) if args.tiny else (BASE_ROUNDS, DOSE_ROUNDS)
        self.doses = [int(part) for part in (args.doses or ("2,1" if args.tiny else ",".join(str(m) for m in dose.M_ORDER))).split(",")]
        self.faults = [part for part in (args.faults or ("none,flip20" if args.tiny else ",".join(dose.FAULTS))).split(",")]
        if "none" not in self.faults or not set(self.faults) <= set(dose.FAULTS) or not set(self.doses) <= set(range(6)):
            raise ValueError("--faults needs 'none' plus flip20 and/or flip1; --doses takes values from 0 to 5")
        self.assets = {}
        if self.tiny:
            from ho_fl.testing import build_tiny_assets
            self.assets = build_tiny_assets(self.root / "assets", datasets=("ag_news",))
        self.hardware = grid.hardware_slug(self.device)
        self.head = grid.git_head(REPO)
        self.base_name = f"{self.model}-s{self.seed}-dose-base"
        self.base_dir = self.root / "runs" / self.base_name

    def common(self, fields):
        if self.tiny:
            return {key: value for key, value in self.assets["ag_news"].items() if key not in ("model", "dataset")}
        return {"device": self.device, "cache_dir": self.cache_dir}

    def base_spec(self):
        fields = {"model": self.model, **BASE, "method": "hofl", "beta": 0.75, "seed": self.seed, "rounds": self.base_rounds,
                  "evaluate_every": self.base_rounds, "save_state_rounds": str(self.base_rounds)}
        return {"name": self.base_name, "m": None, "aggregation": "base", "fault": None, "seed": self.seed,
                "timing_key": f"{self.model}/hofl", "fields": fields}

    def dose_specs(self):
        """The arms, or ``None`` while the base run (which fixes the tiers and the starting state) is missing."""
        config = harness.read_json(self.base_dir / "config.json")
        if harness.run_status(self.base_dir) != "COMPLETED" or not config:
            return None, None
        inputs = dose.write_inputs(self.root / "inputs", config["resource_cuts"], config["k"], self.seed, self.dose_rounds)
        init_state = str(self.base_dir / "states" / f"round_{self.base_rounds:04d}.safetensors")
        return inputs, dose.specs(self.model, self.seed, inputs, init_state, BASE, doses=self.doses, faults=self.faults)


def queue(ctx, plan, label):
    from ho_fl.config import Config
    say(f"== {label} ({ctx.model}, seed {ctx.seed}, {len(plan)} runs, up to {ctx.jobs} at a time, {ctx.hardware}) ==")
    for spec in plan:  # every configuration is validated before anything is launched
        Config(**{**spec["fields"], **ctx.common(spec["fields"])}).resolve()
    record = grid.run_queue(plan, root=ctx.root, tree=REPO, hardware=ctx.hardware, revision=ctx.head, common=ctx.common,
                            deadline=ctx.deadline, jobs=ctx.jobs)
    history = harness.read_json(ctx.root / "queue_all.json", {"entries": []})
    fresh = {entry["name"]: entry for entry in record["queue"]}
    history["entries"] = [fresh.pop(entry["name"], entry) for entry in history["entries"]] + list(fresh.values())
    history.update({key: record[key] for key in ("recorded_at", "hardware", "head", "jobs", "jobs_requested", "preflight_failures")})
    harness.write_json(ctx.root / "queue_all.json", history)
    counts = {}
    for entry in record["queue"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    say(f"{label}: {counts}")
    return all(entry["status"] in harness.FINISHED for entry in record["queue"])


def stage_base(ctx):
    return queue(ctx, [ctx.base_spec()], "base run") and harness.run_status(ctx.base_dir) == "COMPLETED"


def stage_dose(ctx):
    inputs, plan = ctx.dose_specs()
    if plan is None:
        say("The base run is not complete; run the base stage first.")
        return False
    say(f"unreliable device: client {inputs['device']}; doses {ctx.doses}; faults {ctx.faults}; {ctx.dose_rounds} rounds per arm")
    return queue(ctx, plan, "dose-response arms")


# --- feedback -------------------------------------------------------------------------------------------------------

def collect(ctx):
    inputs, plan = ctx.dose_specs()
    arms = dose.collect(ctx.root / "runs", ctx.model, ctx.seed) if plan else {}
    base = harness.load_run(ctx.base_dir)
    timings = harness.read_json(ctx.root / f"timings-{ctx.hardware}.json", {})
    metas = [harness.read_json(path) or {} for path in sorted(ctx.root.rglob("run_meta.json"))]
    finished = sum(1 for spec in plan or [] if (spec["m"], spec["aggregation"], spec["fault"]) in arms)
    return {"generated_at": grid.now(), "tiny": ctx.tiny, "model": ctx.model, "seed": ctx.seed, "head": ctx.head, "hardware": ctx.hardware,
            "base": {"status": base["status"], "rounds": ctx.base_rounds, "accuracy_pct": 100 * base["summary"]["accuracy"] if "accuracy" in base["summary"] else None,
                     "loss": base["summary"].get("task_loss")},
            "device": inputs["device"] if inputs else None, "dose_rounds": ctx.dose_rounds, "doses": sorted(ctx.doses), "faults": ctx.faults,
            "planned_runs": len(plan or []), "finished_runs": finished,
            "tables": {fault: dose.table_rows(arms, fault, sorted(ctx.doses)) for fault in ctx.faults if fault != "none"},
            "clean": dose.table_rows(arms, "none", sorted(ctx.doses)),
            "h2": {fault: entry for fault, entry in dose.decide_h2(arms, sorted(ctx.doses)).items() if fault in ctx.faults} if arms else {},
            "arms": [{"m": m, "aggregation": aggregation, "fault": fault, **outcome} for (m, aggregation, fault), outcome in sorted(arms.items())],
            "queue": harness.read_json(ctx.root / "queue_all.json"), "timings": timings,
            "wall_hours": sum(meta.get("wall_s") or 0 for meta in metas) / 3600,
            "regression": grid.regression_verdict(ctx.root.parent, ctx.head, ctx.hardware)}


def cell_table(rows, doses, text):
    return table(["aggregation", *[f"m={m} ({m + 1} first-order on S1)" for m in doses]],
                 [[row["aggregation"], *["not run" if cell is None else text(cell) for cell in row["cells"]]] for row in rows])


def render(data):
    out = []
    add = out.append
    doses = data["doses"]
    add("# P2 feedback: dose-response (E2)" + (" (TINY DRY RUN: plumbing check on CPU, the numbers mean nothing)" if data["tiny"] else ""))
    add(f"\nGenerated {data['generated_at']} for revision `{data['head'][:12]}` on `{data['hardware']}`. Model `{data['model']}`, AG News Dirichlet 0.5, "
        f"seed {data['seed']}; {data['finished_runs']} of {data['planned_runs']} planned arms finished.\n")
    add("## 1. Baseline regression in this session\n")
    regression = data["regression"]
    add("No regression verdict for this revision and GPU was found next to this root; the runs below are not known to start from an intact baseline."
        if regression is None else f"I1: {regression['i1']} (golden mode: {regression['golden_mode']}); I2 replay: {regression['i2']}.")
    add("\n## 2. Set-up\n")
    base = data["base"]
    add(f"Starting state: round {base['rounds']} of an HO-FL run with the default settings ({base['status']}"
        + (f", test accuracy {base['accuracy_pct']:.2f}%, test loss {base['loss']:.4f}" if base["accuracy_pct"] is not None else "") + "). "
        f"Each arm trains {data['dose_rounds']} more rounds on the same cohorts. Client {data['device']} (a T1 device) is in every cohort and, in the faulty "
        "arms, sends its update on segment S1 negated (`flip1`) or negated and scaled by 20 (`flip20`). The dose m is the number of honest "
        "first-order reports on S1 next to it. `quorum_robust` is filled in from the arm it coincides with (`fo_robust` from m = 2 on, `robust_all` below). "
        "`quorum_hold` and `quorum_dilute` run without zeroth-order updates.")
    for fault, rows in data["tables"].items():
        add(f"\n## 3. Fault `{fault}`\n")
        add("Test loss after the arm, with the loss of the paired run without the fault in parentheses (lower is better):\n")
        add(cell_table(rows, doses, lambda c: f"DIVERGED" if c["status"] == "DIVERGED" else f"{c['loss']:.4f} ({fmt(c['clean_loss'], 4)})"))
        add("\nDamage: test loss with the fault minus test loss of the paired run:\n")
        add(cell_table(rows, doses, lambda c: "diverged" if c["status"] == "DIVERGED" else (f"{c['damage']:+.4f}" if c["damage"] is not None else "n/a")))
        add("\nTest accuracy (%) with the fault:\n")
        add(cell_table(rows, doses, lambda c: "diverged" if c["status"] == "DIVERGED" else f"{c['accuracy_pct']:.2f}"))
        add("\nRelative influence on S1: how far the fault moved the aggregate, in units of what it moves the equal-weight mean (mean over rounds and layers):\n")
        add(cell_table(rows, doses, lambda c: fmt(c["influence_s1"], 2)))
    add("\n## 4. Without the fault\n")
    add("Test loss of the paired runs (the cost or gain of each rule when nothing is wrong):\n")
    add(cell_table(data["clean"], doses, lambda c: "diverged" if c["status"] == "DIVERGED" else f"{c['loss']:.4f}"))
    add("\n## 5. H2: the quorum crossover\n")
    for fault, entry in data["h2"].items():
        if fault in data["tables"]:
            add(f"- `{fault}`: **{entry['verdict']}** (one seed, preliminary): {entry['reason']}.")
    add("\n## 6. Queue, timing and GPU hours\n")
    queue_record = data["queue"]
    if queue_record:
        add(f"Runs at a time: {queue_record.get('jobs')} (requested {queue_record.get('jobs_requested')}).\n")
        add(table(["run", "status", "wall min", "estimated min", "note"],
                  [[e["name"], e["status"] + (" (reused)" if e.get("reused") else ""), fmt(e["wall_s"] / 60, 1) if e.get("wall_s") else "",
                    fmt(e["estimated_s"] / 60, 1) if e.get("estimated_s") else "", e.get("reason") or e.get("error") or ""] for e in queue_record["entries"]]))
    if data["timings"]:
        add("\nMeasured alone on the GPU (three-round preflight):\n")
        add(table(["cost profile", "s per round", "s per evaluation", "start-up s", "peak GPU memory GiB"],
                  [[key, fmt(t["round_s"], 2), fmt(t["eval_s"], 2), fmt(t["startup_s"], 1), fmt(t["peak_mb"] / 1024, 1) if t.get("peak_mb") is not None else "n/a"]
                   for key, t in data["timings"].items()]))
    add(f"\nWall time of all runs recorded under this root: {data['wall_hours']:.2f} h (sum over runs).")
    add("\n## 7. What this means and what to do next\n")
    for line in findings(data):
        add(f"- {line}")
    return "\n".join(out) + "\n"


def findings(data):
    lines = []
    if data["tiny"]:
        lines.append("This was the tiny dry run. It only shows that every stage executes; switch it off for real results.")
    if data["regression"] is None and not data["tiny"]:
        lines.append("Run the baseline regression (P0 stages env, golden, regress) in the same session: these results build on I1.")
    if data["base"]["status"] != "COMPLETED":
        lines.append("The base run did not complete, so no arm could start.")
    for fault, entry in data["h2"].items():
        if fault in data["tables"]:
            lines.append(f"H2 under `{fault}`: {entry['verdict']}; {entry['reason']}.")
    diverged = [arm for arm in data["arms"] if arm["status"] == "DIVERGED" and arm["aggregation"] != "quorum_robust"]
    if diverged:
        lines.append(f"{len(diverged)} arms diverged (a result, counted as the worst outcome): "
                     + ", ".join(f"m={arm['m']} {arm['aggregation']} {arm['fault']} at round {arm['diverged_round']}" for arm in diverged[:12]) + ".")
    for fault, rows in data["tables"].items():
        influence = {row["aggregation"]: [cell["influence_s1"] for cell in row["cells"] if cell and cell["influence_s1"] is not None] for row in rows}
        if influence.get("mean_all") and influence.get("robust_all"):
            lines.append(f"Influence on S1 under `{fault}`: the equal-weight mean is 1 by definition (measured {min(influence['mean_all']):.2f} to {max(influence['mean_all']):.2f}); "
                         f"robust averaging over all reports {min(influence['robust_all']):.2f} to {max(influence['robust_all']):.2f}"
                         + (f"; first-order-only robust averaging {min(influence['fo_robust']):.2f} to {max(influence['fo_robust']):.2f}." if influence.get("fo_robust") else "."))
    missing = data["planned_runs"] - data["finished_runs"]
    if missing:
        lines.append(f"{missing} planned arms did not finish (budget or failures, see the queue table); a later session has to repeat them.")
    lines.append("Each arm runs only a few rounds from a common state, so differences in test loss are small by design; "
                 "the influence table says what the aggregation did, the loss tables what it cost. One seed: every statement is preliminary.")
    return lines


def stage_feedback(ctx):
    say("== feedback ==")
    data = collect(ctx)
    out = ctx.root / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    harness.write_json(out / "feedback_P2.json", data)
    text = render(data)
    (out / "feedback_P2.md").write_text(text)
    analysis.write_csv(out / "dose.csv", data["arms"])
    grid.write_bundle(ctx.root, out / "feedback_P2_bundle.zip")
    say("H2: " + ", ".join(f"{fault} {entry['verdict']}" for fault, entry in data["h2"].items()) + f"; {data['finished_runs']} of {data['planned_runs']} arms finished"
        if ctx.quiet else text)
    say(f"written: {out / 'feedback_P2.md'}, dose.csv and {out / 'feedback_P2_bundle.zip'}")
    return True


STAGES = {"base": stage_base, "dose": stage_dose, "feedback": stage_feedback}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], help="one or more stages, run in the order given")
    parser.add_argument("--root", default="outputs/p2", help="directory that receives every result of the pipeline")
    parser.add_argument("--model", choices=["qwen", "opt"], default="qwen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doses", default="", help="comma-separated doses m in queue order (default 2,1,3,0,4,5: the predicted crossover first)")
    parser.add_argument("--faults", default="", help="comma-separated subset of none,flip20,flip1 (default all)")
    parser.add_argument("--jobs", type=int, default=1, help="runs that may share the GPU; limited further by the measured memory per run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="cache", help="download and cache directory shared by all runs")
    parser.add_argument("--budget-hours", type=float, default=0.0, help="time budget of the queue, counted from now")
    parser.add_argument("--deadline", type=float, default=0.0, help="alternative to --budget-hours: absolute deadline as a Unix timestamp")
    parser.add_argument("--tiny", action="store_true", help="dry run on CPU with synthetic tiny assets")
    parser.add_argument("--quiet", action="store_true", help="feedback stage: print one line instead of the whole report")
    args = parser.parse_args(argv)
    ctx = Context(args)
    results = {}
    for name in list(STAGES) if "all" in args.stages else args.stages:
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
