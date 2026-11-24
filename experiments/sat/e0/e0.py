#!/usr/bin/env python
"""E0 of the satellite scenario: correctness checks, memory calibration and the alignment runs with HO-FL
(brief v2, sections 7.5, 7.6 and 10.1). Stages (each can be re-run; finished work is reused):

  env        interpreter, packages, GPU and git state                                   -> <root>/env.json
  tests      the CPU test suite, upstream tests and tests/sat           [E0-a item 1]   -> <root>/pytest.log
  trace      build the S1 trace and compare it with appendix A          [E0-a item 6]   -> <cache>/sat/traces/s1
  data       HumAID and EuroSAT, the partitions of three seeds and both data variants   -> <root>/data.json
  selfcheck  explicit cuts, participation against upstream, aggregation against the
             central rule, replay across replicas, on the real models   [E0-a items 2-5] -> <root>/selfcheck/
  memory     peak GPU memory of a participation per cut, Qwen and ViT   [section 7.6]    -> <root>/memory/
  smoke      two simulated hours of the full method on both tasks and of the three other
             methods on the vision task                                 [E0-a item 7]   -> <root>/smoke/
  t0         HO-FL on AG News with the zeroth-order part on and off, seeds 43 and 44
             (seed 42 comes from the session of 2026-11-21)             [E0-b]          -> <root>/e0b/
  feedback   everything above condensed into feedback_E0.md / .json and a zip           -> <root>/feedback/
  all        every stage in this order (with --tiny: all but tests)

Every run is a subprocess of this repository's code (nothing is installed). ``--tiny`` replaces models and data by
the synthetic assets of ho_fl.testing and ho_fl.sat.testing and runs on the CPU: it checks that the pipeline works
and its numbers mean nothing.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ho_fl import harness  # noqa: E402

HERE = Path(__file__).parent
EXPECTED = json.loads((HERE / "expected.json").read_text())
SEEDS = (42, 43, 44)
# The smoke runs: (name, task, method). The first is item 7 of E0-a; the others exercise every implemented method.
SMOKE = [("t2-m7", "t2", "m7"), ("t2-m0", "t2", "m0"), ("t2-m2", "t2", "m2"), ("t2-m4", "t2", "m4"), ("t1-m7", "t1", "m7")]
SMOKE_HOURS = 2.0
# E0-b: the settings of the status report of 2026-11-22 (Qwen2.5-1.5B, AG News, Dirichlet 0.5, beta 0.75).
T0_BASE = {"model": "qwen", "dataset": "ag_news", "partition": "dirichlet_05", "method": "hofl", "beta": 0.75}
T0_SEEDS = (43, 44)
T0_MINUTES = {"on": 12.0, "off": 8.0}     # expected duration of a run on the G4, from the measured seconds per round
T0_REFERENCE = {"file": "experiments/p1/references/v1_1121_qwen.json", "on": "A-b0.75", "off": "B-b0.75"}
STAGES = ("env", "tests", "trace", "data", "selfcheck", "memory", "smoke", "t0", "feedback")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(text=""):
    print(text, flush=True)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}"


def num(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


class Context:
    def __init__(self, args):
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tiny = args.tiny
        self.device = "cpu" if args.tiny else args.device
        self.deadline = args.deadline or (time.time() + 3600 * args.budget_hours if args.budget_hours else None)
        self.python = sys.executable
        if self.tiny:
            from ho_fl.sat.testing import build_tiny_assets as sat_assets
            from ho_fl.testing import build_tiny_assets as upstream_assets

            self.assets = sat_assets(self.root / "assets")
            self.upstream = upstream_assets(self.root / "assets-upstream", datasets=("sst2", "ag_news"))
            self.cache_dir = self.assets["t1"]["cache_dir"]
        else:
            self.assets, self.upstream = None, None
            self.cache_dir = str(Path(args.cache_dir).resolve())

    def sat_fields(self, task):
        """Options that route a satellite command to the model and data of ``task``."""
        if self.tiny:
            return dict(self.assets[task])
        return {"task": task, "device": self.device, "cache_dir": self.cache_dir}

    def remaining(self):
        return float("inf") if self.deadline is None else self.deadline - time.time()


def stream(ctx, command, log_path, label=None, show=None):
    """Run a command of this repository; its output goes to ``log_path`` and, filtered by ``show``, to the page."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=str(REPO), env=harness.tree_environment(REPO), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if show is None or show(line):
                say(("    " if label is None else f"    [{label}] ") + line.rstrip("\n"))
        code = process.wait()
    if code != 0:
        say(f"    exit code {code}; end of {log_path}:")
        say("\n".join("      " + line for line in harness.tail(log_path, 30).splitlines()))
    return code, time.monotonic() - started


def quiet(line):
    """Lines worth showing from a command that prints tables: everything but library warnings."""
    return bool(line.strip()) and "Warning" not in line and "warnings.warn" not in line and "newly initialized" not in line \
        and "TRAIN this model" not in line


def progress(line):
    return line.startswith(("round ", "EVAL", "COMPLETED", "DIVERGED"))


def module(ctx, name, fields, *positional):
    return [ctx.python, "-m", name, *positional, *harness.cli(fields)]


def run_sat(ctx, label, fields, output):
    """One satellite training run, unless ``output`` holds a finished one. Unfinished attempts are kept aside."""
    output = Path(output)
    status = (harness.read_json(output / "status.json") or {}).get("status", "MISSING")
    if status in harness.FINISHED:
        say(f"  [{label}] already finished ({status}), reusing {output}")
        return status
    if output.exists() and any(output.iterdir()):
        index = 1
        while output.with_name(f"{output.name}.stale-{index}").exists():
            index += 1
        output.rename(output.with_name(f"{output.name}.stale-{index}"))
        say(f"  [{label}] unfinished attempt ({status}) moved to {output.name}.stale-{index}")
    say(f"  [{label}] starting")
    command = module(ctx, "ho_fl.sat.cli", {**fields, "output": str(output)}, "train")
    code, wall = stream(ctx, command, output.with_name(output.name + ".log"), label, progress)
    status = (harness.read_json(output / "status.json") or {}).get("status", "FAILED")
    output.mkdir(parents=True, exist_ok=True)
    harness.write_json(output / "run_meta.json", {"label": label, "command": command, "returncode": code, "wall_s": wall,
                                                  "status": status, "finished_at": now()})
    return status


# --- stages --------------------------------------------------------------------------------------------------------------

def stage_env(ctx):
    say("== env ==")
    import torch

    def version(name):
        try:
            return __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing optional package is a fact to record, not an error
            return None

    cuda = torch.cuda.is_available()
    record = {"recorded_at": now(), "python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
              "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda else None,
              "packages": {name: version(name) for name in ("transformers", "peft", "datasets", "numpy", "scipy", "PIL", "pyarrow", "huggingface_hub")},
              "land_mask_installed": version("global_land_mask") is not None or _importable("global_land_mask"),
              "git": {"head": harness_git("rev-parse", "HEAD"), "branch": harness_git("rev-parse", "--abbrev-ref", "HEAD"),
                      "dirty": bool(harness_git("status", "--porcelain", "--untracked-files=no"))},
              "tiny": ctx.tiny, "device": ctx.device}
    harness.write_json(ctx.root / "env.json", record)
    say(json.dumps({key: record[key] for key in ("python", "torch", "cuda_runtime", "gpu_name", "land_mask_installed", "git")}))
    ok = ctx.tiny or cuda
    if not ok:
        say("No CUDA device: the GPU stages cannot run here.")
    if not record["land_mask_installed"]:
        say("NOTE: global-land-mask is not installed; the trace will use the 30% duty-cycle substitute for imaging.")
    return ok


def _importable(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def harness_git(*arguments):
    result = subprocess.run(["git", "-C", str(REPO), *arguments], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def stage_tests(ctx):
    say("== tests ==")
    report = ctx.root / "pytest.xml"
    code, wall = stream(ctx, [ctx.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}"], ctx.root / "pytest.log",
                        show=lambda line: "passed" in line or "failed" in line or "error" in line.lower())
    counts = {}
    if report.exists():
        suite = ElementTree.parse(report).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        counts = {key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}
    harness.write_json(ctx.root / "tests.json", {"returncode": code, "wall_s": wall, **counts})
    return code == 0


def stage_trace(ctx):
    say("== trace ==")
    out = Path(ctx.cache_dir) / "sat" / "traces" / "s1"
    code, _ = stream(ctx, [ctx.python, "-m", "ho_fl.sat.cli", "trace", "--scenario", "s1", "--output", str(out), "--tables"],
                     ctx.root / "trace.log", show=quiet)
    meta, sanity = harness.read_json(out / "meta.json") or {}, harness.read_json(out / "sanity.json") or {}
    expected = EXPECTED["trace_sha256"].get(meta.get("land_source"))
    record = {"returncode": code, "path": str(out), "trace_sha256": meta.get("trace_sha256"), "land_source": meta.get("land_source"),
              "expected_sha256": expected, "matches_report": None if expected is None else expected == meta.get("trace_sha256"), "sanity": sanity}
    harness.write_json(ctx.root / "trace.json", record)
    if expected is not None:
        say(f"    trace hash {'equals' if record['matches_report'] else 'DIFFERS from'} the one recorded in the milestone report (P-S1)")
    return code == 0


def stage_data(ctx):
    say("== data ==")
    rows, ok = [], True
    for task in ("t1", "t2"):
        for variant in ("", "zonly"):
            for seed in SEEDS:
                name = f"{task}-{variant or 'base'}-s{seed}"
                fields = {**ctx.sat_fields(task), "seed": seed, **({"variant": variant} if variant else {})}
                show = quiet if seed == SEEDS[0] else (lambda line: False)   # the full table once per task and variant
                code, _ = stream(ctx, module(ctx, "ho_fl.sat.datasets", fields), ctx.root / "data" / f"{name}.log", show=show)
                ok &= code == 0
                tag = "-tiny" if ctx.tiny else ""
                manifest = harness.read_json(Path(ctx.cache_dir) / "sat" / "manifests" / f"{task}{tag}-{variant or 'base'}-s{seed}.json") or {}
                expected = None if ctx.tiny else EXPECTED["partition_sha256"].get(name)
                rows.append({"name": name, "task": task, "variant": variant or "base", "seed": seed, "returncode": code,
                             "partition_sha256": manifest.get("partition_sha256"), "labels_sha256": manifest.get("labels_sha256"),
                             "data_sha256": manifest.get("data_sha256"), "examples_per_satellite": manifest.get("examples_per_satellite"),
                             "unused": len(manifest.get("unused_indices", [])), "expected_sha256": expected,
                             "matches_report": None if expected is None else expected == manifest.get("partition_sha256")})
    harness.write_json(ctx.root / "data.json", {"rows": rows})
    say()
    say(table(["partition", "examples per satellite", "unused", "partition sha256", "as in the milestone report (P-S2)"],
              [[row["name"], row["examples_per_satellite"], row["unused"], (row["partition_sha256"] or "n/a")[:16],
                {True: "yes", False: "NO", None: "n/a"}[row["matches_report"]]] for row in rows]))
    return ok


def stage_selfcheck(ctx):
    say("== selfcheck ==")
    ok = True
    for task in ("t2", "t1"):
        out = ctx.root / "selfcheck" / task
        if (harness.read_json(out / "selfcheck.json") or {}).get("status") == "PASSED":
            say(f"  [{task}] already passed, reusing {out}")
            continue
        code, _ = stream(ctx, module(ctx, "ho_fl.sat.selfcheck", {**ctx.sat_fields(task), "output": str(out)}), out.with_suffix(".log"), show=quiet)
        ok &= code == 0
    out = ctx.root / "selfcheck" / "upstream"
    if (harness.read_json(out / "selfcheck.json") or {}).get("status") == "PASSED":
        say(f"  [upstream] already passed, reusing {out}")
        return ok
    if ctx.tiny:
        fields = {key: value for key, value in ctx.upstream["sst2"].items()}
    else:
        fields = {**{key: T0_BASE[key] for key in ("model", "dataset", "partition")}, "device": ctx.device, "cache_dir": ctx.cache_dir}
    code, _ = stream(ctx, module(ctx, "ho_fl.sat.selfcheck", {**fields, "output": str(out)}, "upstream"), out.with_suffix(".log"), show=quiet)
    return ok and code == 0


def stage_memory(ctx):
    say("== memory ==")
    ok = True
    for task in ("t1", "t2"):
        out = ctx.root / "memory" / task
        if (harness.read_json(out / "memory.json") or {}).get("status") == "COMPLETED":
            say(f"  [{task}] already measured, reusing {out}")
            say((out / "memory.md").read_text())
            continue
        if out.exists() and not any(out.iterdir()):
            out.rmdir()
        code, _ = stream(ctx, module(ctx, "ho_fl.sat.calibrate", {**ctx.sat_fields(task), "output": str(out)}), out.with_suffix(".log"), show=quiet)
        ok &= code == 0
    return ok


def stage_smoke(ctx):
    say("== smoke ==")
    statuses = {}
    for name, task, method in SMOKE:
        fields = {**ctx.sat_fields(task), "method": method, "seed": 42, "hours": SMOKE_HOURS}
        statuses[name] = run_sat(ctx, name, fields, ctx.root / "smoke" / name)
    harness.write_json(ctx.root / "smoke.json", {"statuses": statuses, "verdicts": {name: smoke_verdict(ctx.root / "smoke" / name) for name, _, _ in SMOKE}})
    return all(status == "COMPLETED" for status in statuses.values())


def smoke_verdict(path):
    """What E0-a item 7 asks of a short run: it completes, the training loss goes down, every value is finite."""
    summary = harness.read_json(Path(path) / "summary.json")
    if summary is None:
        return {"status": "MISSING", "passed": False}
    rounds = harness.read_jsonl(Path(path) / "rounds.jsonl")
    losses = [row["task_loss"] for row in rounds if row["task_loss"] is not None]
    third = max(1, len(losses) // 3)
    first, last = statistics.mean(losses[:third]), statistics.mean(losses[-third:])
    return {"status": summary["status"], "loss_first_third": first, "loss_last_third": last, "loss_decreased": last < first,
            "passed": summary["status"] == "COMPLETED" and last < first}


def t0_runs(ctx):
    runs = []
    for seed in T0_SEEDS:
        for mode in ("on", "off"):
            if ctx.tiny:
                fields = {**{k: v for k, v in ctx.upstream["ag_news"].items()}, "partition": "dirichlet_05", "method": "hofl", "beta": 0.75,
                          "rounds": 3, "evaluate_every": 3, "seed": seed}
            else:
                fields = {**T0_BASE, "seed": seed, "device": ctx.device, "cache_dir": ctx.cache_dir}
            if mode == "off":
                fields["zo_mode"] = "off"
            runs.append({"name": f"qwen-zo{mode}-s{seed}", "seed": seed, "mode": mode, "fields": fields})
    return runs


def stage_t0(ctx):
    say("== t0 ==")
    ok = True
    for spec in t0_runs(ctx):
        out = ctx.root / "e0b" / spec["name"]
        finished = harness.run_status(out) in harness.FINISHED
        need = 0 if ctx.tiny or finished else 60 * T0_MINUTES[spec["mode"]] * 1.25
        if ctx.remaining() < need:
            say(f"  [{spec['name']}] skipped: about {need / 60:.0f} min needed, {max(0, ctx.remaining()) / 60:.0f} min of budget left")
            ok = False
            continue
        record = harness.run_training(spec["fields"], out, tree=REPO, label=spec["name"], echo=say)
        ok &= record["status"] == "COMPLETED"
    return ok


# --- feedback ------------------------------------------------------------------------------------------------------------

def collect(ctx):
    root = ctx.root
    data = {"generated_at": now(), "tiny": ctx.tiny, "env": harness.read_json(root / "env.json"), "tests": harness.read_json(root / "tests.json"),
            "trace": harness.read_json(root / "trace.json"), "partitions": (harness.read_json(root / "data.json") or {}).get("rows"),
            "selfcheck": {name: harness.read_json(root / "selfcheck" / name / "selfcheck.json") for name in ("t2", "t1", "upstream")},
            "memory": {task: harness.read_json(root / "memory" / task / "memory.json") for task in ("t1", "t2")}, "smoke": [], "t0": []}
    for name, task, method in SMOKE:
        path = root / "smoke" / name
        summary, evals = harness.read_json(path / "summary.json"), harness.read_jsonl(path / "eval.jsonl")
        data["smoke"].append({"name": name, "task": task, "method": method, "summary": summary, "verdict": smoke_verdict(path),
                              "curve": [{"sim_hours": row["sim_hours"], "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "loss": row["loss"]}
                                        for row in evals if not row.get("final")],
                              "eval_wall_s": [row["wall_s"] for row in evals],
                              "participants_per_round": statistics.mean([row["participant_count"] for row in harness.read_jsonl(path / "rounds.jsonl")] or [0])})
    for spec in t0_runs(ctx):
        run = harness.load_run(root / "e0b" / spec["name"])
        data["t0"].append({"name": spec["name"], "seed": spec["seed"], "mode": spec["mode"], "status": run["status"],
                           "accuracy": run["summary"].get("accuracy"), "task_loss": run["summary"].get("task_loss"),
                           "wall_s": run["meta"].get("wall_s"), "timing": run["meta"].get("timing")})
    reference = harness.read_json(REPO / T0_REFERENCE["file"]) or {"runs": {}}
    data["t0_reference"] = {mode: (reference["runs"].get(T0_REFERENCE[mode]) or {}).get("final_accuracy_pct") for mode in ("on", "off")}
    return data


def checklist(data):
    """E0-a as a list of ``(item, what, verdict, evidence)``; verdicts are PASS, FAIL or NOT RUN."""
    def verdict(flag):
        return "NOT RUN" if flag is None else ("PASS" if flag else "FAIL")

    tests, trace = data["tests"], data["trace"]
    rows = [(1, "existing tests and the new tests pass", verdict(None if tests is None else tests["returncode"] == 0),
             "not run" if tests is None else f"{tests.get('tests', '?')} tests, {tests.get('failures', '?')} failures, {tests.get('errors', '?')} errors")]
    items = {2: "explicit cuts in hybrid_step (bitwise for the upstream cuts; cut 0; cut at the depth)",
             3: "train_participation is bitwise train_client for the tier cuts",
             4: "ideal network: aggregation equals central aggregate + apply (first-order bitwise, zeroth-order within 1e-7)",
             5: "zeroth-order replay is bitwise the same on every replica"}
    for item, what in items.items():
        found = [(name, row) for name, check in data["selfcheck"].items() if check for row in check["rows"] if row["item"] == item]
        sources = sorted({name for name, _ in found})
        rows.append((item, what, verdict(all(row["passed"] for _, row in found) if found else None),
                     "not run" if not found else f"{sum(row['passed'] for _, row in found)} of {len(found)} checks on {', '.join(sources)}"))
    rows.append((6, "trace statistics agree with appendix A", verdict(None if trace is None else trace["returncode"] == 0),
                 "not run" if trace is None else f"land from {trace['land_source']}, trace sha256 {(trace['trace_sha256'] or 'n/a')[:16]}"))
    first = next((run for run in data["smoke"] if run["name"] == "t2-m7"), None)
    done = first is not None and first["summary"] is not None
    rows.append((7, "vision task, full method, two simulated hours: loss goes down, values stay finite",
                 verdict(first["verdict"]["passed"] if done else None),
                 "not run" if not done else f"{first['verdict']['status']}, training loss {first['verdict']['loss_first_third']:.4f} -> {first['verdict']['loss_last_third']:.4f}"))
    return rows


def render(data):
    lines = []
    add = lines.append
    env = data["env"] or {}
    add("# E0 feedback: correctness, memory and alignment with HO-FL (satellite scenario)\n")
    add(f"Generated {data['generated_at']}. " + ("**CPU dry run on the tiny synthetic assets: the numbers below mean nothing; only the pipeline was checked.**"
                                                  if data["tiny"] else f"GPU {env.get('gpu_name')}, torch {env.get('torch')}, code {((env.get('git') or {}).get('head') or '?')[:12]}."))
    add("\n## 1. E0-a checklist (gate G0)\n")
    rows = checklist(data)
    add(table(["item", "check", "verdict", "evidence"], rows))
    failed, missing = [row[0] for row in rows if row[2] == "FAIL"], [row[0] for row in rows if row[2] == "NOT RUN"]
    add("\nGate G0: " + ("**a check FAILED (items " + ", ".join(map(str, failed)) + "): stop and report.**" if failed else
                          ("not decided yet, items " + ", ".join(map(str, missing)) + " were not run." if missing else "**all items pass.**")))
    for name, check in data["selfcheck"].items():
        if check:
            add(f"\nSelf-check `{name}` ({check.get('model')}, {check.get('device')}, {check.get('gpu') or 'CPU'}): {check['status']}\n")
            add(table(["item", "check", "result", "detail"], [[row["item"], row["check"], "pass" if row["passed"] else "FAIL", row["detail"]] for row in check["rows"]]))
    add("\n## 2. Trace and partitions\n")
    trace = data["trace"]
    if trace is None:
        add("Trace: not built.")
    else:
        add(f"Trace `{trace['path']}`: sha256 `{trace['trace_sha256']}`, land from {trace['land_source']}; hash recorded in the milestone report: "
            f"{ {True: 'the same', False: 'DIFFERENT', None: 'none for this land source'}[trace['matches_report']] }.")
    if data["partitions"]:
        add("")
        add(table(["partition", "examples per satellite", "unused", "partition sha256", "as in the milestone report"],
                  [[row["name"], row["examples_per_satellite"], row["unused"], row["partition_sha256"], {True: "yes", False: "NO", None: "n/a"}[row["matches_report"]]]
                   for row in data["partitions"]]))
        add("\nData content hashes (texts or decoded pixels): " + ", ".join(f"{task} `{value}`" for task, value in
                                                                              sorted({row["task"]: row["data_sha256"] for row in data["partitions"]}.items())))
    add("\n## 3. Peak GPU memory of a participation (brief section 7.6)\n")
    for task, memory in data["memory"].items():
        if memory is None:
            add(f"Task {task}: not measured.")
            continue
        add(f"Task {task} ({memory['model']}, batch {'x'.join(map(str, memory['batch_shape']))}, {memory['device']}):\n")
        add(table(["cut", "below the cut", "first-order update (MB)", "peak allocated (GB)"],
                  [[row["cut"], {"inference": "inference only", "hybrid": "forward-only training", "frozen": "frozen"}[row["mode"]],
                    num(row["first_order_bytes"] / 1e6) if row["mode"] != "inference" else "0", num(row["peak_allocated_gb"], 2)] for row in memory["rows"]]))
        states = memory["memory_states_gb"]
        add(f"\nMemory states: backbone {num(states['backbone'], 2)} GB, hybrid idle {num(states['hybrid_idle'], 2)} GB, hybrid imaging "
            f"{num(states['hybrid_imaging'], 2)} GB, forward-only {num(states['forward_only'], 2)} GB.\n")
    add("\n## 4. Smoke runs (two simulated hours, seed 42, server step 1)\n")
    rows = []
    for run in data["smoke"]:
        summary, verdict = run["summary"], run["verdict"]
        if summary is None:
            rows.append([run["name"], "not run"] + [""] * 7)
            continue
        curve = " / ".join(pct(point["accuracy"]) for point in run["curve"])
        rows.append([run["name"], summary["status"], curve, num(verdict.get("loss_first_third"), 4), num(verdict.get("loss_last_third"), 4),
                     num(run["participants_per_round"], 1), num(summary["wall_s_per_round"], 2), num(statistics.mean(run["eval_wall_s"]), 1) if run["eval_wall_s"] else "n/a",
                     num(summary["wall_s"] / 60, 1)])
    add(table(["run", "status", "test accuracy % at 0 / 1 / 2 h", "train loss, first third", "last third", "participants per round",
               "s per round", "s per evaluation", "minutes"], rows))
    add("\nA 24-hour run has 144 rounds and 25 evaluations; its duration follows from the two timing columns.")
    add("\n## 5. E0-b: zeroth-order part on and off in HO-FL (Qwen2.5-1.5B, AG News, Dirichlet 0.5, beta 0.75)\n")
    by_seed = {42: {mode: data["t0_reference"][mode] for mode in ("on", "off")}}
    for run in data["t0"]:
        by_seed.setdefault(run["seed"], {})[run["mode"]] = None if run["accuracy"] is None else 100 * run["accuracy"]
    rows, gaps = [], []
    for seed in sorted(by_seed):
        on, off = by_seed[seed].get("on"), by_seed[seed].get("off")
        gap = None if on is None or off is None else on - off
        if gap is not None:
            gaps.append(gap)
        rows.append([seed, "session of 2026-11-21 (reference file)" if seed == 42 else "this session", num(on, 2), num(off, 2), num(gap, 2)])
    add(table(["seed", "source", "accuracy % with the zeroth-order part", "without it", "difference (points)"], rows))
    if data["tiny"]:
        add("\n(Tiny dry run: three rounds on a random model; the seed-42 row is the real reference.)")
    elif len(gaps) >= 2:
        add(f"\nDifference over {len(gaps)} seeds: mean {statistics.mean(gaps):+.2f} points, standard deviation {statistics.stdev(gaps):.2f}. "
            "The HO-FL authors report a drop of 3.32 points when the zeroth-order part is removed.")
    else:
        add("\nNot enough finished runs for an estimate over seeds.")
    timed = [run for run in data["t0"] if run["wall_s"]]
    if timed:
        add("\n" + table(["run", "status", "minutes"], [[run["name"], run["status"], num(run["wall_s"] / 60, 1)] for run in timed]))
    return "\n".join(lines) + "\n"


def stage_feedback(ctx):
    say("== feedback ==")
    out = ctx.root / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    data = collect(ctx)
    harness.write_json(out / "feedback_E0.json", data)
    text = render(data)
    (out / "feedback_E0.md").write_text(text)
    with zipfile.ZipFile(out / "feedback_E0_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ctx.root.rglob("*")):
            relative = path.relative_to(ctx.root)
            if path.is_file() and relative.parts[0] not in ("assets", "assets-upstream") and path.suffix in (".json", ".jsonl", ".md", ".log", ".xml") \
                    and path.name != "feedback_E0_bundle.zip" and path.stat().st_size < 20_000_000:
                bundle.write(path, str(relative))
        for name in ("meta.json", "sanity.json"):
            path = Path(ctx.cache_dir) / "sat" / "traces" / "s1" / name
            if path.exists():
                bundle.write(path, f"trace/{name}")
    if not ctx.quiet:
        say(text)
    say(f"feedback written to {out}")
    return not [row for row in checklist(data) if row[2] == "FAIL"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    parser.add_argument("--root", default="outputs/sat/e0")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiny", action="store_true", help="CPU dry run on synthetic assets")
    parser.add_argument("--budget-hours", type=float, default=0.0)
    parser.add_argument("--deadline", type=float, default=0.0, help="time.time() value after which no new long run starts")
    parser.add_argument("--quiet", action="store_true", help="do not print the feedback report")
    args = parser.parse_args(argv)
    ctx = Context(args)
    ctx.quiet = args.quiet
    stages = list(args.stages)
    if "all" in stages:
        stages = [stage for stage in STAGES if not (ctx.tiny and stage == "tests")]
    functions = {"env": stage_env, "tests": stage_tests, "trace": stage_trace, "data": stage_data, "selfcheck": stage_selfcheck,
                 "memory": stage_memory, "smoke": stage_smoke, "t0": stage_t0, "feedback": stage_feedback}
    results = {}
    for stage in stages:
        results[stage] = bool(functions[stage](ctx))
        say()
    say("stage results: " + ", ".join(f"{stage}={'ok' if ok else 'ATTENTION'}" for stage, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
