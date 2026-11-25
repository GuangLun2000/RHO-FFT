#!/usr/bin/env python
"""The first experiment of the satellite scenario: gate G0 (brief v2, sections 7.5, 7.6 and 10.1).

The session answers six questions; every stage exists for one of them, cheap stages that can invalidate the rest
come first, and a stage does not start when a stage it builds on ran and did not pass (``--force`` overrides):

  env        interpreter, packages, GPU and git state                                   -> <root>/env.json
  baseline   the regression of experiments/p0: HO-FL, FedAvg and DeComFL on this code
             against the upstream commit, and ho-fl-replay; not repeated while
             verified.json covers this code, GPU and PyTorch          [E0-a item 1]     -> <root>/baseline.json
  tests      the CPU test suite, upstream tests and tests/sat         [E0-a item 1]     -> <root>/tests.json
  trace      build the S1 trace and compare it with appendix A        [E0-a item 6]     -> <cache>/sat/traces/s1
  data       HumAID and EuroSAT, the partitions of three seeds and both data variants   -> <root>/data.json
  selfcheck  explicit cuts, participation against upstream, aggregation against the
             central rule, replay across replicas, on the real models [E0-a items 2-5];
             and how faithful the zeroth-order scalars are            [diagnostic]      -> <root>/selfcheck/
  memory     peak GPU memory of a participation per cut, Qwen and ViT [section 7.6]     -> <root>/memory/
  smoke      two simulated hours of the full method on both tasks and of the three other
             methods on the vision task                               [E0-a item 7]     -> <root>/smoke/
  pilot      a whole simulated day of the full method and of the centralised reference
             on the vision task: does EuroSAT saturate (gate G3), what does a run cost,
             does a repeated run reproduce                            [diagnostic]      -> <root>/pilot/
  t0         HO-FL on AG News with the zeroth-order part on and off, seeds 43 and 44
             (seed 42 comes from the session of 2026-11-21); run on
             2026-11-24 and no longer part of "all"                   [E0-b]            -> <root>/e0b/
  feedback   what the session showed and what should change           -> <root>/feedback/feedback_E0.md, .json, .zip
  all        every stage but t0 in this order (with --tiny: also without tests)

Requirements between the stages: selfcheck and memory need baseline and tests; smoke and pilot also need trace,
data and selfcheck; pilot needs the two-hour run of the same method to have passed; t0 needs baseline only (it
runs upstream code). A stage that was never run blocks nothing.

Every run is a subprocess of this repository's code (nothing is installed). ``--tiny`` replaces models and data by
the synthetic assets of ho_fl.testing and ho_fl.sat.testing and runs on the CPU: it checks that the pipeline works
and its numbers mean nothing. The training runs evaluate on the validation split: no server step has been selected
yet, so nothing in this session looks at the test split.
"""
import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ElementTree
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ho_fl import harness  # noqa: E402

HERE = Path(__file__).parent


def _sibling(name):
    """A module next to this script (the script is run by path, so it has no package to import from)."""
    spec = importlib.util.spec_from_file_location(f"sat_e0_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


digest = _sibling("digest")

EXPECTED = json.loads((HERE / "expected.json").read_text())
SEEDS = (42, 43, 44)
RUN_SEED, RUN_SPLIT = 42, "validation"
# The two-hour runs: (name, task, method). The first is item 7 of E0-a; the others exercise every implemented method.
SMOKE = [("t2-m7", "t2", "m7"), ("t2-m0", "t2", "m0"), ("t2-m2", "t2", "m2"), ("t2-m4", "t2", "m4"), ("t1-m7", "t1", "m7")]
SMOKE_HOURS = 2.0
# The day-long runs; each repeats a two-hour run, so its first rounds also show whether a run reproduces on this GPU.
DAY = [("t2-m7-day", "t2", "m7"), ("t2-m0-day", "t2", "m0")]
DAY_HOURS, TINY_DAY_HOURS, DAY_FALLBACK_MINUTES = 24.0, 3.0, 10.0
# E0-b: the settings of the status report of 2026-11-22 (Qwen2.5-1.5B, AG News, Dirichlet 0.5, beta 0.75).
T0_BASE = {"model": "qwen", "dataset": "ag_news", "partition": "dirichlet_05", "method": "hofl", "beta": 0.75}
T0_SEEDS = (43, 44)
T0_MINUTES = {"on": 12.0, "off": 8.0}     # expected duration of a run on the G4, from the measured seconds per round
T0_REFERENCE = {"file": "experiments/p1/references/v1_1121_qwen.json", "on": "A-b0.75", "off": "B-b0.75"}
STAGES = ("env", "baseline", "tests", "trace", "data", "selfcheck", "memory", "smoke", "pilot", "t0", "feedback")
# What "all" runs. The HO-FL runs of E0-b were done on 2026-11-24 (seed 44 completes them); the stage stays callable.
SESSION = tuple(stage for stage in STAGES if stage != "t0")
# The baseline regression that passed on Colab, and the paths it covers: everything outside the satellite package
# that a run of upstream HO-FL executes. While these are unchanged on the same GPU and PyTorch, it is not repeated.
VERIFIED = json.loads((HERE / "verified.json").read_text())
BASELINE_PATHS = ("ho_fl", ":(exclude)ho_fl/sat", "pyproject.toml", "experiments/p0")
# What each stage is about to do, printed under its heading: the page says what is running before it runs.
ABOUT = {"env": "interpreter, packages, GPU and commit of this session",
         "baseline": "is upstream HO-FL unchanged: its three smoke configurations on the upstream commit and on this code",
         "tests": "the CPU test suite in this environment (about 4 minutes; one line per 72 tests)",
         "trace": "24 hours of the 120-satellite constellation, compared with the reference values of the brief",
         "data": "HumAID and EuroSAT; 12 partitions: 2 tasks x 2 data variants x 3 seeds",
         "selfcheck": "correctness on ViT / EuroSAT, on Qwen / HumAID and against upstream on Qwen / AG News; models are downloaded on first use",
         "memory": "peak GPU memory of one participation at every cut: 9 measurements on Qwen, then 9 on ViT",
         "smoke": "5 training runs of 2 simulated hours (12 rounds) each; one line per round",
         "pilot": "2 training runs of 24 simulated hours (144 rounds) each on EuroSAT; one line per round",
         "t0": "HO-FL on Qwen / AG News with the zeroth-order part on and off, seeds 43 and 44; 160 rounds each",
         "feedback": "the report of this session"}
HEARTBEAT_S = 30   # the longest a running command stays silent on the page
REQUIRES = {"selfcheck": ("baseline", "tests", "data"), "memory": ("baseline", "tests"),
            "smoke": ("baseline", "tests", "trace", "data", "selfcheck"), "pilot": ("baseline", "tests", "trace", "data", "selfcheck"),
            "t0": ("baseline",)}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(text=""):
    print(text, flush=True)


class Context:
    def __init__(self, args):
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tiny, self.force, self.quiet = args.tiny, args.force, args.quiet
        self.pending = [name for name in args.pending.split(",") if name]
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


def clock(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def stream(ctx, command, log_path, label=None, show=None, timeout=None, tail=True):
    """Run a command of this repository; its output goes to ``log_path`` and, filtered by ``show``, to the page.
    Whatever the command prints or ``show`` hides, the page gets a line at least every ``HEARTBEAT_S`` seconds with
    the time so far and the command's last line, so a slow command can be told from a stuck one. ``timeout``
    (seconds) ends a command that hangs, so that the session can go on to its report. Returns the exit code, the
    wall time and whether the timeout ended the command."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started, ended = time.monotonic(), []
    prefix = "    " if label is None else f"    [{label}] "
    state, finished = {"shown": started, "last": ""}, threading.Event()

    def heartbeat():
        while not finished.wait(2):
            if time.monotonic() - state["shown"] >= HEARTBEAT_S:
                state["shown"] = time.monotonic()
                say(f"{prefix}... running, {clock(time.monotonic() - started)} so far" + (f" | last line: {state['last'][:110]}" if state["last"] else ""))

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(REPO), env={**harness.tree_environment(REPO), "PYTHONUTF8": "1"}, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=heartbeat, daemon=True).start()

        def end():
            ended.append(True)
            process.kill()

        timer = threading.Timer(timeout, end) if timeout else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.strip():
                    state["last"] = line.strip()
                if show is None or show(line):
                    say(prefix + line.rstrip("\n"))
                    state["shown"] = time.monotonic()
            code = process.wait()
        finally:
            finished.set()
            if timer:
                timer.cancel()
            if process.poll() is None:   # reading its output failed: the command must not outlive this call
                process.kill()
                process.wait()
    if ended:
        say(f"    stopped after {timeout / 60:.0f} minutes without finishing; see {log_path}")
    elif code != 0 and tail:
        say(f"    exit code {code}; end of {log_path}:")
        say("\n".join("      " + line for line in harness.tail(log_path, 30).splitlines()))
    return code, time.monotonic() - started, bool(ended)


def quiet(line):
    """Lines worth showing from a command that prints tables: everything but library warnings."""
    return bool(line.strip()) and "Warning" not in line and "warnings.warn" not in line and "newly initialized" not in line \
        and "TRAIN this model" not in line


def progress(line):
    """What a training run shows on the page: its start, every round (with the time left), every evaluation and
    how it ended."""
    return line.startswith(("START", "round ", "EVAL", "COMPLETED", "DIVERGED"))


def module(ctx, name, fields, *positional):
    return [ctx.python, "-m", name, *positional, *harness.cli(fields)]


def run_fields(ctx, task, method, hours):
    return {**ctx.sat_fields(task), "method": method, "seed": RUN_SEED, "hours": hours, "eval_split": RUN_SPLIT}


def run_sat(ctx, label, fields, output, timeout=None):
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
    code, wall, ended = stream(ctx, command, output.with_name(output.name + ".log"), label, progress, timeout)
    status = (harness.read_json(output / "status.json") or {}).get("status", "FAILED")
    if status not in harness.FINISHED:
        status = "TIMEOUT" if ended else "FAILED"
    output.mkdir(parents=True, exist_ok=True)
    harness.write_json(output / "run_meta.json", {"label": label, "command": command, "returncode": code, "wall_s": wall,
                                                  "status": status, "finished_at": now()})
    return status


# --- which stage may start -----------------------------------------------------------------------------------------------

def stage_records(root):
    return harness.read_json(Path(root) / "stages.json") or {}


def record_stage(root, name, status, wall_s, blocked_by=None, error=None):
    """One entry per stage in ``<root>/stages.json``: the outcome of its last call and the time of all its calls.
    ``error`` is the end of the traceback when the stage itself broke."""
    stages = stage_records(root)
    previous = stages.get(name) or {}
    stages[name] = {"status": status, "wall_s": wall_s + previous.get("wall_s", 0.0), "calls": previous.get("calls", 0) + 1,
                    "blocked_by": blocked_by, "error": error, "finished_at": now()}
    harness.write_json(Path(root) / "stages.json", stages)


def blocked(root, stage):
    """The stage that keeps ``stage`` from starting (it ran in this root and did not pass), or ``None``."""
    stages = stage_records(root)
    for name in REQUIRES.get(stage, ()):
        if name in stages and stages[name]["status"] != "ok":
            return name
    if stage == "pilot":
        verdicts = (harness.read_json(Path(root) / "smoke.json") or {}).get("verdicts")
        if verdicts is not None and not all((verdicts.get(f"{task}-{method}") or {}).get("passed" if method == "m7" else "completed")
                                             for _, task, method in DAY):
            return "smoke"
    return None


# --- stages --------------------------------------------------------------------------------------------------------------

def stage_env(ctx):
    import torch

    def version(name):
        try:
            return __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing optional package is a fact to record, not an error
            return None

    cuda = torch.cuda.is_available()
    record = {"recorded_at": now(), "python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
              "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda else None,
              "packages": {name: version(name) for name in ("transformers", "peft", "datasets", "numpy", "scipy", "PIL", "pyarrow", "huggingface_hub")},
              "land_mask_installed": version("global_land_mask") is not None or _importable("global_land_mask"),
              "git": {"head": harness_git("rev-parse", "HEAD"), "branch": harness_git("rev-parse", "--abbrev-ref", "HEAD"),
                      "dirty": bool(harness_git("status", "--porcelain", "--untracked-files=no"))},
              "tiny": ctx.tiny, "device": ctx.device}
    harness.write_json(ctx.root / "env.json", record)
    say(json.dumps({key: record[key] for key in ("python", "torch", "cuda_runtime", "gpu_name", "cpus", "land_mask_installed", "git")}))
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


def baseline_record(root, returncode=None):
    """The verdicts of the regression pipeline (experiments/p0) in the form the report reads. ``PASSED`` when every
    smoke configuration equals the upstream commit (bitwise or within the tolerance of the device) and the replay
    passes; ``INCONCLUSIVE`` when the device itself does not reproduce the reference to the tolerance."""
    files = sorted(Path(root).glob("regress/*/verdicts.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        return {"status": "INCOMPLETE", "returncode": returncode}
    verdicts = harness.read_json(files[-1]) or {"i1": {}, "i2": {"verdict": "FAIL"}}
    configs = {name: {"verdict": entry["verdict"], "max_abs_state_diff": entry.get("max_abs_state_diff"),
                      "first_difference": (entry.get("differences") or [None])[0]} for name, entry in verdicts["i1"].items()}
    replay = verdicts["i2"]
    passed = bool(configs) and all(entry["verdict"].startswith("PASS") for entry in configs.values()) and replay["verdict"] == "PASS"
    failed = any(entry["verdict"] in ("FAIL", "RUN_FAILED") for entry in configs.values()) or replay["verdict"] != "PASS"
    head = replay.get("head") or {}
    return {"status": "PASSED" if passed else ("FAILED" if failed else "INCONCLUSIVE"), "returncode": returncode, "head": verdicts.get("head"),
            "golden_mode": verdicts.get("golden_mode"), "configs": configs,
            "replay": {"verdict": replay["verdict"], "max_abs_error": head.get("max_abs_error"), "comparisons": head.get("comparisons")}}


def baseline_verified(ctx):
    """``(record, reason)``: the recorded baseline regression if it still stands, otherwise ``None`` and why it has
    to be run. It stands while the code it covers is unchanged since the commit it passed on and, for a real
    session, GPU and PyTorch are the ones it passed with."""
    record = VERIFIED.get("baseline")
    if not record:
        return None, "no regression is on record"
    env = harness.read_json(ctx.root / "env.json") or {}
    if not ctx.tiny and (env.get("gpu_name"), env.get("torch")) != (record["gpu_name"], record["torch"]):
        return None, (f"the record is for {record['gpu_name']} with torch {record['torch']}; this session has "
                      f"{env.get('gpu_name') or 'no recorded GPU'} with torch {env.get('torch') or 'unknown'}")
    result = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", record["commit"], "--", *BASELINE_PATHS], capture_output=True, text=True)
    if result.returncode != 0:
        return None, f"git could not compare with commit {record['commit'][:12]} (exit code {result.returncode}: {(result.stderr or '').strip()[:160]})"
    changed = result.stdout.split()
    if changed:
        return None, f"changed since commit {record['commit'][:12]}: {', '.join(changed[:6])}" + (f" and {len(changed) - 6} more" if len(changed) > 6 else "")
    return record, ""


def stage_baseline(ctx):
    record, reason = baseline_verified(ctx)
    if record is not None:
        harness.write_json(ctx.root / "baseline.json", {"status": "PASSED", "returncode": 0, "configs": record["configs"], "replay": record["replay"],
                                                        "reused": {key: record[key] for key in ("commit", "date", "gpu_name", "torch")}})
        say(f"    passed on {record['date']} at commit {record['commit'][:12]} on this GPU and PyTorch; the code of upstream HO-FL has not changed "
            "since, so the regression is not repeated")
        return True
    say(f"    running the regression: {reason}")
    say("    13 short runs (9 for the reference values, 3 on this code, 1 replay); the first ones download OPT-125M and SST-2")
    root = ctx.root / "p0"
    command = [ctx.python, str(REPO / "experiments" / "p0" / "p0.py"), "env", "golden", "regress", "feedback", "--quiet", "--root", str(root),
               "--device", ctx.device, *(["--tiny"] if ctx.tiny else ["--cache-dir", ctx.cache_dir])]
    # The page shows every run as it finishes, the stages, the verdict tables and anything unusual; the rest is in the log.
    code, _, _ = stream(ctx, command, ctx.root / "baseline.log",
                        show=lambda line: quiet(line) and (not line.startswith((" ", "{", "}")) or "completed in" in line or "FAILED" in line or "TIMEOUT" in line))
    record = baseline_record(root, code)
    harness.write_json(ctx.root / "baseline.json", record)
    say(f"    baseline regression: {record['status']}")
    return record["status"] == "PASSED"


def read_junit(path):
    """Counts and the failed tests (``{"id", "message"}``) of a JUnit report written by pytest; empty when there is none."""
    path = Path(path)
    if not path.exists():
        return {}, []
    suite = ElementTree.parse(path).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    counts, failed = {key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}, []
    for case in suite.iter("testcase"):
        problem = next((child for child in case if child.tag in ("failure", "error")), None)
        if problem is not None:
            message = (problem.get("message") or problem.text or "").strip().splitlines()
            failed.append({"id": f"{case.get('classname')}::{case.get('name')}", "message": (message[0] if message else "")[:240]})
    return counts, failed


def stage_tests(ctx):
    cache = ["-o", f"cache_dir={ctx.root / 'pytest-cache'}"]   # outside the source tree; the repeat below reads it
    # The page shows the result line and the names of failed tests; their tracebacks are in the log and in the report.
    show = lambda line: line.startswith("FAILED ") or line.rstrip().endswith("%]") or (" in " in line and (" passed" in line or " failed" in line))  # noqa: E731
    code, wall, _ = stream(ctx, [ctx.python, "-m", "pytest", "-q", *cache, f"--junitxml={ctx.root / 'pytest.xml'}"], ctx.root / "pytest.log", show=show, tail=False)
    counts, failed = read_junit(ctx.root / "pytest.xml")
    flaky = []
    if code != 0 and 0 < len(failed) <= 10:
        # A test that fails once and passes when repeated is an unreliable test, not a defect of the code it tests.
        # It is reported as something to fix, but it does not cost the session its GPU stages.
        say(f"    {len(failed)} test(s) failed; running them once more")
        again, extra, _ = stream(ctx, [ctx.python, "-m", "pytest", "-q", "--last-failed", "--last-failed-no-failures", "none", *cache,
                                       f"--junitxml={ctx.root / 'pytest-repeat.xml'}"], ctx.root / "pytest-repeat.log", show=show, tail=False)
        wall += extra
        _, still = read_junit(ctx.root / "pytest-repeat.xml")
        if again == 0 and not still:
            flaky, failed, code = failed, [], 0
            counts.update(failures=0, errors=0)
        elif still:
            failed = still
    harness.write_json(ctx.root / "tests.json", {"returncode": code, "wall_s": wall, **counts, "failed": failed, "flaky": flaky})
    return code == 0


def stage_trace(ctx):
    out = Path(ctx.cache_dir) / "sat" / "traces" / "s1"
    # The page shows the verdict and any statistic that disagrees; the full comparison is in the log.
    code, _, _ = stream(ctx, [ctx.python, "-m", "ho_fl.sat.cli", "trace", "--scenario", "s1", "--output", str(out)], ctx.root / "trace.log",
                        show=lambda line: "statistics agree" in line or "| NO |" in line or line.startswith(("scenario ", "NOTE")))
    meta, sanity = harness.read_json(out / "meta.json") or {}, harness.read_json(out / "sanity.json") or {}
    expected = EXPECTED["trace_sha256"].get(meta.get("land_source"))
    record = {"returncode": code, "path": str(out), "trace_sha256": meta.get("trace_sha256"), "land_source": meta.get("land_source"),
              "expected_sha256": expected, "matches_report": None if expected is None else expected == meta.get("trace_sha256"), "sanity": sanity}
    harness.write_json(ctx.root / "trace.json", record)
    if expected is not None:
        say(f"    trace hash {'equals' if record['matches_report'] else 'DIFFERS from'} the one recorded in the milestone report (P-S1)")
    return code == 0


def stage_data(ctx):
    rows, ok = [], True
    for task in ("t1", "t2"):
        for variant in ("", "zonly"):
            for seed in SEEDS:
                name = f"{task}-{variant or 'base'}-s{seed}"
                fields = {**ctx.sat_fields(task), "seed": seed, **({"variant": variant} if variant else {})}
                # The statistics of each partition go to its log; the page gets the summary below.
                code, _, _ = stream(ctx, module(ctx, "ho_fl.sat.datasets", fields), ctx.root / "data" / f"{name}.log", show=lambda line: False)
                ok &= code == 0
                tag = "-tiny" if ctx.tiny else ""
                manifest = harness.read_json(Path(ctx.cache_dir) / "sat" / "manifests" / f"{task}{tag}-{variant or 'base'}-s{seed}.json") or {}
                expected = None if ctx.tiny else EXPECTED["partition_sha256"].get(name)
                say(f"    {len(rows) + 1:2d}/12 {name}: " + ("FAILED, see data/" + name + ".log" if code else
                    f"{manifest.get('examples_per_satellite')} examples per satellite, partition {(manifest.get('partition_sha256') or 'n/a')[:12]}"
                    + {True: ", as recorded", False: ", NOT the recorded hash", None: ""}[None if expected is None else expected == manifest.get("partition_sha256")]))
                rows.append({"name": name, "task": task, "variant": variant or "base", "seed": seed, "returncode": code,
                             "partition_sha256": manifest.get("partition_sha256"), "labels_sha256": manifest.get("labels_sha256"),
                             "data_sha256": manifest.get("data_sha256"), "examples_per_satellite": manifest.get("examples_per_satellite"),
                             "unused": len(manifest.get("unused_indices", [])), "expected_sha256": expected,
                             "matches_report": None if expected is None else expected == manifest.get("partition_sha256")})
    harness.write_json(ctx.root / "data.json", {"rows": rows})
    same = sum(row["matches_report"] is True for row in rows)
    say(f"    {sum(not row['returncode'] for row in rows)} of {len(rows)} partitions built; {same} have the hash recorded in the milestone report (P-S2)")
    other = [row for row in rows if row["returncode"] or (row["matches_report"] is False)]
    if other:
        say(digest.table(["partition", "built", "partition sha256", "as recorded"],
                         [[row["name"], "no" if row["returncode"] else "yes", row["partition_sha256"] or "n/a", {True: "yes", False: "NO", None: "n/a"}[row["matches_report"]]]
                          for row in other]))
    return ok


def verdicts(line):
    """What a self-check shows on the page: what it is loading and checking as it goes, how many checks passed and
    every check that failed. The passing rows and the estimator table are in its log and, condensed, in the report."""
    return line.startswith(("> ", "Self-check", "Traceback")) or "checks passed" in line or "| FAIL |" in line or "Error" in line


def stage_selfcheck(ctx):
    ok = True
    for task in ("t2", "t1"):
        out = ctx.root / "selfcheck" / task
        if (harness.read_json(out / "selfcheck.json") or {}).get("status") == "PASSED":
            say(f"  [{task}] already passed, reusing {out}")
            continue
        code, _, _ = stream(ctx, module(ctx, "ho_fl.sat.selfcheck", {**ctx.sat_fields(task), "output": str(out)}), out.with_suffix(".log"), show=verdicts)
        ok &= code == 0
    out = ctx.root / "selfcheck" / "upstream"
    if (harness.read_json(out / "selfcheck.json") or {}).get("status") == "PASSED":
        say(f"  [upstream] already passed, reusing {out}")
        return ok
    if ctx.tiny:
        fields = {key: value for key, value in ctx.upstream["sst2"].items()}
    else:
        fields = {**{key: T0_BASE[key] for key in ("model", "dataset", "partition")}, "device": ctx.device, "cache_dir": ctx.cache_dir}
    code, _, _ = stream(ctx, module(ctx, "ho_fl.sat.selfcheck", {**fields, "output": str(out)}, "upstream"), out.with_suffix(".log"), show=verdicts)
    return ok and code == 0


def stage_memory(ctx):
    ok = True
    for task in ("t1", "t2"):
        out = ctx.root / "memory" / task
        if (harness.read_json(out / "memory.json") or {}).get("status") == "COMPLETED":
            say(f"  [{task}] already measured, reusing {out}")
            continue
        if out.exists() and not any(out.iterdir()):
            out.rmdir()
        code, _, _ = stream(ctx, module(ctx, "ho_fl.sat.calibrate", {**ctx.sat_fields(task), "output": str(out)}), out.with_suffix(".log"),
                            show=lambda line: line.startswith(("> ", "Memory states")))
        ok &= code == 0
    return ok


def stage_smoke(ctx):
    statuses = {}
    for name, task, method in SMOKE:
        statuses[name] = run_sat(ctx, name, run_fields(ctx, task, method, SMOKE_HOURS), ctx.root / "smoke" / name, timeout=1800)
    harness.write_json(ctx.root / "smoke.json", {"statuses": statuses, "verdicts": {name: smoke_verdict(ctx.root / "smoke" / name) for name, _, _ in SMOKE}})
    return all(status == "COMPLETED" for status in statuses.values())


def smoke_verdict(path):
    """What E0-a item 7 asks of a short run: it completes, the training loss goes down, every value is finite."""
    summary = harness.read_json(Path(path) / "summary.json")
    if summary is None:
        return {"status": "MISSING", "passed": False, "completed": False}
    rounds = harness.read_jsonl(Path(path) / "rounds.jsonl")
    losses = [row["task_loss"] for row in rounds if row["task_loss"] is not None]
    if not losses:
        return {"status": summary["status"], "passed": False, "completed": summary["status"] == "COMPLETED"}
    third = max(1, len(losses) // 3)
    first, last = statistics.mean(losses[:third]), statistics.mean(losses[-third:])
    return {"status": summary["status"], "loss_first_third": first, "loss_last_third": last, "loss_decreased": last < first,
            "passed": summary["status"] == "COMPLETED" and last < first, "completed": summary["status"] == "COMPLETED"}


def day_estimate_s(short, hours, round_s=600, eval_every=6):
    """Seconds a run of ``hours`` simulated hours should take, from the finished two-hour run of the same method:
    its start-up plus its seconds per round and per evaluation. ``None`` when that run is not there."""
    summary, evals = harness.read_json(Path(short) / "summary.json"), harness.read_jsonl(Path(short) / "eval.jsonl")
    if not summary or not evals or "wall_breakdown_s" not in summary:
        return None
    rounds = round(hours * 3600 / round_s)
    return summary["wall_breakdown_s"]["startup"] + rounds * summary["wall_s_per_round"] + (rounds // eval_every + 1) * statistics.mean(row["wall_s"] for row in evals)


def stage_pilot(ctx):
    hours = TINY_DAY_HOURS if ctx.tiny else DAY_HOURS
    statuses, skipped, estimates = {}, {}, {}
    for name, task, method in DAY:
        output = ctx.root / "pilot" / name
        estimates[name] = estimate = day_estimate_s(ctx.root / "smoke" / f"{task}-{method}", hours)
        done = (harness.read_json(output / "status.json") or {}).get("status") in harness.FINISHED
        need = 0.0 if ctx.tiny or done else 1.25 * (estimate or 60 * DAY_FALLBACK_MINUTES)
        if ctx.remaining() < need:
            skipped[name] = {"needed_min": need / 60, "left_min": max(0.0, ctx.remaining()) / 60}
            statuses[name] = "SKIPPED"
            say(f"  [{name}] skipped: about {need / 60:.0f} min needed, {max(0.0, ctx.remaining()) / 60:.0f} min of budget left")
            continue
        if estimate and not done:
            say(f"  [{name}] expected to take about {estimate / 60:.1f} min (from the two-hour run of the same method)")
        statuses[name] = run_sat(ctx, name, run_fields(ctx, task, method, hours), output, timeout=max(1800.0, 4 * (estimate or 0.0)))
    harness.write_json(ctx.root / "pilot.json", {"statuses": statuses, "skipped": skipped, "estimates_s": estimates, "hours": hours})
    return all(status == "COMPLETED" for status in statuses.values())


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
    ok, skipped = True, {}
    for spec in t0_runs(ctx):
        out = ctx.root / "e0b" / spec["name"]
        finished = harness.run_status(out) in harness.FINISHED
        need = 0 if ctx.tiny or finished else 60 * T0_MINUTES[spec["mode"]] * 1.25
        if ctx.remaining() < need:
            say(f"  [{spec['name']}] skipped: about {need / 60:.0f} min needed, {max(0, ctx.remaining()) / 60:.0f} min of budget left")
            skipped[spec["name"]] = {"needed_min": need / 60, "left_min": max(0.0, ctx.remaining()) / 60}
            ok = False
            continue
        # A run that takes three times what it should is taken to hang; the session then goes on to its report.
        record = harness.run_training(spec["fields"], out, tree=REPO, label=spec["name"], echo=say,
                                      stop_at=None if ctx.tiny else time.time() + 3 * 60 * T0_MINUTES[spec["mode"]])
        ok &= record["status"] == "COMPLETED"
    harness.write_json(ctx.root / "t0.json", {"skipped": skipped})
    return ok


# --- feedback ------------------------------------------------------------------------------------------------------------

def read_run(path, name, group, task, method, hours, started, note=None):
    """A training run as the report reads it: status, curve, who took part, what went unexpectedly, where the wall
    time went. ``started`` says whether its stage ran at all; ``note`` why the run itself was skipped."""
    path = Path(path)
    summary, meta = harness.read_json(path / "summary.json"), harness.read_json(path / "run_meta.json") or {}
    rounds, ledger, evals = harness.read_jsonl(path / "rounds.jsonl"), harness.read_jsonl(path / "ledger.jsonl"), harness.read_jsonl(path / "eval.jsonl")
    status = (summary or {}).get("status") or meta.get("status") or (harness.read_json(path / "status.json") or {}).get("status")
    if note is not None:
        status = "SKIPPED"
    elif status is None:
        status = "MISSING" if started else "NOT RUN"
    elif status == "RUNNING":
        status = "FAILED"
    run = {"name": name, "group": group, "task": task, "method": method, "hours": hours, "status": status, "note": note, "rounds": len(rounds),
           "depth": (harness.read_json(path / "config.json") or {}).get("depth"),
           "verdict": smoke_verdict(path), "wall_s": (summary or {}).get("wall_s") or meta.get("wall_s"),
           "diverged_round": (summary or {}).get("diverged_round"),
           "curve": [{"round": row["round"], "sim_hours": row["sim_hours"], "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "loss": row["loss"]}
                     for row in evals if not row.get("final")]}
    if not rounds:
        return run
    total = sum(row["wall_s"] for row in rounds)
    parts = Counter()
    for row in rounds:
        parts.update(row.get("wall_parts_s") or {})
    seconds = defaultdict(list)
    for row in ledger:
        if row.get("wall_s") is not None:
            seconds[row["kind"], row["cut"], bool(row["zo"])].append(row["wall_s"])
    batteries = {kind: [row["battery_min"][kind] for row in rounds if row["battery_min"].get(kind) is not None] for kind in ("H", "Z")}
    differences = [row["replica_difference"] for row in evals if row.get("replica_difference") is not None]
    run.update({
        "participants": {kind: statistics.mean(row["participants"][kind] for row in rounds) for kind in "BHZ"},
        "invited": {kind: statistics.mean(row["invited"][kind] for row in rounds) for kind in "BHZ"},
        "skipped": {reason: sum(row["skipped"].get(reason, 0) for row in rounds) for reason in ("memory", "eclipse", "energy", "unreachable")},
        "late": sum(row["first_order_late"] for row in rounds), "moved_up": sum(row["moved_up"] for row in rounds),
        "zo": {key: sum(row["zo_reports"][key] for row in rounds) for key in ("sent", "used", "dropped")},
        "replica_groups_max": max(row["replica_groups"] for row in rounds), "replica_difference_max": max(differences, default=0.0),
        "battery_min": {kind: min(values) if values else None for kind, values in batteries.items()},
        "seconds": {"round": total / len(rounds), "evaluation": statistics.mean(row["wall_s"] for row in evals) if evals else None,
                    "startup": ((summary or {}).get("wall_breakdown_s") or {}).get("startup", 0.0),
                    "parts": {key: value / total for key, value in parts.items()} if total else {}},
        "participation_seconds": [{"kind": kind, "cut": cut, "zo": zo, "count": len(values), "mean_s": statistics.mean(values)}
                                  for (kind, cut, zo), values in sorted(seconds.items())],
        "participations": (summary or {}).get("participations"), "energy_wh_per_satellite_day": (summary or {}).get("energy_wh_per_satellite_day"),
        "bytes_per_day": (summary or {}).get("bytes_per_day")})
    return run


def compare_repeat(short, day):
    """The first rounds of a day-long run against the two-hour run with the same configuration: the same numbers,
    or how far apart."""
    first, second = harness.read_jsonl(Path(short) / "rounds.jsonl"), harness.read_jsonl(Path(day) / "rounds.jsonl")
    count = min(len(first), len(second))
    if not count:
        return None
    losses = [(a["task_loss"], b["task_loss"]) for a, b in zip(first[:count], second[:count])]
    evals = {row["round"]: row for row in harness.read_jsonl(Path(day) / "eval.jsonl") if not row.get("final")}
    pairs = [(row, evals[row["round"]]) for row in harness.read_jsonl(Path(short) / "eval.jsonl") if not row.get("final") and row["round"] in evals and row["round"] <= count]
    same = (all(a == b for a, b in losses) and all(a["participant_count"] == b["participant_count"] for a, b in zip(first[:count], second[:count]))
            and all(a["accuracy"] == b["accuracy"] and a["loss"] == b["loss"] for a, b in pairs))
    return {"rounds_compared": count, "evaluations_compared": len(pairs), "identical": same,
            "max_loss_difference": max((abs(a - b) for a, b in losses if a is not None and b is not None), default=0.0),
            "max_accuracy_difference": max((abs(a["accuracy"] - b["accuracy"]) for a, b in pairs), default=0.0)}


def log_tail(path, lines=25, width=220):
    return "\n".join(line[:width] for line in harness.tail(path, lines).splitlines())


def collect(ctx):
    root = ctx.root
    stages = stage_records(root)
    data = {"generated_at": now(), "tiny": ctx.tiny, "env": harness.read_json(root / "env.json"), "stages": stages, "pending": ctx.pending,
            "session": harness.read_json(root / "session.json"), "baseline": harness.read_json(root / "baseline.json"),
            "tests": harness.read_json(root / "tests.json"), "trace": harness.read_json(root / "trace.json"),
            "partitions": (harness.read_json(root / "data.json") or {}).get("rows"),
            "selfcheck": {name: harness.read_json(root / "selfcheck" / name / "selfcheck.json") for name in ("t2", "t1", "upstream")},
            "memory": {task: harness.read_json(root / "memory" / task / "memory.json") for task in ("t1", "t2")},
            "runs": [], "repeats": [], "t0": [], "failures": [], "skipped_for_budget": []}
    if data["session"] and data["session"].get("started_at"):
        data["session"]["elapsed_hours"] = (time.time() - data["session"]["started_at"]) / 3600
    failures = data["failures"]

    def failure(stage, name, log):
        if Path(log).exists():
            failures.append({"stage": stage, "name": name, "log": str(Path(log).relative_to(root)), "tail": log_tail(log)})

    if "baseline" in stages and (data["baseline"] or {}).get("status") not in ("PASSED",):
        failure("baseline", "", root / "baseline.log")
    if data["tests"] and data["tests"]["returncode"] != 0:
        failure("tests", "", root / "pytest.log")
    if data["trace"] and data["trace"]["returncode"] != 0:
        failure("trace", "", root / "trace.log")
    for row in data["partitions"] or []:
        if row["returncode"]:
            failure("data", row["name"], root / "data" / f"{row['name']}.log")
    for name, check in data["selfcheck"].items():
        if check is None and "selfcheck" in stages and stages["selfcheck"]["status"] != "skipped":
            failure("selfcheck", name, root / "selfcheck" / f"{name}.log")
    for task, memory in data["memory"].items():
        if memory is None and "memory" in stages and stages["memory"]["status"] != "skipped":
            failure("memory", task, root / "memory" / f"{task}.log")
    pilot = harness.read_json(root / "pilot.json") or {}
    hours = pilot.get("hours") or (TINY_DAY_HOURS if ctx.tiny else DAY_HOURS)
    groups = [("short", "smoke", SMOKE, SMOKE_HOURS, {}), ("day", "pilot", DAY, hours, pilot.get("skipped") or {})]
    for group, stage, specs, duration, skipped in groups:
        started = stage in stages and stages[stage]["status"] != "skipped"
        for name, task, method in specs:
            note = skipped.get(name)
            run = read_run(root / stage / name, name, group, task, method, duration, started, note)
            data["runs"].append(run)
            if note is not None:
                data["skipped_for_budget"].append(name)
            if run["status"] in ("FAILED", "TIMEOUT"):
                failure(stage, name, root / stage / f"{name}.log")
    for name, task, method in DAY:
        repeat = compare_repeat(root / "smoke" / f"{task}-{method}", root / "pilot" / name)
        if repeat:
            data["repeats"].append({"task": task, "method": method, **repeat})
    skipped = (harness.read_json(root / "t0.json") or {}).get("skipped") or {}
    for spec in t0_runs(ctx):
        run = harness.load_run(root / "e0b" / spec["name"])
        status = run["status"] if run["status"] != "MISSING" else (run["meta"].get("status") or "NOT RUN")   # a run that never wrote its status
        status = "SKIPPED" if spec["name"] in skipped else status
        data["t0"].append({"name": spec["name"], "seed": spec["seed"], "mode": spec["mode"], "status": status, "note": skipped.get(spec["name"]),
                           "accuracy": run["summary"].get("accuracy"), "task_loss": run["summary"].get("task_loss"),
                           "wall_s": run["meta"].get("wall_s"), "timing": run["meta"].get("timing")})
        if spec["name"] in skipped:
            data["skipped_for_budget"].append(spec["name"])
        elif status in ("FAILED", "TIMEOUT", "RUNNING"):
            failure("t0", spec["name"], root / "e0b" / f"{spec['name']}.log")
    reference = harness.read_json(REPO / T0_REFERENCE["file"]) or {"runs": {}}
    data["t0_reference"] = {mode: (reference["runs"].get(T0_REFERENCE[mode]) or {}).get("final_accuracy_pct") for mode in ("on", "off")}
    return data


def stage_feedback(ctx):
    out = ctx.root / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    data = collect(ctx)
    rows = digest.checklist(data)
    state, items = digest.gate(rows)
    harness.write_json(out / "feedback_E0.json", {**data, "checklist": [list(row) for row in rows], "gate": {"state": state, "items": items},
                                                  "findings": digest.findings(data)})
    try:
        text = digest.render(data)
    except Exception:   # a report that cannot be written must not cost the session its record
        text = digest.fallback(data, traceback.format_exc())
    (out / "feedback_E0.md").write_text(text, encoding="utf-8")
    with zipfile.ZipFile(out / "feedback_E0_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ctx.root.rglob("*")):
            relative = path.relative_to(ctx.root)
            if path.is_file() and relative.parts[0] not in ("assets", "assets-upstream") and "trees" not in relative.parts and "assets" not in relative.parts \
                    and path.suffix in (".json", ".jsonl", ".md", ".log", ".xml") and path.name != "feedback_E0_bundle.zip" and path.stat().st_size < 20_000_000:
                bundle.write(path, str(relative))
        for name in ("meta.json", "sanity.json"):
            path = Path(ctx.cache_dir) / "sat" / "traces" / "s1" / name
            if path.exists():
                bundle.write(path, f"trace/{name}")
    if not ctx.quiet:
        say(text)
    say(f"gate G0: {state}" + (f" (items {', '.join(items)})" if items else "") + f"; feedback written to {out}")
    return state != "FAIL"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    parser.add_argument("--root", default="outputs/sat/e0")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiny", action="store_true", help="CPU dry run on synthetic assets")
    parser.add_argument("--budget-hours", type=float, default=0.0)
    parser.add_argument("--deadline", type=float, default=0.0, help="time.time() value after which no new long run starts")
    parser.add_argument("--force", action="store_true", help="start a stage even when a stage it builds on did not pass")
    parser.add_argument("--quiet", action="store_true", help="do not print the feedback report")
    parser.add_argument("--pending", default="", help="feedback stage: comma-separated stages the session will still run (an interim report says so)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the report is in Chinese whatever the locale of the pipe
    ctx = Context(args)
    stages = list(args.stages)
    if "all" in stages:
        stages = [stage for stage in SESSION if not (ctx.tiny and stage == "tests")]
    functions = {"env": stage_env, "baseline": stage_baseline, "tests": stage_tests, "trace": stage_trace, "data": stage_data,
                 "selfcheck": stage_selfcheck, "memory": stage_memory, "smoke": stage_smoke, "pilot": stage_pilot, "t0": stage_t0,
                 "feedback": stage_feedback}
    results = {}
    for stage in stages:
        reason, error = None if ctx.force else blocked(ctx.root, stage), None
        started = time.monotonic()
        say(f"== {stage} ==")
        if reason:
            say(f"  skipped: the stage '{reason}' did not pass (--force runs it anyway)")
            results[stage] = "skipped"
        else:
            say(f"  {ABOUT[stage]}")
            try:
                results[stage] = "ok" if functions[stage](ctx) else "attention"
            except Exception:   # a stage that breaks is recorded like one that fails: what builds on it is skipped, the report shows why
                error = traceback.format_exc()
                say(error)
                results[stage] = "attention"
        record_stage(ctx.root, stage, results[stage], time.monotonic() - started, reason, error and "\n".join(error.splitlines()[-20:]))
        say(f"  {stage}: {results[stage].upper() if results[stage] != 'ok' else 'ok'} in {clock(time.monotonic() - started)}\n")
    say("stage results: " + ", ".join(f"{stage}={result.upper() if result != 'ok' else result}" for stage, result in results.items()))
    return 0 if all(result == "ok" for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
