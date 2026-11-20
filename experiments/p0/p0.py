#!/usr/bin/env python
"""P0 pipeline: golden values, regression against the pre-refactor revisions, the E0 reproduction and a feedback report.

Stages (each can be re-run; finished runs are reused, unfinished ones are set aside and repeated):

  env        record interpreter, packages, GPU and git state                        -> <root>/env.json
  tests      run the CPU test suite (add --gpu for the GPU-marked tests)            -> <root>/pytest.xml
  golden     smoke configurations on the upstream and the instrumented revision,
             twice on the latter, plus ho-fl-replay                                 -> <root>/golden/<rev>-<gpu>/golden.json
  regress    smoke configurations and replay on the current tree, compared with
             the golden values (invariants I1 and I2)                               -> <root>/regress/<rev>-<gpu>/verdicts.json
  e0         3-round preflight of every E0 configuration, then the E0 queue under
             a time budget                                                          -> <root>/e0/queue.json
  feedback   everything above condensed into feedback_P0.md / .json and a zip       -> <root>/feedback/
  all        env, tests, golden, regress, e0, feedback

A completed E0 run is reused whatever revision produced it (its revision is listed in the feedback); the regress
stage is the per-revision check and always runs on the current tree. To repeat a run, rename its directory.

Every run is `python -m ho_fl.train` in a subprocess. `--tiny` replaces the real model and data by the synthetic
tiny assets of ho_fl.testing and runs on CPU; it checks the plumbing of this pipeline in about two minutes and its
numbers mean nothing.
"""
import argparse
import io
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ElementTree
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ho_fl import harness  # noqa: E402

REVISIONS = json.loads((Path(__file__).parent / "revisions.json").read_text())
SMOKE_STATE = "states/round_0003.safetensors"
SAFETY = 1.25           # estimated durations are multiplied by this before they are compared with the budget
DEFAULT_ROUNDS = 160
EVALUATE_EVERY = 10     # default of ho-fl-train for E = 5

# The three smoke configurations of the golden-value protocol (brief, section 8).
SMOKE = {
    "hofl": {"model": "opt", "dataset": "sst2", "method": "hofl", "beta": 0.75, "rounds": 3, "evaluate_every": 3, "seed": 42},
    "fedavg": {"model": "opt", "dataset": "sst2", "method": "fedavg", "rounds": 3, "evaluate_every": 3, "seed": 42},
    "decomfl": {"model": "opt", "dataset": "sst2", "method": "decomfl", "rounds": 3, "evaluate_every": 3, "seed": 42},
}

# E0 groups as (model, dataset, partition, method, beta), with the accuracy (%) and standard deviation that the
# brief (section 7) quotes from the HO-FL paper. These are reference values, not results of this repository.
REFERENCE = {
    ("opt", "sst2", "iid", "hofl", 0.75): (86.54, 0.24),
    ("opt", "ag_news", "dirichlet_05", "hofl", 0.75): (62.85, 6.52),
    ("opt", "ag_news", "dirichlet_05", "hofl", 0.0): (59.58, 2.24),
    ("opt", "ag_news", "dirichlet_05", "fedavg", 0.0): (78.24, 2.97),
    ("opt", "ag_news", "dirichlet_05", "decomfl", 0.0): (23.38, 0.08),
    ("qwen", "ag_news", "dirichlet_05", "hofl", 0.75): (86.69, 0.56),
    ("qwen", "ag_news", "dirichlet_05", "hofl", 0.0): (85.51, 0.43),
    ("qwen", "ag_news", "dirichlet_05", "fedavg", 0.0): (88.15, 0.21),
    ("qwen", "ag_news", "dirichlet_05", "decomfl", 0.0): (55.83, 0.08),
}
PLANNED_SEEDS = {"opt": [42, 43, 44, 45, 46], "qwen": [42, 43, 44]}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(text=""):
    print(text, flush=True)


def git(*arguments):
    result = subprocess.run(["git", "-C", str(REPO), *arguments], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def head_revision():
    sha = git("rev-parse", "HEAD")
    if sha is None:
        return "unknown"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def short(revision):
    return revision[:12] + ("-dirty" if revision.endswith("-dirty") else "")


def hardware_slug(device):
    """Names the device and torch version, e.g. ``tesla-t4-torch2.6.0``. Golden values are only valid per slug."""
    import torch

    name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + "-torch" + torch.__version__.split("+")[0]


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if abs(value) < 1e-3 and value != 0 else f"{value:.{digits}f}"
    return str(value)


class Context:
    def __init__(self, args):
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tiny = args.tiny
        self.device = "cpu" if args.tiny else args.device
        self.mirror = Path(args.mirror).resolve() if args.mirror else None
        self.plan = "tiny" if args.tiny else args.plan
        self.rounds = args.rounds if args.rounds else (3 if args.tiny else DEFAULT_ROUNDS)
        self.gpu_tests = args.gpu
        self.quiet = args.quiet
        self.deadline = None
        if args.deadline:
            self.deadline = args.deadline
        elif args.budget_hours:
            self.deadline = time.time() + 3600 * args.budget_hours
        self.cache_dir = str(Path(args.cache_dir).resolve())
        self.assets = {}
        if self.tiny:
            from ho_fl.testing import build_tiny_assets
            self.assets = build_tiny_assets(self.root / "assets", datasets=("sst2", "ag_news"))
        # Golden values depend on the hardware, so results recorded on another GPU model are never reused.
        self.hardware = hardware_slug(self.device)
        self.golden_root = self.root / "golden" / f"{short(REVISIONS['instrumented'])}-{self.hardware}"
        self.head = head_revision()
        self.regress_root = self.root / "regress" / f"{short(self.head)}-{self.hardware}"
        self.e0_root = self.root / "e0"

    def common(self, dataset):
        """Fields added to every run: where the model and data come from and which device to use."""
        if self.tiny:
            return {key: value for key, value in self.assets[dataset].items() if key not in ("model", "dataset")}
        return {"device": self.device, "cache_dir": self.cache_dir}

    def remaining(self):
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def sync(self):
        """Copy the small result files of this root to the mirror (run directories are mirrored as they finish)."""
        if self.mirror is None:
            return
        for path in self.root.rglob("*.json"):
            relative = path.relative_to(self.root)
            if relative.parts[0] in ("trees", "assets") or ".stale-" in str(relative):
                continue
            target = self.mirror / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name in ("pytest.xml", "pytest.log"):
            if (self.root / name).exists():
                shutil.copy2(self.root / name, self.mirror / name)
        if (self.root / "feedback").exists():
            shutil.copytree(self.root / "feedback", self.mirror / "feedback", dirs_exist_ok=True)

    def restore(self):
        """Bring back results of an earlier session from the mirror; files that exist locally are kept."""
        if self.mirror is None or not self.mirror.exists():
            return
        restored = 0
        for path in self.mirror.rglob("*"):
            relative = path.relative_to(self.mirror)
            if path.is_dir() or relative.parts[0] == "feedback":
                continue
            target = self.root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                restored += 1
        if restored:
            say(f"restored {restored} files from {self.mirror}")


def run(ctx, fields, output, tree, module="ho_fl.train", revision=None, label=None):
    merged = {**fields, **ctx.common(fields.get("dataset", "sst2"))}
    mirror = ctx.mirror / Path(output).relative_to(ctx.root) if ctx.mirror else None
    return harness.run_training(merged, output, tree=tree, module=module, label=label, revision=revision,
                                mirror=mirror, echo=say)


def export_tree(ctx, name):
    """Source tree of a recorded revision, extracted with `git archive` (the repository itself is not touched)."""
    revision = REVISIONS[name]
    target = ctx.root / "trees" / f"{name}-{revision[:12]}"
    if not (target / "ho_fl" / "train.py").exists():
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", revision], capture_output=True)
        if archive.returncode != 0:
            raise RuntimeError(f"Cannot export revision {revision} ({name}): {archive.stderr.decode(errors='replace').strip()}. "
                               "The pipeline needs a git clone with full history, not a source archive.")
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            try:
                tar.extractall(target, filter="data")
            except TypeError:
                tar.extractall(target)
    return target


# --- env -----------------------------------------------------------------------------------------------------------

def stage_env(ctx):
    import importlib.metadata as metadata

    say("== env ==")
    try:
        import tomllib
        pinned = dict(spec.split("==") for spec in tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"])
    except Exception as error:
        pinned = {"error": repr(error)}
    packages = {}
    for name in [*[n for n in pinned if n != "error"], "pytest", "tokenizers"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info = {"recorded_at": now(), "python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": packages, "pinned": pinned,
            "mismatched": {name: {"installed": packages.get(name), "pinned": version}
                           for name, version in pinned.items() if name != "error" and (packages.get(name) or "").split("+")[0] != version},
            "git": {"head": ctx.head, "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
            "revisions": {name: {"sha": sha, "available": git("cat-file", "-e", sha + "^{commit}") is not None}
                          for name, sha in REVISIONS.items()},
            "device": ctx.device, "tiny": ctx.tiny}
    try:
        import torch
        info.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if ctx.device.startswith("cuda") and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
            x = torch.randn(256, 256, device=ctx.device, dtype=torch.bfloat16)
            info["bf16_matmul_finite"] = bool(torch.isfinite((x @ x).float()).all().item())
    except Exception as error:
        info["torch_error"] = repr(error)
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    info["nvidia_smi"] = query.stdout.strip() if query is not None and query.returncode == 0 else None
    if info["nvidia_smi"]:
        info["driver_version"] = info["nvidia_smi"].splitlines()[0].split(",")[1].strip()
    harness.write_json(ctx.root / "env.json", info)
    say(json.dumps({k: info.get(k) for k in ("python", "torch", "cuda_runtime", "cuda_available", "gpu_name",
                                              "driver_version", "bf16_matmul_finite", "mismatched", "git")}, indent=2))
    problems = []
    if not ctx.tiny and not info.get("cuda_available"):
        problems.append("no CUDA device is visible; the GPU stages will fail")
    if info["mismatched"]:
        problems.append("installed package versions differ from pyproject.toml (see 'mismatched')")
    problems += [f"revision {name} ({r['sha'][:12]}) is not in this clone" for name, r in info["revisions"].items() if not r["available"]]
    for problem in problems:
        say(f"WARNING: {problem}")
    return not any("revision" in p or "CUDA" in p for p in problems)


# --- tests ---------------------------------------------------------------------------------------------------------

def stage_tests(ctx):
    say("== tests ==")
    command = [sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", f"--junitxml={ctx.root / 'pytest.xml'}"]
    if ctx.gpu_tests:
        command += ["-m", "gpu"]
    started = time.time()
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, env=harness.tree_environment(REPO))
    (ctx.root / "pytest.log").write_text(result.stdout + result.stderr)
    say("\n".join((result.stdout + result.stderr).splitlines()[-40:]))
    say(f"pytest exit code {result.returncode} in {time.time() - started:.0f}s")
    return result.returncode == 0


def parse_junit(path):
    if not Path(path).exists():
        return None
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    total["time_s"] = round(sum(float(s.get("time", 0)) for s in suites), 1)
    total["failed"] = []
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                total["failed"].append({"test": f"{case.get('classname')}::{case.get('name')}", "kind": kind,
                                        "message": (node.get("message") or "")[:400]})
    return total


# --- golden --------------------------------------------------------------------------------------------------------

def replay_summary(path):
    summary = harness.read_json(Path(path) / "summary.json") or {}
    return {key: summary.get(key) for key in ("status", "all_bitwise_equal", "comparisons", "max_abs_error",
                                               "max_relative_l2_error", "device_backend", "torch_version")}


def stage_golden(ctx):
    say("== golden ==")
    trees = {name: export_tree(ctx, name) for name in ("baseline", "instrumented")}
    root = ctx.golden_root
    records = {}
    for name, fields in SMOKE.items():
        for repeat in ("a", "b"):
            records[f"instrumented-{name}-{repeat}"] = run(ctx, {**fields, "save_state_rounds": "3"}, root / f"instrumented-{name}-{repeat}",
                                                           trees["instrumented"], revision=REVISIONS["instrumented"])
        records[f"baseline-{name}"] = run(ctx, fields, root / f"baseline-{name}", trees["baseline"], revision=REVISIONS["baseline"])
    records["instrumented-replay"] = run(ctx, {}, root / "instrumented-replay", trees["instrumented"], module="ho_fl.replay",
                                         revision=REVISIONS["instrumented"])
    failed = sorted(name for name, record in records.items() if record["status"] != "COMPLETED")
    configs = {}
    for name, fields in SMOKE.items():
        first, second = root / f"instrumented-{name}-a", root / f"instrumented-{name}-b"
        reference = harness.fingerprint(first)
        repeat = harness.compare_fingerprints(reference, harness.fingerprint(second))
        repeat["max_abs_state_diff"] = harness.state_difference(first / SMOKE_STATE, second / SMOKE_STATE)
        upstream = harness.compare_fingerprints(harness.fingerprint(root / f"baseline-{name}"), reference)
        configs[name] = {"fields": fields, **{key: reference[key] for key in harness.FINGERPRINT_KEYS},
                         "repeat": repeat, "upstream": upstream}
    environment = harness.read_json(ctx.root / "env.json") or {}
    reproducible = all(c["repeat"]["bitwise"] and c["repeat"]["hash_equal"] for c in configs.values())
    golden = {"schema": 1, "recorded_at": now(), "complete": not failed, "failed_runs": failed, "tiny": ctx.tiny,
              "device": ctx.device, "revisions": REVISIONS, "mode": "bitwise" if reproducible else "tolerance",
              "environment": {key: environment.get(key) for key in ("gpu_name", "driver_version", "cuda_runtime", "torch",
                                                                    "python", "platform", "packages")},
              "note": "Instrumented runs add --save-state-rounds 3 to the commands of the brief; upstream runs use them unchanged.",
              "configs": configs, "replay": replay_summary(root / "instrumented-replay")}
    harness.write_json(root / "golden.json", golden)
    say()
    say(table(["config", "repeat runs bitwise", "max |state diff| between repeats", "logs identical to upstream", "final hash"],
              [[name, fmt(c["repeat"]["bitwise"] and c["repeat"]["hash_equal"]), fmt(c["repeat"]["max_abs_state_diff"]),
                fmt(all(c["upstream"]["files_identical"].values())), (c["final_parameter_sha256"] or "n/a")[:16]]
               for name, c in configs.items()]))
    say(f"replay on the instrumented revision: {golden['replay']}")
    say(f"golden mode: {golden['mode']}; written to {root / 'golden.json'}")
    if failed:
        say(f"FAILED runs: {failed}")
    ctx.sync()
    return not failed


# --- regress -------------------------------------------------------------------------------------------------------

def i1_verdict(repeat, comparison, status):
    """Verdict on invariant I1 for one smoke configuration (see the brief, section 5)."""
    if status != "COMPLETED":
        return "RUN_FAILED"
    if comparison["bitwise"] and comparison["hash_equal"]:
        return "PASS_BITWISE"
    if repeat["bitwise"] and repeat["hash_equal"]:
        return "FAIL"  # the reference revision reproduces itself on this device, the current tree does not match it
    difference, noise = comparison.get("max_abs_state_diff"), repeat.get("max_abs_state_diff")
    if difference is None:
        return "INCONCLUSIVE"
    if difference <= harness.STATE_TOLERANCE:
        return "PASS_TOLERANCE"
    if noise is not None and noise > harness.STATE_TOLERANCE:
        return "INCONCLUSIVE"  # two runs of the reference revision already differ by more than the tolerance
    return "FAIL"


def stage_regress(ctx):
    say("== regress ==")
    golden = harness.read_json(ctx.golden_root / "golden.json")
    if golden is None or not golden.get("complete"):
        say(f"No complete golden values at {ctx.golden_root}; run the golden stage first.")
        return False
    root = ctx.regress_root
    statuses = {name: run(ctx, {**fields, "save_state_rounds": "3"}, root / f"head-{name}", REPO, revision=ctx.head)["status"]
                for name, fields in SMOKE.items()}
    replay = run(ctx, {}, root / "head-replay", REPO, module="ho_fl.replay", revision=ctx.head)
    i1 = {}
    for name in SMOKE:
        comparison = harness.compare_fingerprints(golden["configs"][name], harness.fingerprint(root / f"head-{name}"))
        comparison["max_abs_state_diff"] = harness.state_difference(ctx.golden_root / f"instrumented-{name}-a" / SMOKE_STATE,
                                                                    root / f"head-{name}" / SMOKE_STATE)
        comparison["verdict"] = i1_verdict(golden["configs"][name]["repeat"], comparison, statuses[name])
        i1[name] = comparison
    head_replay = replay_summary(root / "head-replay")
    i2 = {"head": head_replay, "instrumented": golden["replay"],
          "verdict": "PASS" if replay["status"] == "COMPLETED" and head_replay["status"] == "PASSED" else "FAIL"}
    verdicts = {"recorded_at": now(), "head": ctx.head, "golden_mode": golden["mode"], "tiny": ctx.tiny, "i1": i1, "i2": i2}
    harness.write_json(root / "verdicts.json", verdicts)
    say()
    say(table(["config", "I1 verdict", "losses equal", "evaluations equal", "final hash equal", "max |state diff|", "first difference"],
              [[name, c["verdict"], fmt(c["rounds_equal"]), fmt(c["evaluation_equal"]), fmt(c["hash_equal"]),
                fmt(c["max_abs_state_diff"]), (c["differences"] or [""])[0]] for name, c in i1.items()]))
    say(f"I2 (ho-fl-replay on the current tree): {i2['verdict']} {head_replay}")
    ctx.sync()
    return all(c["verdict"].startswith("PASS") for c in i1.values()) and i2["verdict"] == "PASS"


# --- e0 ------------------------------------------------------------------------------------------------------------

def e0_specs(plan, rounds):
    opt = [key for key in REFERENCE if key[0] == "opt"]
    qwen = [key for key in REFERENCE if key[0] == "qwen"]
    tiers = {"tiny": [(opt, [42])], "opt": [(opt, [42])], "lite": [(opt, [42]), (qwen, [42])],
             "full": [(opt, [42]), (qwen, [42]), (opt, [43, 44, 45, 46]), (qwen, [43, 44])]}[plan]
    specs = []
    for groups, seeds in tiers:
        for seed in seeds:
            for model, dataset, partition, method, beta in groups:
                fields = {"model": model, "dataset": dataset, "partition": partition, "method": method, "seed": seed, "rounds": rounds}
                name = f"{model}-{dataset}-{partition}-{method}"
                if method == "hofl":
                    fields["beta"] = beta
                    name += f"-b{beta:g}"
                    if rounds >= 40:
                        fields["save_state_rounds"] = "40"  # starting point of the dose-response experiment (E2)
                name += f"-s{seed}" + ("" if rounds == DEFAULT_ROUNDS else f"-r{rounds}")
                specs.append({"name": name, "group": [model, dataset, partition, method, beta], "seed": seed,
                              "timing_key": f"{model}/{dataset}/{partition}/{method}", "fields": fields})
    return specs


def stage_e0(ctx):
    say(f"== e0 (plan {ctx.plan}, {ctx.rounds} rounds) ==")
    specs = e0_specs(ctx.plan, ctx.rounds)
    root = ctx.e0_root
    timings = harness.read_json(root / "timings.json", {})
    preflight_failures = {}
    for spec in specs:
        key = spec["timing_key"]
        if key in timings or key in preflight_failures:
            continue
        if harness.run_status(root / "runs" / spec["name"]) == "COMPLETED":
            continue
        if ctx.remaining() < 300:
            say(f"  preflight of {key} skipped: less than five minutes of budget left")
            preflight_failures[key] = "SKIPPED_BUDGET"
            continue
        fields = {k: v for k, v in spec["fields"].items() if k != "save_state_rounds"}
        record = run(ctx, {**fields, "rounds": 3, "evaluate_every": 3}, root / "preflight" / key.replace("/", "-"), REPO,
                     revision=ctx.head, label=f"preflight {key}")
        if record["status"] == "COMPLETED" and record.get("timing"):
            timings[key] = {**record["timing"], "source": "preflight"}
            harness.write_json(root / "timings.json", timings)
            estimate = harness.estimate_seconds(timings[key], ctx.rounds, EVALUATE_EVERY)
            say(f"  {key}: {timings[key]['round_s']:.1f}s per round, {timings[key]['eval_s']:.1f}s per evaluation, "
                f"{timings[key]['startup_s']:.0f}s start-up -> about {estimate / 60:.0f} min for {ctx.rounds} rounds")
        else:
            preflight_failures[key] = record.get("error") or "preflight run failed"
    queue = []
    for spec in specs:
        key, output = spec["timing_key"], root / "runs" / spec["name"]
        entry = {"name": spec["name"], "group": spec["group"], "seed": spec["seed"], "output": str(output)}
        if harness.run_status(output) == "COMPLETED":
            record = run(ctx, spec["fields"], output, REPO, revision=ctx.head)
            entry.update(status="COMPLETED", reused=True, wall_s=record.get("wall_s"))
        elif key not in timings:
            entry.update(status="SKIPPED_PREFLIGHT", reason=str(preflight_failures.get(key, "no timing available")))
        else:
            estimate = SAFETY * harness.estimate_seconds(timings[key], ctx.rounds, EVALUATE_EVERY)
            entry["estimated_s"] = round(estimate)
            if estimate > ctx.remaining():
                entry.update(status="SKIPPED_BUDGET", remaining_s=round(ctx.remaining()))
                say(f"  [{spec['name']}] skipped: needs about {estimate / 60:.0f} min, {max(ctx.remaining(), 0) / 60:.0f} min of budget left")
            else:
                record = run(ctx, spec["fields"], output, REPO, revision=ctx.head)
                entry.update(status=record["status"], reused=False, wall_s=record.get("wall_s"), error=record.get("error"))
                if record["status"] == "COMPLETED" and record.get("timing"):
                    timings[key] = {**record["timing"], "source": spec["name"]}
                    harness.write_json(root / "timings.json", timings)
        queue.append(entry)
        harness.write_json(root / "queue.json", {"recorded_at": now(), "plan": ctx.plan, "rounds": ctx.rounds, "head": ctx.head,
                                                  "preflight_failures": preflight_failures, "queue": queue})
        ctx.sync()
    say()
    say(table(["run", "status", "wall min", "estimate min"],
              [[e["name"], e["status"], fmt(e["wall_s"] / 60, 1) if e.get("wall_s") else "", fmt(e["estimated_s"] / 60, 1) if e.get("estimated_s") else ""]
               for e in queue]))
    return all(e["status"] == "COMPLETED" for e in queue)


# --- feedback ------------------------------------------------------------------------------------------------------

def e0_results(ctx):
    runs = []
    for path in sorted((ctx.e0_root / "runs").glob("*")) if (ctx.e0_root / "runs").exists() else []:
        if not path.is_dir() or ".stale-" in path.name:
            continue
        loaded = harness.load_run(path)
        config, summary = loaded["config"], loaded["summary"]
        if loaded["status"] != "COMPLETED" or not config:
            continue
        beta = config["beta"] if config["method"] == "hofl" else 0.0
        runs.append({"name": path.name, "group": [config["model"], config["dataset"], config["partition"], config["method"], beta],
                     "seed": config["seed"], "rounds": config["rounds"], "accuracy": 100 * summary["accuracy"],
                     "evaluation_loss": summary.get("task_loss"), "wall_s": loaded["meta"].get("wall_s"),
                     "revision": loaded["meta"].get("revision"),
                     "final_parameter_sha256": summary.get("final_parameter_sha256"),
                     "curve": [{"round": row["round"], "accuracy": 100 * row["accuracy"], "evaluation_loss": row["task_loss"],
                                "training_loss": row.get("training_task_loss")} for row in loaded["curves"] if "accuracy" in row]})
    groups = []
    for key, (reference, reference_sd) in REFERENCE.items():
        members = [r for r in runs if tuple(r["group"]) == key]
        entry = {"group": list(key), "reference": reference, "reference_sd": reference_sd,
                 "tolerance": max(1.0, 2 * reference_sd), "planned_seeds": len(PLANNED_SEEDS[key[0]]),
                 "seeds": sorted(r["seed"] for r in members), "accuracies": [r["accuracy"] for r in sorted(members, key=lambda r: r["seed"])]}
        if not members:
            entry["verdict"] = "NOT_RUN"
        else:
            entry["mean"] = statistics.mean(entry["accuracies"])
            entry["sd"] = statistics.stdev(entry["accuracies"]) if len(members) > 1 else None
            entry["difference"] = entry["mean"] - reference
            if any(r["rounds"] != DEFAULT_ROUNDS for r in members):
                entry["verdict"] = "NOT_COMPARABLE"
            elif abs(entry["difference"]) > entry["tolerance"]:
                entry["verdict"] = "OUTSIDE"
            else:
                entry["verdict"] = "WITHIN" if len(members) >= entry["planned_seeds"] else "WITHIN_PRELIMINARY"
        groups.append(entry)
    return runs, groups


def inventory(ctx):
    rows = []
    for meta_path in sorted(ctx.root.rglob("run_meta.json")):
        relative = meta_path.parent.relative_to(ctx.root)
        if relative.parts[0] in ("trees", "assets"):
            continue
        meta = harness.read_json(meta_path) or {}
        rows.append({"run": str(relative), "status": meta.get("status"), "wall_s": meta.get("wall_s"),
                     "revision": (meta.get("revision") or "")[:12], "error": meta.get("error"),
                     "log_tail": meta.get("log_tail") if meta.get("status") != "COMPLETED" else None})
    return rows


def checklist(data):
    tests, regress, groups = data["tests"], data["regress"], data["e0"]["groups"]
    items = {}
    if tests is None:
        items["tests"] = "NOT_RUN"
    else:
        items["tests"] = "PASS" if tests["tests"] > 0 and not tests["failures"] and not tests["errors"] else "FAIL"
    if regress is None:
        items["I1"] = items["I2"] = "NOT_RUN"
    else:
        verdicts = [c["verdict"] for c in regress["i1"].values()]
        items["I1"] = "PASS" if all(v.startswith("PASS") for v in verdicts) else ("FAIL" if "FAIL" in verdicts or "RUN_FAILED" in verdicts else "INCONCLUSIVE")
        items["I2"] = regress["i2"]["verdict"]
    verdicts = [g["verdict"] for g in groups]
    if "OUTSIDE" in verdicts:
        items["E0"] = "CHECK"
    elif all(v == "WITHIN" for v in verdicts):
        items["E0"] = "PASS"
    elif all(v == "NOT_RUN" for v in verdicts):
        items["E0"] = "NOT_RUN"
    else:
        items["E0"] = "INCOMPLETE"
    if data["tiny"]:
        items["G0"] = "DRY_RUN"
    elif all(v == "PASS" for v in items.values()):
        items["G0"] = "PASS"
    elif any(v in ("FAIL", "CHECK") for v in items.values()):
        items["G0"] = "BLOCKED"
    else:
        items["G0"] = "INCOMPLETE"
    return items


def collect(ctx):
    runs, groups = e0_results(ctx)
    queue = harness.read_json(ctx.e0_root / "queue.json")
    timings = harness.read_json(ctx.e0_root / "timings.json", {})
    rows = inventory(ctx)
    data = {"generated_at": now(), "tiny": ctx.tiny, "head": ctx.head, "hardware": ctx.hardware, "revisions": REVISIONS,
            "environment": harness.read_json(ctx.root / "env.json"), "tests": parse_junit(ctx.root / "pytest.xml"),
            "golden": harness.read_json(ctx.golden_root / "golden.json"),
            "regress": harness.read_json(ctx.regress_root / "verdicts.json"),
            "e0": {"queue": queue, "timings": timings, "runs": runs, "groups": groups,
                   "estimated_run_s": {key: harness.estimate_seconds(t, DEFAULT_ROUNDS, EVALUATE_EVERY) for key, t in timings.items()}},
            "runs": rows, "used_gpu_hours": sum(r["wall_s"] or 0 for r in rows) / 3600}
    full = e0_specs("full", DEFAULT_ROUNDS)
    done = {r["name"] for r in runs}
    pending = [s for s in full if s["name"] not in done]
    data["e0"]["remaining_full_plan"] = {
        "runs": len(pending), "without_timing": sorted({s["timing_key"] for s in pending if s["timing_key"] not in timings}),
        "estimated_hours": sum(data["e0"]["estimated_run_s"].get(s["timing_key"], 0) for s in pending) / 3600}
    data["checklist"] = checklist(data)
    return data


def render(data):
    out = []
    add = out.append
    env, tests, golden, regress, e0 = data["environment"] or {}, data["tests"], data["golden"], data["regress"], data["e0"]
    add("# P0 feedback" + (" (TINY DRY RUN: plumbing check on CPU, the numbers mean nothing)" if data["tiny"] else ""))
    add(f"\nGenerated {data['generated_at']} for revision `{data['head']}` on `{data['hardware']}`.\n")
    add("## 1. Gate G0 checklist\n")
    labels = {"tests": "CPU test suite", "I1": "I1 baseline equivalence (smoke configurations)", "I2": "I2 ho-fl-replay",
              "E0": "E0 reproduction", "G0": "Gate G0"}
    add(table(["item", "status"], [[labels[key], value] for key, value in data["checklist"].items()]))

    add("\n## 2. Environment\n")
    add(table(["field", "value"], [[key, env.get(key)] for key in ("gpu_name", "gpu_memory_gib", "driver_version", "torch", "cuda_runtime",
                                                                 "python", "platform", "bf16_matmul_finite")]))
    if env.get("mismatched"):
        add("\nPackages that differ from `pyproject.toml`: " + ", ".join(f"{name} {v['installed']} (pinned {v['pinned']})" for name, v in env["mismatched"].items()))

    add("\n## 3. Tests\n")
    if tests is None:
        add("Not run.")
    else:
        add(f"{tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped, {tests['time_s']} s.")
        for failure in tests["failed"]:
            add(f"- `{failure['test']}` ({failure['kind']}): {failure['message']}")

    add("\n## 4. Golden values and reproducibility of the reference revision\n")
    if golden is None:
        add("Not run.")
    else:
        add(f"Mode: **{golden['mode']}** (bitwise = two runs of the instrumented revision are identical on this device). "
            f"Complete: {fmt(golden['complete'])}." + (f" Failed runs: {golden['failed_runs']}." if golden["failed_runs"] else ""))
        add("")
        add(table(["config", "repeat bitwise", "max abs state diff between repeats", "logs identical to upstream revision", "task_loss per round", "final accuracy"],
                  [[name, fmt(c["repeat"]["bitwise"] and c["repeat"]["hash_equal"]), fmt(c["repeat"]["max_abs_state_diff"]),
                    fmt(all(c["upstream"]["files_identical"].values())), ", ".join(f"{v:.6f}" for v in c["task_loss"]),
                    fmt(c["evaluation"][-1].get("accuracy")) if c["evaluation"] else "n/a"] for name, c in golden["configs"].items()]))
        add(f"\nReplay on the instrumented revision: {golden['replay']}")

    add("\n## 5. Invariants I1 and I2 on the current tree\n")
    if regress is None:
        add("Not run for this revision.")
    else:
        add(table(["config", "verdict", "losses equal", "evaluations equal", "final hash equal", "max abs state diff", "first difference"],
                  [[name, c["verdict"], fmt(c["rounds_equal"]), fmt(c["evaluation_equal"]), fmt(c["hash_equal"]), fmt(c["max_abs_state_diff"]),
                    (c["differences"] or [""])[0]] for name, c in regress["i1"].items()]))
        add(f"\nI2 replay on the current tree: **{regress['i2']['verdict']}** {regress['i2']['head']}")

    add("\n## 6. E0 reproduction\n")
    add("Reference values are those the brief quotes from the HO-FL paper; tolerance is max(1.0, 2 x paper sd) on the mean over seeds.\n")
    add(table(["model", "dataset", "partition", "method", "beta", "seeds run", "accuracy % per seed", "mean", "sd", "reference (sd)", "difference", "tolerance", "verdict"],
              [[*g["group"], f"{len(g['seeds'])}/{g['planned_seeds']}", ", ".join(f"{a:.2f}" for a in g["accuracies"]), fmt(g.get("mean"), 2),
                fmt(g.get("sd"), 2), f"{g['reference']:.2f} ({g['reference_sd']:.2f})", fmt(g.get("difference"), 2), fmt(g["tolerance"], 2),
                g["verdict"]] for g in e0["groups"]]))
    if e0["queue"]:
        add("\nQueue of this session (plan `{plan}`, {rounds} rounds):\n".format(**e0["queue"]))
        add(table(["run", "status", "wall min", "estimated min", "note"],
                  [[q["name"], q["status"], fmt(q["wall_s"] / 60, 1) if q.get("wall_s") else "", fmt(q["estimated_s"] / 60, 1) if q.get("estimated_s") else "",
                    q.get("reason") or q.get("error") or ""] for q in e0["queue"]["queue"]]))
        if e0["queue"]["preflight_failures"]:
            add("\nPreflight failures: " + json.dumps(e0["queue"]["preflight_failures"]))
    for run_record in e0["runs"]:
        add(f"\nCurve of `{run_record['name']}` (round: accuracy %, evaluation loss, training loss):")
        add("; ".join(f"{p['round']}: {p['accuracy']:.2f}, {p['evaluation_loss']:.4f}, " + (f"{p['training_loss']:.4f}" if p["training_loss"] is not None else "n/a")
                      for p in run_record["curve"]))

    add("\n## 7. Timing and GPU hours\n")
    if e0["timings"]:
        add(table(["configuration", "s per round", "s per evaluation", "start-up s", "estimated min per 160-round run", "measured on"],
                  [[key, fmt(t["round_s"], 2), fmt(t["eval_s"], 2), fmt(t["startup_s"], 1), fmt(e0["estimated_run_s"][key] / 60, 1), t.get("source")]
                   for key, t in e0["timings"].items()]))
    remaining = e0["remaining_full_plan"]
    add(f"\nWall time of all runs recorded under this root: {data['used_gpu_hours']:.2f} h. "
        f"Runs of the full E0 plan still missing: {remaining['runs']}, estimated {remaining['estimated_hours']:.1f} h"
        + (f" (no timing yet for {remaining['without_timing']}, not included)." if remaining["without_timing"] else "."))

    add("\n## 8. Run inventory\n")
    add(table(["run", "status", "wall s", "revision"], [[r["run"], r["status"], fmt(r["wall_s"], 0) if r["wall_s"] else "", r["revision"]] for r in data["runs"]]))
    for failed in [r for r in data["runs"] if r["status"] != "COMPLETED"]:
        add(f"\n`{failed['run']}` failed: {failed['error']}\n\n```\n{failed['log_tail'] or ''}\n```")

    add("\n## 9. What this means and what to do next\n")
    for line in findings(data):
        add(f"- {line}")
    return "\n".join(out) + "\n"


def findings(data):
    lines = []
    env, tests, golden, regress, e0 = data["environment"] or {}, data["tests"], data["golden"], data["regress"], data["e0"]
    if data["tiny"]:
        lines.append("This was the tiny dry run. It only shows that every stage of the pipeline executes; switch it off for real results.")
    if env.get("mismatched"):
        lines.append("The environment differs from the pinned versions; results are valid for this environment only and E0 may deviate from the paper.")
    if tests and (tests["failures"] or tests["errors"]):
        lines.append(f"{tests['failures'] + tests['errors']} unit tests failed. Fix these first; send the list in section 3 back.")
    if golden is not None:
        if golden["mode"] == "bitwise":
            lines.append("The reference revision is bitwise reproducible on this device, so I1 is checked bit for bit.")
        else:
            worst = max((c["repeat"]["max_abs_state_diff"] or 0) for c in golden["configs"].values())
            lines.append(f"The reference revision is NOT bitwise reproducible on this device (largest state difference between two identical runs: {worst:.3g}). "
                         f"I1 is checked with the tolerance of replay.py ({harness.STATE_TOLERANCE:g}); report this in P0.md.")
        if not all(all(c["upstream"]["files_identical"].values()) for c in golden["configs"].values()):
            lines.append("The logs of the upstream revision and of the instrumented revision are not byte-identical. If the device is bitwise reproducible, the instrumentation commit changed behaviour and must be inspected.")
    if regress is not None:
        for name, comparison in regress["i1"].items():
            if not comparison["verdict"].startswith("PASS"):
                lines.append(f"I1 {comparison['verdict']} for `{name}`: {(comparison['differences'] or ['no logged difference; final states differ'])[0]}. "
                             "Compare train_client and ho_fl/aggregation.py with tests/legacy_loop.py; the first differing round tells which phase diverges.")
        if regress["i2"]["verdict"] != "PASS":
            lines.append("ho-fl-replay fails on the current tree. Backend.local_step or hybrid_step no longer behaves as replay.py expects.")
    outside = [g for g in e0["groups"] if g["verdict"] == "OUTSIDE"]
    for group in outside:
        lines.append(f"E0 outside tolerance for {group['group']}: mean {group['mean']:.2f} vs reference {group['reference']:.2f} "
                     f"(tolerance {group['tolerance']:.2f}, {len(group['seeds'])} of {group['planned_seeds']} seeds). "
                     "With few seeds this can be seed noise; check the curve in section 6 before concluding.")
    preliminary = [g for g in e0["groups"] if g["verdict"] == "WITHIN_PRELIMINARY"]
    if preliminary:
        lines.append(f"{len(preliminary)} E0 groups are within tolerance on fewer seeds than planned; run the remaining seeds (plan `full`) to close E0.")
    missing = [g for g in e0["groups"] if g["verdict"] == "NOT_RUN"]
    if missing:
        lines.append(f"{len(missing)} E0 groups have no completed run yet: " + ", ".join("/".join(str(x) for x in g["group"]) for g in missing) + ".")
    if e0["queue"]:
        skipped = [q for q in e0["queue"]["queue"] if q["status"] == "SKIPPED_BUDGET"]
        if skipped:
            lines.append(f"{len(skipped)} runs did not fit in the time budget and were skipped; re-running the notebook continues with them.")
        failed = [q for q in e0["queue"]["queue"] if q["status"] in ("FAILED", "SKIPPED_PREFLIGHT")]
        for item in failed:
            lines.append(f"E0 run `{item['name']}` did not run to completion ({item['status']}): {item.get('reason') or item.get('error')}.")
    status = data["checklist"]["G0"]
    lines.append({"PASS": "Gate G0: all four criteria hold. Send this report back so that P0.md can be completed.",
                  "BLOCKED": "Gate G0 is blocked by the items marked FAIL or CHECK above. Send this report back before going on.",
                  "INCOMPLETE": "Gate G0 cannot be decided yet: some stages have not run or E0 lacks seeds. Send this report back; it is enough to plan the next session.",
                  "DRY_RUN": "Gate G0 is not assessed in a dry run."}[status])
    return lines


def stage_feedback(ctx):
    say("== feedback ==")
    data = collect(ctx)
    out = ctx.root / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    harness.write_json(out / "feedback_P0.json", data)
    text = render(data)
    (out / "feedback_P0.md").write_text(text)
    keep = ("status.json", "config.json", "summary.json", "rounds.jsonl", "curves.jsonl", "run_meta.json", "init_state.json",
            "replay_errors.csv")
    with zipfile.ZipFile(out / "feedback_P0_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ctx.root.rglob("*")):
            relative = path.relative_to(ctx.root)
            if not path.is_file() or relative.parts[0] in ("trees", "assets") or path.name == "feedback_P0_bundle.zip":
                continue
            if path.name in keep or path.suffix == ".log" or len(relative.parts) <= 2 or path.name in ("golden.json", "verdicts.json"):
                bundle.write(path, str(relative))
    say("checklist: " + ", ".join(f"{key}={value}" for key, value in data["checklist"].items()) if ctx.quiet else text)
    say(f"written: {out / 'feedback_P0.md'}, {out / 'feedback_P0.json'}, {out / 'feedback_P0_bundle.zip'}")
    ctx.sync()
    return data["checklist"]["G0"] in ("PASS", "DRY_RUN", "INCOMPLETE")


STAGES = {"env": stage_env, "tests": stage_tests, "golden": stage_golden, "regress": stage_regress, "e0": stage_e0,
          "feedback": stage_feedback}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], help="one or more stages, run in the order given")
    parser.add_argument("--root", default="outputs/p0", help="directory that receives every result of the pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="cache", help="download and cache directory shared by all runs")
    parser.add_argument("--mirror", default="", help="optional directory (e.g. on a mounted drive) that receives copies of finished runs")
    parser.add_argument("--tiny", action="store_true", help="dry run on CPU with synthetic tiny assets")
    parser.add_argument("--plan", choices=["opt", "lite", "full"], default="lite", help="E0 runs to schedule")
    parser.add_argument("--rounds", type=int, default=0, help="rounds of the E0 runs (default 160; other values are not comparable with the reference)")
    parser.add_argument("--budget-hours", type=float, default=0.0, help="time budget of the e0 stage, counted from now")
    parser.add_argument("--deadline", type=float, default=0.0, help="alternative to --budget-hours: absolute deadline as a Unix timestamp")
    parser.add_argument("--gpu", action="store_true", help="tests stage: run the GPU-marked tests instead of the CPU suite")
    parser.add_argument("--quiet", action="store_true", help="feedback stage: print the checklist only, not the whole report")
    args = parser.parse_args(argv)
    ctx = Context(args)
    ctx.restore()
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
    ctx.sync()
    say("stage results: " + ", ".join(f"{name}={'ok' if ok else 'ATTENTION'}" for name, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
