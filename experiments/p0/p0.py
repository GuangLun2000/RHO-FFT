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
import collections
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
# Zeroth-order dimension per cut from appendix A of the brief; config.json's bottom_dimensions is authoritative.
APPENDIX_A = {"opt": {"3": 73728, "6": 147456, "9": 221184}, "qwen": {"8": 311296, "16": 622592, "24": 933888}}
# First-order supply of the lowest first-order segment (S1) that appendix A of the brief derives for the samplers.
SUPPLY_REFERENCE = {0.0: "mean width 2.0 / 4.0 / 6.0; S1 width <= 2 in 69.2% of rounds, 0 in 6.4%; top-6 share 60%; N_eff 10; max presence 20%",
                    0.75: "mean width 5.0 / 5.5 / 6.0; S1 width <= 2 in 0.4% of rounds; top-6 share 96%; N_eff 6.49; max presence 80%"}
# Sources that ho-fl-profile and ho-fl-memory depend on; they must not differ from the upstream revision (I2).
PROFILE_MEMORY_SOURCES = ["ho_fl/profile.py", "ho_fl/memory.py", "ho_fl/algorithms.py", "ho_fl/decomfl.py", "ho_fl/model.py",
                          "ho_fl/sampling.py", "ho_fl/randomness.py", "ho_fl/split.py"]


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
        # Timings are only valid for the GPU model they were measured on.
        self.timings_path = self.e0_root / f"timings-{self.hardware}.json"

    def common(self, dataset):
        """Fields added to every run: where the model and data come from and which device to use."""
        if self.tiny:
            return {key: value for key, value in self.assets[dataset].items() if key not in ("model", "dataset")}
        return {"device": self.device, "cache_dir": self.cache_dir}

    def remaining(self):
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def sync(self):
        """Copy the result files of this root to the mirror. Run directories are mirrored as they finish."""
        if self.mirror is None:
            return
        try:
            for path in self.root.rglob("*.json"):
                relative = path.relative_to(self.root)
                # Run directories are copied as whole units by harness.run_training, never file by file.
                if relative.parts[0] in ("trees", "assets") or (path.parent / "run_meta.json").exists() \
                        or (path.parent / "status.json").exists():
                    continue
                target = self.mirror / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            for name in ("pytest.xml", "pytest-gpu.xml", "pytest.log"):
                if (self.root / name).exists():
                    shutil.copy2(self.root / name, self.mirror / name)
            if (self.root / "feedback").exists():
                shutil.copytree(self.root / "feedback", self.mirror / "feedback", dirs_exist_ok=True)
        except (OSError, shutil.Error) as error:
            say(f"WARNING: could not update the mirror {self.mirror}: {error!r}")

    def restore(self):
        """Bring back results of an earlier session from the mirror; files that exist locally are kept."""
        if self.mirror is None or not self.mirror.exists():
            return
        restored = 0
        try:
            incomplete = {meta.parent for meta in self.mirror.rglob("run_meta.json")
                          if not (meta.parent / harness.MIRROR_MARKER).exists()}
            for path in self.mirror.rglob("*"):
                relative = path.relative_to(self.mirror)
                if path.is_dir() or relative.parts[0] == "feedback" or path.name == harness.MIRROR_MARKER \
                        or any(parent in incomplete for parent in path.parents):
                    continue
                target = self.root / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    restored += 1
        except (OSError, shutil.Error) as error:
            say(f"WARNING: could not read the mirror {self.mirror}: {error!r}")
        if restored:
            say(f"restored {restored} files from {self.mirror}")


def run(ctx, fields, output, tree, module="ho_fl.train", revision=None, label=None, cache_tag=None):
    merged = {**fields, **ctx.common(fields.get("dataset", "sst2"))}
    if cache_tag:
        # Each source tree builds its own token and manifest caches, so a change in the data path cannot hide
        # behind files written by another revision.
        merged["cache_dir"] = f"{merged['cache_dir']}-{cache_tag}"
    mirror = ctx.mirror / Path(output).relative_to(ctx.root) if ctx.mirror else None
    return harness.run_training(merged, output, tree=tree, module=module, label=label, revision=revision,
                                mirror=mirror, echo=say, extra_meta={"hardware": ctx.hardware})


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
            "device": ctx.device, "hardware": ctx.hardware, "tiny": ctx.tiny}
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
    if "torch_error" in info:
        problems.append(f"the torch probe raised {info['torch_error']}")
    elif not ctx.tiny and info.get("cuda_available") and info.get("bf16_matmul_finite") is not True:
        problems.append("the bfloat16 probe on the CUDA device did not give finite values; choose another GPU type")
    for problem in problems:
        say(f"WARNING: {problem}")
    return not any(marker in p for p in problems for marker in ("revision", "CUDA", "torch probe"))


# --- tests ---------------------------------------------------------------------------------------------------------

def stage_tests(ctx):
    say("== tests ==")
    report = ctx.root / ("pytest-gpu.xml" if ctx.gpu_tests else "pytest.xml")
    report.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", f"--junitxml={report}"]
    if ctx.gpu_tests:
        command += ["-m", "gpu"]
    started = time.time()
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                            env={**harness.tree_environment(REPO), "HO_FL_CACHE_DIR": ctx.cache_dir})
    (ctx.root / "pytest.log").write_text(result.stdout + result.stderr)
    say("\n".join((result.stdout + result.stderr).splitlines()[-40:]))
    say(f"pytest exit code {result.returncode} in {time.time() - started:.0f}s")
    if not ctx.gpu_tests:
        # The result only counts for the revision it was obtained on.
        harness.write_json(ctx.root / "tests.json", {"head": ctx.head, "returncode": result.returncode, "recorded_at": now()})
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
    if (harness.read_json(ctx.root / "env.json") or {}).get("hardware") != ctx.hardware:
        stage_env(ctx)  # golden values are recorded together with the hardware they belong to
    trees = {name: export_tree(ctx, name) for name in ("baseline", "instrumented")}
    root = ctx.golden_root
    records = {}
    for name, fields in SMOKE.items():
        for repeat in ("a", "b"):
            records[f"instrumented-{name}-{repeat}"] = run(ctx, {**fields, "save_state_rounds": "3"}, root / f"instrumented-{name}-{repeat}",
                                                           trees["instrumented"], revision=REVISIONS["instrumented"], cache_tag="instrumented")
        records[f"baseline-{name}"] = run(ctx, fields, root / f"baseline-{name}", trees["baseline"], revision=REVISIONS["baseline"],
                                          cache_tag="baseline")
    records["instrumented-replay"] = run(ctx, {}, root / "instrumented-replay", trees["instrumented"], module="ho_fl.replay",
                                         revision=REVISIONS["instrumented"], cache_tag="instrumented")
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
    stamp = {"recorded_at": now(), "environment": {key: environment.get(key) for key in (
        "gpu_name", "driver_version", "cuda_runtime", "torch", "python", "platform", "packages")}}
    previous = harness.read_json(root / "golden.json") or {}
    if previous.get("environment") and all(record["reused"] for record in records.values()):
        # Nothing was run in this session: keep the time and environment the values were recorded with.
        stamp = {"recorded_at": previous["recorded_at"], "environment": previous["environment"]}
    reproducible = all(c["repeat"]["bitwise"] and c["repeat"]["hash_equal"] for c in configs.values())
    golden = {"schema": 1, "complete": not failed, "failed_runs": failed, "tiny": ctx.tiny, "hardware": ctx.hardware,
              "device": ctx.device, "revisions": REVISIONS, **stamp,
              "mode": "unknown" if failed else ("bitwise" if reproducible else "tolerance"),
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
        # Cohorts, the starting point and the number of logged rows cannot be noisy on any device.
        exact = comparison["shape_equal"] and comparison["selection_equal"] and comparison["same_start"]
        return "PASS_TOLERANCE" if exact else "FAIL"
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
    if ctx.head.endswith("-dirty"):
        say("WARNING: the working tree has uncommitted changes; these results cannot be attributed to a commit, and "
            "runs already stored for this revision are reused even if the code changed again.")
    statuses = {name: run(ctx, {**fields, "save_state_rounds": "3"}, root / f"head-{name}", REPO, revision=ctx.head, cache_tag="head")["status"]
                for name, fields in SMOKE.items()}
    replay = run(ctx, {}, root / "head-replay", REPO, module="ho_fl.replay", revision=ctx.head, cache_tag="head")
    i1 = {}
    for name in SMOKE:
        comparison = harness.compare_fingerprints(golden["configs"][name], harness.fingerprint(root / f"head-{name}"))
        comparison["max_abs_state_diff"] = harness.state_difference(ctx.golden_root / f"instrumented-{name}-a" / SMOKE_STATE,
                                                                    root / f"head-{name}" / SMOKE_STATE)
        comparison["verdict"] = i1_verdict(golden["configs"][name]["repeat"], comparison, statuses[name])
        i1[name] = comparison
    head_replay = replay_summary(root / "head-replay")
    unchanged = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", REVISIONS["baseline"], "--", *PROFILE_MEMORY_SOURCES]).returncode
    i2 = {"head": head_replay, "instrumented": golden["replay"],
          "profile_memory_sources_unchanged": {0: True, 1: False}.get(unchanged),
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
    say(f"== e0 (plan {ctx.plan}, {ctx.rounds} rounds, {ctx.hardware}) ==")
    specs = e0_specs(ctx.plan, ctx.rounds)
    root = ctx.e0_root
    timings = harness.read_json(ctx.timings_path, {})
    preflight_failures, queue = {}, []

    def preflight(spec):
        """Three rounds of a configuration: a smoke check and the timing the budget decisions are based on."""
        key = spec["timing_key"]
        if ctx.remaining() < 300:
            preflight_failures[key] = "less than five minutes of budget left"
            return
        fields = {k: v for k, v in spec["fields"].items() if k != "save_state_rounds"}
        record = run(ctx, {**fields, "rounds": 3, "evaluate_every": 3}, root / "preflight" / ctx.hardware / key.replace("/", "-"),
                     REPO, revision=ctx.head, label=f"preflight {key}")
        if record["status"] == "COMPLETED" and record.get("timing"):
            timings[key] = {**record["timing"], "source": "preflight"}
            harness.write_json(ctx.timings_path, timings)
            estimate = harness.estimate_seconds(timings[key], ctx.rounds, EVALUATE_EVERY)
            say(f"  {key}: {timings[key]['round_s']:.1f}s per round, {timings[key]['eval_s']:.1f}s per evaluation, "
                f"{timings[key]['startup_s']:.0f}s start-up -> about {estimate / 60:.0f} min for {ctx.rounds} rounds")
        else:
            preflight_failures[key] = record.get("error") or "preflight run failed"

    for spec in specs:
        key, output = spec["timing_key"], root / "runs" / spec["name"]
        entry = {"name": spec["name"], "group": spec["group"], "seed": spec["seed"], "output": str(output)}
        if harness.run_status(output) == "COMPLETED":
            record = run(ctx, spec["fields"], output, REPO, revision=ctx.head)
            entry.update(status="COMPLETED", reused=True, wall_s=record.get("wall_s"))
        else:
            # The preflight of a configuration runs right before its first queued run, so cheap configurations
            # deliver results before expensive ones are even probed.
            if key not in timings and key not in preflight_failures:
                preflight(spec)
            if key not in timings:
                entry.update(status="SKIPPED_PREFLIGHT", reason=str(preflight_failures.get(key, "no timing available")))
            else:
                estimate = SAFETY * harness.estimate_seconds(timings[key], ctx.rounds, EVALUATE_EVERY)
                entry["estimated_s"] = round(estimate)
                if estimate > ctx.remaining():
                    entry.update(status="SKIPPED_BUDGET", remaining_s=round(max(ctx.remaining(), 0)))
                    say(f"  [{spec['name']}] skipped: needs about {estimate / 60:.0f} min, {max(ctx.remaining(), 0) / 60:.0f} min of budget left")
                else:
                    record = run(ctx, spec["fields"], output, REPO, revision=ctx.head)
                    entry.update(status=record["status"], reused=False, wall_s=record.get("wall_s"), error=record.get("error"))
                    if record["status"] == "COMPLETED" and record.get("timing"):
                        timings[key] = {**record["timing"], "source": spec["name"]}
                        harness.write_json(ctx.timings_path, timings)
        queue.append(entry)
        harness.write_json(root / "queue.json", {"recorded_at": now(), "plan": ctx.plan, "rounds": ctx.rounds, "head": ctx.head,
                                                  "hardware": ctx.hardware, "preflight_failures": preflight_failures, "queue": queue})
        ctx.sync()
    say()
    say(table(["run", "status", "wall min", "estimate min"],
              [[e["name"], e["status"] + (" (reused)" if e.get("reused") else ""), fmt(e["wall_s"] / 60, 1) if e.get("wall_s") else "",
                fmt(e["estimated_s"] / 60, 1) if e.get("estimated_s") else ""] for e in queue]))
    pending = sum(e.get("estimated_s", 0) for e in queue if e["status"] == "SKIPPED_BUDGET") / 3600
    if pending > 20:
        say(f"WARNING: the runs of this plan that are still pending are estimated at {pending:.1f} GPU hours, above the 20 hours "
            "the brief asks to be reported before a grid is launched.")
    return all(e["status"] == "COMPLETED" for e in queue)


# --- feedback ------------------------------------------------------------------------------------------------------

def supply_from_ledger(path, config):
    """First-order supply per segment as recorded in a run's ledger.

    A report is first-order on the layers at or above its cut, so the first-order width of segment S_i (the layers
    from the i-th cut up to the next one) in a round is the number of selected clients whose cut is at most that cut.
    """
    rows = harness.read_jsonl(Path(path) / "ledger.jsonl")
    cuts = sorted(set(config.get("resource_cuts") or []))
    if not rows or len(cuts) != 3:
        return None
    cohorts = collections.defaultdict(list)
    for row in rows:
        cohorts[row["round"]].append(row)
    widths = {cut: [sum(1 for row in cohort if row["cut"] <= cut) for cohort in cohorts.values()] for cut in cuts}
    scarce = widths[cuts[0]]
    presence = collections.Counter(row["client"] for row in rows)
    sources = collections.Counter(row["client"] for row in rows if row["cut"] <= cuts[0])
    total = sum(sources.values())
    return {"rounds": len(cohorts), "mean_width": [statistics.mean(widths[cut]) for cut in cuts],
            "s1_width_zero": sum(w == 0 for w in scarce) / len(scarce), "s1_width_le_2": sum(w <= 2 for w in scarce) / len(scarce),
            "s1_top6_share": sum(count for _, count in sources.most_common(6)) / total if total else None,
            "s1_effective_sources": 1 / sum((count / total) ** 2 for count in sources.values()) if total else None,
            "max_presence": max(presence.values()) / len(cohorts)}


def e0_results(ctx):
    """Completed E0 runs of the plan (one per group and seed, by the directory names of the plan) and group statistics."""
    runs = []
    for spec in e0_specs("full", ctx.rounds):
        path = ctx.e0_root / "runs" / spec["name"]
        loaded = harness.load_run(path)
        if loaded["status"] != "COMPLETED" or "accuracy" not in loaded["summary"]:
            continue
        summary, config = loaded["summary"], loaded["config"]
        curve = [row for row in loaded["curves"] if "accuracy" in row]
        runs.append({"name": spec["name"], "group": spec["group"], "seed": spec["seed"], "rounds": ctx.rounds,
                     "accuracy": 100 * summary["accuracy"], "evaluation_loss": summary.get("task_loss"),
                     "uplink_gb": curve[-1]["cumulative_logical_uplink_bytes"] / 1e9 if curve else None,
                     "wall_s": loaded["meta"].get("wall_s"), "revision": loaded["meta"].get("revision"),
                     "hardware": loaded["meta"].get("hardware"), "final_parameter_sha256": summary.get("final_parameter_sha256"),
                     "bottom_dimensions": config.get("bottom_dimensions"),
                     "supply": supply_from_ledger(path, config) if spec["group"][3] == "hofl" else None,
                     "curve": [{"round": row["round"], "accuracy": 100 * row["accuracy"], "evaluation_loss": row["task_loss"],
                                "training_loss": row.get("training_task_loss")} for row in curve]})
    groups = []
    for key, (reference, reference_sd) in REFERENCE.items():
        members = sorted((r for r in runs if tuple(r["group"]) == key), key=lambda r: r["seed"])
        planned = PLANNED_SEEDS[key[0]]
        entry = {"group": list(key), "reference": reference, "reference_sd": reference_sd, "tolerance": max(1.0, 2 * reference_sd),
                 "planned_seeds": len(planned), "seeds": [r["seed"] for r in members], "accuracies": [r["accuracy"] for r in members],
                 "revisions": sorted({short(r["revision"] or "unknown") for r in members}),
                 "uplink_gb": statistics.mean(r["uplink_gb"] for r in members) if members and all(r["uplink_gb"] is not None for r in members) else None}
        if not members:
            entry["verdict"] = "NOT_RUN"
        else:
            entry["mean"] = statistics.mean(entry["accuracies"])
            entry["sd"] = statistics.stdev(entry["accuracies"]) if len(members) > 1 else None
            entry["difference"] = entry["mean"] - reference
            if ctx.rounds != DEFAULT_ROUNDS:
                entry["verdict"] = "NOT_COMPARABLE"
            else:
                # The acceptance rule of the brief is defined for the mean over all planned seeds.
                entry["verdict"] = ("WITHIN" if abs(entry["difference"]) <= entry["tolerance"] else "OUTSIDE") \
                    + ("" if set(entry["seeds"]) >= set(planned) else "_PRELIMINARY")
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
    tests, golden, regress, groups = data["tests"], data.get("golden"), data["regress"], data["e0"]["groups"]
    items = {}
    if tests is None:
        items["tests"] = "NOT_RUN"
    else:
        items["tests"] = "PASS" if tests["tests"] > tests["skipped"] and not tests["failures"] and not tests["errors"] else "FAIL"
    if regress is None:
        items["I1"] = items["I2"] = "NOT_RUN"
    else:
        verdicts = [c["verdict"] for c in regress["i1"].values()]
        items["I1"] = "PASS" if all(v.startswith("PASS") for v in verdicts) else ("FAIL" if "FAIL" in verdicts or "RUN_FAILED" in verdicts else "INCONCLUSIVE")
        # I1 is defined against the upstream revision: on a reproducible device its logs must equal the instrumented ones.
        if golden is not None and golden.get("mode") == "bitwise" and not all(
                all(c["upstream"]["files_identical"].values()) for c in golden["configs"].values()):
            items["I1"] = "FAIL"
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
    elif items["I1"] == "INCONCLUSIVE":
        items["G0"] = "NEEDS_DECISION"
    else:
        items["G0"] = "INCOMPLETE"
    return items


def collect(ctx):
    runs, groups = e0_results(ctx)
    queue = harness.read_json(ctx.e0_root / "queue.json")
    timings = harness.read_json(ctx.timings_path, {})
    rows = inventory(ctx)
    tested = harness.read_json(ctx.root / "tests.json") or {}
    golden = harness.read_json(ctx.golden_root / "golden.json")
    dimensions = {}
    for record in runs:
        if record["group"][3] == "hofl" and record["bottom_dimensions"]:
            dimensions.setdefault(record["group"][0], record["bottom_dimensions"])
    smoke = harness.read_json(ctx.golden_root / "instrumented-hofl-a" / "config.json") or {}
    if smoke.get("bottom_dimensions"):
        dimensions.setdefault("opt", smoke["bottom_dimensions"])
    data = {"generated_at": now(), "tiny": ctx.tiny, "head": ctx.head, "hardware": ctx.hardware, "revisions": REVISIONS,
            "environment": harness.read_json(ctx.root / "env.json"),
            # A test result only counts for the revision it was obtained on.
            "tests": parse_junit(ctx.root / "pytest.xml") if tested.get("head") == ctx.head else None,
            "golden": golden, "regress": harness.read_json(ctx.regress_root / "verdicts.json"),
            "bottom_dimensions": None if ctx.tiny else {model: {"measured": measured, "appendix_a": APPENDIX_A[model],
                                                                "match": measured == APPENDIX_A[model]}
                                                        for model, measured in dimensions.items()},
            "e0": {"queue": queue, "timings": timings, "runs": runs, "groups": groups,
                   "estimated_run_s": {key: harness.estimate_seconds(t, DEFAULT_ROUNDS, EVALUATE_EVERY) for key, t in timings.items()}},
            "runs": rows, "used_gpu_hours": sum(r["wall_s"] or 0 for r in rows) / 3600}
    done = {r["name"] for r in runs}
    pending = [s for s in e0_specs("full", DEFAULT_ROUNDS) if s["name"] not in done]
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
    add(f"\nGenerated {data['generated_at']} for revision `{short(data['head'])}` on `{data['hardware']}`.\n")
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
        add("Not run on this revision.")
    else:
        add(f"{tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped, {tests['time_s']} s.")
        for failure in tests["failed"]:
            add(f"- `{failure['test']}` ({failure['kind']}): {failure['message']}")

    add("\n## 4. Golden values and reproducibility of the reference revision\n")
    if golden is None:
        add("Not run on this hardware.")
    else:
        add(f"Recorded {golden['recorded_at']}. Mode: **{golden['mode']}** (bitwise: two runs of the instrumented revision are identical "
            f"on this device; tolerance: they are not; unknown: a reference run failed). Complete: {fmt(golden['complete'])}."
            + (f" Failed runs: {golden['failed_runs']}." if golden["failed_runs"] else ""))
        add("")
        add(table(["config", "repeat bitwise", "max abs state diff between repeats", "logs identical to upstream revision", "task_loss per round", "final accuracy %"],
                  [[name, fmt(c["repeat"]["bitwise"] and c["repeat"]["hash_equal"]), fmt(c["repeat"]["max_abs_state_diff"]),
                    fmt(all(c["upstream"]["files_identical"].values())), ", ".join(f"{v:.6f}" for v in c["task_loss"]),
                    f"{100 * c['evaluation'][-1]['accuracy']:.2f}" if c["evaluation"] and "accuracy" in c["evaluation"][-1] else "n/a"]
                   for name, c in golden["configs"].items()]))
        add(f"\nReplay on the instrumented revision: {golden['replay']}")

    add("\n## 5. Invariants I1 and I2 on the current tree\n")
    if regress is None:
        add("Not run for this revision on this hardware.")
    else:
        add(table(["config", "verdict", "losses equal", "evaluations equal", "final hash equal", "max abs state diff", "first difference"],
                  [[name, c["verdict"], fmt(c["rounds_equal"]), fmt(c["evaluation_equal"]), fmt(c["hash_equal"]), fmt(c["max_abs_state_diff"]),
                    (c["differences"] or [""])[0]] for name, c in regress["i1"].items()]))
        add(f"\nI2 replay on the current tree: **{regress['i2']['verdict']}** {regress['i2']['head']}")
        add("\nho-fl-profile and ho-fl-memory were not executed. Their sources and the modules they import "
            + {True: "are unchanged since the upstream revision.", False: "DIFFER from the upstream revision; run them before relying on I2."}.get(
                regress["i2"].get("profile_memory_sources_unchanged"), "could not be compared with the upstream revision."))
    if data["bottom_dimensions"]:
        add("\nZeroth-order dimension per cut (`bottom_dimensions` in config.json) against appendix A of the brief:\n")
        add(table(["model", "measured", "appendix A", "match"],
                  [[model, json.dumps(d["measured"]), json.dumps(d["appendix_a"]), fmt(d["match"])] for model, d in data["bottom_dimensions"].items()]))

    add("\n## 6. E0 reproduction\n")
    add("Reference values are those the brief quotes from the HO-FL paper. The acceptance rule is |mean over all planned seeds - reference| "
        "<= max(1.0, 2 x paper sd); verdicts ending in `_PRELIMINARY` are computed on fewer seeds and decide nothing.\n")
    add(table(["model", "dataset", "partition", "method", "beta", "seeds run", "accuracy % per seed", "mean", "sd", "reference (sd)", "difference",
               "tolerance", "verdict", "uplink GB", "code revisions"],
              [[*g["group"], f"{len(g['seeds'])}/{g['planned_seeds']}", ", ".join(f"{a:.2f}" for a in g["accuracies"]), fmt(g.get("mean"), 2),
                fmt(g.get("sd"), 2), f"{g['reference']:.2f} ({g['reference_sd']:.2f})", fmt(g.get("difference"), 2), fmt(g["tolerance"], 2),
                g["verdict"], fmt(g.get("uplink_gb"), 3), ", ".join(g["revisions"])] for g in e0["groups"]]))
    supplied = [r for r in e0["runs"] if r["supply"]]
    if supplied:
        add("\nFirst-order supply recorded in the ledgers of the HO-FL runs. S1, S2, S3 are the segments above the three cuts; "
            "the width of a segment in a round is the number of selected clients that report first-order updates for it.\n")
        add(table(["run", "mean width S1 / S2 / S3", "rounds with S1 width 0", "rounds with S1 width <= 2", "S1 share of top-6 clients",
                   "S1 effective sources", "max client presence"],
                  [[r["name"], " / ".join(f"{w:.2f}" for w in r["supply"]["mean_width"]), f"{100 * r['supply']['s1_width_zero']:.1f}%",
                    f"{100 * r['supply']['s1_width_le_2']:.1f}%", f"{100 * r['supply']['s1_top6_share']:.1f}%" if r["supply"]["s1_top6_share"] is not None else "n/a",
                    fmt(r["supply"]["s1_effective_sources"], 2), f"{100 * r['supply']['max_presence']:.1f}%"] for r in supplied]))
        add("\nExpected values from appendix A of the brief (expectations over rounds and seeds; a single run scatters around them): "
            + "; ".join(f"beta={beta:g}: {text}" for beta, text in SUPPLY_REFERENCE.items()) + ".")
    if e0["queue"]:
        queue = e0["queue"]
        add(f"\nLast E0 stage (recorded {queue['recorded_at']}, revision `{short(queue['head'])}`, plan `{queue['plan']}`, {queue['rounds']} rounds):\n")
        add(table(["run", "status", "wall min", "estimated min", "note"],
                  [[q["name"], q["status"] + (" (reused)" if q.get("reused") else ""), fmt(q["wall_s"] / 60, 1) if q.get("wall_s") else "",
                    fmt(q["estimated_s"] / 60, 1) if q.get("estimated_s") else "", q.get("reason") or q.get("error") or ""] for q in queue["queue"]]))
        if queue["preflight_failures"]:
            add("\nPreflight failures: " + json.dumps(queue["preflight_failures"]))
    for run_record in e0["runs"]:
        add(f"\nCurve of `{run_record['name']}` (round: accuracy %, evaluation loss, training loss):")
        add("; ".join(f"{p['round']}: {p['accuracy']:.2f}, {p['evaluation_loss']:.4f}, " + (f"{p['training_loss']:.4f}" if p["training_loss"] is not None else "n/a")
                      for p in run_record["curve"]))

    add("\n## 7. Timing and GPU hours\n")
    if e0["timings"]:
        add(f"Measured on `{data['hardware']}`.\n")
        add(table(["configuration", "s per round", "s per evaluation", "start-up s", "estimated min per 160-round run", "measured on"],
                  [[key, fmt(t["round_s"], 2), fmt(t["eval_s"], 2), fmt(t["startup_s"], 1), fmt(e0["estimated_run_s"][key] / 60, 1), t.get("source")]
                   for key, t in e0["timings"].items()]))
    remaining = e0["remaining_full_plan"]
    add(f"\nWall time of all runs recorded under this root (every attempt, every stage): {data['used_gpu_hours']:.2f} h. "
        f"Runs of the full E0 plan still missing: {remaining['runs']}, estimated {remaining['estimated_hours']:.1f} h on this hardware"
        + (f" (no timing yet for {remaining['without_timing']}, not included)." if remaining["without_timing"] else "."))

    add("\n## 8. Run inventory\n")
    add(table(["run", "status", "wall s", "revision"], [[r["run"], r["status"], fmt(r["wall_s"], 0) if r["wall_s"] else "", r["revision"]] for r in data["runs"]]))
    for failed in [r for r in data["runs"] if r["status"] != "COMPLETED"]:
        add(f"\n`{failed['run']}` did not complete ({failed['status']}): {failed['error']}\n\n```\n{failed['log_tail'] or ''}\n```")

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
        if golden["mode"] == "unknown":
            lines.append("The golden values are incomplete because a reference run failed (section 8); whether this device is reproducible is not known yet.")
        elif golden["mode"] == "bitwise":
            lines.append("The reference revision is bitwise reproducible on this device, so I1 is checked bit for bit.")
        else:
            worst = max((c["repeat"]["max_abs_state_diff"] or 0) for c in golden["configs"].values())
            lines.append(f"The reference revision is NOT bitwise reproducible on this device (largest difference between the final states of two identical runs: {worst:.3g}). "
                         f"I1 is then checked on the final state with an absolute tolerance of {harness.STATE_TOLERANCE:g}; report this in P0.md.")
        if golden["mode"] == "bitwise" and not all(all(c["upstream"]["files_identical"].values()) for c in golden["configs"].values()):
            lines.append("The logs of the upstream revision and of the instrumented revision are not byte-identical although the device is reproducible: "
                         "the instrumentation commit changed behaviour and must be inspected. I1 is marked FAIL for this reason.")
    if regress is not None:
        for name, comparison in regress["i1"].items():
            if comparison["verdict"] == "FAIL":
                lines.append(f"I1 FAIL for `{name}`: {(comparison['differences'] or ['no logged difference; the final states differ'])[0]}. "
                             "Compare train_client and ho_fl/aggregation.py with tests/legacy_loop.py; the first differing round tells which phase diverges.")
            elif comparison["verdict"] == "INCONCLUSIVE":
                noise = golden["configs"][name]["repeat"].get("max_abs_state_diff") if golden else None
                lines.append(f"I1 is inconclusive for `{name}`: two runs of the reference revision differ by {fmt(noise)} and the current tree differs from them by "
                             f"{fmt(comparison.get('max_abs_state_diff'))}, above the tolerance {harness.STATE_TOLERANCE:g}. This says nothing against the refactoring; "
                             "the device is not reproducible to that tolerance and the research lead has to decide how I1 is checked on it.")
            elif comparison["verdict"] == "RUN_FAILED":
                lines.append(f"The smoke run `{name}` did not complete on the current tree; its log is in section 8.")
        if regress["i2"]["verdict"] != "PASS":
            lines.append("ho-fl-replay fails on the current tree. Backend.local_step or hybrid_step no longer behaves as replay.py expects.")
    for group in e0["groups"]:
        name = "/".join(str(x) for x in group["group"])
        if group["verdict"] == "OUTSIDE":
            lines.append(f"E0 outside tolerance for {name}: mean {group['mean']:.2f} vs reference {group['reference']:.2f} over all {group['planned_seeds']} seeds "
                         f"(tolerance {group['tolerance']:.2f}). The brief asks to stop and report.")
        elif group["verdict"] == "OUTSIDE_PRELIMINARY":
            lines.append(f"E0 {name}: {len(group['seeds'])} of {group['planned_seeds']} seeds give {group['mean']:.2f} against {group['reference']:.2f}, "
                         f"{abs(group['difference']) / group['reference_sd']:.1f} paper standard deviations away. This is not a verdict; run the remaining seeds and look at the curve.")
    counts = collections.Counter(g["verdict"] for g in e0["groups"])
    if counts["WITHIN_PRELIMINARY"]:
        lines.append(f"{counts['WITHIN_PRELIMINARY']} E0 groups are within tolerance on fewer seeds than planned; the remaining seeds are needed to close E0.")
    if counts["NOT_COMPARABLE"]:
        lines.append(f"{counts['NOT_COMPARABLE']} E0 groups were run with a number of rounds other than {DEFAULT_ROUNDS} and are not compared with the reference.")
    missing = [g for g in e0["groups"] if g["verdict"] == "NOT_RUN"]
    if missing:
        lines.append(f"{len(missing)} E0 groups have no completed run yet: " + ", ".join("/".join(str(x) for x in g["group"]) for g in missing) + ".")
    by_key = {(tuple(r["group"]), r["seed"]): r for r in e0["runs"]}
    for (model, dataset, partition), seed in sorted({(tuple(r["group"][:3]), r["seed"]) for r in e0["runs"]}):
        biased, uniform, fedavg = (by_key.get(((model, dataset, partition, method, beta), seed))
                                   for method, beta in (("hofl", 0.75), ("hofl", 0.0), ("fedavg", 0.0)))
        parts = []
        if biased and uniform:
            paper = REFERENCE[(model, dataset, partition, "hofl", 0.75)][0] - REFERENCE[(model, dataset, partition, "hofl", 0.0)][0]
            parts.append(f"HO-FL beta 0.75 minus beta 0 is {biased['accuracy'] - uniform['accuracy']:+.2f} points (difference of the paper means: {paper:+.2f})")
        if fedavg and uniform:
            paper = REFERENCE[(model, dataset, partition, "fedavg", 0.0)][0] - REFERENCE[(model, dataset, partition, "hofl", 0.0)][0]
            parts.append(f"FedAvg minus HO-FL beta 0 is {fedavg['accuracy'] - uniform['accuracy']:+.2f} points (paper: {paper:+.2f})")
        if parts:
            lines.append(f"{model}/{dataset}, seed {seed}: " + "; ".join(parts) + ". These are single-seed paired differences, to be read with the seed-to-seed sd in mind.")
    if e0["queue"]:
        skipped = [q for q in e0["queue"]["queue"] if q["status"] == "SKIPPED_BUDGET"]
        if skipped:
            lines.append(f"{len(skipped)} runs did not fit in the time budget of the last E0 stage and were skipped; running the notebook again continues with them.")
        for item in [q for q in e0["queue"]["queue"] if q["status"] in ("FAILED", "SKIPPED_PREFLIGHT")]:
            lines.append(f"E0 run `{item['name']}` did not run to completion ({item['status']}): {item.get('reason') or item.get('error')}.")
    if data["bottom_dimensions"] and not all(d["match"] for d in data["bottom_dimensions"].values()):
        lines.append("The measured zeroth-order dimensions differ from appendix A of the brief (section 5); the code is authoritative, report the difference.")
    lines.append({"PASS": "Gate G0: all four criteria hold on this revision. Send this report back so that P0.md can be completed.",
                  "BLOCKED": "Gate G0 is blocked by the items marked FAIL or CHECK above. Send this report back before going on.",
                  "NEEDS_DECISION": "Gate G0 needs a decision: I1 cannot be checked to the agreed tolerance on this device. Send this report back.",
                  "INCOMPLETE": "Gate G0 cannot be decided yet: some stages have not run or E0 lacks seeds. Send this report back; it is enough to plan the next session.",
                  "DRY_RUN": "Gate G0 is not assessed in a dry run."}[data["checklist"]["G0"]])
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
