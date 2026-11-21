"""Queues of training runs under a wall-clock budget, for the experiment pipelines under ``experiments/``.

A queue is a list of specs in priority order. Before the queue starts, each cost profile (``timing_key``) is probed
with a three-round run that measures seconds per round and per evaluation and the peak GPU memory. A run is started
only if its estimated duration fits the remaining budget; runs that do not fit are skipped and reported. With
``jobs > 1`` several runs share the GPU: each is its own process, so its numerics are those of a run on its own.

Nothing persists between sessions here: a queue reuses finished runs under the same root, nothing else.
Imports neither torch nor transformers.
"""
import shutil
import statistics
import subprocess
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from . import harness

SAFETY = 1.25            # estimated durations are multiplied by this before they are compared with the budget
GRACE_S = 900            # a run that is still going this long after the deadline is terminated
MEMORY_HEADROOM = 0.85   # share of the GPU memory that concurrent runs may fill, by their measured peaks


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(text=""):
    print(text, flush=True)


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


def git_head(repo):
    """Commit of ``repo``'s working tree, with ``-dirty`` when tracked files are modified."""
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if sha is None:
        return "unknown"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def hardware_slug(device):
    """Names the device and torch version, e.g. ``tesla-t4-torch2.6.0``; timings and golden values are valid per slug."""
    import re

    import torch

    name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + "-torch" + torch.__version__.split("+")[0]


def regression_verdict(parent, head, hardware):
    """Verdicts of the P0 regression stage for this revision and hardware, if it was run under a sibling ``p0*`` root."""
    for candidate in sorted(Path(parent).glob("p0*/regress/*/verdicts.json")):
        verdicts = harness.read_json(candidate) or {}
        if verdicts.get("head") == head and hardware in str(candidate):
            return {"i1": {name: entry["verdict"] for name, entry in verdicts["i1"].items()}, "i2": verdicts["i2"]["verdict"],
                    "golden_mode": verdicts.get("golden_mode"), "path": str(candidate)}
    return None


def gpu_memory():
    """``(used, total)`` memory of the first GPU in MiB, or ``None`` when ``nvidia-smi`` is not available."""
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True)
    try:
        used, total = (float(part) for part in result.stdout.strip().splitlines()[0].split(","))
        return used, total
    except (IndexError, ValueError):
        return None


class MemoryWatch:
    """Peak GPU memory in use while a run executes, above what was in use before it started."""

    def __init__(self, interval=2.0):
        self.interval, self.peak, self.baseline = interval, None, None
        self.stopped = threading.Event()
        self.thread = None

    def _loop(self):
        while not self.stopped.wait(self.interval):
            reading = gpu_memory()
            if reading is not None:
                self.peak = max(self.peak or 0.0, reading[0])

    def start(self):
        reading = gpu_memory()
        if reading is not None:
            self.baseline = reading[0]
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def stop(self):
        """Stop watching and return the peak above the baseline in MiB (``None`` without a GPU)."""
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        if self.peak is None or self.baseline is None:
            return None
        return max(0.0, self.peak - self.baseline)


def preflight_fields(fields):
    """The three-round version of a run: same code path, including one probe round when the run has probes."""
    result = {key: value for key, value in fields.items() if key != "save_state_rounds"}
    result.update(rounds=3, evaluate_every=3)
    if result.get("probe_rounds"):
        result["probe_rounds"] = "1"
    return result


def default_evaluate_every(fields):
    return fields.get("evaluate_every") or max(1, 50 // fields.get("E", 5))


def allowed_jobs(jobs, timings, keys, memory=None):
    """How many runs may share the GPU: the requested number, limited by the measured peak memory of one run."""
    memory = gpu_memory() if memory is None else memory
    peaks = [timings[key].get("peak_mb") for key in keys if key in timings]
    if jobs <= 1 or memory is None or not peaks:
        return max(1, jobs)
    if any(peak is None for peak in peaks):
        return 1
    return max(1, min(jobs, int(MEMORY_HEADROOM * memory[1] // max(max(peaks), 1.0))))


def run_queue(specs, *, root, tree, hardware, revision, common, deadline=None, jobs=1, echo=say, run=harness.run_training):
    """Run ``specs`` in order under ``root`` and return the record of the queue (also written to ``queue.json``).

    A spec is ``{"name", "fields", "timing_key", ...}``; other keys are copied to its entry. ``common(fields)``
    returns the fields added to every run (device, cache directory). ``deadline`` is a ``time.time()`` value.
    Entry statuses: ``COMPLETED``, ``DIVERGED``, ``FAILED``, ``TIMEOUT``, ``SKIPPED_BUDGET``, ``SKIPPED_PREFLIGHT``.
    """
    root = Path(root)
    timings_path = root / f"timings-{hardware}.json"
    timings = harness.read_json(timings_path, {})
    failures = {}

    def remaining():
        return float("inf") if deadline is None else deadline - time.time()

    def start(fields, output, label, stop_at=None):
        return run({**fields, **common(fields)}, output, tree=tree, label=label, revision=revision, echo=echo,
                   extra_meta={"hardware": hardware}, stop_at=stop_at)

    # Phase 1: probe every cost profile that still has runs to do, one at a time, alone on the GPU.
    for spec in specs:
        key = spec["timing_key"]
        if key in timings or key in failures or harness.run_status(root / "runs" / spec["name"]) in harness.FINISHED:
            continue
        if remaining() < 300:
            failures[key] = "less than five minutes of budget left"
            continue
        watch = MemoryWatch().start()
        record = start(preflight_fields(spec["fields"]), root / "preflight" / hardware / key, f"preflight {key}")
        peak = watch.stop()
        if record["status"] == "COMPLETED" and record.get("timing"):
            timings[key] = {**record["timing"], "peak_mb": peak, "source": "preflight"}
            harness.write_json(timings_path, timings)
            echo(f"  {key}: {timings[key]['round_s']:.1f}s per round, {timings[key]['eval_s']:.1f}s per evaluation, "
                 f"{timings[key]['startup_s']:.0f}s start-up" + (f", peak GPU memory {peak / 1024:.1f} GiB" if peak is not None else ""))
        else:
            failures[key] = record.get("error") or "preflight run failed"

    workers = allowed_jobs(jobs, timings, {spec["timing_key"] for spec in specs})
    if workers != jobs:
        echo(f"  running {workers} job(s) at a time instead of {jobs}: limited by the measured GPU memory per run")
    observed = []                   # wall time over solo estimate of the runs finished while sharing the GPU
    start_lock = threading.Lock()   # data preparation of two runs never overlaps (they share cache files)
    entries, record_path = [], root / "queue.json"

    def solo_estimate(spec):
        fields = spec["fields"]
        return harness.estimate_seconds(timings[spec["timing_key"]], fields["rounds"], default_evaluate_every(fields),
                                        initial=fields.get("eval_initial", True))

    def work(spec, stop_at):
        output = root / "runs" / spec["name"]
        finished = threading.Event()

        def release_when_prepared():
            # config.json appears once the model, the token cache and the manifest are loaded.
            while not finished.is_set() and not (output / "config.json").exists():
                time.sleep(0.5)
            start_lock.release()

        start_lock.acquire()
        threading.Thread(target=release_when_prepared, daemon=True).start()
        try:
            return start(spec["fields"], output, spec["name"], stop_at)
        finally:
            finished.set()

    def write_record():
        harness.write_json(record_path, {"recorded_at": now(), "hardware": hardware, "head": revision, "jobs_requested": jobs,
                                         "jobs": workers, "preflight_failures": failures, "queue": entries})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        running, position = {}, 0
        while position < len(specs) or running:
            while position < len(specs) and len(running) < workers:
                spec = specs[position]
                position += 1
                entry = {key: value for key, value in spec.items() if key != "fields"}
                entry["output"] = str(root / "runs" / spec["name"])
                entries.append(entry)
                if harness.run_status(entry["output"]) in harness.FINISHED:
                    running[pool.submit(work, spec, None)] = (spec, entry, None)
                elif spec["timing_key"] not in timings:
                    entry.update(status="SKIPPED_PREFLIGHT", reason=str(failures.get(spec["timing_key"], "no timing available")))
                else:
                    solo = solo_estimate(spec)
                    # Sharing the GPU slows every run down: assume the worst (no gain at all) until it is measured.
                    factor = 1.0 if workers == 1 else (max(1.0, statistics.mean(observed)) if observed else float(workers))
                    estimate = SAFETY * solo * factor
                    entry["estimated_s"] = round(estimate)
                    if estimate > remaining():
                        entry.update(status="SKIPPED_BUDGET", remaining_s=round(max(remaining(), 0)))
                        echo(f"  [{spec['name']}] skipped: needs about {estimate / 60:.0f} min, {max(remaining(), 0) / 60:.0f} min of budget left")
                    else:
                        stop_at = None if deadline is None else deadline + GRACE_S
                        running[pool.submit(work, spec, stop_at)] = (spec, entry, solo)
            if not running:
                continue
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                spec, entry, solo = running.pop(future)
                try:
                    record = future.result()
                except Exception as error:  # a broken launch must not take the queue down
                    record = {"status": "FAILED", "error": repr(error), "reused": False, "wall_s": None}
                entry.update(status=record["status"], reused=bool(record.get("reused")), wall_s=record.get("wall_s"), error=record.get("error"))
                if record["status"] == "COMPLETED" and not record.get("reused") and record.get("timing"):
                    key = spec["timing_key"]
                    if workers == 1:
                        timings[key] = {**record["timing"], "peak_mb": timings[key].get("peak_mb"), "source": spec["name"]}
                    else:
                        timings[key]["shared"] = {**record["timing"], "jobs": workers, "source": spec["name"]}
                        if solo and record.get("wall_s"):
                            observed.append(record["wall_s"] / solo)
                    harness.write_json(timings_path, timings)
            write_record()
    write_record()
    return harness.read_json(record_path)


def write_bundle(root, path, skip_directories=("trees", "assets", "hf-cache"), skip_suffixes=(".safetensors", ".gz", ".zip", ".tmp")):
    """Zip the logs and results under ``root`` (no model states, no caches) into ``path``."""
    root, path = Path(root), Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(root.rglob("*")):
            relative = file.relative_to(root)
            if file.is_file() and file != path and relative.parts[0] not in skip_directories and file.suffix not in skip_suffixes:
                bundle.write(file, str(relative))
    return path
