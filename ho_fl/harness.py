"""Run management for experiments: launching runs with live progress, reading and comparing run directories.

Used by the pipelines under ``experiments/`` and by the GPU regression test. Nothing here changes how training
computes: a run is always ``python -m ho_fl.train`` (or another module of the package) in a subprocess, which is
what the command-line entry points do. This module imports neither torch nor transformers at import time.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import sha256

LOG_FILES = ("rounds.jsonl", "curves.jsonl", "ledger.jsonl")
FINGERPRINT_KEYS = ("task_loss", "selected_clients", "evaluation", "final_parameter_sha256",
                    "initial_parameter_sha256", "partition_id", "file_sha256")
# Largest absolute difference between final states accepted for invariant I1 when a device is not bitwise
# reproducible. It is the absolute part of replay.py's tolerance (replay.py also allows a relative 1e-5).
STATE_TOLERANCE = 1e-7
# Written last into a mirrored run directory; a mirror copy without it is incomplete and is never restored.
MIRROR_MARKER = ".mirror-complete"


# --- reading run directories ---------------------------------------------------------------------------------------

def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def read_jsonl(path):
    """Rows of a JSON-lines file; a trailing line that is still being written is ignored."""
    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run_status(path, module="ho_fl.train"):
    """``COMPLETED``, ``FAILED``, ``RUNNING`` or ``MISSING``. Replay has no status.json; its summary decides."""
    path = Path(path)
    if module == "ho_fl.replay":
        summary = read_json(path / "summary.json")
        if summary is None:
            return "MISSING" if not path.exists() or not any(path.iterdir()) else "FAILED"
        return "COMPLETED" if summary.get("status") == "PASSED" else "FAILED"
    return (read_json(path / "status.json") or {}).get("status", "MISSING")


def load_run(path):
    path = Path(path)
    status = read_json(path / "status.json") or {}
    return {"path": str(path), "status": status.get("status", "MISSING"), "error": status.get("error"),
            "last_completed_round": status.get("last_completed_round"),
            "config": read_json(path / "config.json") or {}, "summary": read_json(path / "summary.json") or {},
            "rounds": read_jsonl(path / "rounds.jsonl"), "curves": read_jsonl(path / "curves.jsonl"),
            "meta": read_json(path / "run_meta.json") or {}}


def fingerprint(path):
    """What two runs of the same configuration must share: losses, cohorts, evaluations and parameter hashes."""
    path = Path(path)
    run = load_run(path)
    return {"status": run["status"],
            "task_loss": [row["task_loss"] for row in run["rounds"]],
            "selected_clients": [row["selected_clients"] for row in run["rounds"]],
            "evaluation": run["curves"],
            "final_parameter_sha256": run["summary"].get("final_parameter_sha256"),
            "initial_parameter_sha256": run["summary"].get("initial_parameter_sha256"),
            "partition_id": run["summary"].get("partition_id"),
            "file_sha256": {name: sha256(path / name) if (path / name).exists() else None for name in LOG_FILES}}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_fingerprints(reference, candidate):
    """Compare a candidate fingerprint with a reference one.

    Evaluation rows are compared on the keys of the reference rows, so a candidate whose logs gained keys still
    matches. ``hash_equal`` is ``None`` when the reference has no final parameter hash (the upstream revision).
    ``bitwise`` means that everything that could be compared is identical.
    """
    differences = []
    loss_diff = eval_diff = 0.0
    rounds_equal = len(reference["task_loss"]) == len(candidate["task_loss"]) > 0
    if not rounds_equal:
        differences.append(f"number of rounds: {len(reference['task_loss'])} vs {len(candidate['task_loss'])}")
    for index, (a, b) in enumerate(zip(reference["task_loss"], candidate["task_loss"]), 1):
        if a != b:
            rounds_equal = False
            loss_diff = max(loss_diff, abs(a - b))
            differences.append(f"round {index}: task_loss {a!r} vs {b!r}")
    selection_equal = reference["selected_clients"] == candidate["selected_clients"]
    if not selection_equal:
        differences.append("selected_clients differ")
    evaluation_equal = len(reference["evaluation"]) == len(candidate["evaluation"])
    if not evaluation_equal:
        differences.append(f"number of evaluation rows: {len(reference['evaluation'])} vs {len(candidate['evaluation'])}")
    for a, b in zip(reference["evaluation"], candidate["evaluation"]):
        for key, value in a.items():
            other = b.get(key)
            if other != value:
                evaluation_equal = False
                if _number(value) and _number(other):
                    eval_diff = max(eval_diff, abs(value - other))
                differences.append(f"evaluation at round {a.get('round')}: {key} {value!r} vs {other!r}")
    for key in ("initial_parameter_sha256", "partition_id"):
        if reference.get(key) != candidate.get(key):
            differences.append(f"{key}: {reference.get(key)} vs {candidate.get(key)}")
    same_start = not any(d.startswith(("initial_parameter_sha256", "partition_id")) for d in differences)
    hash_equal = None
    if reference.get("final_parameter_sha256") is not None:
        hash_equal = reference["final_parameter_sha256"] == candidate.get("final_parameter_sha256")
        if not hash_equal:
            differences.append(f"final_parameter_sha256: {reference['final_parameter_sha256']} vs {candidate.get('final_parameter_sha256')}")
    reference_files, candidate_files = reference.get("file_sha256") or {}, candidate.get("file_sha256") or {}
    files_identical = {name: reference_files.get(name) is not None and reference_files.get(name) == candidate_files.get(name)
                       for name in LOG_FILES}
    shape_equal = (len(reference["task_loss"]) == len(candidate["task_loss"]) > 0
                   and len(reference["evaluation"]) == len(candidate["evaluation"]))
    return {"rounds_equal": rounds_equal, "selection_equal": selection_equal, "evaluation_equal": evaluation_equal,
            "shape_equal": shape_equal,
            "same_start": same_start, "hash_equal": hash_equal, "files_identical": files_identical,
            "max_abs_task_loss_diff": loss_diff, "max_abs_evaluation_diff": eval_diff,
            "bitwise": bool(rounds_equal and selection_equal and evaluation_equal and same_start and hash_equal is not False),
            "differences": differences[:8], "difference_count": len(differences)}


def state_difference(path_a, path_b):
    """Largest absolute difference between two saved states; ``None`` when either file is missing."""
    path_a, path_b = Path(path_a), Path(path_b)
    if not (path_a.exists() and path_b.exists()):
        return None
    from safetensors.torch import load_file

    a, b = load_file(str(path_a)), load_file(str(path_b))
    if set(a) != set(b):
        raise ValueError(f"{path_a} and {path_b} hold different tensors")
    return max(float((a[name].double() - b[name].double()).abs().max()) for name in a)


# --- timing --------------------------------------------------------------------------------------------------------

def timing(progress):
    """Seconds per round, per evaluation and for start-up, from the progress times recorded by ``launch``.

    Within a run the order of events is: start-up, evaluation of round 0, then for each round the local training
    (its row appears in rounds.jsonl) followed, on evaluation rounds, by the evaluation (its row in curves.jsonl).
    Returns ``None`` when the run did not get far enough.
    """
    rounds = sorted((int(r), t) for r, t in progress.get("round_times", {}).items())
    evals = {int(r): t for r, t in progress.get("eval_times", {}).items()}
    if not rounds or 0 not in evals:
        return None
    round_durations, eval_durations = [], []
    previous = evals[0]
    for round_id, at in rounds:
        round_durations.append(at - previous)
        previous = at
        if round_id in evals:
            eval_durations.append(evals[round_id] - at)
            previous = evals[round_id]
    eval_s = statistics.median(eval_durations) if eval_durations else 0.0
    return {"round_s": statistics.median(round_durations), "eval_s": eval_s,
            "startup_s": max(0.0, evals[0] - eval_s), "rounds_measured": len(round_durations),
            "evaluations_measured": len(eval_durations)}


def evaluation_count(rounds, evaluate_every):
    """Number of evaluations of a run: round 0, every ``evaluate_every`` rounds and the last round."""
    return 1 + sum(1 for r in range(1, rounds + 1) if r % evaluate_every == 0 or r == rounds)


def estimate_seconds(timing_record, rounds, evaluate_every):
    return (timing_record["startup_s"] + timing_record["eval_s"] * evaluation_count(rounds, evaluate_every)
            + timing_record["round_s"] * rounds)


# --- launching -----------------------------------------------------------------------------------------------------

def cli(fields):
    """Config fields as command-line arguments of the entry points (booleans become ``--flag`` / ``--no-flag``)."""
    arguments = []
    for key, value in fields.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(flag if value else "--no-" + flag[2:])
        else:
            arguments += [flag, str(value)]
    return arguments


def tree_environment(tree):
    """Environment in which ``python -m ho_fl.<module>`` imports the package from ``tree``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(tree).resolve())] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return env


def _describe(row):
    parts = [f"{key}={row[key]:.4f}" for key in ("accuracy", "exact_match", "f1", "task_loss") if _number(row.get(key))]
    return " ".join(parts)


def launch(command, output, log_path, cwd=None, env=None, label="", progress_every=10, poll=0.2, echo=print, stop_at=None):
    """Run ``command`` and follow ``<output>/rounds.jsonl`` and ``curves.jsonl`` while it runs.

    Standard output and error of the command go to ``log_path``. Returns the exit code, the wall time and the
    times (seconds since launch) at which each round and each evaluation appeared in the logs. ``stop_at`` is an
    optional ``time.time()`` value after which the command is terminated (``timed_out`` is then true).
    """
    output, log_path = Path(output), Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    round_times, eval_times = {}, {}
    offsets = {"rounds.jsonl": 0, "curves.jsonl": 0}
    last_echo = [started]

    def drain():
        now = time.monotonic()
        elapsed = now - started
        for name in offsets:
            path = output / name
            if not path.exists():
                continue
            with path.open("rb") as stream:
                stream.seek(offsets[name])
                data = stream.read()
            complete = data.rfind(b"\n") + 1
            offsets[name] += complete
            for line in data[:complete].splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if name == "rounds.jsonl":
                    round_times[str(row["round"])] = elapsed
                    if row["round"] == 1 or row["round"] % progress_every == 0 or now - last_echo[0] > 120:
                        last_echo[0] = now
                        echo(f"    [{label}] round {row['round']} train_loss={row['task_loss']:.4f} t={elapsed:.0f}s")
                else:
                    eval_times[str(row["round"])] = elapsed
                    last_echo[0] = now
                    echo(f"    [{label}] eval @ round {row['round']}: {_describe(row)} t={elapsed:.0f}s")

    def terminate(process):
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    timed_out = False
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                time.sleep(poll)
                drain()
                if stop_at is not None and time.time() > stop_at:
                    timed_out = True
                    terminate(process)
        except BaseException:
            terminate(process)
            raise
        drain()
    return {"returncode": process.returncode, "wall_s": time.monotonic() - started, "round_times": round_times,
            "eval_times": eval_times, "timed_out": timed_out}


def tail(path, lines=40):
    path = Path(path)
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def mirror_run(output, mirror, echo=print):
    """Copy a finished run directory and its log to ``mirror``. Errors are reported, never raised.

    The marker file is removed first and written last, so an interrupted copy is recognisable.
    """
    output, mirror = Path(output), Path(mirror)
    try:
        (mirror / MIRROR_MARKER).unlink(missing_ok=True)
        shutil.copytree(output, mirror, dirs_exist_ok=True)
        log = output.with_name(output.name + ".log")
        if log.exists():
            shutil.copy2(log, mirror.with_name(mirror.name + ".log"))
        (mirror / MIRROR_MARKER).write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")
        return True
    except (OSError, shutil.Error) as error:
        echo(f"  WARNING: could not mirror {output.name} to {mirror}: {error!r}")
        return False


def _set_aside(output):
    """Move an unfinished attempt out of the way; it is kept because failed runs are part of the record."""
    if not (output / "run_meta.json").exists():
        status = read_json(output / "status.json") or {}
        write_json(output / "run_meta.json", {"label": output.name, "status": "INTERRUPTED",
                                              "last_completed_round": status.get("last_completed_round")})
    index = 1
    while output.with_name(f"{output.name}.stale-{index}").exists():
        index += 1
    target = output.with_name(f"{output.name}.stale-{index}")
    output.rename(target)
    log = output.with_name(output.name + ".log")
    if log.exists():
        log.rename(target.with_name(target.name + ".log"))
    return target


def run_training(fields, output, tree=".", module="ho_fl.train", label=None, revision=None, mirror=None,
                 python=None, progress_every=10, echo=print, extra_meta=None, stop_at=None):
    """Run one entry point of the package in a subprocess of ``tree``, unless ``output`` holds a finished run.

    ``fields`` are config fields; ``output`` is added to them. An unfinished or failed earlier attempt is moved
    aside to ``<output>.stale-N``, because the entry points refuse to write into a non-empty directory. After the
    run, ``<output>/run_meta.json`` records the command, the code revision, the timing and, on failure, the end of
    the log; ``extra_meta`` is merged into it. ``mirror`` is an optional directory that receives a copy of the
    finished run. ``stop_at`` is an optional ``time.time()`` value after which the run is terminated. Returns a
    record with ``status`` (``COMPLETED``, ``FAILED`` or ``TIMEOUT``), ``reused`` and the timing.
    """
    output = Path(output).resolve()
    label = label or output.name
    log_path = output.with_name(output.name + ".log")
    status = run_status(output, module)
    if status == "COMPLETED":
        meta = read_json(output / "run_meta.json") or {}
        echo(f"  [{label}] already completed, reusing {output}")
        if mirror is not None and not (Path(mirror) / MIRROR_MARKER).exists():
            mirror_run(output, mirror, echo)
        return {"label": label, "output": str(output), "status": "COMPLETED", "reused": True,
                "wall_s": meta.get("wall_s"), "timing": meta.get("timing"), "revision": meta.get("revision")}
    if output.exists() and any(output.iterdir()):
        stale = _set_aside(output)
        echo(f"  [{label}] unfinished attempt ({status}) moved to {stale.name}")
        if mirror is not None:
            mirror_run(stale, Path(mirror).with_name(stale.name), echo)
    command = [python or sys.executable, "-m", module, *cli({**fields, "output": str(output)})]
    echo(f"  [{label}] starting: {' '.join(command[2:])}")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress = launch(command, output, log_path, cwd=str(Path(tree).resolve()), env=tree_environment(tree),
                      label=label, progress_every=progress_every, echo=echo, stop_at=stop_at)
    status = run_status(output, module)
    if status != "COMPLETED":
        status = "TIMEOUT" if progress["timed_out"] else "FAILED"
    meta = {**(extra_meta or {}), "label": label, "module": module, "command": command, "tree": str(Path(tree).resolve()),
            "revision": revision, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "returncode": progress["returncode"], "wall_s": progress["wall_s"], "status": status,
            "round_times": progress["round_times"], "eval_times": progress["eval_times"],
            "timing": timing(progress), "log": str(log_path)}
    if status != "COMPLETED":
        meta["log_tail"] = tail(log_path)
        meta["error"] = (read_json(output / "status.json") or {}).get("error")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_meta.json", meta)
    if mirror is not None:
        mirror_run(output, mirror, echo)
    if status == "COMPLETED":
        echo(f"  [{label}] completed in {progress['wall_s']:.0f}s")
    else:
        echo(f"  [{label}] {status} (exit code {progress['returncode']}); end of {log_path}:")
        echo("\n".join("      " + line for line in meta["log_tail"].splitlines()[-25:]))
    return {"label": label, "output": str(output), "status": status, "reused": False, "wall_s": progress["wall_s"],
            "timing": meta["timing"], "revision": revision, "returncode": progress["returncode"],
            "log_tail": meta.get("log_tail"), "error": meta.get("error")}
