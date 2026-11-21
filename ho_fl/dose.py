"""Dose-response experiment E2 (brief section 6.10): damage of one unreliable first-order source as a function of
how many honest first-order reports share its layers.

Everything except the dose is held fixed. Each round the cohort is a designated T1 device ``f`` plus five devices
drawn once from the other 29; all arms use the same cohorts. The dose ``m`` is set by cut overrides: the first
``m`` of the five train like T1 devices (first-order on the scarce segment S1), the others like T2 devices
(zeroth-order on S1, first-order above it). So S1 receives ``m + 1`` first-order reports, one of them from ``f``,
while S2 and S3 always receive six. ``f`` sends its S1 update negated (``flip1``) or negated and scaled by 20
(``flip20``); every arm has a paired run without the fault. All arms start from the same saved state.

The cut overrides are a device of the experiment and say nothing about real hardware.

Aggregation arms. ``quorum_robust`` is not run: with these cohorts it coincides exactly with ``fo_robust`` when
``m + 1`` reaches the quorum of 3 and with ``robust_all`` below it (S2 and S3 hold six first-order reports, so
"first-order only" and "all" are the same electorate there). ``quorum_hold`` and ``quorum_dilute`` switch the
zeroth-order updates off and, below the quorum, leave S1 unchanged or apply the first-order sum over K: the first
session showed zeroth-order reports to be noise, so falling back to "all reports" falls back to noise.
"""
import json
import statistics
from pathlib import Path

import numpy as np

from . import harness
from .faults import role_members

QUORUM = 3
_ROBUST_FO = {"agg_electorate": "fo", "agg_stat": "geomed", "fo_quorum": QUORUM, "zo_mode": "off"}
AGGREGATIONS = {
    "mean_all": {},
    "fo_mean": {"agg_electorate": "fo", "agg_weight": "count"},
    "robust_all": {"agg_stat": "geomed"},
    "fo_robust": {"agg_electorate": "fo", "agg_stat": "geomed"},
    "quorum_hold": {**_ROBUST_FO, "thin_policy": "hold"},
    "quorum_dilute": {**_ROBUST_FO, "thin_policy": "dilute"},
}
TIMING_KEYS = {"mean_all": "dose", "fo_mean": "dose", "robust_all": "dose-robust", "fo_robust": "dose-robust",
               "quorum_hold": "dose-robust-zo-off", "quorum_dilute": "dose-robust-zo-off"}
FAULTS = {"none": None, "flip20": 20.0, "flip1": 1.0}
M_ORDER = (2, 1, 3, 0, 4, 5)   # the predicted crossover first


def write_inputs(directory, resource_cuts, k, seed, rounds):
    """Write the schedule file of every dose and the two fault files; return what the arms need to refer to them."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    strong, medium = sorted(set(resource_cuts))[:2]
    device = role_members("preferred_t1", resource_cuts, k, seed)[0]
    others = [cid for cid in range(len(resource_cuts)) if cid != device]
    rng = np.random.default_rng(70000 + seed)
    cohorts = [[device] + [int(cid) for cid in rng.choice(others, size=k - 1, replace=False)] for _ in range(rounds)]
    schedules = {}
    for m in range(k):
        path = directory / f"schedule_m{m}.jsonl"
        path.write_text("".join(json.dumps({"round": index, "clients": cohort, "cut_override": {
            str(cid): strong if position < m else medium for position, cid in enumerate(cohort[1:])}}) + "\n" for index, cohort in enumerate(cohorts, 1)))
        schedules[m] = str(path)
    faults = {}
    for name, scale in FAULTS.items():
        if scale is None:
            faults[name] = None
            continue
        path = directory / f"fault_{name}.json"
        path.write_text(json.dumps({"faults": [{"select": {"by_id": [device]}, "rounds": [1, rounds], "kind": "sign_flip",
                                                "params": {"scale": scale, "layers": "S1"}}]}) + "\n")
        faults[name] = str(path)
    record = {"device": device, "cuts": [strong, medium], "rounds": rounds, "seed": seed, "cohorts": cohorts, "schedules": schedules, "faults": faults}
    harness.write_json(directory / "inputs.json", record)
    return record


def arm_name(m, aggregation, fault):
    return f"m{m}-{aggregation}-{fault}"


def specs(model, seed, inputs, init_state, base, doses=M_ORDER, faults=tuple(FAULTS)):
    """Runs of the experiment in queue order: dose by dose (the crossover first), each arm with its paired clean run."""
    result = []
    for m in doses:
        for aggregation, fields in AGGREGATIONS.items():
            for fault in faults:
                run = {**base, **fields, "model": model, "seed": seed, "method": "hofl", "sampler": "schedule_file",
                       "schedule_file": inputs["schedules"][m], "init_state": init_state, "rounds": inputs["rounds"],
                       "evaluate_every": inputs["rounds"], "on_nonfinite": "stop"}
                if inputs["faults"][fault]:
                    run["fault_spec"] = inputs["faults"][fault]
                result.append({"name": f"{model}-s{seed}-dose-{arm_name(m, aggregation, fault)}", "m": m, "aggregation": aggregation, "fault": fault,
                               "seed": seed, "timing_key": f"{model}/{TIMING_KEYS[aggregation]}", "fields": run})
    return result


def read_arm(run_dir):
    """Outcome of one arm: final test loss and accuracy, whether it diverged, mean relative influence on S1."""
    run = harness.load_run(run_dir)
    if run["status"] not in harness.FINISHED or "accuracy" not in run["summary"]:
        return None
    influence = [row["relative"] for row in harness.read_jsonl(Path(run_dir) / "influence.jsonl") if row["segment"] == 1]
    diverged = run["status"] == "DIVERGED"
    return {"status": run["status"], "diverged_round": run["summary"].get("diverged_round"),
            "loss": None if diverged else run["summary"]["task_loss"], "accuracy_pct": None if diverged else 100 * run["summary"]["accuracy"],
            "influence_s1": statistics.mean(influence) if influence else None, "wall_s": run["meta"].get("wall_s")}


def collect(runs_dir, model, seed, doses=range(6)):
    """``{(m, aggregation, fault): outcome}`` of the finished arms, with ``quorum_robust`` filled in from the arm it equals."""
    arms = {}
    for m in doses:
        for aggregation in AGGREGATIONS:
            for fault in FAULTS:
                outcome = read_arm(Path(runs_dir) / f"{model}-s{seed}-dose-{arm_name(m, aggregation, fault)}")
                if outcome is not None:
                    arms[(m, aggregation, fault)] = outcome
        for fault in FAULTS:
            twin = arms.get((m, "fo_robust" if m + 1 >= QUORUM else "robust_all", fault))
            if twin is not None:
                arms[(m, "quorum_robust", fault)] = {**twin, "same_as": "fo_robust" if m + 1 >= QUORUM else "robust_all"}
    return arms


def table_rows(arms, fault, doses=range(6)):
    """One row per aggregation rule: loss with the fault, damage against the paired clean run, influence on S1."""
    rows = []
    for aggregation in [*AGGREGATIONS, "quorum_robust"]:
        row = {"aggregation": aggregation, "cells": []}
        for m in doses:
            hit, clean = arms.get((m, aggregation, fault)), arms.get((m, aggregation, "none"))
            if hit is None:
                row["cells"].append(None)
                continue
            damage = hit["loss"] - clean["loss"] if hit["loss"] is not None and clean and clean["loss"] is not None else None
            row["cells"].append({"m": m, "loss": hit["loss"], "clean_loss": clean["loss"] if clean else None, "damage": damage,
                                 "accuracy_pct": hit["accuracy_pct"], "influence_s1": hit["influence_s1"], "status": hit["status"]})
        rows.append(row)
    return rows


def decide_h2(arms, doses=range(6)):
    """Hypothesis H2 (quorum crossover), per fault strength, on the test loss reached with the fault (lower is better;
    a diverged arm counts as worst).

    Supported when averaging robustly among first-order reports only beats robust averaging over all reports
    wherever the honest first-order reports are a majority (m >= 2), and loses to it where they are not (m <= 1).
    """
    result = {}
    for fault in ("flip1", "flip20"):
        loss = lambda aggregation, m: float("inf") if arms[(m, aggregation, fault)]["loss"] is None else arms[(m, aggregation, fault)]["loss"]  # noqa: E731
        available = [m for m in doses if (m, "fo_robust", fault) in arms and (m, "robust_all", fault) in arms]
        low, high = [m for m in available if m + 1 < QUORUM], [m for m in available if m + 1 >= QUORUM]
        entry = {"doses": available, "fo_robust_better_at": [m for m in available if loss("fo_robust", m) < loss("robust_all", m)]}
        if not low or not high:
            entry["verdict"], entry["reason"] = "CANNOT_JUDGE", "needs doses on both sides of the quorum"
        else:
            holds = all(m in entry["fo_robust_better_at"] for m in high) and not any(m in entry["fo_robust_better_at"] for m in low)
            entry["verdict"] = "SUPPORTED" if holds else "NOT_SUPPORTED"
            entry["reason"] = f"first-order-only robust averaging is better at m = {entry['fo_robust_better_at']}; the hypothesis expects exactly m >= {QUORUM - 1}"
        result[fault] = entry
    return result
