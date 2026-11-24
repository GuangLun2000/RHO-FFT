"""The round loop of a satellite run (brief v2, sections 4.2 and 9.1).

A round: every satellite decides from its state in the round's first slot (``policy``); the participants train,
each from the replica of its plane (``client``); the network delivers the reports (``network``); every replica
group applies the sum of what it received (``aggregate``); batteries are updated and the round is logged.

Simulated time and compute time are separate things: the clock of the scenario advances by nominal compute and
link times, never by the wall time of this process.

Evaluations take place on the state after 0, ``eval_every``, 2 x ``eval_every``, ... completed rounds and after
the last one, i.e. at whole simulated hours with the defaults: 25 points from 0 h to 24 h.

Files written to the run directory:

    config.json    the configuration and everything that identifies the inputs (trace, partition and git hashes)
    rounds.jsonl   one row per round
    ledger.jsonl   one row per participation
    eval.jsonl     one row per evaluation
    summary.json   final metrics and the status: COMPLETED, or DIVERGED when values stopped being finite
    status.json    RUNNING / COMPLETED / DIVERGED / FAILED, as for upstream runs (the run harness reads it)

Wall times of this process are logged next to the results and never enter them: ``wall_s`` of a participation,
of a round (with its parts in ``wall_parts_s``) and of an evaluation, and their sums in ``wall_breakdown_s`` of the
summary. They are what the GPU budget of the experiments is planned from. All of them are differences of a
monotonic clock, so an adjustment of the system time cannot make one negative.
"""
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..algorithms import restore
from ..grid import git_head
from ..io import append_jsonl, tensor_hash, write_json
from .aggregate import Groups, replica_difference, round_updates
from .backend import SatBackend
from .client import train_participation
from .config import CLASSES
from .energy import Batteries
from .links import LinkBudget
from .methods import METHODS
from .network import deliver, timeline
from .policy import SKIP_REASONS, decide
from .scenario import ensure_trace

REPO = Path(__file__).resolve().parents[2]


def _by_class(roles, values):
    """Sum of a per-satellite array for each class."""
    return {kind: float(sum(values[sid] for sid in roles.members(kind))) for kind in CLASSES}


def _clock(cuda):
    """A monotonic clock, read after the device has finished what was queued (GPU work is asynchronous)."""
    if cuda:
        torch.cuda.synchronize()
    return time.perf_counter()


def _all_finite(tensors):
    """Whether every value of every tensor is finite (one synchronisation with the device)."""
    flags = [torch.isfinite(tensor).all() for tensor in tensors]
    return bool(torch.stack(flags).all().item()) if flags else True


def evaluate(b, groups, initial, rounds_done, cfg, final=False):
    """One evaluation row: every replica group on the evaluation set, their plane-weighted mean and how far the
    replicas are apart. The last evaluation of a test run uses the whole split (HumAID curves use a subset)."""
    indices = b.final_indices if final else b.curve_indices
    started, rows = time.perf_counter(), []
    for group in groups.ids:
        restore(b.params, groups.state(group))
        metrics = b.evaluate(b.eval_split, indices)
        row = {"group": group, "planes": groups.members(group), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
               "loss": metrics["loss"], "per_class_accuracy": metrics["per_class_accuracy"]}
        if b.held_out_indices:
            held = b.evaluate(b.eval_split, b.held_out_indices)
            row.update(held_out_accuracy=held["accuracy"], held_out_macro_f1=held["macro_f1"], held_out_count=held["count"])
        rows.append(row)
    weight = {row["group"]: len(row["planes"]) / groups.planes for row in rows}
    result = {"round": rounds_done, "sim_hours": rounds_done * cfg.round_s / 3600, "split": b.eval_split,
              "scope": "whole split" if indices is None else f"fixed subset of {len(indices)}", "final": final,
              "count": metrics["count"], "groups": rows, "replica_groups": len(rows),
              "replica_difference": replica_difference(groups, initial), "wall_s": None}
    for key in ("accuracy", "macro_f1", "loss") + (("held_out_accuracy", "held_out_macro_f1") if b.held_out_indices else ()):
        result[key] = sum(weight[row["group"]] * row[key] for row in rows)
    result["wall_s"] = time.perf_counter() - started
    return result


def run(cfg):
    """Run the configuration (a resolved ``SatConfig``) and return its summary."""
    out = Path(cfg.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    status = {"status": "RUNNING", "last_completed_round": 0}
    write_json(out / "status.json", status)
    started = time.perf_counter()
    try:
        method = METHODS[cfg.method]
        trace = ensure_trace(cfg.trace_dir, cfg.scenario, slot_s=cfg.slot_s)
        if trace.meta["scenario"] != cfg.scenario or trace.slot_s != cfg.slot_s or trace.slots < cfg.rounds * cfg.slots_per_round:
            raise ValueError(f"The trace at {cfg.trace_dir} does not fit this run (scenario, slot length or duration)")
        roles = trace.roles
        b = SatBackend(cfg)
        batteries = Batteries(roles)
        initial = {name: param.detach().clone() for name, param in b.params.items()}
        groups = Groups(initial, roles.planes)
        cuda = torch.device(cfg.device).type == "cuda"
        config = {**cfg.to_dict(), "rounds": cfg.rounds, "method_label": method.label, "method_switches": asdict(method),
                  "task_model": b.task.model, "task_dataset": b.task.dataset, "depth": b.task.depth, "cuts": b.cuts,
                  "first_order_bytes_by_cut": {str(cut): b.fo_bytes(cut) for cut in b.cuts}, "report_bytes": b.zo_bytes(),
                  "trainable_parameters": b.total, "classes": b.class_names, "prompt_tokens": b.token_meta,
                  "trace_sha256": trace.sha256, "land_source": trace.meta["land_source"],
                  "partition_sha256": b.manifest["partition_sha256"], "manifest_sha256": b.manifest_id,
                  "data_sha256": b.data_sha256, "initial_parameter_sha256": b.initial_hash, "git": git_head(REPO),
                  "evaluation": {"split": b.eval_split, "curve_examples": len(b.curve_indices) if b.curve_indices else b.split_size(b.eval_split),
                                 "held_out_examples": len(b.held_out_indices) if b.held_out_indices else 0},
                  "aggregation": "mean_all: sum over the participants in ascending satellite number, divided by their number",
                  "direction_seed_scheme": "stable_seed(seed, model, round, sid, 'zo_master')",
                  "lottery_seed_scheme": "stable_seed(seed, 'sat-lottery', round, sid) / 2**32",
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if cuda else None}
        write_json(out / "config.json", config)
        if cfg.faults:
            raise NotImplementedError("Fault injection arrives in milestone P-S4")

        def log_evaluation(rounds_done, final=False):
            row = evaluate(b, groups, initial, rounds_done, cfg, final)
            append_jsonl(out / "eval.jsonl", row)
            extra = f" | held-out accuracy {100 * row['held_out_accuracy']:.2f}%" if "held_out_accuracy" in row else ""
            print(f"EVAL {row['sim_hours']:5.1f} h (round {rounds_done}/{cfg.rounds}) | {row['split']} accuracy {100 * row['accuracy']:.2f}% | "
                  f"macro-F1 {row['macro_f1']:.4f} | loss {row['loss']:.4f}{extra} | {row['replica_groups']} replica group(s) | "
                  f"{row['scope']} | {row['wall_s']:.1f} s", flush=True)
            return row

        def log_evaluation_or_stop(rounds_done, final=False):
            """An evaluation, or ``None`` when the model no longer gives finite values (the run then ends as DIVERGED)."""
            try:
                return log_evaluation(rounds_done, final)
            except (AssertionError, ValueError):
                return None

        wall = Counter(startup=time.perf_counter() - started)   # loading the model, the data, the trace and the partition
        evaluation_started = time.perf_counter()
        last = log_evaluation(0)
        wall["evaluation"] += time.perf_counter() - evaluation_started
        diverged, total_loss, energy_total = None, [], {"compute": np.zeros(roles.count), "transmit": np.zeros(roles.count)}
        bytes_total, participations, wall_training = Counter(), Counter(), 0.0
        for r in range(cfg.rounds):
            round_started = _clock(cuda)
            view = trace.view(r, cfg.slots_per_round)
            budget = LinkBudget(trace, view, cfg.rf_gain_dbi, cfg.laser_gbps)
            times = timeline(cfg, budget, trace, b.full_bytes)
            decisions = decide(r, view, budget, times, batteries, method, cfg, b, roles)
            participants = sorted(sid for sid, decision in decisions.items() if decision.participate)
            # Local training, each participant from the replica of its plane.
            reports, seconds, mark = {}, {}, _clock(cuda)
            for sid in participants:
                reports[sid] = train_participation(b, groups.state(groups.of_plane(roles.plane(sid))), r, sid,
                                                   decisions[sid].cut, zo=decisions[sid].zo)
                finished = _clock(cuda)
                seconds[sid], mark = finished - mark, finished
            if not _all_finite(value for report in reports.values() for value in [*report.losses, *report.delta.values()]):
                diverged = r   # detected before anything of this round is aggregated or logged
                break
            trained = _clock(cuda)
            delivery = deliver(trace, budget, times, decisions, b, method, cfg.round_s)
            delivered = time.perf_counter()
            updates, _ = round_updates(groups, reports, delivery, b.cfg)
            groups.apply(updates, cfg.server_lr)
            if not _all_finite(tensor for group in groups.ids for tensor in groups.state(group).values()):
                diverged = r
                break
            aggregated = _clock(cuda)
            # Energy: what the participants computed and what every satellite sent; then the batteries.
            compute = np.zeros(roles.count)
            for sid in participants:
                compute[sid] = decisions[sid].energy_cmp_j
            before = [batteries.share(sid) for sid in range(roles.count)]
            if method.exchange == "orbit":
                batteries.update(view.sun_seconds, cfg.round_s, compute + delivery.tx_energy_j)
            energy_total["compute"] += compute
            energy_total["transmit"] += delivery.tx_energy_j
            bytes_total.update(delivery.bytes)
            # Logs.
            losses = [report.train_loss for report in reports.values()]
            round_loss = float(np.mean(losses)) if losses else None
            total_loss += losses
            for sid in participants:
                decision, report = decisions[sid], reports[sid]
                participations[decision.kind] += 1
                append_jsonl(out / "ledger.jsonl", {
                    "round": r, "sid": sid, "kind": decision.kind, "plane": roles.plane(sid), "slot": roles.slot(sid),
                    "cut": decision.cut, "zo": decision.zo, "moved_up": decision.moved_up, "imaging": decision.imaging,
                    "sunlit": decision.sunlit, "tau_s": decision.tau_s, "hops": decision.hops,
                    "arrival_s": delivery.arrival_s.get(sid), "first_order_used": sid in delivery.on_time,
                    "late": sid in delivery.late, "report_received": sid in delivery.beacons,
                    "zo_used": sid in delivery.zo_received, "fo_bytes": decision.fo_bytes,
                    "report_bytes": b.zo_bytes() if decision.zo else b.beacon_bytes,
                    "energy_compute_wh": decision.energy_cmp_j / 3600, "energy_transmit_wh": float(delivery.tx_energy_j[sid]) / 3600,
                    "battery_before": before[sid], "battery_after": batteries.share(sid), "train_loss": report.train_loss,
                    "lottery": decision.lottery, "direction_master_seed": report.direction_master_seed, "fault": report.fault,
                    "wall_s": seconds[sid]})
            logged = time.perf_counter()
            invited = Counter(decision.kind for decision in decisions.values() if decision.invited)
            taking_part = Counter(decisions[sid].kind for sid in participants)
            skipped = Counter(decision.reason for decision in decisions.values() if decision.reason)
            append_jsonl(out / "rounds.jsonl", {
                "round": r, "sim_start_s": r * cfg.round_s, "sim_hours": (r + 1) * cfg.round_s / 3600,
                "invited": {kind: invited.get(kind, 0) for kind in CLASSES},
                "participants": {kind: taking_part.get(kind, 0) for kind in CLASSES}, "participant_count": len(participants),
                "counted": len(delivery.beacons),
                "cut_histogram": {str(cut): count for cut, count in sorted(Counter(decisions[sid].cut for sid in participants).items())},
                "moved_up": sum(decisions[sid].moved_up > 0 for sid in participants),
                "deadline_s": times.deadline_s, "ring_s": times.ring_s, "broadcast_s": times.broadcast_s, "flood_s": delivery.flood_s,
                "first_order_sent": len(delivery.arrival_s) if method.exchange == "orbit" else len(delivery.on_time),
                "first_order_late": len(delivery.late),
                "zo_reports": {"sent": sum(decisions[sid].zo for sid in participants), "used": len(delivery.zo_received),
                               "dropped": len(delivery.zo_dropped)},
                "replica_groups": len(groups.ids), "bytes": dict(delivery.bytes),
                "energy_compute_wh": {kind: value / 3600 for kind, value in _by_class(roles, compute).items()},
                "energy_transmit_wh": {kind: value / 3600 for kind, value in _by_class(roles, delivery.tx_energy_j).items()},
                "skipped": {reason: skipped.get(reason, 0) for reason in SKIP_REASONS},
                "battery_min": {kind: min((batteries.share(sid) for sid in roles.members(kind)), default=None) if kind != "B" else None
                                for kind in CLASSES},
                "task_loss": round_loss, "wall_s": logged - round_started,
                # Where the round's wall time went: local training, the network model (CPU), replay and aggregation, logging.
                "wall_parts_s": {"train": trained - round_started, "deliver": delivered - trained,
                                 "aggregate": aggregated - delivered, "log": logged - aggregated}})
            wall_training += logged - round_started
            wall.update(train=trained - round_started, deliver=delivered - trained, aggregate=aggregated - delivered,
                        log=logged - aggregated)
            status["last_completed_round"] = r + 1
            write_json(out / "status.json", status)
            done = r + 1
            if done % cfg.eval_every == 0 or done == cfg.rounds:
                shown = "n/a" if round_loss is None else f"{round_loss:.4f}"
                print(f"round {done}/{cfg.rounds} ({done * cfg.round_s / 3600:.1f} h) | participants B {taking_part.get('B', 0)} "
                      f"H {taking_part.get('H', 0)} Z {taking_part.get('Z', 0)} | train loss {shown} | "
                      f"{wall_training / done:.2f} s per round", flush=True)
                evaluation_started = time.perf_counter()
                evaluated = log_evaluation_or_stop(done)
                wall["evaluation"] += time.perf_counter() - evaluation_started
                if evaluated is None:
                    diverged = r
                    break
                last = evaluated
            del reports, updates
        final = None
        if diverged is None and b.final_indices != b.curve_indices:
            evaluation_started = time.perf_counter()
            final = log_evaluation_or_stop(cfg.rounds, final=True)
            wall["evaluation"] += time.perf_counter() - evaluation_started
        outcome = "COMPLETED" if diverged is None else "DIVERGED"
        days = cfg.rounds * cfg.round_s / 86400
        members = {kind: len(roles.members(kind)) for kind in CLASSES}
        summary = {
            "status": outcome, "task": cfg.task, "method": cfg.method, "method_label": method.label, "variant": cfg.variant,
            "seed": cfg.seed, "hours": cfg.hours, "rounds": cfg.rounds, "server_lr": cfg.server_lr, "eval_split": b.eval_split,
            "last_evaluation": {key: last[key] for key in ("round", "sim_hours", "accuracy", "macro_f1", "loss", "scope", "replica_groups", "replica_difference")},
            "final_evaluation": None if final is None else {key: final[key] for key in ("accuracy", "macro_f1", "loss", "scope", "count")},
            "held_out": {key: last[key] for key in ("held_out_accuracy", "held_out_macro_f1")} if "held_out_accuracy" in last else None,
            "train_loss_first_rounds": float(np.mean(total_loss[:max(1, len(total_loss) // 10)])) if total_loss else None,
            "train_loss_last_rounds": float(np.mean(total_loss[-max(1, len(total_loss) // 10):])) if total_loss else None,
            "participations": {kind: participations.get(kind, 0) for kind in CLASSES},
            "energy_wh_per_satellite_day": {name: {kind: float(sum(values[sid] for sid in roles.members(kind))) / 3600 / members[kind] / days
                                                    for kind in CLASSES} for name, values in energy_total.items()},
            "bytes_per_day": {link: bytes_total.get(link, 0) / days for link in ("rf", "laser", "ground")},
            "trace_sha256": trace.sha256, "partition_sha256": b.manifest["partition_sha256"], "manifest_sha256": b.manifest_id,
            "data_sha256": b.data_sha256, "git": config["git"], "gpu": config["gpu"],
            "initial_parameter_sha256": b.initial_hash,
            "final_parameter_sha256": {str(group): tensor_hash(groups.state(group)) for group in groups.ids},
            "wall_s": time.perf_counter() - started, "wall_s_per_round": wall_training / max(1, status["last_completed_round"]),
            "wall_breakdown_s": {key: float(wall.get(key, 0.0)) for key in ("startup", "train", "deliver", "aggregate", "log", "evaluation")}}
        if diverged is not None:
            summary["diverged_round"] = status["diverged_round"] = diverged
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {**status, "status": outcome})
        print(f"{outcome}: {cfg.task} {cfg.method} seed {cfg.seed} | {b.eval_split} accuracy {100 * last['accuracy']:.2f}% at {last['sim_hours']:.1f} h"
              + (f" | whole {b.eval_split} split {100 * final['accuracy']:.2f}%" if final else "") + f" | {summary['wall_s'] / 60:.1f} min", flush=True)
        return summary
    except BaseException as error:
        write_json(out / "status.json", {**status, "status": "FAILED", "error": repr(error)})
        raise
