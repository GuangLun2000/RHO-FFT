import gzip
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .aggregation import aggregate, apply as apply_update
from .algorithms import hybrid_step, optimizer_for, restore, task_loss
from .batches import ClientStream, collate, collate_answers
from .classification import evaluate as classification_evaluate
from .config import SPECS, parse_config, parse_rounds
from .costs import BACKWARD_RATIO, forward_equivalents
from .data import load_manifest, load_tokens
from .decomfl import CentralEstimator, local_step as decomfl_step
from .io import append_jsonl, sha256, tensor_hash, write_json
from .model import load_model, partition_parameters
from .probes import Probes, replay_participation
from .faults import Faults, influence_rows
from .randomness import client_directions
from .reports import ClientReport
from .sampling import DepRoundSampler
from .scheduling import make_scheduler
from .states import load_state, save_state, state_path


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.wrapper, self.params, self.tokenizer, self.tokens = load_model(cfg)
        self.initial_hash = tensor_hash(self.params)
        self.data = load_tokens(cfg, self.tokenizer)
        self.manifest, self.partition_id, self.manifest_bytes = load_manifest(cfg, self.data["train"])
        self.resource_cuts = [c["mixed_L_zo"] for c in self.manifest["clients"]]
        if cfg.method == "hofl":
            # uniform_cut: every client trains with the same cut (the paper's "uniform boundary" variant); tiers keep
            # their resource cut for bookkeeping.
            self.cuts = [cfg.uniform_cut] * cfg.n if cfg.uniform_cut else self.resource_cuts
        else:
            self.cuts = [SPECS[cfg.model][2] if cfg.method == "decomfl" else 0] * cfg.n
        self.bottoms = {c: partition_parameters(self.params, c)[0] for c in set(self.cuts)}
        self.dimensions = {c: sum(p.numel() for p in bottom.values()) for c, bottom in self.bottoms.items()}
        self.estimator = CentralEstimator(self.params, cfg.mu, cfg.q) if cfg.method == "decomfl" else None
        self.processed_tokens = self.answer_tokens = 0

    def optimizer(self, params=None, lr=None):
        return optimizer_for(self.params if params is None else params, self.cfg, lr=lr)

    def local_step(self, cid, indices, optimizer, directions, e, reference, *, cut=None, zo=True, label_map=None):
        cfg = self.cfg
        if cfg.dataset == "squad":
            from .squad import AnswerLossWrapper
            batch = collate_answers(self.data["train"], indices, self.tokenizer.pad_token_id, cfg.device)
            wrapper, tokens = AnswerLossWrapper(self.wrapper, batch), None
            self.answer_tokens += int(batch["prediction_mask"].sum().item())
        else:
            batch = collate(self.data["train"], indices, self.tokenizer.pad_token_id, cfg.device)
            wrapper, tokens = self.wrapper, self.tokens
            if label_map is not None:
                batch["labels"] = label_map(batch["labels"], e)
        self.processed_tokens += int(batch["attention_mask"].sum().item())
        if cfg.method in {"fedavg", "fedprox"}:
            optimizer.zero_grad(set_to_none=True)
            raw = task_loss(wrapper, batch, tokens)
            loss = raw
            if cfg.method == "fedprox":
                loss = raw + cfg.prox_lambda / 2 * sum((p - reference[n]).square().sum() for n, p in self.params.items())
            loss.backward()
            optimizer.step()
            return raw.detach(), []
        if cfg.method == "decomfl":
            return decomfl_step(self.estimator, wrapper, batch, tokens, optimizer, directions, e)
        return hybrid_step(cfg, wrapper, self.params, self.bottoms[self.cuts[cid] if cut is None else cut], batch, tokens, optimizer, directions, e, zo=zo)

    def evaluate(self, round_id, final):
        cfg = self.cfg
        if cfg.dataset != "squad":
            scores = classification_evaluate(self.wrapper, self.data["validation"], self.tokens, self.tokenizer.pad_token_id, cfg.eval_batch_size, cfg.device, **({"per_class": True} if cfg.per_class_eval else {}))
            row = {"accuracy": scores["accuracy"], "task_loss": scores["validation_loss"], "count": scores["validation_count"], "evaluation_scope": "full_official_test" if cfg.dataset in {"sciq", "ag_news"} else "full_validation"}
            if cfg.per_class_eval:
                row["per_class_accuracy"] = scores["per_class_accuracy"]
            return [row]
        from .squad import answer_loss, generate_answers
        subset = self.data["evaluation_indices"]
        score, _ = generate_answers(self.wrapper, self.tokenizer, self.data["validation"], subset)
        score.update(task_loss=answer_loss(self.wrapper, self.tokenizer, self.data["validation"], subset), evaluation_scope="fixed_1024_subset")
        rows = [score]
        if final or round_id == 0:
            score, _ = generate_answers(self.wrapper, self.tokenizer, self.data["validation"], list(range(len(self.data["validation"]["ids"]))))
            score.update(evaluation_scope="full_10570_validation", metric_version="official_squad_1.1_empty_overlap_returns_zero")
            rows.append(score)
        return rows


class Sampler:
    def __init__(self, cfg, backend):
        self.cfg = cfg
        self.dr = DepRoundSampler([backend.dimensions[c] for c in backend.cuts], cfg.q, cfg.k, cfg.beta, cfg.seed)
        self.torch_rng = torch.Generator().manual_seed(30000 + cfg.seed)
        self.numpy_rng = np.random.default_rng(30000 + cfg.seed)

    def sample(self):
        if self.cfg.sampler == "historical_torch":
            return torch.multinomial(torch.ones(self.cfg.n, dtype=torch.float64), self.cfg.k, replacement=False, generator=self.torch_rng).tolist()
        if self.cfg.sampler == "historical_numpy_uniform":
            self.numpy_rng.random()
            return self.numpy_rng.choice(self.cfg.n, size=self.cfg.k, replace=False).tolist()
        return self.dr.sample()


def logical_uplink(cfg, total, bottom, zo=True):
    scalar_bytes = 4 * cfg.E * cfg.q if zo and cfg.method in {"hofl", "decomfl"} else 0
    delta_bytes = 4 * (total - bottom)
    return {"header_bytes": 18, "zo_scalar_bytes": scalar_bytes, "fo_delta_bytes": delta_bytes,
            "logical_uplink_bytes": 18 + scalar_bytes + delta_bytes,
            "accounting_scope": "logical_seed_scalar_FO_delta_not_actual_dense_server_transfer"}


def train_client(b, state, streams, r, cid, cut, fault=None, history=None):
    """One participation of client ``cid`` in round ``r``: local training from ``state``, returned as a report.

    This is the inner loop of the upstream ``run``: directions, restore, fresh optimizer, E local steps, then the
    difference to ``state``. It advances ``streams[cid]`` and leaves the client's parameters in ``b.params``.

    ``cut`` is the cut the client trains with. It may differ from the client's own cut only for method hofl and
    only among the cuts of the model's tiers (a device of one tier then behaves like a device of another).

    ``fault`` holds the training-level overrides of ``Faults.before_local`` (learning-rate factor, label map, lag);
    with a lag the client starts from the state ``lag`` rounds back in ``history`` and its update is the change
    relative to that starting point, which is what such a device would upload.
    """
    cfg = b.cfg
    if cut != b.cuts[cid] and (cfg.method != "hofl" or cut not in b.bottoms):
        raise ValueError(f"Cut {cut} cannot be used by client {cid}")
    zo = cfg.zo_mode == "on"
    directions = client_directions(cfg.seed, cfg.model, r, cid)
    origin, extra = state, {}
    if fault and fault.get("lag") and history:
        origin = history[max(0, len(history) - 1 - fault["lag"])]
    if fault and fault.get("label_map"):
        extra["label_map"] = fault["label_map"]
    restore(b.params, origin)
    optimizer = b.optimizer(lr=cfg.lr * fault["lr_scale"]) if fault and "lr_scale" in fault else b.optimizer()
    losses, scalars, batches = [], [], []
    for e in range(cfg.E):
        indices = streams[cid].next()
        loss, step_scalars = b.local_step(cid, indices, optimizer, directions, e, origin, cut=cut, zo=zo, **extra)
        losses.append(loss)
        scalars.append(step_scalars)
        batches.append(indices)
    with torch.no_grad():
        delta = {name: param.detach() - origin[name] for name, param in b.params.items()}
    return ClientReport(cid=cid, round=r, cut=cut, tier=SPECS[cfg.model][3].index(b.resource_cuts[cid]),
                        delta=delta, scalars=scalars, batches=batches, losses=losses,
                        direction_master_seed=directions.master_seed, zo=zo,
                        fault="+".join(fault["kinds"]) if fault else None)


def clean_twin(b, state, report):
    """The update the participation behind ``report`` would have produced without a training-level fault.

    Same batches, the current global state, the configured learning rate and labels. Nothing observable changes:
    no data stream advances and the token counters are put back (the parameters are restored by the caller's
    next step, as after any client).
    """
    return replay_participation(b, state, report)


def run(cfg):
    out = Path(cfg.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    status = {"status": "RUNNING", "last_completed_round": 0}
    write_json(out / "status.json", status)
    try:
        b = Backend(cfg)
        sampler = Sampler(cfg, b)
        scheduler = make_scheduler(cfg, b.cuts, b.resource_cuts, sampler)
        probes = Probes(cfg, out)
        faults = Faults(cfg, b.resource_cuts) if cfg.fault_spec else None
        history = deque(maxlen=faults.max_lag + 1) if faults is not None and faults.max_lag else None
        config = {**asdict(cfg), "method_label": cfg.method_label, "partition_id": b.partition_id,
                  "initial_parameter_sha256": b.initial_hash, "resource_cuts": b.resource_cuts,
                  "actual_cuts": b.cuts, "bottom_dimensions": b.dimensions,
                  "sampler_metadata": scheduler.metadata(), "server_aggregation": "dense_mean_delta",
                  "backbone_dtype": "bfloat16", "adapter_dtype": "float32", "lora_rank": 8,
                  "lora_alpha": 16, "lora_dropout": 0, "betas": cfg.betas,
                  "optimizer_reset": "client_participation", "model_revision": SPECS[cfg.model][1],
                  "direction_seed_scheme": "stable_seed(seed, model, round, client_id, 'zo_master')",
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
        config.update(aggregation=cfg.aggregation_name, cost_model={"unit": "forward_equivalent", "backward_ratio": BACKWARD_RATIO})
        if faults is not None:
            config["faults"] = faults.metadata()
        write_json(out / "config.json", config)
        (out / "manifest.json.gz").write_bytes(gzip.compress(b.manifest_bytes, mtime=0))
        state = {n: p.detach().clone() for n, p in b.params.items()}
        if cfg.init_state:
            state, loaded = load_state(cfg.init_state, state)
            restore(b.params, state)
            matches = {"seed": loaded["metadata"].get("seed") == str(cfg.seed),
                       "partition_id": loaded["metadata"].get("partition_id") == b.partition_id}
            write_json(out / "init_state.json", {"path": cfg.init_state, "file_sha256": sha256(cfg.init_state),
                                                  "matches_run": matches, **loaded})
            if not all(matches.values()):
                print(json.dumps({"event": "WARNING", "message": "init_state was saved by a run with another seed or partition", **matches}), flush=True)
        streams = [ClientStream(c, cfg.batch_size) for c in b.manifest["clients"]]
        total = sum(p.numel() for p in b.params.values())
        cumulative, training_loss = 0, None
        save_rounds = set(parse_rounds(cfg.save_state_rounds))

        def evaluate(r):
            rows = b.evaluate(r, r == cfg.rounds)
            for row in rows:
                row.update(round=r, processed_samples=r * cfg.k * cfg.E * cfg.batch_size,
                           cumulative_logical_uplink_bytes=cumulative,
                           training_task_loss=training_loss)
                append_jsonl(out / "curves.jsonl", row)
            print(json.dumps({"event": "EVAL", **rows[-1]}), flush=True)
            return rows[-1]

        def evaluate_or_stop(r):
            """An evaluation, or ``None`` when the model no longer gives finite values and the run is allowed to stop."""
            try:
                return evaluate(r)
            except (AssertionError, ValueError):
                if cfg.on_nonfinite != "stop":
                    raise
                return None

        # eval_initial=False skips the evaluation of the starting point; training is the same either way.
        final, diverged = (evaluate(0) if cfg.eval_initial else {}), None
        for r in range(1, cfg.rounds + 1):
            if history is not None:
                history.append({name: tensor.clone() for name, tensor in state.items()})
            # Schedule: the cohort of this round, in the order the sampler returned it.
            plan = scheduler.sample(r)
            selected = plan.clients
            # Local training: one report per client.
            reports = []
            for cid in selected:
                if faults is None:
                    report = train_client(b, state, streams, r, cid, plan.cut(cid))
                else:
                    report = train_client(b, state, streams, r, cid, plan.cut(cid), fault=faults.before_local(r, cid), history=history)
                    if report.fault is not None:
                        report.clean = clean_twin(b, state, report)
                    report = faults.after_local(r, report)
                reports.append(report)
                counts = logical_uplink(cfg, total, b.dimensions[report.cut], zo=report.zo)
                cumulative += counts["logical_uplink_bytes"]
                p = scheduler.probability(cid)
                append_jsonl(out / "ledger.jsonl", {"round": r, "client": cid, "cut": report.cut,
                             "resource_cut": b.resource_cuts[cid], "d_ZO": b.dimensions[report.cut],
                             "direction_master_seed": report.direction_master_seed if cfg.method in {"hofl", "decomfl"} else None,
                             "p": p, "inclusion_probability": cfg.k * p, **counts,
                             "tier": report.tier, "fault": report.fault,
                             "forward_equivalents": forward_equivalents(cfg.method, cut=report.cut, depth=SPECS[cfg.model][2],
                                                                        E=cfg.E, q=cfg.q, zo=report.zo)})
            if cfg.on_nonfinite == "stop" and not all(torch.isfinite(value).all().item() for report in reports
                                                      for value in [*report.losses, *report.delta.values()]):
                diverged = r  # detected before anything of this round is aggregated or logged
                break
            # Arrivals: every report is on time until deadlines are modelled.
            arrivals = None
            # Aggregation, then the server step on the global state.
            update, _ = aggregate(reports, arrivals, cfg)
            if faults is not None:
                for row in influence_rows(reports, update, cfg):
                    append_jsonl(out / "influence.jsonl", {"round": r, **row})
            # Read-only probes: they see the reports and the update before the state changes.
            probes.run(r, b, state, reports, arrivals, update)
            apply_update(state, update, cfg.server_lr)
            restore(b.params, state)
            scheduler.observe(r, arrivals)
            if b.estimator is not None:
                b.estimator.round_scalars.clear()
            # Logging and evaluation.
            training_loss = torch.stack([loss for report in reports for loss in report.losses]).mean().item()
            if not (np.isfinite(training_loss) and all(torch.isfinite(p).all().item() for p in b.params.values())):
                assert cfg.on_nonfinite == "stop", "non-finite training loss or parameters"
                diverged = r
                break
            if r in save_rounds:
                save_state(state_path(out, r), state, {"round": r, "seed": cfg.seed, "model": cfg.model,
                           "partition_id": b.partition_id, "initial_parameter_sha256": b.initial_hash})
            append_jsonl(out / "rounds.jsonl", {"round": r, "processed_samples": r * cfg.k * cfg.E * cfg.batch_size,
                         "selected_clients": selected, "task_loss": training_loss,
                         "processed_input_tokens": b.processed_tokens, "processed_answer_tokens": b.answer_tokens})
            status["last_completed_round"] = r
            write_json(out / "status.json", status)
            if r % cfg.evaluate_every == 0 or r == cfg.rounds:
                evaluated = evaluate_or_stop(r)
                if evaluated is None:
                    diverged = r
                    break
                final = evaluated
            del reports, report, update
        # A diverged run is a result: it keeps its last finite evaluation and says in which round it broke down.
        outcome = "COMPLETED" if diverged is None else "DIVERGED"
        summary = {"status": outcome, "method": cfg.method_label, "seed": cfg.seed,
                   "partition_id": b.partition_id, "initial_parameter_sha256": b.initial_hash, **final}
        if diverged is not None:
            summary["diverged_round"] = status["diverged_round"] = diverged
        summary["final_parameter_sha256"] = tensor_hash(state)
        if cfg.init_state:
            summary["init_state_parameter_sha256"] = loaded["parameter_sha256"]
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {**status, "status": outcome})
        return summary
    except BaseException as error:
        write_json(out / "status.json", {**status, "status": "FAILED", "error": repr(error)})
        raise


def main():
    run(parse_config())


if __name__ == "__main__":
    main()
