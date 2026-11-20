import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .aggregation import aggregate, apply as apply_update
from .algorithms import hybrid_step, optimizer_for, restore, task_loss
from .batches import ClientStream, collate, collate_answers
from .classification import evaluate as classification_evaluate
from .config import SPECS, parse_config, parse_rounds
from .data import load_manifest, load_tokens
from .decomfl import CentralEstimator, local_step as decomfl_step
from .io import append_jsonl, sha256, tensor_hash, write_json
from .model import load_model, partition_parameters
from .randomness import client_directions
from .reports import ClientReport
from .sampling import DepRoundSampler
from .scheduling import LegacyScheduler
from .states import load_state, save_state, state_path


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.wrapper, self.params, self.tokenizer, self.tokens = load_model(cfg)
        self.initial_hash = tensor_hash(self.params)
        self.data = load_tokens(cfg, self.tokenizer)
        self.manifest, self.partition_id, self.manifest_bytes = load_manifest(cfg, self.data["train"])
        self.resource_cuts = [c["mixed_L_zo"] for c in self.manifest["clients"]]
        self.cuts = self.resource_cuts if cfg.method == "hofl" else [SPECS[cfg.model][2] if cfg.method == "decomfl" else 0] * cfg.n
        self.bottoms = {c: partition_parameters(self.params, c)[0] for c in set(self.cuts)}
        self.dimensions = {c: sum(p.numel() for p in bottom.values()) for c, bottom in self.bottoms.items()}
        self.estimator = CentralEstimator(self.params, cfg.mu, cfg.q) if cfg.method == "decomfl" else None
        self.processed_tokens = self.answer_tokens = 0

    def optimizer(self, params=None):
        return optimizer_for(self.params if params is None else params, self.cfg)

    def local_step(self, cid, indices, optimizer, directions, e, reference):
        cfg = self.cfg
        if cfg.dataset == "squad":
            from .squad import AnswerLossWrapper
            batch = collate_answers(self.data["train"], indices, self.tokenizer.pad_token_id, cfg.device)
            wrapper, tokens = AnswerLossWrapper(self.wrapper, batch), None
            self.answer_tokens += int(batch["prediction_mask"].sum().item())
        else:
            batch = collate(self.data["train"], indices, self.tokenizer.pad_token_id, cfg.device)
            wrapper, tokens = self.wrapper, self.tokens
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
        return hybrid_step(cfg, wrapper, self.params, self.bottoms[self.cuts[cid]], batch, tokens, optimizer, directions, e)

    def evaluate(self, round_id, final):
        cfg = self.cfg
        if cfg.dataset != "squad":
            scores = classification_evaluate(self.wrapper, self.data["validation"], self.tokens, self.tokenizer.pad_token_id, cfg.eval_batch_size, cfg.device)
            return [{"accuracy": scores["accuracy"], "task_loss": scores["validation_loss"], "count": scores["validation_count"], "evaluation_scope": "full_official_test" if cfg.dataset in {"sciq", "ag_news"} else "full_validation"}]
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


def logical_uplink(cfg, total, bottom):
    scalar_bytes = 4 * cfg.E * cfg.q if cfg.method in {"hofl", "decomfl"} else 0
    delta_bytes = 4 * (total - bottom)
    return {"header_bytes": 18, "zo_scalar_bytes": scalar_bytes, "fo_delta_bytes": delta_bytes,
            "logical_uplink_bytes": 18 + scalar_bytes + delta_bytes,
            "accounting_scope": "logical_seed_scalar_FO_delta_not_actual_dense_server_transfer"}


def train_client(b, state, streams, r, cid, cut):
    """One participation of client ``cid`` in round ``r``: local training from ``state``, returned as a report.

    This is the inner loop of the upstream ``run``: directions, restore, fresh optimizer, E local steps, then the
    difference to ``state``. It advances ``streams[cid]`` and leaves the client's parameters in ``b.params``.
    """
    cfg = b.cfg
    if cut != b.cuts[cid]:
        raise NotImplementedError("Cut overrides are not supported yet")
    directions = client_directions(cfg.seed, cfg.model, r, cid)
    restore(b.params, state)
    optimizer = b.optimizer()
    losses, scalars, batches = [], [], []
    for e in range(cfg.E):
        indices = streams[cid].next()
        loss, step_scalars = b.local_step(cid, indices, optimizer, directions, e, state)
        losses.append(loss)
        scalars.append(step_scalars)
        batches.append(indices)
    with torch.no_grad():
        delta = {name: param.detach() - state[name] for name, param in b.params.items()}
    return ClientReport(cid=cid, round=r, cut=cut, tier=SPECS[cfg.model][3].index(b.resource_cuts[cid]),
                        delta=delta, scalars=scalars, batches=batches, losses=losses,
                        direction_master_seed=directions.master_seed)


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
        scheduler = LegacyScheduler(sampler, b.cuts)
        config = {**asdict(cfg), "method_label": cfg.method_label, "partition_id": b.partition_id,
                  "initial_parameter_sha256": b.initial_hash, "resource_cuts": b.resource_cuts,
                  "actual_cuts": b.cuts, "bottom_dimensions": b.dimensions,
                  "sampler_metadata": sampler.dr.metadata(), "server_aggregation": "dense_mean_delta",
                  "backbone_dtype": "bfloat16", "adapter_dtype": "float32", "lora_rank": 8,
                  "lora_alpha": 16, "lora_dropout": 0, "betas": cfg.betas,
                  "optimizer_reset": "client_participation", "model_revision": SPECS[cfg.model][1],
                  "direction_seed_scheme": "stable_seed(seed, model, round, client_id, 'zo_master')",
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
        write_json(out / "config.json", config)
        (out / "manifest.json.gz").write_bytes(gzip.compress(b.manifest_bytes, mtime=0))
        state = {n: p.detach().clone() for n, p in b.params.items()}
        if cfg.init_state:
            state, loaded = load_state(cfg.init_state, state)
            restore(b.params, state)
            write_json(out / "init_state.json", {"path": cfg.init_state, "file_sha256": sha256(cfg.init_state), **loaded})
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

        evaluate(0)
        for r in range(1, cfg.rounds + 1):
            # Schedule: the cohort of this round, in the order the sampler returned it.
            plan = scheduler.sample(r)
            selected = plan.clients
            # Local training: one report per client.
            reports = []
            for cid in selected:
                report = train_client(b, state, streams, r, cid, plan.cut(cid))
                reports.append(report)
                counts = logical_uplink(cfg, total, b.dimensions[b.cuts[cid]])
                cumulative += counts["logical_uplink_bytes"]
                p = scheduler.probability(cid)
                append_jsonl(out / "ledger.jsonl", {"round": r, "client": cid, "cut": b.cuts[cid],
                             "resource_cut": b.resource_cuts[cid], "d_ZO": b.dimensions[b.cuts[cid]],
                             "direction_master_seed": report.direction_master_seed if cfg.method in {"hofl", "decomfl"} else None,
                             "p": p, "inclusion_probability": cfg.k * p, **counts})
            # Arrivals: every report is on time until deadlines are modelled.
            arrivals = None
            # Aggregation, then the server step on the global state.
            update, _ = aggregate(reports, arrivals, cfg)
            apply_update(state, update)
            restore(b.params, state)
            scheduler.observe(r, arrivals)
            if b.estimator is not None:
                b.estimator.round_scalars.clear()
            # Logging and evaluation.
            training_loss = torch.stack([loss for report in reports for loss in report.losses]).mean().item()
            assert np.isfinite(training_loss) and all(torch.isfinite(p).all().item() for p in b.params.values())
            if r in save_rounds:
                save_state(state_path(out, r), state, {"round": r, "seed": cfg.seed, "model": cfg.model,
                           "partition_id": b.partition_id, "initial_parameter_sha256": b.initial_hash})
            append_jsonl(out / "rounds.jsonl", {"round": r, "processed_samples": r * cfg.k * cfg.E * cfg.batch_size,
                         "selected_clients": selected, "task_loss": training_loss,
                         "processed_input_tokens": b.processed_tokens, "processed_answer_tokens": b.answer_tokens})
            status["last_completed_round"] = r
            write_json(out / "status.json", status)
            if r % cfg.evaluate_every == 0 or r == cfg.rounds:
                final = evaluate(r)
            del reports, report, update
        summary = {"status": "COMPLETED", "method": cfg.method_label, "seed": cfg.seed,
                   "partition_id": b.partition_id, "initial_parameter_sha256": b.initial_hash, **final}
        summary["final_parameter_sha256"] = tensor_hash(state)
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {**status, "status": "COMPLETED"})
        return summary
    except BaseException as error:
        write_json(out / "status.json", {**status, "status": "FAILED", "error": repr(error)})
        raise


def main():
    run(parse_config())


if __name__ == "__main__":
    main()
