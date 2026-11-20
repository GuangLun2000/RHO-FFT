"""Read-only probes of a training run (brief section 6.9).

- Layer statistics (``layer_stats``): for every round and layer, how many first-order and zeroth-order components
  there are, how large they are, how well they agree with each other, and how large the applied update is.
- Counterfactual first-order probe (``probe_rounds``): for every selected client, the local training is repeated
  from the same global state on the same batches with backpropagation through all trainable tensors. Comparing the
  client's actual update with this counterfactual, layer by layer, shows how much of the first-order direction the
  zeroth-order components carry.

Probes run between aggregation and the server step. They never advance a data stream, never touch the token
counters and never draw a random number, and they put the parameters back, so a run with probes reaches the same
parameters and writes the same logs as a run without them.
"""
from pathlib import Path

import torch

from .aggregation import plan_units
from .algorithms import optimizer_for, restore, task_loss
from .batches import collate
from .config import SPECS, parse_rounds
from .io import append_jsonl
from .reports import segment_bounds, segment_index


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _mean_norm(gram, members):
    """Norm of the mean of the listed components, from their Gram matrix."""
    if not members:
        return None
    index = torch.tensor(members)
    return float(gram[index][:, index].sum().clamp_min(0).sqrt() / len(members))


def _mean_cosine(gram, norms, pairs):
    values = [float(gram[i, j] / (norms[i] * norms[j])) for i, j in pairs if norms[i] > 0 and norms[j] > 0]
    return sum(values) / len(values) if values else None


@torch.no_grad()
def layer_stats(reports, arrivals, update, cfg):
    """One row per layer for the reports of a round. Reads tensors only; nothing is modified."""
    bounds = segment_bounds(SPECS[cfg.model][3], SPECS[cfg.model][2])
    rows = []
    for plan in plan_units(reports, arrivals, cfg):
        matrix = torch.stack([torch.cat([report.delta[name].reshape(-1) for name in plan.names]) for report in reports]).double()
        gram = (matrix @ matrix.T).cpu()
        norms = gram.diagonal().clamp_min(0).sqrt()
        fo = plan.first_order
        # With zo_mode=off the components below the cut are exact zeros: they are absent, not zeroth-order votes.
        zo = [] if cfg.zo_mode == "off" else [index for index in range(len(reports)) if index not in fo]
        squared = sum(float((cfg.server_lr * update.tensors[name].double() / update.divisor[name]).square().sum())
                      for name in plan.names if name in update.tensors)
        rows.append({
            "layer": plan.layer, "segment": segment_index(plan.layer, bounds), "n_fo": len(fo), "n_zo": len(zo),
            "fo_norm": float(norms[fo].mean()) if fo else None, "zo_norm": float(norms[zo].mean()) if zo else None,
            "fo_mean_norm": _mean_norm(gram, fo), "zo_mean_norm": _mean_norm(gram, zo),
            "cos_fo_fo": _mean_cosine(gram, norms, [(i, j) for i in fo for j in fo if i < j]),
            "cos_zo_zo": _mean_cosine(gram, norms, [(i, j) for i in zo for j in zo if i < j]),
            "cos_fo_zo": _mean_cosine(gram, norms, [(i, j) for i in fo for j in zo]),
            "update_norm": squared ** 0.5, "rule": plan.rule, "electorate": len(plan.members)})
    return rows


def first_order_delta(b, state, batches):
    """Local training from ``state`` on ``batches`` with backpropagation through every trainable tensor.

    This is the first-order (FedAvg) step of ``Backend.local_step`` written out, so that neither the method
    dispatch nor the token counters of the backend are involved. Leaves the trained parameters in ``b.params``.
    """
    cfg = b.cfg
    restore(b.params, state)
    optimizer = optimizer_for(b.params, cfg)
    for indices in batches:
        optimizer.zero_grad(set_to_none=True)
        batch = collate(b.data["train"], indices, b.tokenizer.pad_token_id, cfg.device)
        task_loss(b.wrapper, batch, b.tokens).backward()
        optimizer.step()
    with torch.no_grad():
        return {name: param.detach() - state[name] for name, param in b.params.items()}


def _compare(actual, reference, names):
    """Dot product and squared norms (float64) of two updates restricted to ``names``."""
    dot = sum(float((actual[name].double() * reference[name].double()).sum()) for name in names)
    a = sum(float(actual[name].double().square().sum()) for name in names)
    c = sum(float(reference[name].double().square().sum()) for name in names)
    return dot, a, c, sum(actual[name].numel() for name in names)


def _row(dot, a, c, dimension):
    return {"cosine": _ratio(dot, (a * c) ** 0.5), "norm_ratio": _ratio(a ** 0.5, c ** 0.5), "actual_norm": a ** 0.5,
            "counterfactual_norm": c ** 0.5, "dimension": dimension}


def counterfactual_rows(report, counterfactual, cfg):
    """Rows comparing a report with its first-order counterfactual: one per layer and pooled ones per order."""
    plan = plan_units([report], None, cfg)
    rows, pooled = [], {}
    base = {"client": report.cid, "tier": report.tier, "cut": report.cut}
    for unit in plan:
        order = "fo" if unit.first_order else "zo"
        rows.append({**base, "scope": "layer", "layer": unit.layer, "group": order, **_row(*_compare(report.delta, counterfactual, unit.names))})
        groups = [order] + ([f"zo_{kind}" for kind in ("lora_A", "lora_B")] if order == "zo" else [])
        for group in groups:
            names = unit.names if group in ("fo", "zo") else [name for name in unit.names if group[3:] in name]
            totals = pooled.setdefault(group, [0.0, 0.0, 0.0, 0])
            for position, value in enumerate(_compare(report.delta, counterfactual, names)):
                totals[position] += value
    for group in sorted(pooled):
        rows.append({**base, "scope": "pooled", "layer": None, "group": group, **_row(*pooled[group])})
    return rows


class Probes:
    """The probes of one run, switched on by ``cfg.layer_stats`` and ``cfg.probe_rounds``."""

    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, Path(out)
        self.rounds = set(parse_rounds(cfg.probe_rounds))

    def run(self, r, b, state, reports, arrivals, update):
        cfg = self.cfg
        if cfg.layer_stats:
            for row in layer_stats(reports, arrivals, update, cfg):
                append_jsonl(self.out / "layers.jsonl", {"round": r, **row})
        if r in self.rounds:
            try:
                for report in reports:
                    if report.fault is None:
                        counterfactual = first_order_delta(b, state, report.batches)
                        for row in counterfactual_rows(report, counterfactual, cfg):
                            append_jsonl(self.out / "probe.jsonl", {"round": r, **row})
                        del counterfactual
            except torch.cuda.OutOfMemoryError as error:
                # A probe must never end a run: record the failure and carry on with training.
                append_jsonl(self.out / "probe.jsonl", {"round": r, "scope": "error", "error": repr(error)[:300]})
                torch.cuda.empty_cache()
            finally:
                restore(b.params, state)
                for param in b.params.values():
                    param.grad = None
