"""Read-only probes of a training run (brief section 6.9, and the zeroth-order checks added after session 1).

- Layer statistics (``layer_stats``): for every round and layer, how many first-order and zeroth-order components
  there are, how large they are, how well they agree with each other, and how large the applied update is.
- Counterfactual first-order probe (``probe_rounds``): for every selected client, the local training is repeated
  from the same global state on the same batches with backpropagation through all trainable tensors. Comparing the
  client's actual update with this counterfactual, layer by layer, shows how much of the first-order direction the
  zeroth-order components carry.
- Attribution (``attribution_every``): the gradient of the loss on a fixed subset of the evaluation split gives, to
  first order, how much the applied update lowers that loss. For averaging rules the update is a sum of the
  reports' components, so the predicted decrease splits exactly into the part due to first-order components and
  the part due to zeroth-order components, segment by segment.
- Counterfactual dose probe (``dose_probe``, experiment E2): in a run on the cohorts of a dose schedule, every
  client except the designated unreliable one is retrained at the other of the two lowest cuts on the same batches,
  so that every dose m (honest first-order reports on segment S1 next to the unreliable one) can be formed from the
  same round. Every aggregation rule is then applied with and without a fault on that client's S1 component, and
  the probe records how far each rule moves the aggregate and how much it lowers the evaluation loss to first order.

Probes run between aggregation and the server step. They never advance a data stream, never touch the token
counters and never draw a random number, and they put the parameters back, so a run with probes reaches the same
parameters and writes the same logs as a run without them.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .aggregation import aggregate, plan_units
from .algorithms import optimizer_for, restore, task_loss
from .batches import collate
from .config import SPECS, parse_rounds
from .io import append_jsonl
from .model import block_index
from .randomness import client_directions, stable_seed
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
    for plan in plan_units(reports, arrivals, cfg, unit="layer"):
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
    plan = plan_units([report], None, cfg, unit="layer")
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


# --- retraining and the gradient of the evaluation loss ---------------------------------------------------------------

def replay_participation(b, state, report, cut=None):
    """The update of the participation behind ``report``, retrained from ``state`` on its recorded batches.

    ``cut`` overrides the cut (a device of one tier trained like a device of another); by default the report's own
    cut is used, with the configured learning rate and labels, which gives the clean twin of a training-level fault.
    No data stream advances, the token counters are put back and no global random number is drawn. The trained
    parameters are left in ``b.params``; callers restore them.
    """
    cfg, counters = b.cfg, (b.processed_tokens, b.answer_tokens)
    directions = client_directions(cfg.seed, cfg.model, report.round, report.cid)
    restore(b.params, state)
    optimizer = b.optimizer()
    for e, indices in enumerate(report.batches):
        b.local_step(report.cid, indices, optimizer, directions, e, state, cut=report.cut if cut is None else cut, zo=report.zo)
    with torch.no_grad():
        delta = {name: param.detach() - state[name] for name, param in b.params.items()}
    b.processed_tokens, b.answer_tokens = counters
    return delta


def validation_subset(b, cfg):
    """Fixed examples of the evaluation split on which probes measure the loss and its gradient (measurement only)."""
    count = len(b.data["validation"]["labels"])
    rng = np.random.default_rng(stable_seed(cfg.seed, cfg.model, cfg.dataset, "probe_validation"))
    return sorted(rng.choice(count, size=min(cfg.probe_val_examples, count), replace=False).tolist())


def validation_gradient(b, state, indices, names=None, from_layer=0):
    """Mean loss on ``indices`` of the evaluation split at ``state`` and its gradient for the tensors in ``names``.

    With ``from_layer`` the blocks below it run without a graph, so only tensors of that block or higher get a
    gradient (cheaper when only those are needed). Gradients are left as ``None`` afterwards.
    """
    cfg = b.cfg
    restore(b.params, state)
    for param in b.params.values():
        param.grad = None
    data, total = b.data["validation"], 0.0
    for start in range(0, len(indices), cfg.eval_batch_size):
        batch = collate(data, indices[start:start + cfg.eval_batch_size], b.tokenizer.pad_token_id, cfg.device)
        if from_layer:
            with torch.no_grad():
                hidden, context = b.wrapper.forward_prefix(batch["input_ids"], batch["attention_mask"], from_layer)
            logits = b.wrapper.forward_suffix(hidden, context, from_layer, last_only=True, token_ids=b.tokens)[:, -1].float()
        else:
            logits = b.wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=b.tokens)[:, -1].float()
        loss = F.cross_entropy(logits, batch["labels"], reduction="sum") / len(indices)
        loss.backward()
        total += float(loss.detach())
    grads = {name: param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
             for name, param in b.params.items() if names is None or name in names}
    for param in b.params.values():
        param.grad = None
    return grads, total


def _dot(a, b, names):
    return sum(float((a[name].double() * b[name].double()).sum()) for name in names)


# --- attribution of the predicted decrease to first-order and zeroth-order components ---------------------------------

@torch.no_grad()
def attribution_rows(reports, arrivals, update, cfg, grads, val_loss):
    """Predicted first-order decrease of the evaluation loss caused by the applied update, per segment and order.

    ``pred`` is minus the inner product of the evaluation-loss gradient with an update: the decrease that update
    produces to first order. For averaging rules (statistic mean, no clipping) the applied update is the sum of the
    members' components divided by the unit's divisor, so ``pred_fo + pred_zo = pred_total`` exactly. For robust
    statistics only the total is given.
    """
    bounds = segment_bounds(SPECS[cfg.model][3], SPECS[cfg.model][2])
    additive = cfg.agg_stat == "mean" and not cfg.norm_clip
    segments = {}
    for plan in plan_units(reports, arrivals, cfg, unit="layer"):
        entry = segments.setdefault(segment_index(plan.layer, bounds), {"layers": 0, "pred_total": 0.0, "pred_fo": 0.0, "pred_zo": 0.0,
                                                                         "norm2_total": 0.0, "norm2_fo": 0.0, "norm2_zo": 0.0, "norm2_g": 0.0, "n_fo": 0, "n_zo": 0})
        entry["layers"] += 1
        entry["norm2_g"] += sum(float(grads[name].double().square().sum()) for name in plan.names)
        applied = [name for name in plan.names if name in update.tensors]
        if not applied:
            continue
        scale = cfg.server_lr / update.divisor[applied[0]]
        total = {name: update.tensors[name].double() * scale for name in applied}
        entry["pred_total"] -= _dot(grads, total, applied)
        entry["norm2_total"] += sum(float(value.square().sum()) for value in total.values())
        if not additive:
            continue
        first_order = set(plan.first_order)
        entry["n_fo"] += sum(1 for index in plan.members if index in first_order)
        entry["n_zo"] += sum(1 for index in plan.members if index not in first_order)
        for order, members in (("fo", [i for i in plan.members if i in first_order]), ("zo", [i for i in plan.members if i not in first_order])):
            if not members:
                continue
            part = {name: sum(reports[i].delta[name].double() for i in members) * scale for name in applied}
            entry[f"pred_{order}"] -= _dot(grads, part, applied)
            entry[f"norm2_{order}"] += sum(float(value.square().sum()) for value in part.values())
    rows = []
    for segment, entry in sorted(segments.items()):
        norm_g = entry.pop("norm2_g") ** 0.5
        row = {"segment": segment, "layers": entry.pop("layers"), "decomposed": additive, "val_loss": val_loss, "norm_g": norm_g}
        for key in ("total", "fo", "zo"):
            norm = entry[f"norm2_{key}"] ** 0.5
            pred = entry[f"pred_{key}"] if key == "total" or additive else None
            row.update({f"pred_{key}": pred, f"norm_{key}": norm if key == "total" or additive else None,
                        f"cos_{key}": pred / (norm * norm_g) if pred is not None and norm > 0 and norm_g > 0 else None})
        if additive:
            row.update(n_fo=entry["n_fo"] / row["layers"], n_zo=entry["n_zo"] / row["layers"])
        rows.append(row)
    return rows


# --- counterfactual dose probe (experiment E2) ------------------------------------------------------------------------

@torch.no_grad()
def dose_rows(b, state, reports, cfg, indices):
    """Every dose, aggregation rule and fault strength of experiment E2, formed from the reports of one round.

    The cohort must be laid out as in a dose schedule: the unreliable device first, at the lowest cut c1, followed
    by devices at c1 or at the second cut c2. Each of those is retrained at the other cut on the same batches, so
    both versions exist for every position. Dose m uses the c1 version for the first m positions and the c2 version
    for the rest. Only segment S1 = [c1, c2) differs between doses, so only S1 is aggregated and measured.
    """
    from .dose import COUNTERFACTUAL_FAULTS, COUNTERFACTUAL_RULES

    c1, c2 = sorted(set(SPECS[cfg.model][3]))[:2]
    if len(reports) != cfg.k or reports[0].cut != c1 or any(report.cut not in (c1, c2) for report in reports[1:]):
        raise ValueError(f"dose_probe needs the cohort of a dose schedule: the unreliable device at cut {c1}, the others at {c1} or {c2}")
    names = [name for name in reports[0].delta if c1 <= block_index(name) < c2]
    versions = []
    for report in reports[1:]:
        other = c2 if report.cut == c1 else c1
        with torch.enable_grad():
            alternative = replay_participation(b, state, report, cut=other)
        versions.append({report.cut: {name: report.delta[name] for name in names}, other: {name: alternative[name] for name in names}})
        del alternative
    with torch.enable_grad():
        grads, val_loss = validation_gradient(b, state, indices, names=set(names), from_layer=c1)
    norm_g = _dot(grads, grads, names) ** 0.5
    device = {name: reports[0].delta[name] for name in names}
    norm_device = _dot(device, device, names) ** 0.5
    fo_preds = [-_dot(grads, device, names)] + [-_dot(grads, version[c1], names) for version in versions]
    zo_preds = [-_dot(grads, version[c2], names) for version in versions]
    rows = [{"kind": "reference", "val_loss": val_loss, "norm_g": norm_g, "norm_device": norm_device,
             "pred_fo_report": float(np.mean(fo_preds)), "pred_zo_report": float(np.mean(zo_preds)),
             "pred_fo_reports": fo_preds, "pred_zo_reports": zo_preds}]
    for m in range(cfg.k):
        cohort = [(c1 if position < m else c2, versions[position][c1 if position < m else c2]) for position in range(cfg.k - 1)]
        clean = {}
        for fault, scale in COUNTERFACTUAL_FAULTS.items():
            first = device if not scale else {name: device[name] * -scale for name in names}
            members = [SimpleNamespace(cut=c1, delta=first)] + [SimpleNamespace(cut=cut, delta=delta) for cut, delta in cohort]
            for rule, fields in COUNTERFACTUAL_RULES.items():
                update, _ = aggregate(members, None, replace(cfg, **fields))
                applied = {name: update.tensors[name].double() * (cfg.server_lr / update.divisor[name]) if name in update.tensors
                           else torch.zeros_like(device[name], dtype=torch.float64) for name in names}
                pred, norm = -_dot(grads, applied, names), _dot(applied, applied, names) ** 0.5
                row = {"kind": "candidate", "m": m, "rule": rule, "fault": fault, "pred": pred, "norm": norm,
                       "cos": pred / (norm * norm_g) if norm > 0 and norm_g > 0 else None, "influence": None}
                if not scale:
                    clean[rule] = applied
                elif norm_device > 0:
                    shift = {name: applied[name] - clean[rule][name] for name in names}
                    row["influence"] = _dot(shift, shift, names) ** 0.5 / ((1 + scale) * norm_device / cfg.k)
                rows.append(row)
    return rows


class Probes:
    """The probes of one run, switched on by ``cfg.layer_stats``, ``cfg.probe_rounds``, ``cfg.attribution_every``
    and ``cfg.dose_probe``."""

    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, Path(out)
        self.rounds = set(parse_rounds(cfg.probe_rounds))
        self.indices = None

    def attribution_due(self, r):
        return bool(self.cfg.attribution_every) and (r - 1) % self.cfg.attribution_every == 0

    def run(self, r, b, state, reports, arrivals, update):
        cfg = self.cfg
        if cfg.layer_stats:
            for row in layer_stats(reports, arrivals, update, cfg):
                append_jsonl(self.out / "layers.jsonl", {"round": r, **row})
        if not (r in self.rounds or self.attribution_due(r) or cfg.dose_probe):
            return
        if self.indices is None and (cfg.attribution_every or cfg.dose_probe):
            self.indices = validation_subset(b, cfg)
        current = "probe.jsonl"
        try:
            if r in self.rounds:
                for report in reports:
                    if report.fault is None:
                        counterfactual = first_order_delta(b, state, report.batches)
                        for row in counterfactual_rows(report, counterfactual, cfg):
                            append_jsonl(self.out / "probe.jsonl", {"round": r, **row})
                        del counterfactual
            if self.attribution_due(r):
                current = "attribution.jsonl"
                grads, val_loss = validation_gradient(b, state, self.indices)
                for row in attribution_rows(reports, arrivals, update, cfg, grads, val_loss):
                    append_jsonl(self.out / "attribution.jsonl", {"round": r, **row})
                del grads
            if cfg.dose_probe:
                current = "dose_probe.jsonl"
                for row in dose_rows(b, state, reports, cfg, self.indices):
                    append_jsonl(self.out / "dose_probe.jsonl", {"round": r, **row})
        except torch.cuda.OutOfMemoryError as error:
            # A probe must never end a run: record the failure and carry on with training.
            append_jsonl(self.out / current, {"round": r, "scope": "error", "kind": "error", "error": repr(error)[:300]})
            torch.cuda.empty_cache()
        finally:
            restore(b.params, state)
            for param in b.params.values():
                param.grad = None
