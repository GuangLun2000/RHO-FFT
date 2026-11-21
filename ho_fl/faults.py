"""Fault injection (brief section 6.6): devices that keep sending problematic reports for a while.

A fault specification is a JSON file::

    {"faults": [{"select": {"role": "preferred_t1", "count": 1}, "rounds": [41, 80],
                 "kind": "lr_scale", "params": {"scale": 20}}]}

- ``select``: ``{"by_id": [...]}`` or a role with a count. Roles: ``preferred_t1`` (T1 devices of the preferred
  set), ``ordinary_t1`` (the other T1 devices), ``t2``, ``t3``. The preferred set is always the one of
  ``dep_round_v1`` for the run's seed, whatever scheduler the run uses, and devices are taken in that sampler's
  ranking, so every method compared under one seed faces the same device.
- ``rounds``: first and last round (inclusive) in which the fault is active.
- ``kind`` and ``params``:

  ============  =========================================================  ================================
  training      ``label_noise``  rate, map (``shift`` or ``random``)       local labels are replaced
  training      ``lr_scale``     scale                                     local learning rate is multiplied
  training      ``stale``        lag                                       local training starts from the
                                                                           global state ``lag`` rounds ago
  report        ``noop``                                                   the update is all zeros
  report        ``sign_flip``    scale, layers (optional)                  the update is negated and scaled
  report        ``gauss``        scale, layers (optional)                  the update is replaced by noise of
                                                                           ``scale`` times its norm
  ============  =========================================================  ================================

  ``layers`` restricts a report-level fault to a segment (``"S0"`` to ``"S3"``, the segments between the tier
  cuts) or to a half-open range of layers ``[low, high]``. Timing faults (``late``) belong to the link model
  and are not implemented here.

Every faulty report carries its clean twin (``ClientReport.clean``): for report-level faults the update before the
transformation, for training-level faults the update of the same participation retrained without the fault. The
twin is what the influence of a fault on the aggregate is measured against. The list above is closed on purpose.
"""
import json
from dataclasses import replace
from pathlib import Path

import torch

from .aggregation import aggregate
from .config import SPECS
from .io import sha256
from .model import block_index
from .randomness import stable_seed
from .reports import layer_groups, segment_bounds, segment_index
from .scheduling import preferred_clients, tie_order, tiers_of

TRAINING_KINDS = {"label_noise", "lr_scale", "stale"}
REPORT_KINDS = {"noop", "sign_flip", "gauss"}
ROLES = ("preferred_t1", "ordinary_t1", "t2", "t3")


def role_members(role, resource_cuts, k, seed):
    """Clients of a role in the ranking of ``dep_round_v1`` (best-ranked first)."""
    tiers, preferred = tiers_of(resource_cuts), set(preferred_clients(resource_cuts, k, seed))
    wanted = {"preferred_t1": lambda cid: tiers[cid] == 0 and cid in preferred, "ordinary_t1": lambda cid: tiers[cid] == 0 and cid not in preferred,
              "t2": lambda cid: tiers[cid] == 1, "t3": lambda cid: tiers[cid] == 2}[role]
    return [cid for cid in tie_order(resource_cuts, seed) if wanted(cid)]


def layer_filter(selection, cfg):
    """Predicate on layer indices for the ``layers`` parameter of a report-level fault (``None`` selects every layer)."""
    if selection is None:
        return lambda layer: True
    if isinstance(selection, str):
        low, high = segment_bounds(SPECS[cfg.model][3], SPECS[cfg.model][2])[int(selection.upper().lstrip("S"))]
    else:
        low, high = selection
    return lambda layer: low <= layer < high


class Faults:
    """The faults of one run: which device is affected in which round, and what happens to it."""

    def __init__(self, cfg, resource_cuts):
        self.cfg = cfg
        raw = json.loads(Path(cfg.fault_spec).read_text())
        self.entries = []
        for entry in raw["faults"]:
            kind, params, select = entry["kind"], entry.get("params", {}), entry["select"]
            if kind == "late":
                raise NotImplementedError("Timing faults need the link model, which is not implemented yet")
            if kind not in TRAINING_KINDS | REPORT_KINDS:
                raise ValueError(f"Unknown fault kind {kind!r}")
            if "by_id" in select:
                clients = [int(cid) for cid in select["by_id"]]
            else:
                clients = role_members(select["role"], resource_cuts, cfg.k, cfg.seed)[:int(select.get("count", 1))]
            if not clients or any(not 0 <= cid < cfg.n for cid in clients):
                raise ValueError(f"Fault selection {select} matches no valid client")
            first, last = (int(value) for value in entry["rounds"])
            layer_filter(params.get("layers"), cfg)  # fails early on a malformed selection
            self.entries.append({"clients": clients, "first": first, "last": last, "kind": kind, "params": params})
        self.max_lag = max((int(e["params"]["lag"]) for e in self.entries if e["kind"] == "stale"), default=0)

    def metadata(self):
        return {"path": self.cfg.fault_spec, "sha256": sha256(self.cfg.fault_spec),
                "faults": [{**entry, "params": dict(entry["params"])} for entry in self.entries]}

    def active(self, r, cid, kinds):
        return [entry for entry in self.entries if entry["kind"] in kinds and entry["first"] <= r <= entry["last"] and cid in entry["clients"]]

    def before_local(self, r, cid):
        """Overrides of the local training of ``cid`` in round ``r``, or ``None``: lr_scale, label_map, lag, kinds."""
        entries = self.active(r, cid, TRAINING_KINDS)
        if not entries:
            return None
        context = {"kinds": [entry["kind"] for entry in entries]}
        for entry in entries:
            params = entry["params"]
            if entry["kind"] == "lr_scale":
                context["lr_scale"] = context.get("lr_scale", 1.0) * float(params["scale"])
            elif entry["kind"] == "stale":
                context["lag"] = int(params["lag"])
            else:
                context["label_map"] = self._label_map(params, cid, r)
        return context

    def _label_map(self, params, cid, r):
        rate, mode, classes, seed = float(params.get("rate", 1.0)), params.get("map", "shift"), len(self.cfg.verbalizers), self.cfg.seed
        if mode not in {"shift", "random"}:
            raise ValueError("label_noise map is shift or random")

        def apply(labels, step):
            generator = torch.Generator().manual_seed(stable_seed(seed, "fault_labels", cid, r, step))
            chosen = torch.rand(labels.shape, generator=generator) < rate
            if mode == "shift":
                replacement = (labels + 1) % classes
            else:
                replacement = torch.randint(0, classes, labels.shape, generator=generator).to(labels.device)
            return torch.where(chosen.to(labels.device), replacement, labels)

        return apply

    @torch.no_grad()
    def after_local(self, r, report):
        """Apply the report-level faults of this round to ``report``; the update before them becomes its clean twin."""
        entries = self.active(r, report.cid, REPORT_KINDS)
        if not entries:
            return report
        delta = dict(report.delta)
        for entry in entries:
            params = entry["params"]
            selected = layer_filter(params.get("layers"), self.cfg)
            names = [name for name in delta if selected(block_index(name))]
            if entry["kind"] == "noop":
                for name in names:
                    delta[name] = torch.zeros_like(delta[name])
            elif entry["kind"] == "sign_flip":
                for name in names:
                    delta[name] = delta[name] * -float(params.get("scale", 1.0))
            else:
                noise = {}
                for name in names:
                    generator = torch.Generator().manual_seed(stable_seed(self.cfg.seed, "fault_gauss", report.cid, r, name))
                    noise[name] = torch.randn(delta[name].shape, generator=generator).to(delta[name])
                target = float(params.get("scale", 1.0)) * sum(float(delta[name].double().square().sum()) for name in names) ** 0.5
                norm = sum(float(value.double().square().sum()) for value in noise.values()) ** 0.5
                for name in names:
                    delta[name] = noise[name] * (target / norm if norm > 0 else 0.0)
        kinds = ([report.fault] if report.fault else []) + [entry["kind"] for entry in entries]
        return replace(report, delta=delta, clean=report.clean if report.clean is not None else report.delta, fault="+".join(kinds))


@torch.no_grad()
def influence_rows(reports, update, cfg):
    """How far the faulty reports moved the aggregate, layer by layer (brief section 6.10).

    The aggregation is repeated with every faulty report replaced by its clean twin. Per layer:
    ``influence`` = norm of (aggregate with the faults - aggregate with the twins);
    ``reference`` = norm of the summed change of the faulty reports divided by K, which is exactly the influence
    under the equal-weight mean; ``relative`` = their ratio. Layers the faults did not touch are left out.
    """
    faulty = [index for index, report in enumerate(reports) if report.clean is not None]
    if not faulty:
        return []
    clean_update, _ = aggregate([replace(report, delta=report.clean, clean=None) if report.clean is not None else report for report in reports], None, cfg)
    bounds = segment_bounds(SPECS[cfg.model][3], SPECS[cfg.model][2])
    rows = []
    for layer, names in sorted(layer_groups(list(reports[0].delta)).items()):
        change = sum(float(sum(reports[index].delta[name].double() - reports[index].clean[name].double() for index in faulty).square().sum()) for name in names) ** 0.5
        if change == 0:
            continue
        squared = 0.0
        for name in names:
            with_fault = update.tensors[name].double() / update.divisor[name] if name in update.tensors else 0.0
            with_twin = clean_update.tensors[name].double() / clean_update.divisor[name] if name in clean_update.tensors else 0.0
            difference = with_fault - with_twin
            squared += float(difference.square().sum()) if torch.is_tensor(difference) else 0.0
        reference = change / cfg.k
        rows.append({"layer": layer, "segment": segment_index(layer, bounds), "clients": [reports[index].cid for index in faulty],
                     "fault": "+".join(sorted({reports[index].fault for index in faulty})), "influence": squared ** 0.5,
                     "reference": reference, "relative": squared ** 0.5 / reference})
    return rows
