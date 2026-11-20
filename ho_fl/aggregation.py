"""Server-side aggregation of client reports.

The upstream rule (``mean_all``: every report votes on every tensor and the sum is divided by the cohort size K)
keeps the arithmetic form of the upstream loop, because the order and the form of the floating-point operations
decide whether the baseline is reproduced bit for bit:

    total = 0;  total += delta_i  for each report, in cohort order;  state += total * (server_lr / K)

Every other configuration goes through a plan: for each layer (the LoRA tensors of one block) the reports that
have a component there (A), those whose component is first-order (F), and the electorate that votes. A report's
component on a layer is first-order when the layer is at or above the report's cut. With ``zo_mode=off`` the
zeroth-order components do not exist, so A equals F. Implemented statistics: the mean over the electorate.
"""
from dataclasses import dataclass

import torch

from .reports import is_first_order, layer_groups


@dataclass
class Update:
    tensors: dict    # tensor name -> accumulated update; tensors of units that are not updated are absent
    divisor: dict    # tensor name -> number the accumulated update is divided by when it is applied


@dataclass
class UnitPlan:
    layer: int
    names: list          # tensor names of the unit
    arrived: list        # indices (positions in the cohort) of the reports with a component on this unit: A
    first_order: list    # those whose component is first-order: F
    members: list        # the electorate
    rule: str            # all | fo | thin_all | dilute | hold | empty


def plan_units(reports, arrivals, cfg):
    """Who votes on each layer. Pure: reads the cut of each report and the configuration, never tensor values."""
    if arrivals is not None:
        raise NotImplementedError("Partial arrivals are not supported yet")
    groups = layer_groups(list(reports[0].delta))
    plans = []
    for layer in sorted(groups):
        first_order = [index for index, report in enumerate(reports) if is_first_order(report.cut, layer)]
        arrived = first_order if cfg.zo_mode == "off" else list(range(len(reports)))
        if not arrived:
            members, rule = [], "empty"
        elif cfg.agg_electorate == "all":
            members, rule = arrived, "all"
        elif len(first_order) >= cfg.fo_quorum:
            members, rule = first_order, "fo"
        elif cfg.thin_policy == "all":
            members, rule = arrived, "thin_all"
        elif cfg.thin_policy == "dilute" and first_order:
            members, rule = first_order, "dilute"
        else:
            members, rule = [], "hold"
        plans.append(UnitPlan(layer, groups[layer], arrived, first_order, members, rule))
    return plans


@torch.no_grad()
def aggregate(reports, arrivals, cfg):
    """Combine ``reports`` into an ``Update`` and a log entry. ``arrivals`` is ``None`` when every report is on time."""
    if arrivals is not None:
        raise NotImplementedError("Partial arrivals are not supported yet")
    if cfg.agg_electorate != "all" or cfg.agg_weight != "cohort":
        return _aggregate_planned(reports, arrivals, cfg)
    names = list(reports[0].delta)
    total = {name: torch.zeros_like(reports[0].delta[name]) for name in names}
    for report in reports:
        for name in names:
            total[name].add_(report.delta[name])
    log = {"electorate": "all", "stat": "mean", "weight": "cohort", "reports": len(reports)}
    return Update(total, {name: cfg.k for name in names}), log


def _aggregate_planned(reports, arrivals, cfg):
    """Mean over the electorate of each layer, accumulated in cohort order like the default path."""
    tensors, divisor, units = {}, {}, []
    for plan in plan_units(reports, arrivals, cfg):
        # ``dilute`` keeps the weight 1/K of the cohort whatever agg_weight says: sum of the first-order reports over K.
        count = cfg.k if cfg.agg_weight == "cohort" or plan.rule == "dilute" else len(plan.members)
        units.append({"layer": plan.layer, "arrived": len(plan.arrived), "first_order": len(plan.first_order),
                      "rule": plan.rule, "members": len(plan.members), "divisor": count if plan.members else None})
        if not plan.members:
            continue
        for name in plan.names:
            total = torch.zeros_like(reports[0].delta[name])
            for index in plan.members:
                total.add_(reports[index].delta[name])
            tensors[name], divisor[name] = total, count
    log = {"electorate": cfg.agg_electorate, "stat": "mean", "weight": cfg.agg_weight, "reports": len(reports), "units": units}
    return Update(tensors, divisor), log


@torch.no_grad()
def apply(state, update, server_lr=1.0):
    """Add ``update`` to ``state`` in place."""
    for name, tensor in update.tensors.items():
        state[name].add_(tensor, alpha=server_lr / update.divisor[name])
