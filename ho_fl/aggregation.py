"""Server-side aggregation of client reports (brief section 6.4).

The upstream rule (``mean_all``: every report votes on every tensor and the sum is divided by the cohort size K)
keeps the arithmetic form of the upstream loop, because the order and the form of the floating-point operations
decide whether the baseline is reproduced bit for bit:

    total = 0;  total += delta_i  for each report, in cohort order;  state += total * (server_lr / K)

Every other configuration goes through a plan: for each unit (by default a layer, i.e. the LoRA tensors of one
block) the reports that have a component there (A), those whose component is first-order (F), and the electorate
that votes. A report's component on a layer is first-order when the layer is at or above the report's cut. With
``zo_mode=off`` the zeroth-order components do not exist, so A equals F.

Statistics over the electorate: the mean (accumulated in cohort order, like the default path), the geometric
median, the coordinate-wise median and the coordinate-wise trimmed mean. The robust statistics return a typical
member, so they are applied with weight one; only the mean has a cohort or a count weight.
"""
from dataclasses import dataclass

import torch

from .config import SPECS
from .model import block_index
from .reports import is_first_order, layer_groups, segment_bounds, segment_groups


@dataclass
class Update:
    tensors: dict    # tensor name -> accumulated update; tensors of units that are not updated are absent
    divisor: dict    # tensor name -> number the accumulated update is divided by when it is applied


@dataclass
class UnitPlan:
    layer: int           # lowest layer of the unit
    names: list          # tensor names of the unit
    arrived: list        # indices (positions in the cohort) of the reports with a component on this unit: A
    first_order: list    # those whose component is first-order: F
    members: list        # the electorate
    rule: str            # all | fo | thin_all | dilute | hold | empty


def unit_groups(names, cfg, unit=None):
    """``[(lowest layer, tensor names)]`` of the aggregation units: layers, the segments between cuts, or everything."""
    unit = unit or cfg.agg_unit
    if unit == "layer":
        groups = layer_groups(names)
        return [(layer, groups[layer]) for layer in sorted(groups)]
    if unit == "segment":
        cuts, depth = SPECS[cfg.model][3], SPECS[cfg.model][2]
        groups, bounds = segment_groups(names, cuts, depth), segment_bounds(cuts, depth)
        return [(bounds[index][0], groups[index]) for index in sorted(groups) if groups[index]]
    return [(min(block_index(name) for name in names), list(names))]


def plan_units(reports, arrivals, cfg, unit=None):
    """Who votes on each unit. Pure: reads the cut of each report and the configuration, never tensor values.

    ``unit`` overrides ``cfg.agg_unit`` (the probes always look layer by layer).
    """
    if arrivals is not None:
        raise NotImplementedError("Partial arrivals are not supported yet")
    plans = []
    for layer, names in unit_groups(list(reports[0].delta), cfg, unit):
        # Units never straddle a cut except the whole model, where a report is first-order only if all of it is.
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
        plans.append(UnitPlan(layer, names, arrived, first_order, members, rule))
    return plans


# --- statistics over the rows of a matrix (one row per member, float64) ------------------------------------------------

def geometric_median(matrix, iterations=100, tolerance=1e-6):
    """Smoothed Weiszfeld iteration started at the mean; with one or two rows the mean itself."""
    center = matrix.mean(0)
    if matrix.shape[0] <= 2:
        return center
    floor = 1e-10 * float(matrix.norm(dim=1).mean()) + 1e-300
    for _ in range(iterations):
        weights = 1.0 / (matrix - center).norm(dim=1).clamp_min(floor)
        updated = (weights[:, None] * matrix).sum(0) / weights.sum()
        change = float((updated - center).norm()) / max(float(center.norm()), floor)
        center = updated
        if change < tolerance:
            break
    return center


def coordinate_median(matrix):
    """Coordinate-wise median; with an even number of rows the mean of the two middle values (not the lower one)."""
    ordered = matrix.sort(dim=0).values
    rows = matrix.shape[0]
    return ordered[rows // 2] if rows % 2 else (ordered[rows // 2 - 1] + ordered[rows // 2]) / 2


def trimmed_mean(matrix):
    """Coordinate-wise mean without the largest and the smallest value; with fewer than three rows the mean."""
    rows = matrix.shape[0]
    if rows < 3:
        return matrix.mean(0)
    return (matrix.sum(0) - matrix.max(dim=0).values - matrix.min(dim=0).values) / (rows - 2)


STATISTICS = {"geomed": geometric_median, "cwmed": coordinate_median, "trimmed": trimmed_mean}


def _unit_vector(report, names):
    return torch.cat([report.delta[name].reshape(-1) for name in names]).double()


def clip_scales(reports, plan, limit):
    """Factor per arrived report that brings its component on the unit down to ``limit`` times the median norm."""
    norms = {index: float(_unit_vector(reports[index], plan.names).norm()) for index in plan.arrived}
    ordered = sorted(norms.values())
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {index: min(1.0, limit * median / norm) if norm > 0 else 1.0 for index, norm in norms.items()}


@torch.no_grad()
def aggregate(reports, arrivals, cfg):
    """Combine ``reports`` into an ``Update`` and a log entry. ``arrivals`` is ``None`` when every report is on time."""
    if arrivals is not None:
        raise NotImplementedError("Partial arrivals are not supported yet")
    upstream = (cfg.agg_electorate == "all" and cfg.agg_weight == "cohort" and cfg.agg_stat == "mean"
                and cfg.agg_unit == "layer" and not cfg.norm_clip)
    if not upstream:
        return _aggregate_planned(reports, arrivals, cfg)
    names = list(reports[0].delta)
    total = {name: torch.zeros_like(reports[0].delta[name]) for name in names}
    for report in reports:
        for name in names:
            total[name].add_(report.delta[name])
    log = {"electorate": "all", "stat": "mean", "weight": "cohort", "reports": len(reports)}
    return Update(total, {name: cfg.k for name in names}), log


def _aggregate_planned(reports, arrivals, cfg):
    """Statistic over the electorate of each unit; the mean is accumulated in cohort order like the default path."""
    tensors, divisor, units = {}, {}, []
    for plan in plan_units(reports, arrivals, cfg):
        # ``dilute`` is the sum of the first-order reports over K whatever statistic and weight are configured.
        robust = cfg.agg_stat != "mean" and plan.rule != "dilute"
        count = 1 if robust else (cfg.k if cfg.agg_weight == "cohort" or plan.rule == "dilute" else len(plan.members))
        units.append({"layer": plan.layer, "arrived": len(plan.arrived), "first_order": len(plan.first_order),
                      "rule": plan.rule, "members": len(plan.members), "divisor": count if plan.members else None})
        if not plan.members:
            continue
        scales = clip_scales(reports, plan, cfg.norm_clip) if cfg.norm_clip > 0 else None
        if robust:
            rows = [_unit_vector(reports[index], plan.names) * (scales[index] if scales else 1.0) for index in plan.members]
            center, offset = STATISTICS[cfg.agg_stat](torch.stack(rows)), 0
            for name in plan.names:
                like = reports[0].delta[name]
                tensors[name], divisor[name] = center[offset:offset + like.numel()].reshape(like.shape).to(like.dtype), 1
                offset += like.numel()
            continue
        for name in plan.names:
            total = torch.zeros_like(reports[0].delta[name])
            for index in plan.members:
                if scales is None:
                    total.add_(reports[index].delta[name])
                else:
                    total.add_(reports[index].delta[name], alpha=scales[index])
            tensors[name], divisor[name] = total, count
    log = {"electorate": cfg.agg_electorate, "stat": cfg.agg_stat, "weight": cfg.agg_weight, "unit": cfg.agg_unit,
           "reports": len(reports), "units": units}
    return Update(tensors, divisor), log


@torch.no_grad()
def apply(state, update, server_lr=1.0):
    """Add ``update`` to ``state`` in place."""
    for name, tensor in update.tensors.items():
        state[name].add_(tensor, alpha=server_lr / update.divisor[name])
