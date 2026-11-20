"""Server-side aggregation of client reports.

Only the upstream rule is implemented so far (``mean_all``: every report votes on every tensor and the sum is divided
by the cohort size K). Its arithmetic is kept in the form of the upstream loop, because the order and the form of
the floating-point operations decide whether the baseline is reproduced bit for bit:

    total = 0;  total += delta_i  for each report, in cohort order;  state += total * (server_lr / K)
"""
from dataclasses import dataclass

import torch


@dataclass
class Update:
    tensors: dict    # tensor name -> accumulated update
    divisor: dict    # tensor name -> number the accumulated update is divided by when it is applied


@torch.no_grad()
def aggregate(reports, arrivals, cfg):
    """Combine ``reports`` into an ``Update`` and a log entry. ``arrivals`` is ``None`` when every report is on time."""
    if arrivals is not None:
        raise NotImplementedError("Partial arrivals are not supported yet")
    names = list(reports[0].delta)
    total = {name: torch.zeros_like(reports[0].delta[name]) for name in names}
    for report in reports:
        for name in names:
            total[name].add_(report.delta[name])
    log = {"electorate": "all", "stat": "mean", "weight": "cohort", "reports": len(reports)}
    return Update(total, {name: cfg.k for name in names}), log


@torch.no_grad()
def apply(state, update, server_lr=1.0):
    """Add ``update`` to ``state`` in place."""
    for name, tensor in update.tensors.items():
        state[name].add_(tensor, alpha=server_lr / update.divisor[name])
