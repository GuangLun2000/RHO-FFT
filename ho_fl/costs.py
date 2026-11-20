"""Compute cost of local training in forward-equivalents (brief section 6.7).

The unit is one forward pass through the whole stack of transformer blocks for one batch. Cost is taken to be
proportional to the number of blocks traversed; embedding, head, optimizer step and direction sampling are free. A
backward pass through a block costs ``backward_ratio`` block-forwards. This is an accounting model, not a
measurement: reports show measured seconds next to it.
"""
BACKWARD_RATIO = 2.0


def forward_equivalents(method, *, cut, depth, E, q, zo=True, backward_ratio=BACKWARD_RATIO):
    """Forward-equivalents of one participation (E local steps) of a client.

    hofl: one full forward, a backward over the ``depth - cut`` blocks above the cut and, unless the zeroth-order
    update is switched off, ``q`` forwards through the ``cut`` blocks below it. fedavg and fedprox: one forward and
    one full backward. decomfl: one unperturbed forward and two perturbed forwards per direction.
    """
    if method in {"fedavg", "fedprox"}:
        step = 1 + backward_ratio
    elif method == "decomfl":
        step = 1 + 2 * q
    elif method == "hofl":
        step = 1 + backward_ratio * (depth - cut) / depth + (q * cut / depth if zo else 0)
    else:
        raise ValueError(f"Unsupported method {method}")
    return E * step
