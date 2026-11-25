"""One participation of a satellite (brief v2, sections 1.3 and 3).

``train_participation`` is ``ho_fl.train.train_client`` for the satellite scenario: the same steps in the same
order (directions from the shared seed, restore the starting state, a fresh AdamW, E local steps, the difference
to the starting state), with the cut and the zeroth-order switch decided per participation by the policy instead
of per run by the configuration. For the cuts upstream supports the two are bitwise identical; the tests check it.

``probe_step`` is the zeroth-order part of a forward-only satellite under the variant ``line``: instead of an
estimate of the gradient along random directions it returns the loss at three points along the previous update.
"""
import torch
import torch.nn.functional as F

from ..algorithms import restore
from ..randomness import client_directions, stable_seed
from ..reports import ClientReport
from .config import CLASSES


def train_participation(b, state, r, sid, cut, zo=True):
    """Local training of satellite ``sid`` in round ``r`` from ``state`` with the blocks below ``cut`` forward-only.

    ``zo=False`` leaves those blocks untouched (their gradient stays ``None``, so AdamW applies no weight decay to
    them either). Advances the satellite's data stream and leaves its parameters in ``b.params``. The report holds
    the update of every trainable tensor, the zeroth-order scalars and the batches that were used.
    """
    cfg = b.cfg
    if cut not in b.bottoms:
        raise ValueError(f"Cut {cut} is not a cut of this task")
    directions = client_directions(cfg.seed, cfg.model, r, sid)
    restore(b.params, state)
    optimizer = b.optimizer()
    losses, scalars, batches = [], [], []
    for e in range(cfg.E):
        indices = b.streams[sid].next()
        loss, step_scalars = b.local_step(sid, indices, optimizer, directions, e, cut=cut, zo=zo)
        losses.append(loss)
        scalars.append(step_scalars)
        batches.append(indices)
    with torch.no_grad():
        delta = {name: param.detach() - state[name] for name, param in b.params.items()}
    kind = b.kinds[sid]
    return ClientReport(cid=sid, round=r, cut=cut, tier=CLASSES.index(kind) if kind in CLASSES else 0, delta=delta,
                        scalars=scalars, batches=batches, losses=losses, direction_master_seed=directions.master_seed,
                        zo=zo and cut > 0)


PROBE_POINTS = (0.0, 1.0, 2.0)   # how many previous global updates further the loss is evaluated: here, one, two


@torch.no_grad()
def probe_step(b, state, step, batches):
    """What a forward-only satellite measures for the step-length search (variant ``line``): the task loss of its
    own training batches at ``state``, at ``state + step`` and at ``state + 2 * step``, where ``step`` is the
    previous global update of its replica. Forward passes only; nothing is trained.

    Returns ``{"losses": [sum over the examples at each point], "examples": their number}``; a loss may be
    infinite or not a number when the model breaks that far out, which the aggregation reads as "do not go there".
    On a satellite the pass at ``state`` is the one its head is trained from, so the two others are the extra cost:
    the forward passes of q = 2 directions. Leaves ``state`` in ``b.params``.
    """
    losses = []
    for scale in PROBE_POINTS:
        for name, param in b.params.items():
            param.copy_(state[name])
            if scale and name in step:
                param.add_(step[name], alpha=scale)
        total = torch.zeros((), dtype=torch.float64, device=b.cfg.device)
        for indices in batches:
            batch = b.batch("train", indices)
            logits = b.wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=b.tokens)[:, -1].float()
            total += F.cross_entropy(logits, batch["labels"], reduction="sum").double()
        losses.append(total.item())
    restore(b.params, state)
    return {"losses": losses, "examples": sum(len(indices) for indices in batches)}


def zo_packet(report, cfg):
    """What a zeroth-order report carries, in the form ``ho_fl.replay.replay_client`` reads: per local step the
    seed and the scalar of every direction. The seeds follow from the master seed, which every satellite derives
    from the shared seed, the round and the sender; on the wire the report is the header and the scalars."""
    return [{"step": e, "directions": [{"seed": stable_seed(report.direction_master_seed, e, p), "scalar": float(scalar)}
                                       for p, scalar in enumerate(step)]}
            for e, step in enumerate(report.scalars)]
