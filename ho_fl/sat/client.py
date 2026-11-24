"""One participation of a satellite (brief v2, sections 1.3 and 3).

``train_participation`` is ``ho_fl.train.train_client`` for the satellite scenario: the same steps in the same
order (directions from the shared seed, restore the starting state, a fresh AdamW, E local steps, the difference
to the starting state), with the cut and the zeroth-order switch decided per participation by the policy instead
of per run by the configuration. For the cuts upstream supports the two are bitwise identical; the tests check it.
"""
import torch

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


def zo_packet(report, cfg):
    """What a zeroth-order report carries, in the form ``ho_fl.replay.replay_client`` reads: per local step the
    seed and the scalar of every direction. The seeds follow from the master seed, which every satellite derives
    from the shared seed, the round and the sender; on the wire the report is the header and the scalars."""
    return [{"step": e, "directions": [{"seed": stable_seed(report.direction_master_seed, e, p), "scalar": float(scalar)}
                                       for p, scalar in enumerate(step)]}
            for e, step in enumerate(report.scalars)]
