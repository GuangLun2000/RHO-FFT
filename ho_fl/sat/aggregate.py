"""Replica groups and the dual-channel aggregation (brief v2, sections 4.3 and 6.2).

Every orbital plane holds a replica of the trainable state. Planes whose replicas are bitwise the same form one
replica group and share one set of tensors; in a run without faults there is a single group all the time. A group
is never updated in place: applying an update writes new tensors, so a state shared by several planes (or handed
out to a caller) cannot change behind anyone's back.

The update of a group in a round is

    state <- state + server_lr / |A| * sum over the participants i in A, in ascending satellite number, of term_i

where ``term_i`` is, tensor by tensor, the participant's first-order update for the blocks at or above its cut (if
it arrived in time) and, for the blocks below its cut, the zeroth-order update rebuilt on this group's own state
from the seeds and scalars of the report (``ho_fl.replay.replay_client``). ``A`` is everyone whose report header
arrived, late first-order updates included, which is the divide-by-K rule of HO-FL.

The first-order tensors are the participants' own, so they are bitwise what a central server would sum; the
replayed tensors agree with what the participant computed locally within the tolerance of the replay.
"""
from collections import Counter
from dataclasses import dataclass

import torch

from ..aggregation import Update, apply
from ..model import block_index
from ..replay import replay_client
from .client import zo_packet


class Groups:
    """The replicas of the planes, kept as groups of planes with identical state."""

    def __init__(self, state, planes):
        self.planes = planes
        self.states = {0: state}
        self.group_of = [0] * planes
        self._next = 1

    @property
    def ids(self):
        return sorted(self.states)

    def of_plane(self, plane):
        return self.group_of[plane]

    def state(self, group):
        return self.states[group]

    def members(self, group):
        return [plane for plane in range(self.planes) if self.group_of[plane] == group]

    def split(self, group, parts):
        """Divide ``group`` into the given lists of planes; the first part keeps the group number. The new groups
        start from the same tensors, which is safe because updates never write in place."""
        assert sorted(plane for part in parts for plane in part) == self.members(group)
        numbers = [group]
        for part in parts[1:]:
            number, self._next = self._next, self._next + 1
            self.states[number] = self.states[group]
            for plane in part:
                self.group_of[plane] = number
            numbers.append(number)
        return numbers

    def merge_identical(self):
        """Join groups whose states are bitwise the same (for example after a resynchronisation)."""
        for group in self.ids:
            for other in self.ids:
                if other <= group or group not in self.states or other not in self.states:
                    continue
                a, b = self.states[group], self.states[other]
                if all(torch.equal(a[name], b[name]) for name in a):
                    self.group_of = [group if g == other else g for g in self.group_of]
                    del self.states[other]

    @torch.no_grad()
    def apply(self, updates, server_lr):
        """Give every group in ``updates`` a new state: its old one plus its update (``None`` leaves it as it is)."""
        for group, update in updates.items():
            if update is None:
                continue
            state = {name: tensor.clone() for name, tensor in self.states[group].items()}
            apply(state, update, server_lr)
            self.states[group] = state


@dataclass
class Term:
    sid: int
    first_order: dict    # tensor name -> the participant's own update (blocks at or above its cut)
    zeroth_order: dict   # tensor name -> the update replayed on the group's state (blocks below its cut)


@torch.no_grad()
def replay_zeroth_order(state, report, cfg):
    """The zeroth-order update of ``report`` rebuilt on ``state`` from its seeds and scalars: tensors of the blocks
    below the report's cut. The first-order part plays no role in it, so zeros stand in for it."""
    suffix = {name: tensor for name, tensor in state.items() if block_index(name) >= report.cut}
    zeros = {name: torch.zeros_like(tensor) for name, tensor in suffix.items()}
    with torch.enable_grad():
        final, _ = replay_client(state, report.cut, zo_packet(report, cfg), zeros, cfg)
    return {name: final[name] - state[name] for name in state if name not in suffix}


def round_terms(state, reports, delivery, cfg):
    """What every counted participant contributes to the group holding ``state``, in ascending satellite number."""
    terms = []
    for sid in sorted(reports):
        if sid not in delivery.beacons:
            continue
        report = reports[sid]
        first_order, zeroth_order = {}, {}
        if sid in delivery.on_time:
            first_order = {name: tensor for name, tensor in report.delta.items() if block_index(name) >= report.cut}
        if report.zo and sid in delivery.zo_received:
            zeroth_order = replay_zeroth_order(state, report, cfg)
        terms.append(Term(sid, first_order, zeroth_order))
    return terms


@torch.no_grad()
def sum_terms(state, terms):
    """The ``Update`` of a group: per tensor the sum of the terms in the order given, to be divided by their number.
    ``None`` when nobody is counted. The form is the one of ``ho_fl.aggregation.aggregate``: start from zeros and
    add one participant after the other."""
    if not terms:
        return None
    total = {name: torch.zeros_like(tensor) for name, tensor in state.items()}
    for term in terms:
        for part in (term.first_order, term.zeroth_order):
            for name, tensor in part.items():
                total[name].add_(tensor)
    return Update(total, {name: len(terms) for name in total})


def round_updates(groups, reports, delivery, cfg):
    """``{group: Update}`` for the round and, per group, the terms it was built from."""
    updates, terms = {}, {}
    for group in groups.ids:
        terms[group] = round_terms(groups.state(group), reports, delivery, cfg)
        updates[group] = sum_terms(groups.state(group), terms[group])
    return updates, terms


@torch.no_grad()
def channel_statistics(terms):
    """How the two channels share the update of a round, block by block. A diagnostic: it reads the terms that
    ``round_terms`` built (with accumulators of its own) and changes nothing of the update.

    For every block that received anything (``blocks``, ascending; the head of a vision task is block ``depth``):

    ``fo_n``, ``zo_n``        how many of the ``counted`` participants sent a first-order / a zeroth-order update
    ``fo_norm``, ``zo_norm``  the norms of the two parts of the mean update: each sum divided by the number of
                              counted participants, as the update itself is
    ``cos``                   the cosine between the two sums: above zero when the zeroth-order updates of the
                              satellites below their cut point the way the first-order updates of the others
                              do; ``None`` where a part is missing or zero

    ``None`` when nobody is counted. The divisor of the update counts every participant, so a block to which few
    of them send a first-order update gets a correspondingly smaller first-order step: ``fo_n / counted`` says
    how much smaller, and the two norms say what the zeroth-order updates put in its place.
    """
    if not terms:
        return None
    sums, numbers = ({}, {}), (Counter(), Counter())   # first-order, zeroth-order
    for term in terms:
        for part, total, number in zip((term.first_order, term.zeroth_order), sums, numbers):
            for name, tensor in part.items():
                if name not in total:
                    total[name] = torch.zeros_like(tensor, dtype=torch.float32)
                total[name].add_(tensor)
            number.update({block_index(name) for name in part})
    names = sorted(set(sums[0]) | set(sums[1]))
    blocks = sorted({block_index(name) for name in names})
    if not names:
        return None
    zero = torch.zeros((), dtype=torch.float32, device=next(iter({**sums[1], **sums[0]}.values())).device)
    rows = []
    for block in blocks:
        members = [name for name in names if block_index(name) == block]
        first = sum((sums[0][name].square().sum() for name in members if name in sums[0]), zero)
        zeroth = sum((sums[1][name].square().sum() for name in members if name in sums[1]), zero)
        dot = sum(((sums[0][name] * sums[1][name]).sum() for name in members if name in sums[0] and name in sums[1]), zero)
        rows.append(torch.stack([first, zeroth, dot]))
    table = torch.stack(rows).double().cpu().tolist()   # one synchronisation with the device
    count = len(terms)
    return {"counted": count, "blocks": blocks,
            "fo_n": [numbers[0].get(block, 0) for block in blocks], "zo_n": [numbers[1].get(block, 0) for block in blocks],
            "fo_norm": [first ** 0.5 / count for first, _, _ in table], "zo_norm": [zeroth ** 0.5 / count for _, zeroth, _ in table],
            "cos": [dot / (first * zeroth) ** 0.5 if first > 0 and zeroth > 0 else None for first, zeroth, dot in table]}


@torch.no_grad()
def replica_difference(groups, initial):
    """How far the replicas of the planes are from their mean, relative to how far the mean has moved from the
    initial state: ``sqrt(mean_p ||theta_p - mean||^2) / ||mean - theta_0||``. Zero for a single group."""
    if len(groups.ids) == 1:
        return 0.0
    weights = {group: len(groups.members(group)) / groups.planes for group in groups.ids}
    spread = moved = 0.0
    for name, start in initial.items():
        mean = sum(weights[group] * groups.state(group)[name].double() for group in groups.ids)
        spread += sum(weights[group] * float((groups.state(group)[name].double() - mean).square().sum()) for group in groups.ids)
        moved += float((mean - start.double()).square().sum())
    return (spread / moved) ** 0.5 if moved > 0 else (0.0 if spread == 0 else None)
