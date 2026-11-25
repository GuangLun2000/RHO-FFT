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

That is the rule of every method as the brief defines it, and its code path (``round_updates``, ``Groups.apply``)
is untouched by what follows. The variants of ``ho_fl.sat.methods`` change the rule (``weigh_terms``,
``advance``): the first-order sum of a tensor can be divided by the participants that sent one, the zeroth-order
part can be weighted or left out, and with the step-length search the update of a round is extended along the
previous one,

    step <- first-order part + beta * previous step,      state <- state + step,

where ``beta`` in [0, 1] follows from the losses the forward-only satellites measured along the previous step
(``step_length``, ``extension``): a momentum whose coefficient is measured every round on data the first-order
satellites never saw. A group keeps its previous step next to its state.
"""
import math
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
        self.steps = {0: None}   # the last update a group applied, tensor by tensor; kept by ``place`` only
        self.group_of = [0] * planes
        self._next = 1

    @property
    def ids(self):
        return sorted(self.states)

    def of_plane(self, plane):
        return self.group_of[plane]

    def state(self, group):
        return self.states[group]

    def step(self, group):
        """The update the group applied in its last round, or ``None`` (no round yet, or a method that keeps none)."""
        return self.steps[group]

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
            self.steps[number] = self.steps[group]
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
                if all(torch.equal(a[name], b[name]) for name in a) and _same_step(self.steps[group], self.steps[other]):
                    self.group_of = [group if g == other else g for g in self.group_of]
                    del self.states[other], self.steps[other]

    @torch.no_grad()
    def apply(self, updates, server_lr):
        """Give every group in ``updates`` a new state: its old one plus its update (``None`` leaves it as it is)."""
        for group, update in updates.items():
            if update is None:
                continue
            state = {name: tensor.clone() for name, tensor in self.states[group].items()}
            apply(state, update, server_lr)
            self.states[group] = state

    def place(self, group, state, step=None):
        """Give ``group`` the new state ``state`` (tensors of its own) and remember the step that led to it."""
        self.states[group] = state
        self.steps[group] = step


def _same_step(a, b):
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


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


def round_terms(state, reports, delivery, cfg, replay=True):
    """What every counted participant contributes to the group holding ``state``, in ascending satellite number.
    ``replay=False`` leaves the zeroth-order parts out without rebuilding them (a method that gives them no weight)."""
    terms = []
    for sid in sorted(reports):
        if sid not in delivery.beacons:
            continue
        report = reports[sid]
        first_order, zeroth_order = {}, {}
        if sid in delivery.on_time:
            first_order = {name: tensor for name, tensor in report.delta.items() if block_index(name) >= report.cut}
        if replay and report.zo and sid in delivery.zo_received:
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
def weigh_terms(state, terms, method):
    """The ``Update`` of a group under the switches of ``method`` (``None`` when nobody is counted).

    Per tensor, with ``F`` the sum of the first-order updates, ``n`` the number of participants that sent one,
    ``Z`` the sum of the zeroth-order updates, ``N`` the number of counted participants and ``w`` the weight of the
    zeroth-order part:

        fo_average "all"                       (F + w Z) / N              the rule of HO-FL when w = 1
        fo_average "senders", n > 0            F / n + w Z / N            written as (F + (w n / N) Z) / n
        fo_average "senders", n = 0            w Z / N

    A tensor nobody sent anything for is absent from the update. The sums are accumulated from zeros in the order
    of the terms, like ``sum_terms``, so where every participant sends a first-order update (n = N, no Z) the
    result is bitwise the one of the default rule.
    """
    if not terms:
        return None
    count, first, zeroth, senders = len(terms), {}, {}, Counter()
    for term in terms:
        for name, tensor in term.first_order.items():
            if name not in first:
                first[name] = torch.zeros_like(state[name])
            first[name].add_(tensor)
            senders[name] += 1
        if method.zo_weight:
            for name, tensor in term.zeroth_order.items():
                if name not in zeroth:
                    zeroth[name] = torch.zeros_like(state[name])
                zeroth[name].add_(tensor)
    tensors, divisor = {}, {}
    for name in state:
        own = method.fo_average == "senders" and senders[name] > 0
        number = senders[name] if own else count
        total = first.get(name)
        if name in zeroth:
            weight = method.zo_weight * (number / count if own else 1.0)
            total = zeroth[name].mul_(weight) if total is None else total.add_(zeroth[name], alpha=weight)
        if total is not None:
            tensors[name], divisor[name] = total, number
    return Update(tensors, divisor)


MAX_STEPS = 2.0   # the farthest point a forward-only satellite probes, in previous global updates: the search never goes beyond it


def step_length(losses):
    """How many previous global updates further the pooled loss of the forward-only satellites is smallest, from
    its values at 0, 1 and 2 of them (``client.PROBE_POINTS``): ``(alpha, case)`` with ``alpha`` in [0, 2].

    A parabola through the three values has the slope ``g = (-3 L0 + 4 L1 - L2) / 2`` at 0 and the curvature
    ``c = L2 - 2 L1 + L0``. Going on pays only downhill (g < 0); the minimum is at ``-g / c`` when the parabola
    opens upwards, and beyond the last probe otherwise, where the search stops at the last probe: it never
    extrapolates past a point somebody measured. Values that are not finite, or no values, give 0.
    """
    if losses is None or len(losses) != 3 or not all(math.isfinite(value) for value in losses):
        return 0.0, "no valid probes"
    start, one, two = losses
    slope, curvature = (-3.0 * start + 4.0 * one - two) / 2.0, two - 2.0 * one + start
    if slope >= 0:
        return 0.0, "uphill"
    if curvature <= 0 or -slope / curvature >= MAX_STEPS:
        return MAX_STEPS, "minimum beyond the last probe"
    return -slope / curvature, "minimum between the probes"


@torch.no_grad()
def extension(fresh, previous, alpha):
    """``(beta, kappa)``: how much of the previous step is added to the fresh one, and how far the fresh step
    already reaches along the previous one, ``kappa = <fresh, previous> / <previous, previous>``.

    The probes say the loss along the previous step is smallest ``alpha`` of it further. The fresh first-order
    step, computed at the same point, already covers ``kappa`` of that distance, so the extension is the rest,
    limited to one previous step: ``beta = min(1, max(0, alpha - max(kappa, 0)))``.

    - It is never negative: the worst a search can do is to add nothing, which leaves the first-order update of
      the round as it is.
    - It is never above one, so the step of a round is at most the previous step plus a fresh one: the step length
      can grow by one first-order update per round, not by a factor. A search misled by a few satellites with
      unusual data then costs one step of the current length, and the next search can stop it.
    - Together with ``alpha <= 2`` the model lands between the points the satellites measured whenever the fresh
      step is no longer than the previous one along it (kappa <= 1).
    """
    if not previous:
        return 0.0, 0.0
    names = list(previous)
    zero = torch.zeros((), dtype=torch.float64, device=previous[names[0]].device)
    square = sum((previous[name].double().square().sum() for name in names), zero)
    dot = sum(((fresh[name].double() * previous[name].double()).sum() for name in names if name in fresh), zero)
    square, dot = torch.stack([square, dot]).tolist()   # one synchronisation with the device
    if not math.isfinite(square) or not math.isfinite(dot) or square <= 0:
        return 0.0, 0.0
    kappa = dot / square
    return min(1.0, max(0.0, alpha - max(kappa, 0.0))), kappa


@torch.no_grad()
def advance(groups, reports, delivery, cfg, method, server_lr, probes=None):
    """A round under the switches of a variant: every replica group gets its new state. Returns the terms per group
    (for the diagnostics) and, with the step-length search, what the search of the first group found.

    The first-order part is applied the way the default rule applies its update (a copy of the state, then
    ``ho_fl.aggregation.apply``), so a variant that changes nothing for a method ends in bitwise the same state.
    Every replica of a group holds the same state, the same previous step and the same flooded probes, so all of
    them compute the same extension: the replicas stay identical without exchanging anything more.
    """
    terms, found = {}, None
    for group in groups.ids:
        state = groups.state(group)
        terms[group] = round_terms(state, reports, delivery, cfg, replay=bool(method.zo_weight) and not method.zo_search)
        update = weigh_terms(state, terms[group], method)
        if update is None:
            continue
        new = {name: tensor.clone() for name, tensor in state.items()}
        apply(new, update, server_lr)
        if not method.zo_search:
            groups.place(group, new)
            continue
        previous = groups.step(group)
        held = [probes[sid] for sid in sorted(probes or {}) if sid in delivery.zo_received]   # the probes every replica holds
        examples = sum(probe["examples"] for probe in held)
        pooled = [sum(probe["losses"][point] for probe in held) for point in range(3)] if held else None
        alpha, case = step_length(pooled)
        beta, kappa = extension({name: new[name] - state[name] for name in update.tensors}, previous, alpha)
        if not method.zo_weight:
            beta = 0.0   # the control: the probes are taken and reported, the model does not use them
        if beta:
            for name, tensor in previous.items():
                new[name].add_(tensor, alpha=beta)
        groups.place(group, new, {name: new[name] - state[name] for name in new})
        if group == min(groups.ids):
            found = {"reports": len(held), "examples": examples, "loss": [value / examples for value in pooled] if examples else None,
                     "alpha": alpha, "case": case if previous else "no previous step", "kappa": kappa, "beta": beta}
    return terms, found


@torch.no_grad()
def first_order_turn(terms, previous=None):
    """A diagnostic of how steadily the first-order updates point one way: per block the cosine between the sum of
    this round's first-order updates and the sum of the previous round's (``previous``, as this function returned
    it). Returns ``({block: cosine}, sums)``; a block without a first-order update in either round has no entry.
    Reads the terms with accumulators of its own, like ``channel_statistics``."""
    sums = {}
    for term in terms or ():
        for name, tensor in term.first_order.items():
            if name not in sums:
                sums[name] = torch.zeros_like(tensor, dtype=torch.float32)
            sums[name].add_(tensor)
    shared = sorted(name for name in sums if previous and name in previous)
    if not shared:
        return {}, sums
    blocks = sorted({block_index(name) for name in shared})
    zero = torch.zeros((), dtype=torch.float32, device=sums[shared[0]].device)
    rows = []
    for block in blocks:
        members = [name for name in shared if block_index(name) == block]
        rows.append(torch.stack([sum(((sums[name] * previous[name]).sum() for name in members), zero),
                                 sum((sums[name].square().sum() for name in members), zero),
                                 sum((previous[name].square().sum() for name in members), zero)]))
    table = torch.stack(rows).double().cpu().tolist()
    return {block: dot / (now * before) ** 0.5 for block, (dot, now, before) in zip(blocks, table) if now > 0 and before > 0}, sums


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
