"""Scheduling: which clients take part in a round, and with which cut.

A scheduler offers ``sample(round_id)`` returning a ``RoundPlan``, ``observe(round_id, arrivals)``,
``probability(cid)``, ``inclusion()`` and ``metadata()``. ``probability`` is the per-draw value written to the
ledger as ``p``; the ledger's ``inclusion_probability`` is ``K * p``, the marginal probability of being selected.

Schedulers:
- ``LegacyScheduler`` wraps the upstream samplers without changing what they draw.
- ``TierEqualisedScheduler`` averages the inclusion probabilities of ``dep_round_v1`` within each tier.
- ``TierQuotaScheduler`` selects a fixed number of clients from each tier in every round.

Tiers are defined by the clients' resource cuts (the cuts in the manifest), never by the cuts a method trains with.
This module depends on numpy only, so supply maps can be computed without loading a model.
"""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .sampling import DepRoundSampler, dep_round, probabilities


@dataclass
class RoundPlan:
    round: int
    clients: list                                       # in the order the sampler returned them
    default_cuts: list                                  # cut of every client when nothing is overridden
    cut_override: dict = field(default_factory=dict)    # client -> cut, for controlled experiments

    def cut(self, cid):
        return self.cut_override.get(cid, self.default_cuts[cid])


def tiers_of(resource_cuts):
    """Tier index of every client: 0 for the smallest resource cut (T1, the strongest devices)."""
    order = sorted(set(resource_cuts))
    return [order.index(cut) for cut in resource_cuts]


def tie_order(resource_cuts, seed):
    """All clients ranked as ``dep_round_v1`` ranks them: strongest tier first, ties broken with seed 40000+seed."""
    return probabilities(resource_cuts, 1, len(resource_cuts), 0.0, 40000 + seed)[1].tolist()


def preferred_clients(resource_cuts, k, seed):
    """The preferred set of ``dep_round_v1`` (the K best-ranked clients), whatever scheduler is in use."""
    return tie_order(resource_cuts, seed)[:k]


class LegacyScheduler:
    """``dep_round_v1``, ``historical_torch`` and ``historical_numpy_uniform`` behind the scheduler interface.

    ``sampler`` is a ``ho_fl.train.Sampler``. Every draw is delegated to it, so the sequence of cohorts, the order
    of clients inside a cohort and the random numbers consumed are exactly those of the upstream loop.
    """

    def __init__(self, sampler, cuts):
        self.sampler = sampler
        self.cuts = list(cuts)

    def sample(self, round_id, available=None):
        if available is not None:
            raise NotImplementedError("The upstream samplers do not support availability constraints")
        return RoundPlan(round_id, self.sampler.sample(), self.cuts)

    def observe(self, round_id, arrivals):
        pass

    def probability(self, cid):
        """The per-draw probability ``p_i`` written to the ledger."""
        return float(self.sampler.dr.p[cid])

    def inclusion(self):
        """Marginal probability of each client being selected in a round."""
        return [self.sampler.dr.k * float(p) for p in self.sampler.dr.p]

    def metadata(self):
        return self.sampler.dr.metadata()


class _DepRoundOnly:
    """Stand-in for ``ho_fl.train.Sampler`` when ``dep_round_v1`` is simulated without a backend."""

    def __init__(self, resource_cuts, k, beta, seed):
        # The cut stands in for the zeroth-order dimension: same ranking, hence the same probabilities and draws.
        self.dr = DepRoundSampler(resource_cuts, 1, k, beta, seed)

    def sample(self):
        return self.dr.sample()


class TierEqualisedScheduler:
    """Per-tier average of the ``dep_round_v1`` inclusion probabilities, sampled with ``dep_round``.

    Each tier keeps the expected number of slots it has under ``dep_round_v1`` at the same beta, but no device
    inside a tier is preferred. At beta 0 this draws the same cohorts as ``dep_round_v1``.
    """

    def __init__(self, resource_cuts, cuts, k, beta, seed):
        self.k, self.beta, self.cuts = k, beta, list(cuts)
        self.tie_seed, self.round_seed = 40000 + seed, 30000 + seed
        base, preferred, _ = probabilities(resource_cuts, 1, k, beta, self.tie_seed)
        self.preferred = preferred.tolist()
        self.tiers = tiers_of(resource_cuts)
        tiers = np.asarray(self.tiers)
        self.p = np.empty_like(base)
        for tier in sorted(set(self.tiers)):
            self.p[tiers == tier] = base[tiers == tier].mean()
        self.rng = np.random.default_rng(self.round_seed)

    def sample(self, round_id, available=None):
        if available is not None:
            raise NotImplementedError("Availability constraints are not supported yet")
        return RoundPlan(round_id, dep_round(self.k * self.p, self.rng), self.cuts)

    def observe(self, round_id, arrivals):
        pass

    def probability(self, cid):
        return float(self.p[cid])

    def inclusion(self):
        return [self.k * float(p) for p in self.p]

    def metadata(self):
        return {"sampler": "tier_equalised", "beta": self.beta, "tie_seed": self.tie_seed, "round_seed": self.round_seed,
                "fixed_preferred_clients": self.preferred, "tiers": self.tiers, "p": self.p.tolist(),
                "target_inclusion_probabilities": (self.k * self.p).tolist()}


class TierQuotaScheduler:
    """A fixed number of clients from each tier in every round (``quota`` = counts for T1, T2, T3).

    ``in_tier``: ``rotate`` takes the clients selected least often in the last ``window`` rounds, ties broken at
    random; ``random`` draws uniformly without replacement; ``fixed`` always takes the best-ranked clients of the
    tier in ``tie_order`` (for T1 these are the first clients of the ``dep_round_v1`` preferred set).
    The cohort lists the picks of T1, then T2, then T3.
    """

    def __init__(self, quota, in_tier, window, resource_cuts, cuts, k, seed):
        if sum(quota) != k:
            raise ValueError("The tier quota must sum to k")
        self.quota, self.in_tier, self.window, self.k, self.cuts = list(quota), in_tier, window, k, list(cuts)
        self.tie_seed, self.round_seed = 40000 + seed, 60000 + seed
        self.tiers = tiers_of(resource_cuts)
        self.members = [[cid for cid, tier in enumerate(self.tiers) if tier == index] for index in range(len(quota))]
        if any(count > len(members) for count, members in zip(quota, self.members)):
            raise ValueError("A tier quota exceeds the number of clients of the tier")
        ranked = tie_order(resource_cuts, seed)
        self.preferred = ranked[:k]
        self.fixed = [[cid for cid in ranked if self.tiers[cid] == index][:count] for index, count in enumerate(quota)]
        self.rng = np.random.default_rng(self.round_seed)
        self.history = deque(maxlen=window)
        self.last = None

    def sample(self, round_id, available=None):
        if available is not None:
            raise NotImplementedError("Availability constraints are not supported yet")
        counts = Counter(cid for cohort in self.history for cid in cohort)
        picks = []
        for members, count, fixed in zip(self.members, self.quota, self.fixed):
            if self.in_tier == "fixed":
                picks += fixed
                continue
            rank = self.rng.permutation(len(members)).tolist()
            load = (lambda cid: counts[cid]) if self.in_tier == "rotate" else (lambda cid: 0)
            order = sorted(range(len(members)), key=lambda position: (load(members[position]), rank[position]))
            picks += [members[position] for position in order[:count]]
        self.last = picks
        return RoundPlan(round_id, list(picks), self.cuts)

    def observe(self, round_id, arrivals):
        # Rotation counts selections. Counting on-time deliveries instead needs the link model (a later milestone).
        if self.last is not None:
            self.history.append(self.last)
            self.last = None

    def inclusion(self):
        if self.in_tier == "fixed":
            chosen = {cid for group in self.fixed for cid in group}
            return [1.0 if cid in chosen else 0.0 for cid in range(len(self.tiers))]
        return [self.quota[tier] / len(self.members[tier]) for tier in self.tiers]

    def probability(self, cid):
        return self.inclusion()[cid] / self.k

    def metadata(self):
        inclusion = self.inclusion()
        return {"sampler": "tier_quota", "tier_quota": self.quota, "in_tier": self.in_tier, "rotate_window": self.window,
                "tie_seed": self.tie_seed, "round_seed": self.round_seed, "fixed_preferred_clients": self.preferred,
                "fixed_clients": self.fixed, "tiers": self.tiers, "p": [value / self.k for value in inclusion],
                "target_inclusion_probabilities": inclusion}


def make_scheduler(cfg, cuts, resource_cuts, sampler=None):
    """The scheduler named by ``cfg.sampler``.

    ``cuts`` are the cuts the method trains with, ``resource_cuts`` those of the manifest. ``sampler`` is the
    ``ho_fl.train.Sampler`` of the run; without it only ``dep_round_v1`` and the tier schedulers can be built
    (enough to simulate them without a model).
    """
    if cfg.sampler == "tier_equalised":
        return TierEqualisedScheduler(resource_cuts, cuts, cfg.k, cfg.beta, cfg.seed)
    if cfg.sampler == "tier_quota":
        quota = [int(part) for part in cfg.tier_quota.split(",")]
        return TierQuotaScheduler(quota, cfg.in_tier, cfg.rotate_window, resource_cuts, cuts, cfg.k, cfg.seed)
    if sampler is None:
        if cfg.sampler != "dep_round_v1":
            raise ValueError(f"Sampler {cfg.sampler} needs the run's Sampler object")
        sampler = _DepRoundOnly(resource_cuts, cfg.k, cfg.beta, cfg.seed)
    return LegacyScheduler(sampler, cuts)
