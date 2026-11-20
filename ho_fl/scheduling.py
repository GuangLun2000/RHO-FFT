"""Scheduling: which clients take part in a round, and with which cut.

A scheduler offers ``sample(round_id)`` returning a ``RoundPlan``, ``observe(round_id, arrivals)``,
``probability(cid)``, ``inclusion()`` and ``metadata()``. This module currently holds one implementation,
``LegacyScheduler``, which wraps the upstream samplers without changing what they draw.
"""
from dataclasses import dataclass, field


@dataclass
class RoundPlan:
    round: int
    clients: list                                       # in the order the sampler returned them
    default_cuts: list                                  # cut of every client when nothing is overridden
    cut_override: dict = field(default_factory=dict)    # client -> cut, for controlled experiments

    def cut(self, cid):
        return self.cut_override.get(cid, self.default_cuts[cid])


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
