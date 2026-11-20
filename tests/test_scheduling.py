"""The scheduler wrapper must reproduce the upstream samplers draw for draw."""
from types import SimpleNamespace

import numpy as np
import pytest

from ho_fl.config import Config
from ho_fl.scheduling import LegacyScheduler, RoundPlan
from ho_fl.train import Sampler


def fake_backend(seed=0):
    cuts = np.random.default_rng(seed).permutation(np.repeat([3, 6, 9], 10)).tolist()
    return SimpleNamespace(cuts=cuts, dimensions={3: 73728, 6: 147456, 9: 221184})


@pytest.mark.parametrize("sampler, beta", [("dep_round_v1", 0.75), ("dep_round_v1", 0.0), ("historical_torch", 0.0),
                                           ("historical_numpy_uniform", 0.0)])
def test_wrapped_sampler_draws_the_same_cohorts_in_the_same_order(sampler, beta):
    cfg = Config(sampler=sampler, beta=beta).resolve()
    backend = fake_backend()
    reference = Sampler(cfg, backend)
    scheduler = LegacyScheduler(Sampler(cfg, backend), backend.cuts)
    for round_id in range(1, 41):
        plan = scheduler.sample(round_id)
        assert plan.clients == reference.sample() and plan.round == round_id
        assert len(plan.clients) == cfg.k
        scheduler.observe(round_id, None)


def test_probabilities_and_metadata_are_those_of_the_wrapped_sampler():
    cfg = Config().resolve()
    backend = fake_backend()
    sampler = Sampler(cfg, backend)
    scheduler = LegacyScheduler(sampler, backend.cuts)
    assert [scheduler.probability(cid) for cid in range(30)] == [float(p) for p in sampler.dr.p]
    assert scheduler.inclusion() == [cfg.k * float(p) for p in sampler.dr.p]
    assert np.isclose(sum(scheduler.inclusion()), cfg.k)
    assert sorted(set(np.round(scheduler.inclusion(), 12))) == [0.05, 0.8]
    assert scheduler.metadata() == sampler.dr.metadata()


def test_round_plan_cut_uses_the_override_when_present():
    plan = RoundPlan(1, [4, 7], [3, 3, 3, 3, 6, 6, 6, 9])
    assert (plan.cut(4), plan.cut(7)) == (6, 9)
    assert RoundPlan(1, [4, 7], [3, 3, 3, 3, 6, 6, 6, 9], {7: 3}).cut(7) == 3


def test_availability_constraints_are_not_supported_by_the_upstream_samplers():
    backend = fake_backend()
    scheduler = LegacyScheduler(Sampler(Config().resolve(), backend), backend.cuts)
    with pytest.raises(NotImplementedError):
        scheduler.sample(1, available={0, 1, 2})
