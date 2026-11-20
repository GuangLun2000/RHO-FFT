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


# --- tier schedulers ----------------------------------------------------------------------------------------------

from collections import Counter  # noqa: E402

from ho_fl.scheduling import (TierEqualisedScheduler, TierQuotaScheduler, make_scheduler, preferred_clients,  # noqa: E402
                              tie_order, tiers_of)


def resource_cuts(seed=0):
    return np.random.default_rng(seed).permutation(np.repeat([3, 6, 9], 10)).tolist()


def test_preferred_set_matches_the_upstream_sampler_and_ignores_beta():
    for seed in range(20):
        cuts = resource_cuts(seed)
        backend = SimpleNamespace(cuts=cuts, dimensions={3: 73728, 6: 147456, 9: 221184})
        for beta in (0.0, 0.75):
            upstream = Sampler(Config(beta=beta, seed=seed).resolve(), backend).dr.preferred.tolist()
            assert preferred_clients(cuts, 6, seed) == upstream
        order = tie_order(cuts, seed)
        assert sorted(order) == list(range(30)) and [cuts[cid] for cid in order] == sorted(cuts)
    assert tiers_of([9, 3, 6, 3]) == [2, 0, 1, 0]


def test_tier_equalised_keeps_tier_slots_and_removes_the_preference_inside_a_tier():
    cuts = resource_cuts()
    scheduler = make_scheduler(Config(sampler="tier_equalised").resolve(), cuts, cuts)
    assert isinstance(scheduler, TierEqualisedScheduler)
    inclusion = np.asarray(scheduler.inclusion())
    assert np.allclose(inclusion[np.asarray(cuts) == 3], 0.5) and np.allclose(inclusion[np.asarray(cuts) != 3], 0.05)
    assert np.isclose(inclusion.sum(), 6) and scheduler.probability(0) == pytest.approx(inclusion[0] / 6)
    counts = Counter()
    for round_id in range(1, 3001):
        plan = scheduler.sample(round_id)
        assert len(set(plan.clients)) == 6
        counts.update(plan.clients)
    assert np.abs(np.asarray([counts[cid] for cid in range(30)]) / 3000 - inclusion).max() < 0.04
    assert {"fixed_preferred_clients", "tie_seed", "p"} <= set(scheduler.metadata())


def test_tier_equalised_at_beta_zero_draws_the_cohorts_of_dep_round():
    cuts = resource_cuts()
    equalised = make_scheduler(Config(sampler="tier_equalised", beta=0.0).resolve(), cuts, cuts)
    upstream = make_scheduler(Config(beta=0.0).resolve(), cuts, cuts)
    assert [equalised.sample(r).clients for r in range(1, 31)] == [upstream.sample(r).clients for r in range(1, 31)]


def test_scheduler_built_without_a_backend_draws_what_the_run_draws():
    cuts = resource_cuts()
    backend = SimpleNamespace(cuts=cuts, dimensions={3: 73728, 6: 147456, 9: 221184})
    cfg = Config().resolve()
    standalone, in_run = make_scheduler(cfg, cuts, cuts), make_scheduler(cfg, cuts, cuts, Sampler(cfg, backend))
    assert [standalone.sample(r).clients for r in range(1, 31)] == [in_run.sample(r).clients for r in range(1, 31)]
    assert standalone.inclusion() == in_run.inclusion()
    with pytest.raises(ValueError):
        make_scheduler(Config(sampler="historical_torch", beta=0.0).resolve(), cuts, cuts)


@pytest.mark.parametrize("in_tier", ["rotate", "random", "fixed"])
def test_tier_quota_takes_the_quota_from_every_tier_in_every_round(in_tier):
    cuts = resource_cuts()
    cfg = Config(sampler="tier_quota", tier_quota="3,1,2", in_tier=in_tier).resolve()
    scheduler = make_scheduler(cfg, cuts, cuts)
    assert isinstance(scheduler, TierQuotaScheduler)
    counts = Counter()
    for round_id in range(1, 401):
        plan = scheduler.sample(round_id)
        scheduler.observe(round_id, None)
        assert [cuts[cid] for cid in plan.clients] == [3, 3, 3, 6, 9, 9] and len(set(plan.clients)) == 6
        counts.update(plan.clients)
    frequency = np.asarray([counts[cid] for cid in range(30)]) / 400
    assert np.isclose(sum(scheduler.inclusion()), 6) and scheduler.probability(0) == pytest.approx(scheduler.inclusion()[0] / 6)
    if in_tier == "fixed":
        assert sorted(cid for cid in range(30) if counts[cid]) == sorted(cid for group in scheduler.fixed for cid in group)
        assert scheduler.fixed[0] == preferred_clients(cuts, 6, cfg.seed)[:3] and set(frequency) == {0.0, 1.0}
    else:
        assert np.abs(frequency - np.asarray(scheduler.inclusion())).max() < (0.03 if in_tier == "rotate" else 0.12)  # uniform draws scatter: sd about 0.02 over 400 rounds


def test_rotation_balances_selections_within_the_window():
    cuts = resource_cuts()
    scheduler = make_scheduler(Config(sampler="tier_quota", tier_quota="3,1,2").resolve(), cuts, cuts)
    strong = [cid for cid in range(30) if cuts[cid] == 3]
    history = []
    for round_id in range(1, 201):
        history.append(scheduler.sample(round_id).clients)
        scheduler.observe(round_id, None)
    for start in range(20, 180, 20):  # any 20 rounds: 60 T1 slots over 10 devices, 6 each, give or take one
        window = Counter(cid for cohort in history[start:start + 20] for cid in cohort if cid in strong)
        assert max(window.values()) - min(window[cid] for cid in strong) <= 2
    other = make_scheduler(Config(sampler="tier_quota", tier_quota="3,1,2", seed=43).resolve(), cuts, cuts)
    assert [other.sample(r).clients for r in range(1, 4)] != history[:3]
