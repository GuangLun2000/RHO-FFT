"""Characterisation of the upstream samplers (facts F3 and F5 of the brief)."""
import numpy as np
import pytest

from ho_fl.sampling import DepRoundSampler, dep_round, probabilities

OPT_DIMENSIONS = (73728, 147456, 221184)  # zeroth-order dimension of the three OPT tiers


def tiered(dimensions=OPT_DIMENSIONS, seed=0):
    tiers = np.random.default_rng(seed).permutation(np.repeat([0, 1, 2], 10))
    return [dimensions[t] for t in tiers], tiers


def test_beta_zero_is_uniform():
    dims, _ = tiered()
    p, _, scores = probabilities(dims, 2, 6, 0.0, 40042)
    assert np.allclose(p, 1 / 30) and np.allclose(scores, np.asarray(dims) / 2)


def test_preferred_set_is_six_strongest_clients_with_inclusion_080():
    dims, tiers = tiered()
    p, preferred, _ = probabilities(dims, 2, 6, 0.75, 40042)
    assert len(preferred) == 6 and all(tiers[i] == 0 for i in preferred)
    others = np.setdiff1d(np.arange(30), preferred)
    assert np.allclose(6 * p[preferred], 0.8) and np.allclose(6 * p[others], 0.05)
    assert np.isclose(p.sum(), 1.0)


def test_tie_seed_only_changes_which_strong_clients_are_preferred():
    dims, tiers = tiered()
    sets = {tuple(sorted(probabilities(dims, 2, 6, 0.75, seed)[1].tolist())) for seed in range(40000, 40020)}
    assert len(sets) > 1 and all(tiers[i] == 0 for chosen in sets for i in chosen)


def test_equal_dimensions_stay_uniform_for_any_beta():
    p, _, _ = probabilities([100] * 30, 2, 6, 0.75, 1)
    assert np.allclose(p, 1 / 30)


@pytest.mark.parametrize("args", [
    ([], 2, 6, 0.5, 1), ([1.0, -1.0], 2, 1, 0.5, 1), ([1.0, 2.0], 0, 1, 0.5, 1), ([1.0, 2.0], 2, 3, 0.5, 1),
    ([1.0, 2.0], 2, 1, 1.5, 1), ([1.0, float("nan")], 2, 1, 0.5, 1),
])
def test_probabilities_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        probabilities(*args)


def test_dep_round_returns_k_distinct_clients_and_matches_marginals():
    dims, _ = tiered()
    p, _, _ = probabilities(dims, 2, 6, 0.75, 40042)
    rng = np.random.default_rng(7)
    counts = np.zeros(30)
    draws = 4000
    for _ in range(draws):
        selected = dep_round(6 * p, rng)
        assert len(selected) == 6 and len(set(selected)) == 6
        counts[selected] += 1
    assert np.abs(counts / draws - 6 * p).max() < 0.03


def test_dep_round_rejects_non_integral_totals_and_out_of_range_values():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        dep_round([0.5, 0.2], rng)
    with pytest.raises(ValueError):
        dep_round([1.5, 0.5], rng)
    assert sorted(dep_round([1.0, 0.0, 1.0], rng)) == [0, 2]


def test_sampler_is_reproducible_and_uses_the_documented_seeds():
    dims, _ = tiered()
    first, second = DepRoundSampler(dims, 2, 6, 0.75, 42), DepRoundSampler(dims, 2, 6, 0.75, 42)
    assert [first.sample() for _ in range(20)] == [second.sample() for _ in range(20)]
    other = DepRoundSampler(dims, 2, 6, 0.75, 43)
    assert [first.sample() for _ in range(20)] != [other.sample() for _ in range(20)]
    meta = first.metadata()
    assert (meta["round_seed"], meta["tie_seed"], meta["sampler"]) == (30042, 40042, "dep_round_v1")
    assert set(meta) == {"sampler", "tie_seed", "round_seed", "fixed_preferred_clients", "scores_d_over_q", "p",
                         "target_inclusion_probabilities"}
    assert meta["fixed_preferred_clients"] == first.preferred.tolist()
