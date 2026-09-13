"""Tests for the simulated rival field."""

import numpy as np
import pytest

from stock_competition import rivals


def test_random_weights_follow_the_rules():
    weights = rivals.random_bounded_weights((5_000, 20, 10), np.random.default_rng(0), 0.05, 0.20)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert weights.min() >= 0.05 - 1e-6
    assert weights.max() <= 0.20 + 1e-6


@pytest.mark.parametrize("pool_size", [12, 200])
def test_picks_are_distinct_and_uniform(pool_size):
    picks = rivals.sample_picks((20_000, 5), pool_size, 10, np.random.default_rng(1))
    assert picks.shape == (20_000, 5, 10) and picks.min() >= 0 and picks.max() < pool_size
    ordered = np.sort(picks, axis=-1)
    assert not (ordered[..., 1:] == ordered[..., :-1]).any()
    frequency = np.bincount(picks.ravel(), minlength=pool_size) / picks.size
    assert np.allclose(frequency, 1 / pool_size, rtol=0.05)


def test_pool_must_have_enough_stocks():
    with pytest.raises(ValueError):
        rivals.sample_picks((10,), 5, 10, np.random.default_rng(0))


@pytest.fixture(name="pool")
def fixture_pool():
    return np.random.default_rng(0).normal(0.02, 0.2, (12_000, 40)).astype(np.float32)


def test_field_does_not_depend_on_thread_count(pool):
    single, _ = rivals.simulate_field({"v": pool}, 20, seed=3, workers=1)
    threaded, _ = rivals.simulate_field({"v": pool}, 20, seed=3, workers=8)
    assert np.array_equal(single["v"].best, threaded["v"].best)
    assert np.array_equal(single["v"].third, threaded["v"].third)


def test_field_summary_matches_every_rival(pool):
    fields, _ = rivals.simulate_field({"v": pool}, 20, seed=4)
    every = rivals.rival_returns(pool, 20, seed=4)
    assert np.array_equal(fields["v"].best, every.max(axis=1))
    assert np.array_equal(fields["v"].third, np.sort(every, axis=1)[:, -3])


def test_views_face_the_same_rivals_and_batches_can_be_split(pool):
    both, _ = rivals.simulate_field({"a": pool, "b": pool}, 20, seed=5)
    assert np.array_equal(both["a"].best, both["b"].best)
    first, _ = rivals.simulate_field({"a": pool[:5_000]}, 20, seed=5)
    rest, _ = rivals.simulate_field({"a": pool[5_000:]}, 20, seed=5, first_sim=5_000)
    assert np.array_equal(np.concatenate([first["a"].best, rest["a"].best]), both["a"].best)
    with pytest.raises(ValueError):
        rivals.simulate_field({"a": pool}, 20, seed=5, first_sim=123)


def test_winner_profile_counts_and_finds_the_strong_stock():
    pool = np.random.default_rng(1).normal(0.0, 0.1, (10_000, 60)).astype(np.float32)
    pool[:, 7] += 0.5  # stock 7 always adds a lot of return
    _, profiles = rivals.simulate_field({"v": pool}, 20, seed=6, profile=True)
    profile = profiles["v"]
    assert profile.winner_holdings.sum() == pytest.approx(10_000 * 10)
    assert profile.rival_holdings.sum() == pytest.approx(10_000 * 20 * 10)
    lift = (profile.winner_holdings / 10_000) / (profile.rival_holdings / (10_000 * 20))
    assert int(np.argmax(lift)) == 7 and lift[7] > 3
    assert 0.3 < profile.winner_top3_weight / 10_000 < 0.6
    merged = profile.merge(profile)
    assert merged.n_simulations == 20_000 and merged.winner_holdings.sum() == pytest.approx(2 * 10_000 * 10)


def test_a_random_rival_wins_about_one_time_in_n_plus_one():
    pool = np.random.default_rng(0).normal(0.02, 0.2, (60_000, 60)).astype(np.float32)
    every = rivals.rival_returns(pool, 21, seed=5)
    p_win = (every[:, 0] > every[:, 1:].max(axis=1)).mean()
    assert p_win == pytest.approx(1 / 21, abs=0.006)
