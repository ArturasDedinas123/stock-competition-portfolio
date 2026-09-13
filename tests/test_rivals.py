"""Tests for the simulated rival field."""

import numpy as np
import pytest

from stock_competition import rivals


def test_random_weights_follow_the_rules():
    weights = rivals.random_bounded_weights((5_000, 20, 10), np.random.default_rng(0), 0.05, 0.20)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert weights.min() >= 0.05 - 1e-6
    assert weights.max() <= 0.20 + 1e-6


def test_best_is_never_below_third():
    pool = np.random.default_rng(0).normal(0.02, 0.2, (20_000, 40)).astype(np.float32)
    best, third = rivals.rival_field(pool, 20, seed=1)
    assert np.all(best >= third)


def test_rivals_do_not_depend_on_thread_count():
    pool = np.random.default_rng(0).normal(0.02, 0.2, (12_000, 40)).astype(np.float32)
    single = rivals.rival_field(pool, 20, seed=3, workers=1)
    threaded = rivals.rival_field(pool, 20, seed=3, workers=8)
    assert all(np.array_equal(a, b) for a, b in zip(single, threaded))


def test_a_random_rival_wins_about_one_time_in_n_plus_one():
    pool = np.random.default_rng(0).normal(0.02, 0.2, (60_000, 60)).astype(np.float32)
    _, _, every = rivals.rival_field(pool, 21, seed=5, return_all=True)
    p_win = (every[:, 0] > every[:, 1:].max(axis=1)).mean()
    assert p_win == pytest.approx(1 / 21, abs=0.006)


def test_pool_must_have_enough_stocks():
    with pytest.raises(ValueError):
        rivals.rival_field(np.zeros((10, 5), np.float32), 20, seed=0, n_stocks=10)
