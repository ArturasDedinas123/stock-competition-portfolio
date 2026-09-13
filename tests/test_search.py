"""Tests for the weight grid, portfolio scoring and fine-tuning."""

import numpy as np
import pytest

from stock_competition import search


def test_grid_sizes_match_the_counts_quoted_in_the_notebook():
    assert len(search.enumerate_grid(10, 0.05, 0.20, 0.05)) == 44_803
    assert len(search.enumerate_grid(10, 0.05, 0.20, 0.025)) == 5_266_030


def test_every_grid_portfolio_follows_the_rules():
    weights = search.weights_from_units(search.enumerate_grid(10, 0.05, 0.20, 0.05), 0.05, 0.05)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert weights.min() >= 0.05 - 1e-7
    assert weights.max() <= 0.20 + 1e-7
    assert len(np.unique(weights, axis=0)) == len(weights)


def test_grid_rejects_a_step_that_does_not_fit_the_limits():
    with pytest.raises(ValueError):
        search.enumerate_grid(10, 0.05, 0.20, 0.03)


def test_holdings_label_lists_positions_above_the_minimum():
    weights = [0.2, 0.05, 0.175, 0.05, 0.05, 0.05, 0.075, 0.05, 0.25, 0.05]
    tickers = list("ABCDEFGHIJ")
    assert search.holdings_label(weights, tickers, 0.05) == "I 25% · A 20% · C 17.5% · G 7.5%"
    assert search.holdings_label([0.1] * 10, tickers, 0.1) == "all at the minimum"


@pytest.fixture(name="small_problem")
def fixture_small_problem():
    rng = np.random.default_rng(1)
    units = search.enumerate_grid(10, 0.05, 0.20, 0.05)[:2_000]
    returns = rng.normal(0.02, 0.15, (3_000, 10)).astype(np.float32)
    best = rng.normal(0.10, 0.08, 3_000).astype(np.float32)
    third = best - 0.05
    return units, returns, best, third


def test_threaded_scoring_matches_a_direct_calculation(small_problem):
    units, returns, best, third = small_problem
    counts = search.score_portfolios(units, 0.05, 0.05, returns, best, third, verbose=False)
    portfolio_returns = search.weights_from_units(units, 0.05, 0.05) @ returns.T
    assert np.array_equal(counts["win"], (portfolio_returns > best).sum(axis=1))
    assert np.array_equal(counts["top3"], (portfolio_returns > third).sum(axis=1))
    assert np.array_equal(counts["loss"], (portfolio_returns < 0).sum(axis=1))
    single = search.score_portfolios(units, 0.05, 0.05, returns, best, third, workers=1, verbose=False)
    assert all(np.array_equal(counts[k], single[k]) for k in ("win", "top3", "loss"))


def test_evaluate_weights_and_win_rates_agree_with_scoring(small_problem):
    units, returns, best, third = small_problem
    weights = search.weights_from_units(units, 0.05, 0.05)
    counts = search.score_portfolios(units, 0.05, 0.05, returns, best, third, verbose=False)
    table = search.evaluate_weights(weights, returns, best, third)
    assert np.allclose(table["p_win"], counts["win"] / len(returns))
    assert np.allclose(table["p_top3"], counts["top3"] / len(returns))
    assert np.allclose(search.win_rates(weights, returns, best), table["p_win"])


def test_refine_weights_climbs_to_the_best_portfolio():
    target = np.array([0.20, 0.20, 0.20, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06])

    def closeness(weights):
        return -np.abs(weights - target).sum(axis=1)

    weights, scores = search.refine_weights(np.full((1, 10), 0.1), closeness, 0.05, 0.20, verbose=False)
    assert np.allclose(weights[0], target)
    assert scores[0] == pytest.approx(0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights.min() >= 0.05 - 1e-9 and weights.max() <= 0.20 + 1e-9


def test_portfolio_moments_match_direct_calculation(small_problem):
    units, returns, _, _ = small_problem
    weights = search.weights_from_units(units, 0.05, 0.05).astype(np.float64)
    mu, cov = returns.mean(axis=0, dtype=np.float64), np.cov(returns, rowvar=False)
    means, stds = search.portfolio_moments(units, 0.05, 0.05, mu, cov, chunk=300)
    assert np.allclose(means[:, 0], weights @ mu, atol=1e-6)
    assert np.allclose(stds, np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights)), atol=1e-6)


def test_weighted_quantiles_ignore_a_sample_with_zero_weight():
    rng = np.random.default_rng(2)
    kept, ignored = rng.normal(0, 1, 50_000), rng.normal(10, 1, 50_000)
    quantiles = search.weighted_quantiles([kept, ignored], [1.0, 0.0], [0.05, 0.5, 0.95])
    assert np.allclose(quantiles, np.quantile(kept, [0.05, 0.5, 0.95]), atol=0.01)
