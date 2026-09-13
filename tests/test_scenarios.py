"""Tests for the bootstrap simulation and the expected-return views."""

import numpy as np
import pandas as pd
import pytest

from stock_competition import scenarios
from stock_competition.settings import StrategySettings

from .conftest import BENCHMARK


def test_constant_returns_give_exact_totals():
    log_returns = np.full((300, 3), 0.001)
    totals, expected = scenarios.bootstrap_log_returns(log_returns, 64, 1_000, 21, np.random.default_rng(0))
    assert np.allclose(totals, 0.064, atol=1e-6)
    assert np.allclose(expected, 0.064)


@pytest.mark.parametrize("half_life", [None, 250])
def test_expected_matches_the_simulated_mean(half_life):
    rng = np.random.default_rng(0)
    log_returns = rng.normal(0.0005, 0.02, (1_500, 4))
    totals, expected = scenarios.bootstrap_log_returns(log_returns, 64, 200_000, 21, rng, half_life)
    standard_error = totals.std(axis=0) / np.sqrt(len(totals))
    assert np.all(np.abs(totals.mean(axis=0, dtype=np.float64) - expected) < 5 * standard_error)


def test_half_life_samples_recent_history_more_often():
    log_returns = np.vstack([np.zeros((1_000, 1)), np.full((500, 1), 0.01)])
    rng = np.random.default_rng(0)
    _, uniform = scenarios.bootstrap_log_returns(log_returns, 64, 10, 21, rng)
    _, recent = scenarios.bootstrap_log_returns(log_returns, 64, 10, 21, rng, half_life_days=100)
    assert recent[0] > uniform[0]


def test_views_hit_their_targets_and_keep_volatility():
    rng = np.random.default_rng(0)
    totals, expected = scenarios.bootstrap_log_returns(rng.normal(0, 0.02, (1_500, 5)), 64, 50_000, 21, rng)
    target = np.array([0.01, 0.02, 0.03, -0.01, 0.0])
    returns = scenarios.apply_view(totals, expected, target)
    assert returns.dtype == np.float32
    assert np.allclose(returns.mean(axis=0, dtype=np.float64), target, atol=1e-5)
    assert np.allclose(np.log1p(returns).std(axis=0), totals.std(axis=0), rtol=1e-3)
    assert np.allclose(scenarios.apply_view(totals, expected), np.expm1(totals))


def test_drift_scenarios_use_capm_for_the_benchmark(synthetic_prices):
    settings = StrategySettings(horizon=64, equity_premium=0.05, momentum_tilt=0.04)
    targets, details = scenarios.drift_scenarios(synthetic_prices, synthetic_prices.columns, BENCHMARK,
                                                 settings, risk_free=0.03)
    assert list(targets.columns) == ["neutral", "momentum"]
    assert targets.loc[BENCHMARK, "neutral"] == pytest.approx(1.08 ** (64 / 252) - 1)
    assert targets.loc[BENCHMARK, "momentum"] == pytest.approx(targets.loc[BENCHMARK, "neutral"])
    assert details.loc[BENCHMARK, "beta"] == pytest.approx(1.0)


def test_analyst_view_needs_enough_targets(synthetic_prices):
    settings = StrategySettings(horizon=64)
    cols = synthetic_prices.columns
    last = synthetic_prices.iloc[-1]
    few = pd.Series({cols[0]: last.iloc[0] * 1.2})
    many = last.drop(BENCHMARK) * 1.1
    targets_few, _ = scenarios.drift_scenarios(synthetic_prices, cols, BENCHMARK, settings, 0.03, few)
    targets_many, _ = scenarios.drift_scenarios(synthetic_prices, cols, BENCHMARK, settings, 0.03, many)
    assert "analyst" not in targets_few.columns
    assert "analyst" in targets_many.columns
