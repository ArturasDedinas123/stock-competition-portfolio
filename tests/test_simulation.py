"""Tests for simulated competitions and portfolio scoring."""

import numpy as np
import pandas as pd
import pytest

from stock_competition import scenarios, simulation, stats
from stock_competition.settings import StrategySettings

from .conftest import BENCHMARK, OURS, RIVALS_ONLY

VIEWS = {"neutral": 1.0, "momentum": 1.0, "trend": 2.0}


@pytest.fixture(name="competition")
def fixture_competition(synthetic_prices):
    settings = StrategySettings(horizon=64)
    targets, _ = scenarios.drift_scenarios(synthetic_prices, synthetic_prices.columns, BENCHMARK, settings, 0.03)
    log_returns = stats.log_returns(synthetic_prices).iloc[1:]
    simulated = simulation.simulate(log_returns, targets, settings, n_sims=20_000, seed=1, keep=OURS + [BENCHMARK],
                                    pool=OURS + RIVALS_ONLY, view_weights=VIEWS, chunk=10_000, profile=True)
    return simulated, targets


def test_views_hit_their_targets(competition):
    simulated, targets = competition
    assert simulated.views == ["neutral", "momentum", "trend"]
    assert simulated.view_share.to_dict() == {"neutral": 0.25, "momentum": 0.25, "trend": 0.5}
    for view in ("neutral", "momentum"):
        means = simulated.returns[view].mean(axis=0, dtype=np.float64)
        assert np.allclose(means, targets[view].loc[OURS + [BENCHMARK]], atol=1e-5)


def test_evaluate_matches_a_direct_calculation(competition):
    simulated, _ = competition
    portfolios = pd.DataFrame([np.full(10, 0.1), [0.2, 0.2, 0.2, 0.1] + [0.05] * 6], columns=OURS,
                              index=["equal", "concentrated"])
    table = simulated.evaluate(portfolios)
    for label, weights in portfolios.iterrows():
        per_view = {v: simulated.returns_of(v, OURS) @ weights.to_numpy(np.float32) for v in simulated.views}
        p_win = sum(simulated.view_share[v] * (per_view[v] > simulated.fields[v].best).mean() for v in simulated.views)
        assert table.loc[label, "P(win)"] == pytest.approx(p_win)
        assert table.loc[label, "5th pct"] < table.loc[label, "median"] < table.loc[label, "95th pct"]
    assert np.allclose(simulated.p_win(portfolios.to_numpy(), OURS), table["P(win)"])


def test_benchmark_can_be_scored_like_a_portfolio(competition):
    simulated, _ = competition
    table = simulated.evaluate(pd.DataFrame({BENCHMARK: [1.0]}, index=["index"]), quantiles=False)
    assert list(table.columns[:4]) == ["P(win)", "P(top 3)", "P(loss)", "mean"]
    assert 0 <= table.loc["index", "P(win)"] < 0.05


def test_chunked_runs_have_the_same_statistics(synthetic_prices):
    settings = StrategySettings(horizon=64)
    targets, _ = scenarios.drift_scenarios(synthetic_prices, synthetic_prices.columns, BENCHMARK, settings, 0.03)
    log_returns = stats.log_returns(synthetic_prices).iloc[1:]
    runs = [simulation.simulate(log_returns, targets, settings, n_sims=40_000, seed=2, keep=OURS,
                                pool=OURS + RIVALS_ONLY, view_weights={"neutral": 1}, chunk=chunk)
            for chunk in (40_000, 10_000)]
    assert np.allclose(runs[0].returns["neutral"].std(axis=0), runs[1].returns["neutral"].std(axis=0), rtol=0.05)
    with pytest.raises(ValueError):
        simulation.simulate(log_returns, targets, settings, n_sims=100, seed=2, keep=OURS, pool=OURS,
                            view_weights={"analyst": 1})


def test_winner_table_is_consistent(competition):
    simulated, _ = competition
    table = simulated.winner_table()
    assert set(table.index) == set(OURS + RIVALS_ONLY)
    assert table["held by rivals"].sum() == pytest.approx(10)
    assert table["held by winners"].sum() == pytest.approx(10)
    assert table["lift"].is_monotonic_decreasing
    assert 0.3 < simulated.winner_top3_weight() < 0.6
