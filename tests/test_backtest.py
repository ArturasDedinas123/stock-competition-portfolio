"""Tests for the walk-forward backtest."""

from stock_competition import backtest
from stock_competition.settings import StrategySettings

from .conftest import BENCHMARK, OURS, RIVALS_ONLY


def test_quarter_starts_leave_room_for_history_and_horizon(synthetic_prices):
    dates = synthetic_prices.index
    starts = backtest.quarter_starts(dates, horizon=64, min_history=260)
    assert starts and min(starts) >= 260 and max(starts) + 64 < len(dates)
    assert all(dates[i].month in (3, 6, 9, 12) for i in starts)


def test_walk_forward_produces_a_row_per_quarter(synthetic_prices):
    settings = StrategySettings(horizon=64)
    result = backtest.walk_forward(synthetic_prices, OURS, RIVALS_ONLY, BENCHMARK, None, settings,
                                   grid_step=0.05, n_sims=300, n_field_draws=500, verbose=False)
    assert len(result) == len(backtest.quarter_starts(synthetic_prices.index, 64, 260))
    rates = result.filter(regex="_field_win$")
    assert ((rates >= 0) & (rates <= 1)).all().all()
    assert result["pick_percentile"].between(0, 1).all()
    assert result["pick"].str.len().gt(0).all()
