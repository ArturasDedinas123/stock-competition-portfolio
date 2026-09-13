"""Tests for the free-choice stock search."""

import numpy as np
import pandas as pd
import pytest

from stock_competition import picks, search
from stock_competition.rivals import RivalField
from stock_competition.simulation import Scenarios


@pytest.fixture(name="planted")
def fixture_planted():
    """40 candidate stocks; the first three have much wider returns, which is what wins contests."""
    rng = np.random.default_rng(0)
    n_sims, n_stocks = 20_000, 40
    volatility = np.full(n_stocks, 0.08)
    volatility[:3] = 0.40
    returns = (rng.normal(0.02, 1.0, (n_sims, n_stocks)) * volatility).astype(np.float32)
    best = np.quantile(returns[:, 10:] @ np.full(30, 1 / 30, np.float32), 0.5) + rng.normal(0.08, 0.03, n_sims)
    columns = [f"C{i}" for i in range(n_stocks)]
    field = RivalField(best.astype(np.float32), (best - 0.04).astype(np.float32))
    return Scenarios(columns, {"only": returns}, {"only": field}, pd.Series({"only": 1.0}), columns)


def test_single_stock_rates_rank_the_volatile_stocks_first(planted):
    rates = picks.single_stock_win_rates(planted, planted.columns)
    assert set(rates.index.tolist()[:3]) == {"C0", "C1", "C2"}


def test_best_swap_matches_rebuilding_the_portfolio(planted):
    stocks = [f"C{i}" for i in range(10, 20)]
    weights = np.array([0.2, 0.2, 0.2, 0.1] + [0.05] * 6)
    position, candidate, p_win = picks.best_swap(planted, stocks, weights, planted.columns)
    swapped = list(stocks)
    swapped[position] = candidate
    assert candidate not in stocks
    assert p_win == pytest.approx(planted.p_win(weights, swapped)[0])


def test_search_finds_the_planted_stocks_and_keeps_the_rules(planted):
    start = [f"C{i}" for i in range(10, 20)]
    result = picks.search_free_choice(planted, planted.columns, {"start": start}, 0.05, 0.20, verbose=False)
    assert {"C0", "C1", "C2"} <= set(result.stocks)
    assert len(set(result.stocks)) == 10
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.min() >= 0.05 - 1e-9 and result.weights.max() <= 0.20 + 1e-9
    assert result.p_win > result.history["P(win)"].iloc[0]
    assert result.p_win == pytest.approx(planted.p_win(result.weights, result.stocks)[0])
    heavy = {result.stocks[i] for i in np.argsort(-result.weights)[:3]}
    assert heavy == {"C0", "C1", "C2"}


def test_portfolio_label_lists_every_holding():
    weights = [0.2, 0.2, 0.2, 0.1] + [0.05] * 6
    stocks = list("ABCDEFGHIJ")
    assert picks.portfolio_label(weights, stocks, 0.05) == "A 20% · B 20% · C 20% · D 10% · 5% each: E, F, G, H, I, J"
    assert search.holdings_label(weights, stocks, 0.05) == "A 20% · B 20% · C 20% · D 10%"
    shuffled = [9, 2, 0, 5, 1, 3, 8, 4, 7, 6]
    assert picks.portfolio_label(np.array(weights)[shuffled], [stocks[i] for i in shuffled], 0.05) == \
        picks.portfolio_label(weights, stocks, 0.05)
