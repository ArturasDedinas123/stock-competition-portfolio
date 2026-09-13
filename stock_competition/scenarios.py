"""Simulated returns over the competition horizon.

Horizon returns come from a block bootstrap: runs of consecutive historical days drawn at random,
on the same dates for every stock, so volatility, correlations and crash days carry over. Each
expected-return *view* then re-centers the simulations without changing volatility or correlations.
"""

import numpy as np
import pandas as pd

from .settings import StrategySettings
from .stats import TRADING_DAYS_PER_YEAR, weekly_betas

VIEW_LABELS = {
    "neutral": "Neutral (CAPM)",
    "momentum": "Momentum",
    "analyst": "Analyst targets",
    "trend": "Historical trend",
}


def drift_scenarios(prices: pd.DataFrame, cols, benchmark: str, settings: StrategySettings, risk_free: float,
                    analyst_targets: pd.Series | None = None, analyst_shrink: float = 0.25,
                    max_analyst_tilt: float = 0.15):
    """Expected simple returns over the horizon for every column, under each expected-return view.

    Views:
        neutral   risk-free rate + beta x equity premium (CAPM); only risk separates the stocks
        momentum  neutral + ``momentum_tilt`` per standard deviation of 12-month-minus-1-month return
        analyst   neutral + ``analyst_shrink`` x upside to the mean analyst target, relative to the median
                  stock (only when at least half of the stocks have a target)

    Returns:
        ``(targets, details)``: ``targets`` has one column per view, ``details`` the inputs behind them.
    """
    cols = list(cols)
    p = prices.reindex(columns=cols)
    stocks = [c for c in cols if c != benchmark]
    beta = weekly_betas(p, cols, benchmark)
    capm = risk_free + beta * settings.equity_premium

    momentum = p.iloc[-22] / p.iloc[-253] - 1
    z_score = (momentum[stocks] - momentum[stocks].mean()) / momentum[stocks].std()
    z_score = z_score.clip(-3, 3).reindex(cols).fillna(0.0)

    views = {"neutral": capm, "momentum": capm + settings.momentum_tilt * z_score}
    details = {"beta": beta, "momentum_12_1": momentum, "momentum_z": z_score}
    if analyst_targets is not None:
        upside = analyst_targets.reindex(stocks) / p.reindex(columns=stocks).iloc[-1] - 1
        if upside.notna().mean() >= 0.5:
            tilt = (analyst_shrink * (upside - upside.median())).clip(-max_analyst_tilt, max_analyst_tilt)
            views["analyst"] = capm + tilt.reindex(cols).fillna(0.0)
            details["analyst_upside"] = upside.reindex(cols)

    annual = pd.DataFrame(views).clip(lower=-0.5)
    details = pd.DataFrame(details).join(annual.add_prefix("annual_"))
    return (1 + annual) ** (settings.horizon / TRADING_DAYS_PER_YEAR) - 1, details


def bootstrap_log_returns(log_returns: np.ndarray, horizon: int, n_sims: int, block: int,
                          rng: np.random.Generator, half_life_days: float | None = None):
    """Horizon log returns assembled from random blocks of consecutive days.

    Args:
        log_returns: Daily log returns, shape (days, columns).
        horizon: Number of days per simulated period.
        n_sims: Number of simulated periods.
        block: Days per block (the last block is shorter when ``block`` doesn't divide ``horizon``).
        rng: Random generator.
        half_life_days: Sample recent blocks more often (exponential decay); None samples uniformly.

    Returns:
        ``(totals, expected)``: float32 totals of shape (n_sims, columns), and the exact mean of the
        totals under this sampling scheme (used to de-mean the simulations).
    """
    n_cols = log_returns.shape[1]
    cumulative = np.vstack([np.zeros((1, n_cols)), np.cumsum(log_returns, axis=0)])
    lengths = [block] * (horizon // block) + ([horizon % block] if horizon % block else [])
    totals = np.zeros((n_sims, n_cols), dtype=np.float32)
    expected = np.zeros(n_cols)
    for length in lengths:
        block_sums = cumulative[length:] - cumulative[:-length]  # one row per possible start day
        n_starts = len(block_sums)
        if half_life_days:
            probs = 0.5 ** (np.arange(n_starts)[::-1] / half_life_days)
            probs /= probs.sum()
            starts = rng.choice(n_starts, size=n_sims, p=probs)
            expected += probs @ block_sums
        else:
            starts = rng.integers(n_starts, size=n_sims)
            expected += block_sums.mean(axis=0)
        totals += block_sums.astype(np.float32)[starts]
    return totals, expected


def apply_view(totals: np.ndarray, expected: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Simple horizon returns (float32) for one view.

    With ``target`` (an expected simple return per column), simulations are shifted so each column's
    mean return equals its target exactly; volatility and correlations are unchanged. Without a
    target, the raw historical drift is kept (the "trend" view).
    """
    if target is None:
        return np.expm1(totals)
    out = totals - expected.astype(np.float32)
    gross_mean = np.exp(out).mean(axis=0, dtype=np.float64)
    out += (np.log1p(np.asarray(target, dtype=np.float64)) - np.log(gross_mean)).astype(np.float32)
    np.expm1(out, out=out)
    return out
