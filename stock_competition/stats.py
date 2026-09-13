"""Descriptive statistics for a set of stocks."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Log returns over ``periods`` rows (the first ``periods`` rows are NaN)."""
    return pd.DataFrame(np.log(prices.to_numpy()), index=prices.index, columns=prices.columns).diff(periods)


def weekly_betas(prices: pd.DataFrame, tickers, benchmark: str) -> pd.Series:
    """Beta to the benchmark from weekly log returns (weekly data avoids end-of-day timing noise)."""
    tickers = list(tickers)
    cols = list(dict.fromkeys(tickers + [benchmark]))
    weekly = log_returns(prices.reindex(columns=cols).resample("W-FRI").last()).dropna().to_numpy()
    market = weekly[:, cols.index(benchmark)]
    variance = market.var(ddof=1)
    return pd.Series({t: np.cov(weekly[:, cols.index(t)], market)[0, 1] / variance for t in tickers})


def average_correlation(prices: pd.DataFrame, tickers) -> pd.Series:
    """Each stock's average correlation of daily returns with the other stocks in ``tickers``."""
    corr = log_returns(prices.reindex(columns=list(tickers))).dropna().corr()
    return (corr.sum() - 1) / (len(corr) - 1)


def stock_stats(prices: pd.DataFrame, tickers, benchmark: str, horizon: int) -> pd.DataFrame:
    """Annual return and volatility, beta, worst drawdown, momentum and historical horizon-return range."""
    p = prices.reindex(columns=list(tickers))
    years = (len(p) - 1) / TRADING_DAYS_PER_YEAR
    horizon_returns = (p.shift(-horizon) / p - 1).dropna()
    return pd.DataFrame({
        "annual_return": (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1,
        "annual_vol": log_returns(p).std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "beta": weekly_betas(prices, tickers, benchmark),
        "max_drawdown": (p / p.cummax() - 1).min(),
        "momentum_12_1": p.iloc[-22] / p.iloc[-253] - 1,
        f"{horizon}d_return_p5": horizon_returns.quantile(0.05),
        f"{horizon}d_return_median": horizon_returns.median(),
        f"{horizon}d_return_p95": horizon_returns.quantile(0.95),
    })
