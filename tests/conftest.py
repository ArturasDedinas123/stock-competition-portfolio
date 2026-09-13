"""Shared fixtures: synthetic prices, so the tests never touch the network."""

import numpy as np
import pandas as pd
import pytest

OURS = [f"S{i}" for i in range(10)]
RIVALS_ONLY = [f"R{i}" for i in range(12)]
BENCHMARK = "BENCH"


@pytest.fixture(name="synthetic_prices")
def fixture_synthetic_prices() -> pd.DataFrame:
    """About five years of daily prices driven by one market factor with different betas."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=1300)
    columns = OURS + RIVALS_ONLY + [BENCHMARK]
    betas = np.linspace(0.5, 1.8, len(columns))
    betas[-1] = 1.0
    market = rng.normal(0.0004, 0.01, len(dates))
    idiosyncratic = rng.normal(0, 0.015, (len(dates), len(columns)))
    idiosyncratic[:, -1] = 0.0
    log_returns = market[:, None] * betas + idiosyncratic
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=columns)
