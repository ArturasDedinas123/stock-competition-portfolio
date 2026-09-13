"""Walk-forward backtest: re-run the strategy at past quarter starts using only data available then."""

from typing import NamedTuple

import numpy as np
import pandas as pd

from .rivals import rival_field
from .scenarios import apply_view, bootstrap_log_returns, drift_scenarios
from .search import enumerate_grid, holdings_label, portfolio_moments, score_portfolios, weights_from_units
from .settings import StrategySettings
from .stats import TRADING_DAYS_PER_YEAR

BACKTEST_VIEWS = ("neutral", "momentum")  # past analyst targets aren't available


class QuarterChoice(NamedTuple):
    """Portfolios chosen at a quarter start, as positions in the weight grid."""

    pick: int
    predicted_p_win: float
    max_sharpe: int


def quarter_starts(dates: pd.Index, horizon: int, min_history: int, day: int = 14) -> list[int]:
    """Positions of the first trading day on or after the ``day``-th of Mar, Jun, Sep and Dec.

    Only dates with ``min_history`` days before them and a full horizon after them are kept.
    """
    dates = pd.DatetimeIndex(dates)
    first_year = int(pd.Timestamp(dates.to_numpy()[0]).year)
    last_year = int(pd.Timestamp(dates.to_numpy()[-1]).year)
    candidates = [int(dates.searchsorted(np.datetime64(f"{year}-{month:02d}-{day:02d}")))
                  for year in range(first_year, last_year + 1) for month in (3, 6, 9, 12)]
    return [i for i in candidates if i >= min_history and i + horizon < len(dates)]


def _choose_portfolios(train: pd.DataFrame, n_ours: int, n_pool: int, benchmark: str, risk_free: float,
                       settings: StrategySettings, units: np.ndarray, grid_step: float, n_sims: int,
                       seed: int) -> QuarterChoice:
    """The P(win) pick and the max-Sharpe portfolio for one quarter, using only the ``train`` prices."""
    targets, _ = drift_scenarios(train, train.columns, benchmark, settings, risk_free)
    log_returns = np.diff(np.log(train.to_numpy()), axis=0)
    totals, expected = bootstrap_log_returns(log_returns, settings.horizon, n_sims, settings.block_days,
                                             np.random.default_rng(seed), settings.half_life_days)
    p_win = np.zeros(len(units))
    ours_by_view = {}
    for view in BACKTEST_VIEWS:
        returns = apply_view(totals, expected, targets[view].to_numpy())
        best, _ = rival_field(returns[:, :n_pool], settings.n_rivals, seed, n_ours,
                              settings.min_weight, settings.max_weight)
        ours_by_view[view] = np.ascontiguousarray(returns[:, :n_ours])
        wins = score_portfolios(units, settings.min_weight, grid_step, ours_by_view[view], best, verbose=False)["win"]
        p_win += wins / n_sims / len(BACKTEST_VIEWS)
    neutral = ours_by_view["neutral"]
    means, stds = portfolio_moments(units, settings.min_weight, grid_step,
                                    neutral.mean(axis=0, dtype=np.float64), np.cov(neutral, rowvar=False))
    rf_horizon = (1 + risk_free) ** (settings.horizon / TRADING_DAYS_PER_YEAR) - 1
    return QuarterChoice(int(np.argmax(p_win)), float(p_win.max()), int(np.argmax((means[:, 0] - rf_horizon) / stds)))


def _risk_free_rate(irx: pd.Series | None, date, fallback: float) -> float:
    """Annual 13-week T-bill yield on ``date`` (the latest value up to it), or ``fallback``."""
    if irx is None:
        return fallback
    known = irx.loc[:date].dropna()
    return fallback if known.empty else float(known.iloc[-1]) / 100


def walk_forward(prices: pd.DataFrame, ours, rivals_only, benchmark: str, irx: pd.Series | None,
                 settings: StrategySettings, *, grid_step: float, n_sims: int, seed: int = 0,
                 n_field_draws: int = 20_000, min_history: int = 260, verbose: bool = True) -> pd.DataFrame:
    """Test the strategy on past quarters.

    At each quarter start the whole search (neutral and momentum views) runs on data up to that
    day. The chosen portfolio is then valued at the actual prices over the next ``settings.horizon``
    days, and compared with equal weight, the max-Sharpe portfolio and the benchmark. The realized
    field is ``settings.n_rivals`` random rival portfolios at their actual returns, redrawn
    ``n_field_draws`` times.

    Args:
        prices: Daily prices for ``ours``, ``rivals_only`` and ``benchmark``.
        ours: Our stocks.
        rivals_only: Other stocks rivals can pick.
        benchmark: Benchmark ticker, used for beta.
        irx: 13-week T-bill yield in percent (None to always use the fallback rate).
        settings: Competition rules and model settings.
        grid_step: Weight grid step.
        n_sims: Simulations per quarter.
        seed: Random seed.
        n_field_draws: Random rival fields used to measure the realized win rate.
        min_history: Trading days of history required before the first quarter.
        verbose: Print one line per quarter.
    """
    ours, rivals_only = list(ours), list(rivals_only)
    p = prices[ours + rivals_only + [benchmark]]
    n_ours, n_pool = len(ours), len(ours) + len(rivals_only)
    units = enumerate_grid(n_ours, settings.min_weight, settings.max_weight, grid_step)
    grid_weights = weights_from_units(units, settings.min_weight, grid_step)
    equal = int(np.flatnonzero((units == units[:, :1]).all(axis=1))[0])  # the only row with identical weights

    rows = []
    for number, i0 in enumerate(quarter_starts(p.index, settings.horizon, min_history)):
        t0 = p.index[i0]
        choice = _choose_portfolios(p.iloc[: i0 + 1], n_ours, n_pool, benchmark,
                                    _risk_free_rate(irx, t0, settings.risk_free_fallback),
                                    settings, units, grid_step, n_sims, seed + number)
        row = {"start": t0, "end": p.index[i0 + settings.horizon],
               "pick": holdings_label(grid_weights[choice.pick], ours, settings.min_weight),
               "predicted_p_win": choice.predicted_p_win}
        row |= _score_outcome((p.iloc[i0 + settings.horizon].to_numpy() / p.iloc[i0].to_numpy() - 1).astype(np.float32),
                              n_ours, n_pool, grid_weights,
                              {"pick": choice.pick, "equal_weight": equal, "max_sharpe": choice.max_sharpe},
                              settings, n_field_draws, seed + 10_000 + number)
        rows.append(row)
        if verbose:
            print(f"  {t0:%Y-%m-%d}: {row['pick']:<45} pick {row['pick_return']:+7.1%} · "
                  f"equal weight {row['equal_weight_return']:+7.1%} · {benchmark} {row['spy_return']:+6.1%}")
    return pd.DataFrame(rows).set_index("start")


def _score_outcome(realized: np.ndarray, n_ours: int, n_pool: int, grid_weights: np.ndarray,
                   positions: dict[str, int], settings: StrategySettings, n_field_draws: int, seed: int) -> dict:
    """Actual returns, the pick's percentile among all grid portfolios, and win rates against random fields."""
    grid_realized = grid_weights @ realized[:n_ours]
    field = np.broadcast_to(realized[:n_pool], (n_field_draws, n_pool))
    field_best, _ = rival_field(field, settings.n_rivals, seed, n_ours, settings.min_weight, settings.max_weight)
    returns = {name: grid_realized[position] for name, position in positions.items()} | {"spy": realized[-1]}
    row = {f"{name}_return": float(value) for name, value in returns.items()}
    row["pick_percentile"] = float((grid_realized <= returns["pick"]).mean())
    row |= {f"{name}_field_win": float((value > field_best).mean()) for name, value in returns.items()}
    row["field_winner_median"] = float(np.median(field_best))
    return row
