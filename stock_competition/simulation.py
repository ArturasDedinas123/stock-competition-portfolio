"""Simulated competitions: horizon returns in every expected-return view, plus the rival field."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .rivals import CHUNK, RivalField, WinnerProfile, simulate_field
from .scenarios import VIEW_LABELS, apply_view, bootstrap_log_returns
from .search import outcome_rates, weighted_quantiles, win_rates
from .settings import StrategySettings


@dataclass
class Scenarios:
    """A simulated competition, ready to score portfolios.

    Attributes:
        columns: Stocks whose simulated returns are kept; portfolios can hold any of them.
        returns: View -> simulated simple returns over the horizon, shape (n_sims, len(columns)).
        fields: View -> best and 3rd-best rival return in each simulation.
        view_share: View -> weight in averages across views (sums to 1).
        pool: Stocks the rivals pick from.
        profiles: View -> what the winning rivals held (empty unless simulated with ``profile=True``).
    """

    columns: list[str]
    returns: dict[str, np.ndarray]
    fields: dict[str, RivalField]
    view_share: pd.Series
    pool: list[str] = field(default_factory=list)
    profiles: dict[str, WinnerProfile] = field(default_factory=dict)

    @property
    def views(self) -> list[str]:
        """Simulated views, in order."""
        return list(self.returns)

    @property
    def n_sims(self) -> int:
        """Number of simulated competitions."""
        return len(self.returns[self.views[0]])

    def _positions(self, stocks: Sequence[str]) -> list[int]:
        index = {c: i for i, c in enumerate(self.columns)}
        return [index[s] for s in stocks]

    def average(self, per_view: Mapping[str, np.ndarray]) -> np.ndarray:
        """Weighted average across views of per-view arrays of equal shape."""
        return np.tensordot(self.view_share.reindex(self.views).to_numpy(), np.stack([per_view[v] for v in self.views]),
                            axes=1)

    def returns_of(self, view: str, stocks: Sequence[str]) -> np.ndarray:
        """Simulated returns of ``stocks`` in one view, shape (n_sims, len(stocks))."""
        return self.returns[view][:, self._positions(stocks)]

    def p_win(self, weights: np.ndarray, stocks: Sequence[str]) -> np.ndarray:
        """P(win) averaged across views for each row of ``weights`` (columns ordered as ``stocks``)."""
        positions = self._positions(stocks)
        weights = np.atleast_2d(weights)
        return self.average({v: win_rates(weights, self.returns[v][:, positions], self.fields[v].best)
                              for v in self.views})

    def evaluate(self, portfolios: pd.DataFrame, quantiles: bool = True) -> pd.DataFrame:
        """Contest statistics for each portfolio (rows; columns are stocks, values are weights).

        Returns P(win), P(top 3), P(loss) and mean return averaged across views; with ``quantiles``
        also the 5th percentile, median and 95th percentile of all views pooled; then P(win) per view.
        """
        positions = self._positions(list(portfolios.columns))
        weights = portfolios.to_numpy(np.float32)
        per_view = {v: outcome_rates(weights, self.returns[v][:, positions], self.fields[v].best, self.fields[v].third)
                    for v in self.views}
        table = pd.DataFrame({
            "P(win)": self.average({v: r["p_win"] for v, r in per_view.items()}),
            "P(top 3)": self.average({v: r["p_top3"] for v, r in per_view.items()}),
            "P(loss)": self.average({v: r["p_loss"] for v, r in per_view.items()}),
            "mean": self.average({v: r["mean"] for v, r in per_view.items()}),
        }, index=portfolios.index)
        if quantiles:
            ranges = np.array([self._quantiles(row, [0.05, 0.5, 0.95]) for _, row in portfolios.iterrows()])
            table["5th pct"], table["median"], table["95th pct"] = ranges[:, 0], ranges[:, 1], ranges[:, 2]
        for v in self.views:
            table[f"P(win) {VIEW_LABELS.get(v, v)}"] = per_view[v]["p_win"]
        return table

    def portfolio_returns(self, weights: pd.Series) -> dict[str, np.ndarray]:
        """View -> simulated returns of one portfolio (index = stocks, values = weights)."""
        positions = self._positions(list(weights.index))
        w = weights.to_numpy(np.float32)
        return {v: self.returns[v][:, positions] @ w for v in self.views}

    def _quantiles(self, weights: pd.Series, levels: Sequence[float]) -> np.ndarray:
        per_view = self.portfolio_returns(weights)
        return weighted_quantiles([per_view[v] for v in self.views], self.view_share[self.views].tolist(), levels)

    def pooled(self, weights: pd.Series) -> np.ndarray:
        """Simulated returns of one portfolio, all views stacked (for charts)."""
        return np.concatenate(list(self.portfolio_returns(weights).values()))

    def pooled_best(self) -> np.ndarray:
        """Winning rival returns, all views stacked (for charts)."""
        return np.concatenate([self.fields[v].best for v in self.views])

    def winner_table(self) -> pd.DataFrame:
        """What winning rivals held, per pool stock, averaged across views.

        Columns: share of all rivals holding the stock, share of winners holding it, lift (winner
        share / rival share), and the average weight when held by a rival and by a winner.
        """
        if not self.profiles:
            raise ValueError("Simulate with profile=True to record what the winners held.")
        first = self.profiles[self.views[0]]
        held_by_rivals = first.rival_holdings / (first.n_simulations * first.n_rivals)
        held_by_winners = self.average({v: p.winner_holdings / p.n_simulations for v, p in self.profiles.items()})
        winner_weight = self.average({v: p.winner_weight / np.maximum(p.winner_holdings, 1)
                                       for v, p in self.profiles.items()})
        return pd.DataFrame({
            "held by rivals": held_by_rivals,
            "held by winners": held_by_winners,
            "lift": held_by_winners / held_by_rivals,
            "avg weight (rivals)": first.rival_weight / np.maximum(first.rival_holdings, 1),
            "avg weight (winners)": winner_weight,
        }, index=pd.Index(self.pool, name="ticker")).sort_values("lift", ascending=False)

    def winner_top3_weight(self) -> float:
        """Average combined weight of the winning rivals' three largest positions."""
        return float(sum(self.view_share[v] * p.winner_top3_weight / p.n_simulations for v, p in self.profiles.items()))


def _locate(columns: Sequence[str], stocks: Sequence[str]) -> slice | list[int]:
    """Positions of ``stocks`` in ``columns``: a slice when consecutive (avoids copying), else a list."""
    index = {c: i for i, c in enumerate(columns)}
    positions = [index[s] for s in stocks]
    if positions and positions == list(range(positions[0], positions[0] + len(positions))):
        return slice(positions[0], positions[0] + len(positions))
    return positions


def _views_to_simulate(view_weights: Mapping[str, float], targets: pd.DataFrame) -> list[str]:
    views = [v for v, weight in view_weights.items() if weight > 0]
    missing = [v for v in views if v != "trend" and v not in targets.columns]
    if missing:
        raise ValueError(f"No expected returns for views {missing}")
    return views


def _simulate_chunk(values: np.ndarray, view_targets: Mapping[str, np.ndarray | None], settings: StrategySettings,
                    size: int, rng: np.random.Generator, seed: int, first_sim: int, pool_at: slice | list[int],
                    profile: bool):
    """One batch of simulated futures: per-view returns of every stock, and the rival field."""
    totals, expected = bootstrap_log_returns(values, settings.horizon, size, settings.block_days, rng,
                                             settings.half_life_days)
    view_returns = {v: apply_view(totals, expected, target) for v, target in view_targets.items()}
    fields, profiles = simulate_field({v: r[:, pool_at] for v, r in view_returns.items()}, settings.n_rivals, seed + 1,
                                      settings.portfolio_size, settings.min_weight, settings.max_weight,
                                      first_sim=first_sim, profile=profile)
    return view_returns, fields, profiles


def simulate(log_returns: pd.DataFrame, targets: pd.DataFrame, settings: StrategySettings, *, n_sims: int, seed: int,
             keep: Sequence[str], pool: Sequence[str], view_weights: Mapping[str, float], chunk: int = 100_000,
             profile: bool = False) -> Scenarios:
    """Simulate the competition: horizon returns in each view and the rival field in each simulation.

    Futures are generated ``chunk`` at a time, so memory stays bounded for any ``n_sims``.

    Args:
        log_returns: Daily log returns of every stock involved, without missing values.
        targets: Expected horizon return per stock (rows) for each view (columns), from
            :func:`scenarios.drift_scenarios`. The "trend" view keeps historical drift and needs no column.
        settings: Competition rules and model settings.
        n_sims: Number of simulated competitions.
        seed: Random seed for the bootstrap and the rivals.
        keep: Stocks whose simulated returns are kept for building portfolios.
        pool: Stocks the rivals pick from.
        view_weights: View -> weight in averages; views with weight 0 are skipped.
        chunk: Simulations generated at a time; a multiple of ``rivals.CHUNK``.
        profile: Also record what the winning rivals held.
    """
    if chunk % CHUNK:
        raise ValueError(f"chunk must be a multiple of {CHUNK}")
    views = _views_to_simulate(view_weights, targets)
    keep_at, pool_at = _locate(list(log_returns.columns), keep), _locate(list(log_returns.columns), pool)
    view_targets = {v: None if v == "trend" else targets[v].reindex(log_returns.columns).to_numpy() for v in views}
    rng = np.random.default_rng(seed)

    kept = {v: np.empty((n_sims, len(keep)), np.float32) for v in views}
    fields = {v: RivalField(np.empty(n_sims, np.float32), np.empty(n_sims, np.float32)) for v in views}
    profiles: dict[str, WinnerProfile] = {}
    for start in range(0, n_sims, chunk):
        stop = min(n_sims, start + chunk)
        view_returns, chunk_fields, chunk_profiles = _simulate_chunk(log_returns.to_numpy(), view_targets, settings,
                                                                     stop - start, rng, seed, start, pool_at, profile)
        for v in views:
            kept[v][start:stop] = view_returns[v][:, keep_at]
            fields[v].best[start:stop], fields[v].third[start:stop] = chunk_fields[v].best, chunk_fields[v].third
            if v in chunk_profiles:
                profiles[v] = profiles[v].merge(chunk_profiles[v]) if v in profiles else chunk_profiles[v]

    share = pd.Series({v: float(view_weights[v]) for v in views})
    return Scenarios(list(keep), kept, fields, share / share.sum(), list(pool), profiles)
