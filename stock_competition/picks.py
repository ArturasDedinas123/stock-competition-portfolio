"""Free-choice search: the best contest portfolio when any candidate stock may be picked."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .parallel import run_parallel
from .search import CELLS_PER_TASK, enumerate_grid, holdings_label, score_portfolios, weights_from_units
from .simulation import Scenarios


@dataclass
class FreeChoice:
    """Result of :func:`search_free_choice`.

    Attributes:
        stocks: The chosen stocks.
        weights: Their weights, in the same order.
        p_win: P(win) averaged across views on the search simulations.
        history: One row per step of every search: starting point, round, holdings and P(win).
    """

    stocks: list[str]
    weights: np.ndarray
    p_win: float
    history: pd.DataFrame


def portfolio_label(weights, stocks: Sequence[str], min_w: float) -> str:
    """All holdings in one line, e.g. ``'NVDA 20% · SNAP 20% · 5% each: AAPL, JPM'``.

    Positions are ordered by weight, then ticker, so the same portfolio always gets the same label.
    """
    order = sorted(range(len(stocks)), key=lambda i: (-float(weights[i]), stocks[i]))
    stocks, weights = [stocks[i] for i in order], np.asarray(weights, dtype=float)[order]
    at_minimum = [s for s, w in zip(stocks, weights) if w <= min_w + 1e-6]
    parts = [holdings_label(weights, stocks, min_w)] if len(at_minimum) < len(stocks) else []
    if at_minimum:
        parts.append(f"{min_w:.0%} each: {', '.join(sorted(at_minimum))}")
    return " · ".join(parts)


def single_stock_win_rates(scenarios: Scenarios, stocks: Sequence[str]) -> pd.Series:
    """How often each stock on its own would beat the winning rival, averaged across views.

    This ignores the rules (a single stock at 100%); it is a quick measure of how much winning
    upside a stock brings.
    """
    per_view = {v: np.count_nonzero(scenarios.returns_of(v, stocks) > scenarios.fields[v].best[:, None], axis=0)
                / scenarios.n_sims for v in scenarios.views}
    return pd.Series(scenarios.average(per_view), index=pd.Index(list(stocks), name="ticker")).sort_values(
        ascending=False)


def best_weights(scenarios: Scenarios, stocks: Sequence[str], units: np.ndarray, min_w: float,
                 step: float) -> tuple[np.ndarray, float]:
    """The weights with the highest P(win) for a fixed set of stocks, searched over a weight grid."""
    wins = scenarios.average({v: score_portfolios(units, min_w, step, scenarios.returns_of(v, stocks),
                                                  scenarios.fields[v].best, verbose=False)["win"] / scenarios.n_sims
                              for v in scenarios.views})
    best = int(np.argmax(wins))
    return np.round(weights_from_units(units[best], min_w, step).astype(float), 6), float(wins[best])


def best_swap(scenarios: Scenarios, stocks: Sequence[str], weights: np.ndarray,
              candidates: Sequence[str]) -> tuple[int, str, float]:
    """The single swap with the highest P(win): replace one held stock by a candidate at the same weight.

    Every (held stock, candidate) pair is scored at once from one base portfolio, so a round costs
    about as much as scoring ``len(stocks) * len(candidates)`` portfolios directly would, without
    building them. Returns (position replaced, candidate, P(win)).
    """
    column = {c: i for i, c in enumerate(scenarios.columns)}
    held_at = [column[s] for s in stocks]
    candidate_at = np.array([column[c] for c in candidates])
    blocked = np.isin(candidate_at, held_at)
    block = max(1, CELLS_PER_TASK // len(candidates))
    blocks = list(range(0, scenarios.n_sims, block))
    counts = np.zeros((len(scenarios.views), len(stocks), len(blocks), len(candidates)))
    bases = {v: scenarios.returns[v][:, held_at] @ weights.astype(np.float32) for v in scenarios.views}

    def work(task: tuple[int, int, int]) -> None:
        view_index, position, block_index = task
        view = scenarios.views[view_index]
        rows = slice(blocks[block_index], blocks[block_index] + block)
        returns = scenarios.returns[view][rows]
        weight = np.float32(weights[position])
        partial = bases[view][rows] - weight * returns[:, held_at[position]]
        swapped = partial[:, None] + weight * returns[:, candidate_at]
        counts[view_index, position, block_index] = np.count_nonzero(
            swapped > scenarios.fields[view].best[rows, None], axis=0)

    run_parallel(work, [(v, p, b) for v in range(len(scenarios.views)) for p in range(len(stocks))
                        for b in range(len(blocks))])
    per_view = counts.sum(axis=2) / scenarios.n_sims  # (views, positions, candidates)
    p_win = scenarios.average(dict(zip(scenarios.views, per_view)))
    p_win[:, blocked] = -np.inf
    position, candidate = divmod(int(np.argmax(p_win)), len(candidates))
    return position, candidates[candidate], float(p_win[position, candidate])


def search_free_choice(scenarios: Scenarios, candidates: Sequence[str], starts: Mapping[str, Sequence[str]],
                       min_w: float, max_w: float, step: float = 0.05, min_gain: float = 1e-4, max_rounds: int = 15,
                       verbose: bool = True) -> FreeChoice:
    """Search for the stocks and weights with the highest P(win) when any candidate may be picked.

    From each starting set: find the best weights on a ``step`` grid, then the best single swap;
    keep swapping (and re-optimizing weights) while P(win) improves by more than ``min_gain``.
    The best result over all starting points is returned.
    """
    units = enumerate_grid(len(next(iter(starts.values()))), min_w, max_w, step)
    results = [_climb(scenarios, name, start, candidates, units, min_w, step, min_gain, max_rounds, verbose)
               for name, start in starts.items()]
    best = max(results, key=lambda result: result.p_win)
    history = pd.concat([result.history for result in results], ignore_index=True)
    return FreeChoice(best.stocks, best.weights, best.p_win, history)


def _climb(scenarios: Scenarios, name: str, start: Sequence[str], candidates: Sequence[str], units: np.ndarray,
           min_w: float, step: float, min_gain: float, max_rounds: int, verbose: bool) -> FreeChoice:
    """Swap stocks one at a time from one starting set while P(win) keeps improving."""
    stocks = list(start)
    weights, p_win = best_weights(scenarios, stocks, units, min_w, step)
    rows = [{"start": name, "round": 0, "holdings": portfolio_label(weights, stocks, min_w), "P(win)": p_win}]
    for round_number in range(1, max_rounds + 1):
        position, candidate, swap_p_win = best_swap(scenarios, stocks, weights, candidates)
        if swap_p_win <= p_win + min_gain:
            break
        replaced, stocks[position] = stocks[position], candidate
        weights, p_win = best_weights(scenarios, stocks, units, min_w, step)
        rows.append({"start": name, "round": round_number, "holdings": portfolio_label(weights, stocks, min_w),
                     "P(win)": p_win})
        if verbose:
            print(f"  {name} · round {round_number}: {replaced} → {candidate} · P(win) {p_win:.2%}")
    return FreeChoice(stocks, weights, p_win, pd.DataFrame(rows))
