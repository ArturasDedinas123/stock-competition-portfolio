"""Simulated rival portfolios: random stock picks and weights under the competition rules."""

import math
from dataclasses import dataclass

import numpy as np

from .parallel import run_parallel

CHUNK = 5_000  # simulations per thread task; rivals are seeded per chunk, so this must stay fixed


@dataclass
class RivalField:
    """Best and 3rd-best rival return in every simulation of one view."""

    best: np.ndarray
    third: np.ndarray


@dataclass
class WinnerProfile:
    """What the winning rivals held, compared with all rivals, in one view.

    Holdings arrays are indexed like the stock pool: how many portfolios held each stock, and the
    summed weight of that stock across those portfolios.
    """

    winner_holdings: np.ndarray
    winner_weight: np.ndarray
    winner_top3_weight: float  # summed weight of each winner's three largest positions
    rival_holdings: np.ndarray
    rival_weight: np.ndarray
    n_simulations: int
    n_rivals: int

    def merge(self, other: "WinnerProfile") -> "WinnerProfile":
        """Combine the counts of two batches of simulations."""
        return WinnerProfile(
            self.winner_holdings + other.winner_holdings, self.winner_weight + other.winner_weight,
            self.winner_top3_weight + other.winner_top3_weight, self.rival_holdings + other.rival_holdings,
            self.rival_weight + other.rival_weight, self.n_simulations + other.n_simulations, self.n_rivals,
        )


def random_bounded_weights(shape, rng: np.random.Generator, min_w: float, max_w: float) -> np.ndarray:
    """Random weights along the last axis, each within ``[min_w, max_w]`` and summing to 1.

    The free weight above the minimums is spread uniformly at random (a flat Dirichlet draw);
    draws that break the maximum are redrawn.
    """
    n = shape[-1]
    free, cap = 1 - n * min_w, max_w - min_w
    weights = _uniform_simplex(shape, rng) * free
    bad = (weights > cap).any(axis=-1)
    while bad.any():
        weights[bad] = _uniform_simplex((int(bad.sum()), n), rng) * free
        bad = (weights > cap).any(axis=-1)
    return (min_w + weights).astype(np.float32)


def _uniform_simplex(shape, rng: np.random.Generator) -> np.ndarray:
    draws = rng.standard_exponential(shape, dtype=np.float32)
    return draws / draws.sum(axis=-1, keepdims=True)


def sample_picks(shape: tuple[int, ...], pool_size: int, n_stocks: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct random stock positions, shape ``(*shape, n_stocks)``, uniform over all possible picks.

    Large pools draw positions and redraw portfolios with a repeat, so the cost doesn't grow with
    the pool; small pools (where repeats are common) sort random keys instead.
    """
    if pool_size < n_stocks:
        raise ValueError(f"The pool has {pool_size} stocks; rivals need {n_stocks}.")
    if pool_size < 4 * n_stocks:
        keys = rng.random((*shape, pool_size), dtype=np.float32)
        return np.argpartition(keys, n_stocks - 1, axis=-1)[..., :n_stocks]
    picks = rng.integers(0, pool_size, size=(*shape, n_stocks), dtype=np.int32)
    while True:
        ordered = np.sort(picks, axis=-1)
        repeated = (ordered[..., 1:] == ordered[..., :-1]).any(axis=-1)
        if not repeated.any():
            return picks
        picks[repeated] = rng.integers(0, pool_size, size=(int(repeated.sum()), n_stocks), dtype=np.int32)


def _draw_rivals(seed: int, chunk_id: int, size: int, n_rivals: int, pool_size: int, n_stocks: int,
                 min_w: float, max_w: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, chunk_id])
    picks = sample_picks((size, n_rivals), pool_size, n_stocks, rng)
    return picks, random_bounded_weights((size, n_rivals, n_stocks), rng, min_w, max_w)


def simulate_field(pool_returns: dict[str, np.ndarray], n_rivals: int, seed: int, n_stocks: int = 10,
                   min_w: float = 0.05, max_w: float = 0.20, first_sim: int = 0, profile: bool = False,
                   workers: int | None = None) -> tuple[dict[str, RivalField], dict[str, WinnerProfile]]:
    """Simulate the rival field in every view.

    Each simulation gets ``n_rivals`` fresh rivals; each holds ``n_stocks`` distinct random stocks
    from the pool with random weights. Rivals are drawn once per chunk of simulations and valued in
    every view, so all views compete against exactly the same rivals. They depend only on ``seed``
    and the simulation's position (``first_sim`` + row), not on the number of threads.

    Args:
        pool_returns: View name -> simulated returns of the pool stocks, shape (n_sims, pool_size).
        n_rivals: Rivals per simulation.
        seed: Seed for the rivals' picks and weights.
        n_stocks: Stocks per rival portfolio.
        min_w: Minimum weight per stock.
        max_w: Maximum weight per stock.
        first_sim: Position of the first row in a longer run (a multiple of ``CHUNK``).
        profile: Also record what the winning rivals held.
        workers: Worker threads (default: one per CPU core).

    Returns:
        ``(fields, profiles)``: per-view :class:`RivalField`, and per-view :class:`WinnerProfile`
        (empty unless ``profile``).
    """
    if first_sim % CHUNK:
        raise ValueError(f"first_sim must be a multiple of {CHUNK}")
    views = list(pool_returns)
    n_sims, pool_size = pool_returns[views[0]].shape
    n_chunks = math.ceil(n_sims / CHUNK)
    fields = {v: RivalField(np.empty(n_sims, np.float32), np.empty(n_sims, np.float32)) for v in views}
    winner_counts = np.zeros((n_chunks, len(views), 2, pool_size))
    winner_top3 = np.zeros((n_chunks, len(views)))
    rival_counts = np.zeros((n_chunks, 2, pool_size))
    k = min(2, n_rivals - 1)

    def work(index: int) -> None:
        start, stop = index * CHUNK, min(n_sims, (index + 1) * CHUNK)
        size = stop - start
        picks, weights = _draw_rivals(seed, first_sim // CHUNK + index, size, n_rivals, pool_size, n_stocks,
                                      min_w, max_w)
        rows = np.arange(size)
        if profile:
            rival_counts[index] = [np.bincount(picks.ravel(), minlength=pool_size),
                                   np.bincount(picks.ravel(), weights=weights.ravel(), minlength=pool_size)]
        for j, view in enumerate(views):
            returns = (pool_returns[view][start:stop][rows[:, None, None], picks] * weights).sum(axis=2)
            fields[view].best[start:stop] = returns.max(axis=1)
            fields[view].third[start:stop] = -np.partition(-returns, k, axis=1)[:, k]
            if profile:
                winner = returns.argmax(axis=1)
                held, held_weights = picks[rows, winner], weights[rows, winner]
                winner_counts[index, j] = [np.bincount(held.ravel(), minlength=pool_size),
                                           np.bincount(held.ravel(), weights=held_weights.ravel(), minlength=pool_size)]
                winner_top3[index, j] = np.sort(held_weights, axis=1)[:, -3:].sum()

    run_parallel(work, range(n_chunks), workers)
    profiles = {}
    if profile:
        totals, top3, rival_totals = winner_counts.sum(axis=0), winner_top3.sum(axis=0), rival_counts.sum(axis=0)
        profiles = {view: WinnerProfile(totals[j, 0], totals[j, 1], float(top3[j]), rival_totals[0], rival_totals[1],
                                        n_sims, n_rivals)
                    for j, view in enumerate(views)}
    return fields, profiles


def rival_returns(pool_returns: np.ndarray, n_rivals: int, seed: int, n_stocks: int = 10, min_w: float = 0.05,
                  max_w: float = 0.20, first_sim: int = 0, workers: int | None = None) -> np.ndarray:
    """Every rival's return in every simulation, shape (n_sims, n_rivals).

    Uses the same rivals as :func:`simulate_field` with the same arguments.
    """
    n_sims, pool_size = pool_returns.shape
    every = np.empty((n_sims, n_rivals), np.float32)

    def work(index: int) -> None:
        start, stop = index * CHUNK, min(n_sims, (index + 1) * CHUNK)
        picks, weights = _draw_rivals(seed, first_sim // CHUNK + index, stop - start, n_rivals, pool_size, n_stocks,
                                      min_w, max_w)
        rows = np.arange(stop - start)[:, None, None]
        every[start:stop] = (pool_returns[start:stop][rows, picks] * weights).sum(axis=2)

    run_parallel(work, range(math.ceil(n_sims / CHUNK)), workers)
    return every
