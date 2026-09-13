"""Simulated rival portfolios: random stock picks and weights under the competition rules."""

import math

import numpy as np

from .parallel import run_parallel


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


def _chunk_returns(pool_returns: np.ndarray, index: int, chunk: int, n_rivals: int, seed: int, n_stocks: int,
                   min_w: float, max_w: float) -> np.ndarray:
    """Returns of every rival in simulations ``[index * chunk, (index + 1) * chunk)``."""
    n_sims, pool_size = pool_returns.shape
    if pool_size < n_stocks:
        raise ValueError(f"The pool has {pool_size} stocks; rivals need {n_stocks}.")
    start, stop = index * chunk, min(n_sims, (index + 1) * chunk)
    rng = np.random.default_rng([seed, index])
    keys = rng.random((stop - start, n_rivals, pool_size), dtype=np.float32)
    picks = np.argpartition(keys, n_stocks - 1, axis=2)[..., :n_stocks]
    weights = random_bounded_weights((stop - start, n_rivals, n_stocks), rng, min_w, max_w)
    rows = np.arange(stop - start)[:, None, None]
    return (pool_returns[start:stop][rows, picks] * weights).sum(axis=2)


def rival_returns(pool_returns: np.ndarray, n_rivals: int, seed: int, n_stocks: int = 10, min_w: float = 0.05,
                  max_w: float = 0.20, chunk: int = 5_000, workers: int | None = None) -> np.ndarray:
    """Every rival's return in every simulation, shape (n_sims, n_rivals).

    Each simulation gets ``n_rivals`` fresh rivals; each rival holds ``n_stocks`` distinct random
    stocks from the pool with random weights. Rivals depend only on ``seed`` and ``chunk``, so every
    view and every thread count sees exactly the same rivals.

    Args:
        pool_returns: Simulated returns of the stocks rivals can pick, shape (n_sims, pool_size).
        n_rivals: Rivals per simulation.
        seed: Seed for the rivals' picks and weights.
        n_stocks: Stocks per rival portfolio.
        min_w: Minimum weight per stock.
        max_w: Maximum weight per stock.
        chunk: Simulations per thread task.
        workers: Worker threads (default: one per CPU core).
    """
    every = np.empty((len(pool_returns), n_rivals), np.float32)

    def work(index: int) -> None:
        returns = _chunk_returns(pool_returns, index, chunk, n_rivals, seed, n_stocks, min_w, max_w)
        every[index * chunk: index * chunk + len(returns)] = returns

    run_parallel(work, range(math.ceil(len(pool_returns) / chunk)), workers)
    return every


def rival_field(pool_returns: np.ndarray, n_rivals: int, seed: int, n_stocks: int = 10, min_w: float = 0.05,
                max_w: float = 0.20, chunk: int = 5_000, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Best and 3rd-best rival return in every simulation.

    Uses the same rivals as :func:`rival_returns` with the same arguments, without storing every
    rival's return. Returns ``(best, third)``, each of shape (n_sims,).
    """
    n_sims = len(pool_returns)
    best = np.empty(n_sims, np.float32)
    third = np.empty(n_sims, np.float32)
    k = min(2, n_rivals - 1)

    def work(index: int) -> None:
        returns = _chunk_returns(pool_returns, index, chunk, n_rivals, seed, n_stocks, min_w, max_w)
        start = index * chunk
        best[start:start + len(returns)] = returns.max(axis=1)
        third[start:start + len(returns)] = -np.partition(-returns, k, axis=1)[:, k]

    run_parallel(work, range(math.ceil(n_sims / chunk)), workers)
    return best, third
