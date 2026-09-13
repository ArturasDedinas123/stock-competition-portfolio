"""Exhaustive grid search over portfolio weights, fine-tuning, and portfolio evaluation."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from .parallel import run_parallel

CELLS_PER_TASK = 10_000_000  # portfolios x simulations per thread task (about 40 MB of float32)


def enumerate_grid(n: int = 10, min_w: float = 0.05, max_w: float = 0.20, step: float = 0.025) -> np.ndarray:
    """Every weight vector on a ``step`` grid with ``min_w <= w <= max_w`` that sums to 1.

    Rows are stored compactly as uint8 counts of ``step`` above ``min_w``; convert them with
    :func:`weights_from_units`. With 10 stocks, 5-20% limits and 2.5% steps there are 5,266,030 rows.
    """
    free = round((1 - n * min_w) / step)
    cap = round((max_w - min_w) / step)
    if not (np.isclose(free * step, 1 - n * min_w) and np.isclose(cap * step, max_w - min_w)):
        raise ValueError("step must divide both 1 - n * min_w and max_w - min_w")
    rows = np.zeros((1, 0), dtype=np.uint8)
    sums = np.zeros(1, dtype=np.int16)
    for column in range(n):
        remaining = n - column - 1
        new_rows, new_sums = [], []
        for k in range(cap + 1):
            total = sums + k
            ok = (total <= free) & (free - total <= cap * remaining)
            new_rows.append(np.hstack([rows[ok], np.full((int(ok.sum()), 1), k, dtype=np.uint8)]))
            new_sums.append(total[ok])
        rows, sums = np.vstack(new_rows), np.concatenate(new_sums)
    return rows


def weights_from_units(units: np.ndarray, min_w: float, step: float) -> np.ndarray:
    """Convert grid rows from :func:`enumerate_grid` to float32 weights."""
    return (np.float32(min_w) + units.astype(np.float32) * np.float32(step)).astype(np.float32)


def holdings_label(weights, tickers, min_w: float) -> str:
    """Describe the positions above the minimum weight, e.g. ``'NVDA 20% · SNAP 17.5%'``."""
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(-weights, kind="stable")
    parts = [f"{tickers[i]} {weights[i]:.1%}".replace(".0%", "%") for i in order if weights[i] > min_w + 1e-6]
    return " · ".join(parts) if parts else "all at the minimum"


def _rows_per_task(n_sims: int) -> int:
    return max(1, CELLS_PER_TASK // max(1, n_sims))


def score_portfolios(units: np.ndarray, min_w: float, step: float, returns: np.ndarray, best: np.ndarray,
                     third: np.ndarray | None = None, workers: int | None = None,
                     verbose: bool = True) -> dict[str, np.ndarray]:
    """Count, for every grid portfolio, the simulations where it wins, finishes top 3 and loses money.

    Args:
        units: Grid rows from :func:`enumerate_grid`.
        min_w: Minimum weight used to build the grid.
        step: Grid step used to build the grid.
        returns: Simulated returns of our stocks, shape (n_sims, n_stocks).
        best: Best rival return per simulation.
        third: 3rd-best rival return per simulation (optional).
        workers: Worker threads (default: one per CPU core).
        verbose: Print progress.

    Returns:
        Dict of count arrays ``win``, ``loss`` (and ``top3``), plus ``n_sims``.
    """
    n_sims = len(returns)
    dtype = np.uint16 if n_sims < 2**16 else np.uint32
    returns_t = np.ascontiguousarray(returns.T, dtype=np.float32)
    counts = {"win": np.empty(len(units), dtype), "loss": np.empty(len(units), dtype)}
    if third is not None:
        counts["top3"] = np.empty(len(units), dtype)
    rows = _rows_per_task(n_sims)

    def work(start: int) -> None:
        stop = min(len(units), start + rows)
        port = weights_from_units(units[start:stop], min_w, step) @ returns_t
        counts["win"][start:stop] = np.count_nonzero(port > best, axis=1)
        counts["loss"][start:stop] = np.count_nonzero(port < 0, axis=1)
        if third is not None:
            counts["top3"][start:stop] = np.count_nonzero(port > third, axis=1)

    run_parallel(work, range(0, len(units), rows), workers, label=f"{len(units):,} portfolios" if verbose else None)
    counts["n_sims"] = np.array(n_sims)
    return counts


def evaluate_weights(weights: np.ndarray, returns: np.ndarray, best: np.ndarray, third: np.ndarray,
                     workers: int | None = None) -> pd.DataFrame:
    """P(win), P(top 3), P(loss) and mean return for each row of ``weights``."""
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    returns_t = np.ascontiguousarray(returns.T, dtype=np.float32)
    n_sims = len(returns)
    out = np.empty((len(weights), 4))
    rows = _rows_per_task(n_sims)

    def work(start: int) -> None:
        stop = min(len(weights), start + rows)
        port = weights[start:stop] @ returns_t
        out[start:stop, 0] = np.count_nonzero(port > best, axis=1) / n_sims
        out[start:stop, 1] = np.count_nonzero(port > third, axis=1) / n_sims
        out[start:stop, 2] = np.count_nonzero(port < 0, axis=1) / n_sims
        out[start:stop, 3] = port.mean(axis=1, dtype=np.float64)

    run_parallel(work, range(0, len(weights), rows), workers)
    return pd.DataFrame(out, columns=["p_win", "p_top3", "p_loss", "mean"])


def win_rates(weights: np.ndarray, returns: np.ndarray, best: np.ndarray, workers: int | None = None) -> np.ndarray:
    """P(win) for each row of ``weights``: the share of simulations where it beats the best rival."""
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    returns_t = np.ascontiguousarray(returns.T, dtype=np.float32)
    out = np.empty(len(weights))
    rows = _rows_per_task(len(returns))

    def work(start: int) -> None:
        stop = min(len(weights), start + rows)
        out[start:stop] = np.count_nonzero(weights[start:stop] @ returns_t > best, axis=1) / len(returns)

    run_parallel(work, range(0, len(weights), rows), workers)
    return out


def refine_weights(start_weights: np.ndarray, score: Callable[[np.ndarray], np.ndarray], min_w: float,
                   max_w: float, step: float = 0.01, beam: int = 25, max_rounds: int = 40, verbose: bool = True):
    """Fine-tune portfolios by moving ``step`` of weight from one stock to another.

    Beam search: every round tries all single moves from the ``beam`` best portfolios found so far
    and stops once no new portfolio beats the best score. Weights are handled in basis points, so
    they stay exact and always sum to 100%.

    Args:
        start_weights: Starting portfolios, shape (m, n).
        score: Maps an (k, n) weight array to k scores (higher is better).
        min_w: Minimum weight per stock.
        max_w: Maximum weight per stock.
        step: Size of one move.
        beam: Portfolios expanded per round.
        max_rounds: Upper limit on rounds.
        verbose: Print progress.

    Returns:
        ``(weights, scores)`` for every evaluated portfolio, best first.
    """
    bp = 10_000
    low, high, delta = round(min_w * bp), round(max_w * bp), round(step * bp)
    pool = np.unique(np.rint(np.asarray(start_weights) * bp).astype(np.int32), axis=0)
    scores = np.asarray(score(pool / bp), dtype=float)
    seen = {row.tobytes() for row in pool}
    n = pool.shape[1]
    moves = _single_moves(n, delta)

    for round_number in range(1, max_rounds + 1):
        top = np.argsort(-scores, kind="stable")[:beam]
        best_so_far = scores[top[0]]
        neighbours = (pool[top][:, None, :] + moves[None]).reshape(-1, n)
        neighbours = np.unique(neighbours[((neighbours >= low) & (neighbours <= high)).all(axis=1)], axis=0)
        neighbours = neighbours[[row.tobytes() not in seen for row in neighbours]]
        if neighbours.size == 0:
            break
        seen.update(row.tobytes() for row in neighbours)
        new_scores = np.asarray(score(neighbours / bp), dtype=float)
        pool, scores = np.vstack([pool, neighbours]), np.concatenate([scores, new_scores])
        if verbose:
            print(f"  round {round_number}: {len(neighbours):,} new portfolios · "
                  f"best score {max(best_so_far, new_scores.max()):.3%}")
        if new_scores.max() <= best_so_far:
            break
    order = np.argsort(-scores, kind="stable")
    return pool[order] / bp, scores[order]


def _single_moves(n: int, delta: int) -> np.ndarray:
    """Every way to move ``delta`` from one of ``n`` stocks to another, one move per row."""
    moves = np.zeros((n * (n - 1), n), dtype=np.int32)
    for k, (i, j) in enumerate((i, j) for i in range(n) for j in range(n) if i != j):
        moves[k, i], moves[k, j] = -delta, delta
    return moves


def portfolio_moments(units: np.ndarray, min_w: float, step: float, mu: np.ndarray, cov: np.ndarray,
                      chunk: int = 500_000):
    """Mean return under each row of ``mu`` (shape (views, n)) and standard deviation for every grid portfolio."""
    mu = np.atleast_2d(mu)
    means = np.empty((len(units), len(mu)), np.float32)
    stds = np.empty(len(units), np.float32)
    for start in range(0, len(units), chunk):
        stop = min(len(units), start + chunk)
        weights = weights_from_units(units[start:stop], min_w, step).astype(np.float64)
        means[start:stop] = weights @ mu.T
        stds[start:stop] = np.sqrt(((weights @ cov) * weights).sum(axis=1))
    return means, stds


def weighted_quantiles(samples: list[np.ndarray], sample_weights: list[float], quantiles) -> np.ndarray:
    """Quantiles of a mixture in which each sample array carries a total weight."""
    values = np.concatenate(samples)
    weights = np.concatenate([np.full(len(s), w / len(s)) for s, w in zip(samples, sample_weights)])
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return np.interp(quantiles, cdf / cdf[-1], values[order])
