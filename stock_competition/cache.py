"""Small on-disk cache in ``data/`` for downloads and slow computations."""

import hashlib
import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from .paths import DATA_DIR


def cache_key(**params) -> str:
    """Short, stable hash of the parameters that determine a cached result."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def cached_npz(name: str, key: str, compute: Callable[..., dict], *args, **kwargs) -> dict[str, np.ndarray]:
    """Return ``data/<name>_<key>.npz`` if it exists; otherwise run ``compute(*args, **kwargs)`` and save it."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}_{key}.npz"
    if path.exists():
        print(f"  loaded cached results ({path.name})")
        with np.load(path) as stored:
            return {k: stored[k] for k in stored.files}
    result = compute(*args, **kwargs)
    np.savez(path, **result)
    return result


def cached_frame(name: str, key: str, compute: Callable[..., pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """Return ``data/<name>_<key>.pkl`` if it exists; otherwise run ``compute(*args, **kwargs)`` and save it."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}_{key}.pkl"
    if path.exists():
        print(f"  loaded cached results ({path.name})")
        stored = pd.read_pickle(path)
        if not isinstance(stored, pd.DataFrame):
            raise TypeError(f"{path.name} does not contain a DataFrame")
        return stored
    result = compute(*args, **kwargs)
    result.to_pickle(path)
    return result


def prune_cache(max_age_days: float = 3.0) -> list[str]:
    """Delete cache files older than ``max_age_days`` (prices go stale daily) and return their names."""
    if not DATA_DIR.exists():
        return []
    cutoff = time.time() - max_age_days * 86_400
    removed = []
    for path in DATA_DIR.iterdir():
        if path.is_file() and path.suffix in {".csv", ".npz", ".pkl"} and path.stat().st_mtime < cutoff:
            path.unlink()
            removed.append(path.name)
    return removed
