"""Tests for the offline parts of data handling, caching and statistics."""

import os
import time

import numpy as np
import pandas as pd
import pytest

from stock_competition import cache, data, stats

from .conftest import BENCHMARK, OURS


def test_window_starts_after_the_newest_listing(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[:100, 3] = np.nan  # S3 lists 100 days later
    start = data.shared_window_start(prices, OURS, skip_days=20)
    assert start == prices.index[120]
    assert data.shared_window_start(prices, OURS, 20, "2021-01-01") == pd.Timestamp("2021-01-01")


def test_window_prices_drop_optional_gaps_and_reject_required_gaps(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[500:510, prices.columns.get_loc("R0")] = np.nan
    window, dropped = data.window_prices(prices, OURS, ["R0", "R1"], prices.index[0])
    assert dropped == ["R0"] and "R1" in window.columns
    prices.iloc[600:610, 0] = np.nan
    with pytest.raises(ValueError):
        data.window_prices(prices, OURS, [], prices.index[0])


def test_load_prices_batches_retries_and_caches(tmp_path, monkeypatch, synthetic_prices):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "BATCH_SIZE", 4)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    requests = []
    flaky = {"S5"}  # fails on the first request only

    def fake_batch(tickers, start):
        requests.append((list(tickers), start))
        frame = synthetic_prices.loc[start:, [t for t in tickers if t in synthetic_prices]].copy()
        for ticker in tickers:
            if ticker in flaky or ticker not in synthetic_prices:
                frame[ticker] = np.nan
                flaky.discard(ticker)
        return frame.reindex(columns=tickers)

    monkeypatch.setattr(data, "_download_batch", fake_batch)
    tickers = OURS + ["NOPE"]
    prices = data.load_prices(tickers, start="2020-01-01")
    assert [len(batch) for batch, _ in requests[:3]] == [4, 4, 3]
    assert all(start == "2020-01-01" for _, start in requests)
    assert sorted(requests[3][0]) == ["NOPE", "S5"]  # one retry for tickers without data
    assert bool(prices["S5"].notna().to_numpy().all()) and "NOPE" in prices.columns
    assert prices.index[0] >= pd.Timestamp("2020-01-01")
    calls_before = len(requests)
    again = data.load_prices(OURS[:3], start="2020-01-01")
    assert len(requests) == calls_before and list(again.columns) == OURS[:3]


def test_cached_results_are_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    calls = []

    def compute(value):
        calls.append(value)
        return {"x": np.arange(value)}

    key = cache.cache_key(a=1, b="two")
    assert key == cache.cache_key(b="two", a=1)
    first = cache.cached_npz("demo", key, compute, 3)
    second = cache.cached_npz("demo", key, compute, 3)
    assert calls == [3] and np.array_equal(first["x"], second["x"])


def test_prune_cache_removes_only_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    old, new = tmp_path / "prices_old.csv", tmp_path / "prices_new.csv"
    old.write_text("x")
    new.write_text("x")
    week_ago = time.time() - 7 * 86_400
    os.utime(old, (week_ago, week_ago))
    assert cache.prune_cache(max_age_days=3) == ["prices_old.csv"]
    assert new.exists() and not old.exists()


def test_stock_stats_and_betas(synthetic_prices):
    table = stats.stock_stats(synthetic_prices, OURS + [BENCHMARK], BENCHMARK, horizon=64)
    assert table.loc[BENCHMARK, "beta"] == pytest.approx(1.0)
    assert (table["max_drawdown"] <= 0).all()
    assert table["beta"]["S9"] > table["beta"]["S0"]
    correlation = stats.average_correlation(synthetic_prices, OURS)
    assert correlation.between(-1, 1).all() and len(correlation) == len(OURS)
