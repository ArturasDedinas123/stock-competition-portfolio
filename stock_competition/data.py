"""Prices, analyst targets and earnings dates from Yahoo Finance, cached per calendar day."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from .cache import cache_path, prune_cache, read_frame

# yfinance logs transient HTTP errors it recovers from; keep notebook output readable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BATCH_SIZE = 100  # tickers per download request


def _today() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _download_batch(tickers: list[str], start: str | None) -> pd.DataFrame:
    """Adjusted closes for one request; tickers that fail come back as all-NaN columns."""
    if start:
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    else:
        raw = yf.download(tickers, period="max", auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers, dtype=float)
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    closes = closes.reindex(columns=tickers)
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
    return closes


def _download_in_batches(tickers: list[str], start: str | None) -> pd.DataFrame:
    return pd.concat([_download_batch(tickers[i:i + BATCH_SIZE], start)
                      for i in range(0, len(tickers), BATCH_SIZE)], axis=1)


def _download_closes(tickers: list[str], start: str | None) -> pd.DataFrame:
    """Adjusted closes in batches of ``BATCH_SIZE``; tickers with no data get one more try."""
    closes = _download_in_batches(tickers, start)
    empty = closes.reindex(columns=tickers).isna().all(axis=0).to_numpy()
    failed = [t for t, no_data in zip(tickers, empty) if no_data]
    if failed:
        time.sleep(2)
        closes = closes.drop(columns=failed).join(_download_in_batches(failed, start), how="outer")
    return closes.reindex(columns=tickers).sort_index()


def load_prices(tickers, start: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes (splits and dividends included).

    Args:
        tickers: Ticker symbols.
        start: First date to download (``"YYYY-MM-DD"``); None downloads each ticker's full history.
        refresh: Download again even if today's cache already has the tickers.

    Results are cached per day and start date in ``data/``. Tickers that fail to download come back
    as all-NaN columns. Cache files older than a few days are deleted when a new day's file is created.
    """
    tickers = list(dict.fromkeys(tickers))
    path = cache_path(f"prices_{_today()}_{start or 'max'}.pkl")
    if not path.exists():
        prune_cache()
    cached = read_frame(path) if path.exists() else pd.DataFrame()
    wanted = tickers if refresh else [t for t in tickers if t not in cached.columns]
    if wanted:
        fresh = _download_closes(wanted, start)
        if len(cached.columns):
            fresh = cached.drop(columns=wanted, errors="ignore").join(fresh, how="outer")
        cached = fresh.sort_index()
        cached.to_pickle(path)
    return cached.reindex(columns=tickers).dropna(how="all")


def _fetch_target_price(ticker: str) -> float | None:
    try:
        return yf.Ticker(ticker).info.get("targetMeanPrice")
    except Exception:  # pylint: disable=broad-exception-caught  # any network/parse failure means "no target"
        return None


def load_analyst_targets(tickers, refresh: bool = False) -> pd.Series:
    """Mean analyst 12-month price target per ticker (NaN where unavailable), cached per day."""
    tickers = list(dict.fromkeys(tickers))
    path = cache_path(f"analyst_targets_{_today()}.csv")
    if path.exists() and not refresh:
        cached = pd.read_csv(path, index_col=0)["target_mean_price"]
        if isinstance(cached, pd.Series) and set(tickers) <= set(cached.index):
            return cached.reindex(tickers)

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(_fetch_target_price, tickers))
    # Parallel requests occasionally fail with a transient auth error; retry those one at a time.
    for i, ticker in enumerate(tickers):
        for _ in range(2):
            if values[i] is not None:
                break
            time.sleep(1)
            values[i] = _fetch_target_price(ticker)
    targets = pd.Series(values, index=pd.Index(tickers, name="ticker"), name="target_mean_price", dtype=float)
    targets.to_csv(path)
    return targets


def next_earnings(tickers, start, end) -> pd.DataFrame:
    """Next scheduled earnings date per ticker and whether it falls inside ``(start, end]``."""
    rows = []
    for ticker in tickers:
        try:
            calendar = yf.Ticker(ticker).calendar
            dates = list(calendar.get("Earnings Date", [])) if isinstance(calendar, dict) else []
        except Exception:  # pylint: disable=broad-exception-caught  # informational only
            dates = []
        upcoming = pd.Timestamp(dates[0]) if dates else pd.NaT
        rows.append({
            "ticker": ticker,
            "next_earnings": upcoming,
            "in_competition_window": bool(dates) and pd.Timestamp(start) < upcoming <= pd.Timestamp(end),
        })
    return pd.DataFrame(rows).set_index("ticker")


def shared_window_start(prices: pd.DataFrame, tickers, skip_days: int = 20, history_start="max") -> pd.Timestamp:
    """First date on which every ticker has traded for at least ``skip_days`` days."""
    newest_listing = prices[tickers].apply(pd.Series.first_valid_index).max()
    position = int(prices.index.searchsorted(newest_listing)) + skip_days
    start = pd.Timestamp(prices.index.to_numpy()[position])
    if not isinstance(start, pd.Timestamp):
        raise ValueError("The price index must contain dates")
    limit = pd.Timestamp(history_start) if history_start != "max" else None
    if isinstance(limit, pd.Timestamp) and limit > start:
        start = limit
    return start


def window_prices(prices: pd.DataFrame, required, optional, start, max_gap: int = 5):
    """Prices from ``start`` on. Required tickers must be complete; optional ones with gaps are dropped.

    Returns ``(window, dropped)``. Gaps of up to ``max_gap`` days are forward-filled first.
    """
    window = prices.loc[start:, list(required) + list(optional)].ffill(limit=max_gap)
    missing = window[required].isna().any()
    if missing.any():
        raise ValueError(f"Missing prices for {list(missing[missing].index)} after {start:%Y-%m-%d}")
    dropped = [t for t in optional if window[t].isna().any()]
    return window.drop(columns=dropped), dropped
