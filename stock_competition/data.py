"""Prices, analyst targets and earnings dates from Yahoo Finance, cached per calendar day."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from .cache import prune_cache
from .paths import DATA_DIR

# yfinance logs transient HTTP errors it recovers from; keep notebook output readable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _today() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _download_closes(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, period="max", auto_adjust=True, progress=False, threads=True)
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    closes = closes.reindex(columns=tickers)
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    return closes.sort_index()


def load_prices(tickers, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes (splits and dividends included) over each ticker's full history.

    Results are cached in ``data/prices_<date>.csv``. Tickers that fail to download come back as
    all-NaN columns. Cache files older than a few days are deleted when a new day's file is created.
    """
    DATA_DIR.mkdir(exist_ok=True)
    tickers = list(dict.fromkeys(tickers))
    path = DATA_DIR / f"prices_{_today()}.csv"
    if not path.exists():
        prune_cache()
    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    wanted = tickers if refresh else [t for t in tickers if t not in cached.columns]
    if wanted:
        fresh = _download_closes(wanted)
        if len(cached.columns):
            cached = cached.drop(columns=wanted, errors="ignore").join(fresh, how="outer").sort_index()
        else:
            cached = fresh
        cached.to_csv(path)
    return cached[tickers].dropna(how="all")


def _fetch_target_price(ticker: str) -> float | None:
    try:
        return yf.Ticker(ticker).info.get("targetMeanPrice")
    except Exception:  # pylint: disable=broad-exception-caught  # any network/parse failure means "no target"
        return None


def load_analyst_targets(tickers, refresh: bool = False) -> pd.Series:
    """Mean analyst 12-month price target per ticker (NaN where unavailable), cached per day."""
    DATA_DIR.mkdir(exist_ok=True)
    tickers = list(dict.fromkeys(tickers))
    path = DATA_DIR / f"analyst_targets_{_today()}.csv"
    if path.exists() and not refresh:
        cached = pd.read_csv(path, index_col=0)["target_mean_price"]
        if set(tickers) <= set(cached.index):
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
    start = prices.index[prices.index.get_loc(newest_listing) + skip_days]
    if history_start != "max":
        start = max(start, pd.Timestamp(history_start))
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
