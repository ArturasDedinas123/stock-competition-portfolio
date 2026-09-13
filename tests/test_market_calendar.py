"""Tests for the NYSE trading calendar."""

import pandas as pd

from stock_competition.market_calendar import trading_days


def test_competition_window_has_64_trading_days():
    assert len(trading_days("2026-09-14", "2026-12-14")) == 64


def test_purchase_day_is_not_counted():
    days = trading_days("2026-09-14", "2026-09-18")
    assert pd.Timestamp("2026-09-14") not in days
    assert len(days) == 4


def test_market_holidays_are_skipped():
    assert pd.Timestamp("2026-11-26") not in trading_days("2026-11-20", "2026-11-30")  # Thanksgiving
    assert pd.Timestamp("2026-07-03") not in trading_days("2026-06-30", "2026-07-08")  # July 4 falls on a Saturday
    assert pd.Timestamp("2026-04-03") not in trading_days("2026-03-30", "2026-04-08")  # Good Friday
