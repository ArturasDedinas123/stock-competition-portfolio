"""NYSE trading calendar, used to count the trading days in the competition window."""

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Full-day NYSE market holidays (early closes are ignored)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE trading days after ``start`` up to and including ``end``.

    Buying at the ``start`` close and valuing at the ``end`` close, this is the number of daily
    returns the portfolio experiences.
    """
    days = pd.date_range(start, end, freq=CustomBusinessDay(calendar=NYSEHolidayCalendar()))
    return pd.DatetimeIndex(days[days > pd.Timestamp(start)])
