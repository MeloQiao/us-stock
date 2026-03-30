"""
Market hours helper — determines whether a market is currently open.

Used to skip live quote fetching during closed hours and fall back to
last-close prices from HF Dataset cache instead.
"""

from __future__ import annotations

from datetime import datetime, time

import pytz

# Market sessions in local time (open, close)
_SESSIONS: dict[str, tuple[str, time, time]] = {
    "us": ("America/New_York",   time(9, 30),  time(16, 0)),
    "hk": ("Asia/Hong_Kong",     time(9, 30),  time(16, 0)),
    "cn": ("Asia/Shanghai",      time(9, 30),  time(15, 0)),
}

# Add a 15-minute buffer after close — yfinance data may lag slightly
_CLOSE_BUFFER_MINUTES = 15


def is_market_open(market: str) -> bool:
    """
    Return True if the given market is currently within trading hours
    (weekdays only, no holiday check — holiday check is done separately
    by exchange_calendars in the pipeline).

    During closed hours the UI should use cached OHLCV close prices
    instead of hitting live quote APIs.
    """
    cfg = _SESSIONS.get(market.lower())
    if cfg is None:
        return False

    tz_name, open_time, close_time = cfg
    tz = pytz.timezone(tz_name)
    now_local = datetime.now(tz)

    # Weekends
    if now_local.weekday() >= 5:
        return False

    now_t = now_local.time().replace(second=0, microsecond=0)

    # Add buffer after close so we still fetch live prices briefly after close
    from datetime import timedelta
    close_dt = datetime.combine(now_local.date(), close_time)
    close_with_buffer = (close_dt + timedelta(minutes=_CLOSE_BUFFER_MINUTES)).time()

    return open_time <= now_t <= close_with_buffer
