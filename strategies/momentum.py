"""
Momentum strategies:
5. MACD Signal Line Crossover — standard (12, 26, 9)
6. ROC Momentum Ranking       — 20-day / 60-day rate of change
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ── 5. MACD Crossover ─────────────────────────────────────────────────────

def macd_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict:
    """
    Standard MACD strategy.
    Signal: +1 when MACD line crosses above signal line
            -1 when MACD line crosses below signal line
    """
    close = df["Close"]
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    above = macd_line > signal_line
    raw = pd.Series(0, index=df.index)
    raw[above & ~above.shift(1).fillna(False)] = 1
    raw[~above & above.shift(1).fillna(True)] = -1

    position = raw.replace(0, np.nan).ffill().fillna(0).astype(int)

    return {
        "signal": int(position.iloc[-1]),
        "signal_series": position,
        "indicators": {
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": histogram,
        },
        "raw_signal": raw,
    }


# ── 6. ROC Momentum Ranking ───────────────────────────────────────────────

def roc_momentum(
    df: pd.DataFrame,
    period_short: int = 20,
    period_long: int = 60,
) -> dict:
    """
    Rate of Change momentum.
    Signal: +1 when both short-term and long-term ROC are positive (strong momentum)
            -1 when both are negative (losing momentum)
             0 mixed
    ROC = (Close_t / Close_{t-n} - 1) * 100
    """
    close = df["Close"]
    roc_short = (close / close.shift(period_short) - 1) * 100
    roc_long = (close / close.shift(period_long) - 1) * 100

    signal = pd.Series(0, index=df.index)
    signal[(roc_short > 0) & (roc_long > 0)] = 1
    signal[(roc_short < 0) & (roc_long < 0)] = -1

    return {
        "signal": int(signal.iloc[-1]),
        "signal_series": signal,
        "indicators": {
            f"ROC{period_short}": roc_short,
            f"ROC{period_long}": roc_long,
        },
        "raw_signal": signal,
    }
