"""
Macro timing strategy:
9. VIX Timing — VIX > 30 reduce QQQ / consider SQQQ; VIX < 15 long TQQQ
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def vix_timing(
    df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    fear_threshold: float = 30.0,
    greed_threshold: float = 15.0,
) -> dict:
    """
    VIX-based macro timing.

    Parameters
    ----------
    df : OHLCV DataFrame for the target symbol (used for index alignment)
    vix_df : OHLCV DataFrame for ^VIX. If None, attempts to fetch it internally.
    fear_threshold : VIX above this → bearish (fear), signal = -1
    greed_threshold : VIX below this → bullish (greed), signal = +1

    Signal values:
        +1  = low VIX (complacency / bull market) → long TQQQ
         0  = neutral zone
        -1  = high VIX (fear) → reduce QQQ, consider SQQQ
    """
    if vix_df is None:
        try:
            from data.fetcher import fetch_history
            vix_df = fetch_history("^VIX", years=10)
        except Exception as e:
            logger.warning("Could not fetch VIX data: %s", e)
            # Return neutral if VIX unavailable
            neutral = pd.Series(0, index=df.index)
            return {
                "signal": 0,
                "signal_series": neutral,
                "indicators": {},
                "raw_signal": neutral,
            }

    vix = vix_df["Close"].reindex(df.index, method="ffill")

    signal = pd.Series(0, index=df.index)
    signal[vix > fear_threshold] = -1
    signal[vix < greed_threshold] = 1

    return {
        "signal": int(signal.iloc[-1]),
        "signal_series": signal,
        "indicators": {
            "VIX": vix,
            "Fear_Level": pd.Series(fear_threshold, index=df.index),
            "Greed_Level": pd.Series(greed_threshold, index=df.index),
        },
        "raw_signal": signal,
        "vix_value": round(float(vix.iloc[-1]), 2) if not vix.empty else None,
    }
