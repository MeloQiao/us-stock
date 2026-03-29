"""
Mean reversion strategies (NOT for leveraged ETFs like TQQQ/SQQQ):
7. RSI Overbought / Oversold   — RSI < 30 buy, RSI > 70 sell
8. Bollinger Band Squeeze      — band narrows then price breaks upper band
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 7. RSI Strategy ───────────────────────────────────────────────────────

def rsi_strategy(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> dict:
    """
    RSI mean reversion.
    Signal: +1 when RSI crosses above oversold level (buy dip)
            -1 when RSI crosses below overbought level (take profit / short)
    """
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    was_oversold = rsi < oversold
    was_overbought = rsi > overbought

    raw = pd.Series(0, index=df.index)
    # Buy signal: RSI crosses back above oversold (momentum recovering)
    raw[~was_oversold & was_oversold.shift(1).fillna(False)] = 1
    # Sell signal: RSI crosses back below overbought (momentum fading)
    raw[~was_overbought & was_overbought.shift(1).fillna(False)] = -1

    position = raw.replace(0, np.nan).ffill().fillna(0).astype(int)

    return {
        "signal": int(position.iloc[-1]),
        "signal_series": position,
        "indicators": {
            "RSI": rsi,
            "Oversold": pd.Series(oversold, index=df.index),
            "Overbought": pd.Series(overbought, index=df.index),
        },
        "raw_signal": raw,
    }


# ── 8. Bollinger Band Squeeze Breakout ───────────────────────────────────

def bollinger_squeeze(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    squeeze_threshold: float = 0.1,
) -> dict:
    """
    Bollinger Band squeeze breakout.
    Squeeze = band width / middle band < squeeze_threshold
    Signal: +1 when price breaks above upper band after a squeeze
            -1 when price breaks below lower band after a squeeze
    """
    close = df["Close"]
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / (middle + 1e-9)

    squeezed = bandwidth < squeeze_threshold
    # Squeeze was recently active (in last 5 bars)
    recent_squeeze = squeezed.rolling(5).max().astype(bool)

    raw = pd.Series(0, index=df.index)
    raw[(close > upper) & recent_squeeze] = 1
    raw[(close < lower) & recent_squeeze] = -1

    position = raw.replace(0, np.nan).ffill().fillna(0).astype(int)

    return {
        "signal": int(position.iloc[-1]),
        "signal_series": position,
        "indicators": {
            "BB_Upper": upper,
            "BB_Middle": middle,
            "BB_Lower": lower,
            "BB_Width": bandwidth,
        },
        "raw_signal": raw,
    }
