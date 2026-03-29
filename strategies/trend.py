"""
Trend-following strategies:
1. Golden / Death Cross  — MA50 cross MA200
2. SuperTrend            — ATR-based dynamic trend line
3. Donchian Channel      — Turtle breakout (20-day high / 10-day low)
4. EMA + ADX filter      — ADX>25 and EMA12 > EMA26
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    dm_plus = (high - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)
    # Resolve tie: if both positive, zero the smaller
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0
    mask2 = dm_minus < dm_plus
    dm_minus[mask2] = 0

    atr = _atr(df, period)
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ── 1. Golden / Death Cross ───────────────────────────────────────────────

def golden_cross(
    df: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
) -> dict:
    """
    Signal: +1 when MA_fast crosses above MA_slow (golden cross)
            -1 when MA_fast crosses below MA_slow (death cross)
             0 otherwise
    """
    close = df["Close"]
    ma_fast = _sma(close, fast)
    ma_slow = _sma(close, slow)

    above = ma_fast > ma_slow
    signal = pd.Series(0, index=df.index)
    signal[above & ~above.shift(1).fillna(False)] = 1   # just crossed up
    signal[~above & above.shift(1).fillna(True)] = -1   # just crossed down

    # Carry last non-zero signal forward for position tracking
    position = signal.replace(0, np.nan).ffill().fillna(0).astype(int)

    return {
        "signal": int(position.iloc[-1]),
        "signal_series": position,
        "indicators": {
            f"MA{fast}": ma_fast,
            f"MA{slow}": ma_slow,
        },
        "raw_signal": signal,
    }


# ── 2. SuperTrend ─────────────────────────────────────────────────────────

def supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> dict:
    """
    SuperTrend indicator.
    Signal: +1 = price above upper band (bullish), -1 = price below lower band (bearish)
    """
    close = df["Close"]
    hl2 = (df["High"] + df["Low"]) / 2
    atr = _atr(df, atr_period)

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Final bands with carry-forward logic
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(df)):
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        cur_close = close.iloc[i - 1]

        final_upper.iloc[i] = upper_band.iloc[i] if (upper_band.iloc[i] < prev_upper or cur_close > prev_upper) else prev_upper
        final_lower.iloc[i] = lower_band.iloc[i] if (lower_band.iloc[i] > prev_lower or cur_close < prev_lower) else prev_lower

    # Trend direction
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        prev_trend = trend.iloc[i - 1]
        cur_close = close.iloc[i]
        if prev_trend == -1 and cur_close > final_upper.iloc[i]:
            trend.iloc[i] = 1
        elif prev_trend == 1 and cur_close < final_lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend

    supertrend_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=df.index)

    return {
        "signal": int(trend.iloc[-1]),
        "signal_series": trend,
        "indicators": {
            "SuperTrend": supertrend_line,
            "ST_Upper": final_upper,
            "ST_Lower": final_lower,
        },
        "raw_signal": trend.diff().clip(-1, 1).fillna(0).astype(int),
    }


# ── 3. Donchian Channel (Turtle Breakout) ────────────────────────────────

def donchian_channel(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
) -> dict:
    """
    Turtle trading rules:
    - Buy when price breaks above 20-day high
    - Exit when price drops below 10-day low
    """
    close = df["Close"]
    high_20 = df["High"].rolling(entry_period).max().shift(1)  # shift to avoid lookahead
    low_10 = df["Low"].rolling(exit_period).min().shift(1)
    high_10 = df["High"].rolling(exit_period).max()  # for mid-channel reference

    raw_signal = pd.Series(0, index=df.index)
    raw_signal[close > high_20] = 1
    raw_signal[close < low_10] = -1

    position = raw_signal.replace(0, np.nan).ffill().fillna(0).astype(int)

    return {
        "signal": int(position.iloc[-1]),
        "signal_series": position,
        "indicators": {
            f"Donchian_{entry_period}H": high_20,
            f"Donchian_{exit_period}L": low_10,
        },
        "raw_signal": raw_signal,
    }


# ── 4. EMA + ADX Filter ───────────────────────────────────────────────────

def ema_adx(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> dict:
    """
    Signal triggers only when both conditions are met:
    - ADX > threshold (trending market)
    - EMA_fast > EMA_slow (bullish) or EMA_fast < EMA_slow (bearish)
    """
    close = df["Close"]
    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    adx = _adx(df, adx_period)

    trending = adx > adx_threshold
    bull = ema_f > ema_s
    bear = ema_f < ema_s

    signal = pd.Series(0, index=df.index)
    signal[trending & bull] = 1
    signal[trending & bear] = -1

    return {
        "signal": int(signal.iloc[-1]),
        "signal_series": signal,
        "indicators": {
            f"EMA{ema_fast}": ema_f,
            f"EMA{ema_slow}": ema_s,
            "ADX": adx,
        },
        "raw_signal": signal.diff().clip(-1, 1).fillna(0).astype(int),
    }
