"""
Composite scoring strategy:
10. Multi-strategy weighted voting — 6/10+ signals → bullish
"""

from __future__ import annotations

import pandas as pd

from config import LEVERAGED_ETFS, STRATEGY_PARAMS, STRATEGY_WEIGHTS
from .trend import golden_cross, supertrend, donchian_channel, ema_adx
from .momentum import macd_crossover, roc_momentum
from .mean_reversion import rsi_strategy, bollinger_squeeze
from .macro import vix_timing


def composite_score(
    df: pd.DataFrame,
    symbol: str = "",
    vix_df: pd.DataFrame | None = None,
    buy_threshold: int = 6,
    sell_threshold: int = 4,
    weights: dict | None = None,
) -> dict:
    """
    Aggregate all applicable strategies into a composite score.

    Score = sum of (+1/-1/0) × weight across applicable strategies.
    Signal: +1 if score >= buy_threshold
            -1 if score <= -sell_threshold (i.e., <= -4 when 6+ bearish)
             0 otherwise

    weights: regime-specific weights from walk_forward_optimizer. Falls back to
             STRATEGY_WEIGHTS (all = 1) if None.

    Leveraged ETFs skip mean reversion strategies.
    """
    is_leveraged = symbol.upper() in LEVERAGED_ETFS
    p = STRATEGY_PARAMS
    _weights = weights if weights is not None else STRATEGY_WEIGHTS

    strategy_results = {}

    # Trend strategies
    strategy_results["golden_cross"] = golden_cross(df, **p["golden_cross"])
    strategy_results["supertrend"] = supertrend(df, **p["supertrend"])
    strategy_results["donchian_channel"] = donchian_channel(df, **p["donchian"])
    strategy_results["ema_adx"] = ema_adx(df, **p["ema_adx"])

    # Momentum strategies
    strategy_results["macd_crossover"] = macd_crossover(df, **p["macd"])
    strategy_results["roc_momentum"] = roc_momentum(df, **p["roc"])

    # Mean reversion — skip for leveraged ETFs
    if not is_leveraged:
        strategy_results["rsi_strategy"] = rsi_strategy(df, **p["rsi"])
        strategy_results["bollinger_squeeze"] = bollinger_squeeze(df, **p["bollinger"])

    # Macro
    strategy_results["vix_timing"] = vix_timing(df, vix_df=vix_df, **p["vix"])

    # Build score series (sum of all signal_series × weight aligned to df.index)
    score_series = pd.Series(0.0, index=df.index)
    for name, result in strategy_results.items():
        weight = _weights.get(name, 1)
        score_series += result["signal_series"].reindex(df.index, fill_value=0) * weight

    max_possible = sum(_weights.get(k, 1) for k in strategy_results)

    signal = pd.Series(0, index=df.index)
    signal[score_series >= buy_threshold] = 1
    signal[score_series <= -sell_threshold] = -1

    # Current bar summary (weighted)
    current_scores = {name: res["signal"] for name, res in strategy_results.items()}
    total_score = sum(sig * _weights.get(name, 1) for name, sig in current_scores.items())

    return {
        "signal": int(signal.iloc[-1]),
        "signal_series": signal,
        "indicators": {
            "Composite_Score": score_series,
            "Buy_Threshold": pd.Series(buy_threshold, index=df.index),
            "Sell_Threshold": pd.Series(-sell_threshold, index=df.index),
        },
        "raw_signal": signal,
        "score_breakdown": current_scores,
        "total_score": total_score,
        "max_possible": max_possible,
        "strategy_results": strategy_results,
    }


def run_all_strategies(
    df: pd.DataFrame,
    symbol: str = "",
    vix_df: pd.DataFrame | None = None,
    weights: dict | None = None,
) -> dict:
    """
    Run all individual strategies + composite for a symbol.

    Parameters
    ----------
    weights : regime-specific strategy weights from walk_forward_optimizer.
              If None, falls back to equal weights (1.0 per strategy).

    Returns dict keyed by strategy name, each containing the strategy result dict.
    """
    is_leveraged = symbol.upper() in LEVERAGED_ETFS
    p = STRATEGY_PARAMS

    results = {}
    results["golden_cross"]    = golden_cross(df, **p["golden_cross"])
    results["supertrend"]      = supertrend(df, **p["supertrend"])
    results["donchian_channel"]= donchian_channel(df, **p["donchian"])
    results["ema_adx"]         = ema_adx(df, **p["ema_adx"])
    results["macd_crossover"]  = macd_crossover(df, **p["macd"])
    results["roc_momentum"]    = roc_momentum(df, **p["roc"])

    if not is_leveraged:
        results["rsi_strategy"]     = rsi_strategy(df, **p["rsi"])
        results["bollinger_squeeze"]= bollinger_squeeze(df, **p["bollinger"])

    results["vix_timing"] = vix_timing(df, vix_df=vix_df, **p["vix"])
    results["composite_score"] = composite_score(
        df, symbol=symbol, vix_df=vix_df,
        buy_threshold=p["composite"]["buy_threshold"],
        sell_threshold=p["composite"]["sell_threshold"],
        weights=weights,
    )

    return results
