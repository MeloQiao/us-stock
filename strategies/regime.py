"""
Market regime detection.

Primary gate  : benchmark price vs 200-day MA
                  above → "bull"  (allow all signals)
                  below → "bear"  (block new buy orders; exits still execute)
Secondary info: 50-day MA gives momentum sub-state (bull_strong / bull_caution)

Benchmarks per market:
  us  → SPY
  hk  → 02800  (Tracker Fund / HSI)
  cn  → 510300 (沪深300 ETF)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BENCHMARK = {"us": "SPY", "hk": "02800", "cn": "510300"}


def detect_regime(
    market: str = "us",
    price_data: Optional[dict] = None,
) -> dict:
    """
    Detect current market regime for *market*.

    Parameters
    ----------
    market     : "us" | "hk" | "cn"
    price_data : optional pre-fetched {symbol: DataFrame} dict.
                 If the benchmark symbol is present, use it directly
                 (avoids a redundant yfinance call in the pipeline).

    Returns
    -------
    dict with keys:
        regime       : "bull" | "bear" | "neutral" | "unknown"
        benchmark    : ticker used (e.g. "SPY")
        price        : latest close
        ma200        : 200-day simple MA
        ma50         : 50-day simple MA
        above_ma200  : bool
        above_ma50   : bool
        allow_buy    : bool  — True when new longs are permitted
        sub_state    : "bull_strong" | "bull_caution" | "bear" | "unknown"
        reason       : human-readable explanation
    """
    benchmark = _BENCHMARK.get(market, "SPY")

    try:
        # Try pre-fetched data first
        df = None
        if price_data and benchmark in price_data:
            df = price_data[benchmark]

        if df is None or df.empty:
            from data.fetcher import fetch_history
            df = fetch_history(benchmark, years=2, market=market)

        if df is None or df.empty:
            raise ValueError(f"No price data for {benchmark}")

        close = df["Close"].dropna()
        if len(close) < 50:
            raise ValueError(f"Insufficient history for {benchmark} ({len(close)} bars)")

        price  = float(close.iloc[-1])
        ma200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        ma50   = float(close.rolling(50).mean().iloc[-1])

        above_ma200 = price > ma200
        above_ma50  = price > ma50

        if above_ma200 and above_ma50:
            regime     = "bull"
            sub_state  = "bull_strong"
            allow_buy  = True
            reason     = (
                f"{benchmark} {price:.2f} > MA50 {ma50:.2f} > MA200 {ma200:.2f}"
                " — 强势牛市，买入信号正常执行"
            )
        elif above_ma200 and not above_ma50:
            regime     = "bull"
            sub_state  = "bull_caution"
            allow_buy  = True
            reason     = (
                f"{benchmark} {price:.2f} > MA200 {ma200:.2f} 但 < MA50 {ma50:.2f}"
                " — 牛市但短期承压，谨慎做多"
            )
        else:
            regime     = "bear"
            sub_state  = "bear"
            allow_buy  = False
            reason     = (
                f"{benchmark} {price:.2f} < MA200 {ma200:.2f}"
                " — 熊市机制：屏蔽所有新买入信号，仅执行止损"
            )

        return {
            "regime": regime,
            "sub_state": sub_state,
            "benchmark": benchmark,
            "price": round(price, 4),
            "ma200": round(ma200, 4),
            "ma50": round(ma50, 4),
            "above_ma200": above_ma200,
            "above_ma50": above_ma50,
            "allow_buy": allow_buy,
            "reason": reason,
        }

    except Exception as e:
        logger.warning("Regime detection failed for [%s]: %s", market, e)
        return {
            "regime": "unknown",
            "sub_state": "unknown",
            "benchmark": benchmark,
            "price": None,
            "ma200": None,
            "ma50": None,
            "above_ma200": None,
            "above_ma50": None,
            "allow_buy": True,   # fail-open: don't block if we can't detect
            "reason": f"检测失败，默认放行: {e}",
        }


def apply_regime_gate(
    signals: dict[str, int],
    regime_info: dict,
) -> dict[str, int]:
    """
    Filter signals through regime gate.
    In bear regime: all buy (1) signals → 0 (hold).
    Sell (-1) signals always pass through regardless of regime.

    Returns a new signals dict.
    """
    if regime_info.get("allow_buy", True):
        return dict(signals)

    gated = {}
    for sym, sig in signals.items():
        gated[sym] = 0 if sig == 1 else sig  # block buys, keep sells & holds

    blocked = [s for s, sig in signals.items() if sig == 1]
    if blocked:
        logger.info(
            "Regime gate [%s]: blocked %d buy signal(s) → %s",
            regime_info.get("regime", "bear"),
            len(blocked),
            blocked,
        )
    return gated
