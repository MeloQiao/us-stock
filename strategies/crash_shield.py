"""
Layer 1: Crash Shield — 规则驱动的崩盘预警

逻辑：当宏观/市场结构多个指标同时恶化，触发防御模式。
不依赖 ML，立即可用。

4个子信号（各独立打分），累计 ≥ 3 个触发防御：
  1. VIX 恐慌加速  — VIX > 25 且 5日涨幅 > 20%
  2. 趋势破坏      — SPY 跌破 MA50 且 MA50 本身在下行
  3. 短期急跌      — SPY 20日跌幅 < -8%
  4. 信用利差恶化  — HYG（高收益债 ETF）20日跌幅 < -3%
                     （高收益债跌 = 信用市场在为衰退定价）

防御级别：
  NONE  (0-1个信号)  — 正常
  CAUTION (2个信号)  — 新仓减半，持仓不动
  SHIELD (3-4个信号) — 屏蔽全部新多单，触发仓位减半

Usage
─────
  from strategies.crash_shield import evaluate_crash_shield
  result = evaluate_crash_shield(spy_df, vix_df, hyg_df)
  # result["level"]: "NONE" | "CAUTION" | "SHIELD"
  # result["score"]: 0-4 触发信号数
  # result["signals"]: 各子信号详情
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_crash_shield(
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    hyg_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Evaluate crash shield signals on the latest available data.

    Parameters
    ----------
    spy_df : SPY OHLCV DataFrame (or benchmark index for non-US)
    vix_df : VIX OHLCV DataFrame (optional, US only)
    hyg_df : HYG (iShares High Yield Bond ETF) OHLCV (optional)

    Returns
    -------
    {
        "level":   "NONE" | "CAUTION" | "SHIELD",
        "score":   int (0-4),
        "signals": { signal_name: {"triggered": bool, "value": float, "threshold": float} }
        "position_multiplier": float (1.0 | 0.5 | 0.0 for new entries)
    }
    """
    signals: dict[str, dict] = {}
    score = 0

    spy_close = spy_df["Close"].dropna()
    if len(spy_close) < 200:
        logger.warning("CrashShield: insufficient SPY history (%d bars)", len(spy_close))
        return _default_result()

    # ── Signal 1: VIX 恐慌加速 ──────────────────────────────────────────────
    if vix_df is not None and not vix_df.empty:
        vix = vix_df["Close"].dropna().reindex(spy_close.index, method="ffill")
        vix_now = float(vix.iloc[-1])
        vix_5d_ago = float(vix.iloc[-6]) if len(vix) > 5 else vix_now
        vix_5d_chg = (vix_now - vix_5d_ago) / (vix_5d_ago + 1e-9)

        triggered = vix_now > 25 and vix_5d_chg > 0.20
        signals["vix_panic"] = {
            "triggered": triggered,
            "value": round(vix_now, 1),
            "change_5d": round(vix_5d_chg * 100, 1),
            "threshold": "VIX>25 且 5日涨幅>20%",
            "label": f"VIX恐慌加速: {vix_now:.1f} ({vix_5d_chg*100:+.1f}%)",
        }
        if triggered:
            score += 1
    else:
        signals["vix_panic"] = {"triggered": False, "label": "VIX数据不可用"}

    # ── Signal 2: 趋势破坏（SPY 跌破 MA50 且 MA50 下行）─────────────────────
    ma50 = spy_close.rolling(50, min_periods=25).mean()
    ma50_now = float(ma50.iloc[-1])
    ma50_10d_ago = float(ma50.iloc[-11]) if len(ma50) > 10 else ma50_now
    price_now = float(spy_close.iloc[-1])

    below_ma50 = price_now < ma50_now
    ma50_declining = ma50_now < ma50_10d_ago

    triggered = below_ma50 and ma50_declining
    signals["trend_break"] = {
        "triggered": triggered,
        "value": round(price_now, 2),
        "ma50": round(ma50_now, 2),
        "ma50_declining": ma50_declining,
        "threshold": "价格<MA50 且 MA50本身下行",
        "label": (
            f"趋势破坏: 价格{price_now:.2f} vs MA50 {ma50_now:.2f}"
            f" ({'↓下行' if ma50_declining else '↑上行'})"
        ),
    }
    if triggered:
        score += 1

    # ── Signal 3: 短期急跌（20日跌幅 < -8%）────────────────────────────────
    ret_20d = float(spy_close.iloc[-1] / spy_close.iloc[-21] - 1) if len(spy_close) > 20 else 0.0
    triggered = ret_20d < -0.08
    signals["sharp_decline"] = {
        "triggered": triggered,
        "value": round(ret_20d * 100, 2),
        "threshold": "20日跌幅 < -8%",
        "label": f"短期急跌: SPY 20日 {ret_20d*100:+.2f}%",
    }
    if triggered:
        score += 1

    # ── Signal 4: 信用利差恶化（HYG 20日跌幅 < -3%）────────────────────────
    if hyg_df is not None and not hyg_df.empty:
        hyg = hyg_df["Close"].dropna()
        if len(hyg) > 20:
            hyg_ret = float(hyg.iloc[-1] / hyg.iloc[-21] - 1)
            triggered = hyg_ret < -0.03
            signals["credit_spread"] = {
                "triggered": triggered,
                "value": round(hyg_ret * 100, 2),
                "threshold": "HYG 20日跌幅 < -3%",
                "label": f"信用利差恶化: HYG 20日 {hyg_ret*100:+.2f}%",
            }
            if triggered:
                score += 1
        else:
            signals["credit_spread"] = {"triggered": False, "label": "HYG历史不足"}
    else:
        # 没有 HYG 数据时，用 SPY 60日跌幅作为替代
        ret_60d = float(spy_close.iloc[-1] / spy_close.iloc[-61] - 1) if len(spy_close) > 60 else 0.0
        triggered = ret_60d < -0.15
        signals["credit_spread"] = {
            "triggered": triggered,
            "value": round(ret_60d * 100, 2),
            "threshold": "SPY 60日跌幅 < -15%（HYG替代）",
            "label": f"市场深度下跌: SPY 60日 {ret_60d*100:+.2f}%",
        }
        if triggered:
            score += 1

    # ── 汇总防御级别 ─────────────────────────────────────────────────────────
    if score >= 3:
        level = "SHIELD"
        position_multiplier = 0.0   # 新仓完全屏蔽
        existing_reduce = 0.5       # 现有仓位减半
    elif score == 2:
        level = "CAUTION"
        position_multiplier = 0.5   # 新仓减半
        existing_reduce = 1.0       # 现有仓位不动
    else:
        level = "NONE"
        position_multiplier = 1.0
        existing_reduce = 1.0

    triggered_list = [k for k, v in signals.items() if v.get("triggered")]
    logger.info(
        "CrashShield: level=%s score=%d/4 triggered=%s",
        level, score, triggered_list,
    )

    return {
        "level": level,
        "score": score,
        "signals": signals,
        "triggered": triggered_list,
        "position_multiplier": position_multiplier,
        "existing_reduce": existing_reduce,
    }


def _default_result() -> dict:
    return {
        "level": "NONE",
        "score": 0,
        "signals": {},
        "triggered": [],
        "position_multiplier": 1.0,
        "existing_reduce": 1.0,
    }
