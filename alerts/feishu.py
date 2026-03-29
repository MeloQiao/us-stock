"""
Feishu (Lark) webhook alerts.
Sends strategy signals as rich card messages to a Feishu group bot.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SIGNAL_EMOJI = {1: "🟢", -1: "🔴", 0: "⚪"}
SIGNAL_TEXT = {1: "买入 BUY", -1: "卖出 SELL", 0: "持观望 HOLD"}


def _build_signal_card(
    date: str,
    symbol_signals: list[dict],
    vix_value: Optional[float] = None,
) -> dict:
    """
    Build a Feishu interactive card payload.
    symbol_signals: list of {symbol, strategy, signal, score_breakdown (optional)}
    """
    # Header
    header = {
        "template": "blue",
        "title": {
            "tag": "plain_text",
            "content": f"📊 美股策略信号 · {date}",
        },
    }

    elements = []

    # VIX banner
    if vix_value is not None:
        vix_color = "red" if vix_value > 30 else ("green" if vix_value < 15 else "yellow")
        vix_label = "恐慌" if vix_value > 30 else ("贪婪" if vix_value < 15 else "中性")
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**VIX 指数**: {vix_value:.1f}  →  **{vix_label}**",
            },
        })
        elements.append({"tag": "hr"})

    # Group signals by symbol
    symbol_map: dict[str, list[dict]] = {}
    for item in symbol_signals:
        sym = item["symbol"]
        symbol_map.setdefault(sym, []).append(item)

    for symbol, items in symbol_map.items():
        # Composite signal for this symbol
        composite = next((i for i in items if i["strategy"] == "composite_score"), None)
        headline_signal = composite["signal"] if composite else 0
        emoji = SIGNAL_EMOJI[headline_signal]
        score = composite.get("total_score", "—") if composite else "—"
        max_s = composite.get("max_possible", 9) if composite else 9

        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**{emoji} {symbol}**  综合评分: {score}/{max_s}  →  {SIGNAL_TEXT[headline_signal]}",
            },
        })

        # Individual strategy breakdown
        rows = [f"| 策略 | 信号 |", "|------|------|"]
        for item in items:
            if item["strategy"] == "composite_score":
                continue
            rows.append(f"| {item['strategy_label']} | {SIGNAL_EMOJI[item['signal']]} {SIGNAL_TEXT[item['signal']]} |")
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(rows),
            },
        })
        elements.append({"tag": "hr"})

    # Footer
    elements.append({
        "tag": "note",
        "elements": [{
            "tag": "plain_text",
            "content": "信号基于收盘价计算 · 次日开盘执行 · 仅供参考，不构成投资建议",
        }],
    })

    return {
        "msg_type": "interactive",
        "card": {
            "header": header,
            "elements": elements,
        },
    }


def send_signal_alert(
    webhook_url: str,
    symbol_signals: list[dict],
    vix_value: Optional[float] = None,
    date: Optional[str] = None,
) -> bool:
    """
    Send strategy signal alert to Feishu group via webhook.

    Parameters
    ----------
    webhook_url : Feishu custom bot webhook URL
    symbol_signals : list of signal dicts, each containing:
        {symbol, strategy, strategy_label, signal, total_score, max_possible}
    vix_value : current VIX level (optional)
    date : date string, defaults to today

    Returns True on success.
    """
    if not webhook_url:
        logger.warning("Feishu webhook URL not configured.")
        return False

    date = date or datetime.today().strftime("%Y-%m-%d")
    payload = _build_signal_card(date, symbol_signals, vix_value)

    try:
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") == 0 or result.get("StatusCode") == 0:
            logger.info("Feishu alert sent successfully.")
            return True
        else:
            logger.warning("Feishu returned non-zero code: %s", result)
            return False
    except Exception as e:
        logger.error("Failed to send Feishu alert: %s", e)
        return False


def send_text_message(webhook_url: str, text: str) -> bool:
    """Send a plain text message to Feishu. Useful for error alerts."""
    if not webhook_url:
        return False
    try:
        payload = {"msg_type": "text", "content": {"text": text}}
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Feishu text message failed: %s", e)
        return False


def build_signal_list(all_results: dict[str, dict], strategy_labels: dict) -> list[dict]:
    """
    Convert run_all_strategies() output into the flat list expected by send_signal_alert().

    Parameters
    ----------
    all_results : {symbol: {strategy_name: result_dict}}
    strategy_labels : {strategy_name: display_label}
    """
    items = []
    for symbol, strategies in all_results.items():
        composite = strategies.get("composite_score", {})
        total_score = composite.get("total_score", 0)
        max_possible = composite.get("max_possible", 9)

        for strat_name, result in strategies.items():
            items.append({
                "symbol": symbol,
                "strategy": strat_name,
                "strategy_label": strategy_labels.get(strat_name, strat_name),
                "signal": result["signal"],
                "total_score": total_score,
                "max_possible": max_possible,
            })
    return items
