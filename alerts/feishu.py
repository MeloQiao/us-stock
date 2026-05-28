"""
Feishu (Lark) webhook alerts.
Sends strategy signals as rich card messages to a Feishu group bot.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx

from config import SYMBOL_NAMES

logger = logging.getLogger(__name__)

SIGNAL_EMOJI = {1: "🟢", -1: "🔴", 0: "⚪"}
SIGNAL_TEXT = {1: "买入 BUY", -1: "卖出 SELL", 0: "持观望 HOLD"}

_MARKET_LABEL = {"us": "🇺🇸 美股 (US)", "hk": "🇭🇰 港股 (HK)", "cn": "🇨🇳 A股 (CN)"}
_MARKET_COLOR = {"us": "blue", "hk": "green", "cn": "red"}


def _build_portfolio_section(summary: dict) -> list[dict]:
    """Build card elements for portfolio NAV + open positions."""
    elements = []
    currency = summary.get("currency", "USD")
    market = summary.get("market", "").upper()

    # NAV summary line
    acct = summary.get("account", {})  # Alpaca format
    if acct:
        nav = acct.get("portfolio_value", 0)
        cash = acct.get("cash", 0)
        bp = acct.get("buying_power", 0)
        nav_line = (
            f"**💼 {market} 纸账户 NAV**: ${nav:,.0f} USD  |  "
            f"现金: ${cash:,.0f}  |  买入力: ${bp:,.0f}"
        )
    else:
        total = summary.get("total_capital", 0)
        invested = summary.get("invested", 0)
        avail = summary.get("available", 0)
        unreal = summary.get("unrealized_pnl", 0)
        realized = summary.get("realized_pnl", 0)
        nav = total + realized + unreal
        nav_line = (
            f"**💼 {market} 虚拟账户 NAV**: {nav:,.0f} {currency}  |  "
            f"已投入: {invested:,.0f}  |  可用: {avail:,.0f}  |  "
            f"浮盈: {unreal:+,.0f}  |  已实现: {realized:+,.0f}"
        )

    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": nav_line}})

    # Open positions table
    positions = summary.get("open_positions") or summary.get("positions", [])
    if positions:
        rows = ["| 标的 | 持仓量 | 成本价 | 现价 | 浮盈 |",
                "|------|--------|--------|------|------|"]
        for p in positions:
            sym = p.get("symbol", "")
            qty = p.get("qty") or p.get("shares", 0)
            entry = p.get("avg_entry_price") or p.get("entry_price", 0)
            cur = p.get("current_price") or p.get("entry_price", 0)
            pnl = p.get("unrealized_pl") or p.get("unrealized_pnl", None)
            pnl_pct = p.get("unrealized_plpc") or p.get("unrealized_pct", None)
            pnl_str = f"{pnl:+,.0f} ({pnl_pct:+.1f}%)" if pnl is not None and pnl_pct is not None else "—"
            rows.append(f"| {sym} | {qty} | {entry:,.2f} | {cur:,.2f} | {pnl_str} |")
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(rows)}})
    else:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "_暂无持仓_"}})

    elements.append({"tag": "hr"})
    return elements


def _build_signal_card(
    date: str,
    symbol_signals: list[dict],
    vix_value: Optional[float] = None,
    trades: Optional[list[dict]] = None,
    portfolio_summary: Optional[dict] = None,
    market: str = "us",
    regime_info: Optional[dict] = None,
    price_info: Optional[dict] = None,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    Build a Feishu interactive card payload.

    symbol_signals : list of {symbol, strategy, strategy_label, signal,
                               total_score, max_possible}
    trades         : list of executed paper trade orders (optional),
                     each containing {action, symbol, notional, score}
    market         : "us" | "hk" | "cn" — controls card color and title
    """
    market_label = _MARKET_LABEL.get(market, market.upper())
    header = {
        "template": _MARKET_COLOR.get(market, "blue"),
        "title": {
            "tag": "plain_text",
            "content": f"📊 {market_label} 策略信号 · {date}",
        },
    }

    elements = []

    # ── Regime banner ─────────────────────────────────────────────────────
    if regime_info:
        sub   = regime_info.get("sub_state", regime_info.get("regime", "unknown"))
        emoji = {"bull_strong": "🟢", "bull_caution": "🟡", "bear": "🔴"}.get(sub, "⚪")
        label = {"bull_strong": "牛市强势", "bull_caution": "牛市承压", "bear": "熊市"}.get(sub, "未知")
        bmark = regime_info.get("benchmark", "—")
        price = regime_info.get("price")
        ma200 = regime_info.get("ma200")
        price_str = f"{price:.2f}" if price else "—"
        ma200_str = f"{ma200:.2f}" if ma200 else "—"
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": (
                    f"**市场机制**: {emoji} {label}  "
                    f"|  {bmark} {price_str} vs MA200 {ma200_str}"
                    + ("  |  ⚠️ **熊市：已屏蔽新买入信号**" if sub == "bear" else "")
                ),
            },
        })
        elements.append({"tag": "hr"})

    # ── VIX banner ────────────────────────────────────────────────────────
    if vix_value is not None:
        vix_label = "恐慌" if vix_value > 30 else ("贪婪" if vix_value < 15 else "中性")
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**VIX 指数**: {vix_value:.1f}  →  **{vix_label}**",
            },
        })
        elements.append({"tag": "hr"})

    # ── 3-Layer protection status (US only) ──────────────────────────────
    if extra_info and market == "us":
        lines = []

        # Layer 1: Crash Shield
        cs = extra_info.get("crash_shield", {})
        cs_level = cs.get("level", "NONE")
        cs_score = cs.get("score", 0)
        cs_emoji = {"NONE": "🟢", "CAUTION": "🟡", "SHIELD": "🔴"}.get(cs_level, "⚪")
        cs_label = {"NONE": "正常", "CAUTION": "警惕", "SHIELD": "防御"}.get(cs_level, cs_level)
        triggered = cs.get("triggered", [])
        trig_str  = f" 触发:{','.join(triggered)}" if triggered else ""
        lines.append(f"**🛡️ 崩盘护盾**: {cs_emoji} {cs_label} ({cs_score}/4){trig_str}")

        # Layer 2: ML Regime
        ml = extra_info.get("ml_regime", {})
        if ml:
            ml_prob = ml.get("prob", 0.5)
            ml_mult = ml.get("multiplier", 1.0)
            ml_emoji = "🟢" if ml_prob >= 0.6 else ("🔴" if ml_prob <= 0.4 else "🟡")
            lines.append(
                f"**🤖 ML 制度**: {ml_emoji} 看多概率 {ml_prob*100:.0f}%  →  仓位系数 ×{ml_mult:.2f}"
            )

        # Layer 3: Dual Momentum
        dm = extra_info.get("dual_momentum", {})
        if dm:
            abs_ok      = dm.get("abs_momentum_ok", True)
            spy12m      = dm.get("spy_12m_ret", 0.0)
            crash_prot  = dm.get("crash_protect", False)
            dm_scale    = dm.get("position_scale", 1.0)
            abs_str     = f"✅ SPY 12M {spy12m*100:+.1f}%" if abs_ok else f"❌ SPY 12M {spy12m*100:+.1f}% (屏蔽做多)"
            crash_str   = "  ⚡ 动量崩溃保护触发" if crash_prot else ""
            lines.append(f"**📈 双动量**: {abs_str}{crash_str}  |  全局规模 ×{dm_scale:.2f}")

        if lines:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": "\n".join(lines)},
            })
            elements.append({"tag": "hr"})

    # ── Per-symbol signal breakdown ───────────────────────────────────────
    symbol_map: dict[str, list[dict]] = {}
    for item in symbol_signals:
        symbol_map.setdefault(item["symbol"], []).append(item)

    currency_sym = "¥" if market == "cn" else ("HK$" if market == "hk" else "$")

    # ── Compact signal table: one row per symbol ──────────────────────────
    rows = [
        "| 标的 | 收盘价 | 涨跌 | 评分 | 信号 |",
        "|------|--------|------|------|------|",
    ]
    for symbol, items in symbol_map.items():
        composite = next((i for i in items if i["strategy"] == "composite_score"), None)
        headline_signal = composite["signal"] if composite else 0
        emoji = SIGNAL_EMOJI[headline_signal]
        score   = composite.get("total_score", 0) if composite else 0
        max_s   = composite.get("max_possible", 9) if composite else 9
        name    = SYMBOL_NAMES.get(symbol, "")
        display = f"{symbol} {name}" if name else symbol

        pinfo = (price_info or {}).get(symbol)
        if pinfo:
            close = pinfo["close"]
            chg   = pinfo["change_pct"]
            chg_arrow = "▲" if chg >= 0 else "▼"
            price_col = f"{currency_sym}{close:,.2f}"
            chg_col   = f"{chg_arrow}{abs(chg):.2f}%"
        else:
            price_col = "—"
            chg_col   = "—"

        score_col = f"{score:.2f}/{max_s:.2f}"
        rows.append(
            f"| {display} | {price_col} | {chg_col} | {score_col} | {emoji} {SIGNAL_TEXT[headline_signal]} |"
        )

    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": "\n".join(rows)},
    })
    elements.append({"tag": "hr"})

    # ── Portfolio NAV + positions ─────────────────────────────────────────
    if portfolio_summary is not None:
        elements.extend(_build_portfolio_section(portfolio_summary))

    # ── Paper trade orders ────────────────────────────────────────────────
    if trades is not None:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "**📋 次日开盘纸交易挂单**"},
        })

        # Virtual portfolio actions (cn_tracker)
        buy_orders     = [t for t in trades if t.get("action") in ("enter_long", "buy")]
        exit_orders    = [t for t in trades if t.get("action") in ("exit", "sell")]
        filtered_orders= [t for t in trades if t.get("action") == "filtered"]
        # Real position actions (rh_tracker / hk_tracker compare_signals)
        cp_exit   = [t for t in trades if t.get("action") == "EXIT"]
        cp_enter  = [t for t in trades if t.get("action") == "ENTER"]
        cp_trim   = [t for t in trades if t.get("action") == "TRIM"]
        cp_hold   = [t for t in trades if t.get("action") == "HOLD"]

        currency_sym = "¥" if market == "cn" else ("HK$" if market == "hk" else "$")
        rows = ["| 方向 | 标的 | 参考价 | 估算股数 | 金额 | 评分 |",
                "|------|------|--------|---------|------|------|"]

        # Virtual buy orders
        for t in buy_orders:
            notional = t.get("notional") or 0
            score = t.get("score", "—")
            ref_price = t.get("ref_price") or t.get("price")
            est_shares = t.get("est_shares") or t.get("shares")
            sym = t["symbol"]
            name = SYMBOL_NAMES.get(sym, "")
            display = f"{sym} {name}" if name else sym
            price_str = f"{currency_sym}{ref_price:,.2f}" if ref_price else "—"
            shares_str = str(est_shares) if est_shares else "—"
            rows.append(f"| 🟢 买入 | {display} | {price_str} | {shares_str} | {currency_sym}{notional:,.0f} | {score} |")
        # Virtual exit orders
        for t in exit_orders:
            sym = t["symbol"]
            name = SYMBOL_NAMES.get(sym, "")
            display = f"{sym} {name}" if name else sym
            pnl = t.get("pnl")
            pnl_str = f"{currency_sym}{pnl:+,.0f}" if pnl is not None else "—"
            rows.append(f"| 🔴 平仓 | {display} | — | — | {pnl_str} | — |")

        # Real position EXIT alerts (compare_signals)
        for t in cp_exit:
            sym = t["symbol"]
            name = SYMBOL_NAMES.get(sym, "")
            display = f"{sym} {name}" if name else sym
            price = t.get("price", 0)
            shares = t.get("shares", 0)
            pnl_pct = t.get("pnl_pct", 0)
            rows.append(
                f"| 🔴 卖出 | {display} | {currency_sym}{price:,.2f} | {shares:.0f} | "
                f"{currency_sym}{price*shares:,.0f} | {pnl_pct:+.1f}% |"
            )
        # Real position ENTER alerts
        for t in cp_enter:
            sym = t["symbol"]
            name = SYMBOL_NAMES.get(sym, "")
            display = f"{sym} {name}" if name else sym
            price = t.get("price", 0)
            notional = t.get("notional", 0)
            score = t.get("composite_score", t.get("score", "—"))
            rows.append(
                f"| 🟢 买入 | {display} | {currency_sym}{price:,.2f} | — | "
                f"{currency_sym}{notional:,.0f} | {score} |"
            )
        # Real position TRIM alerts
        for t in cp_trim:
            sym = t["symbol"]
            name = SYMBOL_NAMES.get(sym, "")
            display = f"{sym} {name}" if name else sym
            trim_amt = t.get("trim_usd") or t.get("trim_hkd", 0)
            rows.append(f"| 🟡 减仓 | {display} | — | — | {currency_sym}{trim_amt:,.0f} | — |")

        has_any = buy_orders or exit_orders or cp_exit or cp_enter or cp_trim
        if not has_any:
            rows.append("| — | 无新挂单 | — | — | — | — |")

        # HOLD summary (compact, not per-row)
        if cp_hold:
            hold_names = []
            for t in cp_hold:
                sym = t["symbol"]
                name = SYMBOL_NAMES.get(sym, sym)
                pnl = t.get("pnl_pct", 0)
                hold_names.append(f"{sym}({pnl:+.1f}%)")
            rows.append(f"| 🟢 持有 | {' / '.join(hold_names)} | — | — | — | — |")

        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(rows)},
        })

        # Show which BUY signals were filtered out by portfolio optimizer
        if filtered_orders:
            filtered_names = []
            for t in filtered_orders:
                sym = t["symbol"]
                name = SYMBOL_NAMES.get(sym, "")
                filtered_names.append(f"{sym} {name}" if name else sym)
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        "⚠️ **组合优化过滤**（有BUY信号但未下单）：" +
                        "、".join(filtered_names) +
                        "\n_原因：相关性去重或行业集中度超限_"
                    ),
                },
            })

        elements.append({"tag": "hr"})

    # ── Footer ────────────────────────────────────────────────────────────
    elements.append({
        "tag": "note",
        "elements": [{
            "tag": "plain_text",
            "content": "US Stock Monitor · 信号基于收盘价计算 · 次日开盘执行 · 仅供参考，不构成投资建议",
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
    trades: Optional[list[dict]] = None,
    portfolio_summary: Optional[dict] = None,
    market: str = "us",
    regime_info: Optional[dict] = None,
    price_info: Optional[dict] = None,
    extra_info: Optional[dict] = None,
) -> bool:
    """
    Send strategy signal alert to Feishu group via webhook.

    Parameters
    ----------
    webhook_url       : Feishu custom bot webhook URL
    symbol_signals    : list of signal dicts
    vix_value         : current VIX level (optional)
    date              : date string, defaults to today
    trades            : paper/virtual trade orders (optional)
    portfolio_summary : portfolio NAV + positions snapshot (optional)
    extra_info        : optional dict with crash_shield, ml_regime, dual_momentum status
    """
    if not webhook_url:
        logger.warning("Feishu webhook URL not configured.")
        return False

    date = date or datetime.today().strftime("%Y-%m-%d")
    payload = _build_signal_card(
        date, symbol_signals, vix_value,
        trades=trades, portfolio_summary=portfolio_summary,
        market=market, regime_info=regime_info,
        price_info=price_info,
        extra_info=extra_info,
    )

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
        if "Stock" not in text and "Hugging Face" not in text:
            text = f"[US Stock Monitor] {text}"
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
    all_results    : {symbol: {strategy_name: result_dict}}
    strategy_labels: {strategy_name: display_label}
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
