"""
Options Suggestions Helper

Generates Covered Call and Cash-Secured Put recommendations
based on current composite scores and positions.

NOT automated — output is advisory only.
User executes manually in Robinhood (or any options-enabled broker).

两种建议：
  Covered Call:   已持仓 + 信号仍为买入 → 每月卖虚值 Call 收租金
  Cash-Secured Put: 等待买入期（空仓）→ 卖下方 Put 打折布局

Robinhood 操作路径:
  Call: 持股页面 → Trade → Trade Options → Sell → Call
  Put:  搜索标的 → Trade → Trade Options → Sell → Put (需要现金抵押)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Black-Scholes (no scipy dependency) ──────────────────────────────────────

def _ncdf(x: float) -> float:
    """Normal CDF via Abramowitz & Stegun approximation."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    p = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2) * poly
    return p if x >= 0 else 1.0 - p


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-8 or sigma < 1e-6:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-8 or sigma < 1e-6:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _bs_delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-8 or sigma < 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_monthly_expiry(dte_target: int = 28) -> datetime:
    """Return approximate next monthly expiry date (~dte_target days out)."""
    today = datetime.now()
    target = today + timedelta(days=dte_target)
    # Move to 3rd Friday of target month
    year, month = target.year, target.month
    first_day = datetime(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    # If that's already past, go to next month
    if third_friday.date() <= today.date():
        if month == 12:
            month, year = 1, year + 1
        else:
            month += 1
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
    return third_friday


def _implied_vol(price_history: Optional[list[float]], fallback: float = 0.25) -> float:
    """
    Estimate IV from recent price history (last 21 closes).
    Uses realized vol × 1.25 (vol risk premium).
    """
    if not price_history or len(price_history) < 5:
        return fallback
    import math
    log_rets = [math.log(price_history[i] / price_history[i - 1])
                for i in range(1, len(price_history))]
    n = len(log_rets)
    mean = sum(log_rets) / n
    variance = sum((r - mean) ** 2 for r in log_rets) / max(n - 1, 1)
    rv_daily = math.sqrt(variance)
    rv_annual = rv_daily * math.sqrt(252)
    return max(rv_annual * 1.25, 0.12)   # floor at 12%


# ── Core suggestion functions ─────────────────────────────────────────────────

def suggest_covered_calls(
    positions: list[dict],          # from alpaca_trader.get_positions()
    price_history: dict[str, list[float]],   # {symbol: [close_prices, last 21 days]}
    composite_scores: dict[str, float],
    call_otm: float = 0.05,         # 5% out-of-the-money
    dte_target: int  = 28,          # target days to expiry
    rf: float        = 0.04,        # risk-free rate
    min_shares: int  = 100,         # minimum shares needed for 1 contract
) -> list[dict]:
    """
    For each held position with ≥ 100 shares, generate a Covered Call suggestion.

    Returns a list of suggestion dicts with human-readable fields.
    """
    suggestions = []
    expiry = _next_monthly_expiry(dte_target)
    T      = (expiry - datetime.now()).days / 365.0

    for pos in positions:
        sym    = pos["symbol"]
        qty    = pos.get("qty", 0)
        price  = pos.get("current_price", 0)
        mv     = pos.get("market_value", 0)
        score  = composite_scores.get(sym, 0)

        if qty < min_shares:
            continue       # need at least 100 shares for 1 contract
        if price <= 0:
            continue

        n_contracts = int(qty // 100)
        hist = price_history.get(sym, [])
        iv   = _implied_vol(hist)

        K_call    = price * (1 + call_otm)
        premium   = _bs_call(price, K_call, T, rf, iv)
        delta     = _bs_delta_call(price, K_call, T, rf, iv)
        prem_pct  = premium / price * 100
        ann_yield = prem_pct * (365 / max((expiry - datetime.now()).days, 1))

        # Breakeven on downside: current_price - premium
        breakeven = price - premium

        suggestions.append({
            "symbol":      sym,
            "type":        "covered_call",
            "qty_held":    qty,
            "n_contracts": n_contracts,
            "current_price": round(price, 2),
            "strike":      round(K_call, 2),
            "expiry":      expiry.strftime("%Y-%m-%d"),
            "dte":         (expiry - datetime.now()).days,
            "premium":     round(premium, 2),
            "premium_pct": round(prem_pct, 2),
            "ann_yield":   round(ann_yield, 1),
            "delta":       round(delta, 2),
            "iv_est":      round(iv * 100, 1),
            "breakeven":   round(breakeven, 2),
            "income_1c":   round(premium * 100, 0),     # per contract ($)
            "income_total": round(premium * 100 * n_contracts, 0),
            "composite_score": round(score, 1),
            "note": ("⚠️ 信号已转弱，谨慎卖 Call" if score < 4.0
                     else "✅ 信号仍强，可售虚值 Call 收租"),
        })

    suggestions.sort(key=lambda x: -x["ann_yield"])
    return suggestions


def suggest_cash_puts(
    watch_symbols: list[str],
    prices: dict[str, float],
    price_history: dict[str, list[float]],
    composite_scores: dict[str, float],
    signals: dict[str, int],
    positions: list[dict],
    put_otm_by_score: dict | None = None,
    dte_target: int = 28,
    rf: float       = 0.04,
) -> list[dict]:
    """
    For symbols we're waiting to enter (signal=0, score positive but not triggered,
    or signal=1 but not yet positioned), suggest Cash-Secured Puts.

    Idea: sell a Put at the price we'd WANT to buy at.
    If stock drops there → auto-buy at discount.
    If stock stays up   → keep premium.

    put_otm_by_score: {score_min: otm_pct}, e.g. {4.5: 0.05, 2.5: 0.08}
    """
    if put_otm_by_score is None:
        # Higher score → smaller OTM (more willing to be assigned near current price)
        # Lower score  → larger OTM (only buy if it drops more)
        put_otm_by_score = {7.5: 0.03, 6.0: 0.05, 4.5: 0.07, 2.5: 0.10}

    held_syms = {p["symbol"] for p in positions}
    suggestions = []
    expiry = _next_monthly_expiry(dte_target)
    T      = (expiry - datetime.now()).days / 365.0

    for sym in watch_symbols:
        score  = composite_scores.get(sym, 0)
        signal = signals.get(sym, 0)
        price  = prices.get(sym, 0)

        # Only suggest for symbols with positive score but not yet held
        if sym in held_syms:
            continue     # already holding — Covered Call applies instead
        if price <= 0:
            continue
        if score < 2.5:
            continue     # no bullish case — don't want to be assigned

        # Skip if strong BUY signal already — just buy directly
        if signal == 1 and score >= 7.5:
            continue

        # Determine OTM % by score tier
        put_otm = 0.10
        for score_min in sorted(put_otm_by_score.keys(), reverse=True):
            if score >= score_min:
                put_otm = put_otm_by_score[score_min]
                break

        hist      = price_history.get(sym, [])
        iv        = _implied_vol(hist)
        K_put     = price * (1 - put_otm)
        premium   = _bs_put(price, K_put, T, rf, iv)
        prem_pct  = premium / price * 100
        ann_yield = prem_pct * (365 / max((expiry - datetime.now()).days, 1))
        # Net cost if assigned = K_put - premium (effective buy price)
        eff_buy   = K_put - premium

        suggestions.append({
            "symbol":      sym,
            "type":        "cash_secured_put",
            "current_price": round(price, 2),
            "strike":      round(K_put, 2),
            "put_otm_pct": round(put_otm * 100, 1),
            "expiry":      expiry.strftime("%Y-%m-%d"),
            "dte":         (expiry - datetime.now()).days,
            "premium":     round(premium, 2),
            "premium_pct": round(prem_pct, 2),
            "ann_yield":   round(ann_yield, 1),
            "iv_est":      round(iv * 100, 1),
            "eff_buy_price": round(eff_buy, 2),
            "eff_discount":  round((1 - eff_buy / price) * 100, 1),
            "cash_required": round(K_put * 100, 0),    # per contract ($)
            "composite_score": round(score, 1),
            "note": (f"score={score:.1f} → {put_otm*100:.0f}% OTM target buy ${K_put:.2f}"),
        })

    suggestions.sort(key=lambda x: -x["composite_score"])
    return suggestions


# ── Formatting ────────────────────────────────────────────────────────────────

def format_options_report(
    call_suggestions: list[dict],
    put_suggestions:  list[dict],
) -> str:
    """Format suggestions as a concise text report for logging / Feishu."""
    lines = []

    if call_suggestions:
        lines.append("📊 【Covered Call 建议 — 持仓收租金】")
        lines.append("  在 Robinhood：持股页面 → Trade → Trade Options → Sell → Call")
        for s in call_suggestions:
            lines.append(
                f"  {s['symbol']:6s}  持{s['qty_held']:.0f}股({s['n_contracts']}张合约)  "
                f"行权价 ${s['strike']:.2f}(+{(s['strike']/s['current_price']-1)*100:.1f}%)  "
                f"到期 {s['expiry']}  权利金 ${s['premium']:.2f}/股  "
                f"收入 ${s['income_total']:.0f}  年化 {s['ann_yield']:.1f}%  "
                f"Delta {s['delta']:.2f}  {s['note']}"
            )
    else:
        lines.append("📊 【Covered Call】无建议（持仓不足100股）")

    lines.append("")

    if put_suggestions:
        lines.append("🎯 【Cash-Secured Put 建议 — 空仓等待时打折买入】")
        lines.append("  在 Robinhood：搜索标的 → Trade → Trade Options → Sell → Put")
        for s in put_suggestions:
            lines.append(
                f"  {s['symbol']:6s}  score={s['composite_score']:.1f}  "
                f"行权价 ${s['strike']:.2f}(-{s['put_otm_pct']:.0f}%)  "
                f"到期 {s['expiry']}  权利金 ${s['premium']:.2f}/股  "
                f"年化 {s['ann_yield']:.1f}%  "
                f"实际买入成本 ${s['eff_buy_price']:.2f}(折扣 {s['eff_discount']:.1f}%)  "
                f"所需保证金 ${s['cash_required']:.0f}/合约"
            )
    else:
        lines.append("🎯 【Cash-Secured Put】无建议（无符合条件的等待标的）")

    return "\n".join(lines)


def generate_options_suggestions(
    market: str,
    positions: list[dict],
    prices: dict[str, float],
    composite_scores: dict[str, float],
    signals: dict[str, int],
    watch_symbols: list[str],
    price_history: dict[str, list[float]] | None = None,
) -> dict:
    """
    Main entry point called from scheduler/jobs.py.
    Returns dict with call_suggestions, put_suggestions, report_text.
    """
    if market != "us":
        return {}    # options suggestions only for US market

    ph = price_history or {}

    call_sug = suggest_covered_calls(
        positions=positions,
        price_history=ph,
        composite_scores=composite_scores,
    )
    put_sug = suggest_cash_puts(
        watch_symbols=watch_symbols,
        prices=prices,
        price_history=ph,
        composite_scores=composite_scores,
        signals=signals,
        positions=positions,
    )
    report = format_options_report(call_sug, put_sug)

    logger.info("\n%s", report)

    return {
        "covered_calls":  call_sug,
        "cash_puts":      put_sug,
        "report":         report,
    }
