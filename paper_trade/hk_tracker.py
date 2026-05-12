"""
HK Virtual Portfolio Tracker

Local JSON replacement for the HuggingFace-based VirtualPortfolio.
Tracks virtual HKD 100万 positions driven by daily strategy signals.

Usage
─────
  # Show current virtual portfolio
  python3 paper_trade/hk_tracker.py show

  # Reset to fresh start (HKD 100万 cash, no positions)
  python3 paper_trade/hk_tracker.py reset
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parent / "hk_virtual.json"

TOTAL_CAPITAL = 1_000_000   # HKD 100万 virtual capital
CURRENCY      = "HKD"

# Regime → max fraction of total capital per position
_REGIME_FRACTION: dict[str, float] = {
    "bull_strong":  0.30,
    "bull_caution": 0.20,
    "bear":         0.10,
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load() -> dict:
    if not _DATA_FILE.exists():
        return _fresh_state()
    with open(_DATA_FILE) as f:
        return json.load(f)


def save(data: dict) -> None:
    data["updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _fresh_state() -> dict:
    return {
        "_note": "HK virtual portfolio. Auto-managed by hk_tracker.py via daily signals.",
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "capital": TOTAL_CAPITAL,
        "currency": CURRENCY,
        "cash": TOTAL_CAPITAL,
        "positions": [],
        "trade_history": [],
    }


# ── State helpers ─────────────────────────────────────────────────────────────

def _get_position(data: dict, symbol: str) -> Optional[dict]:
    return next((p for p in data["positions"] if p["symbol"] == symbol), None)


def _available_cash(data: dict) -> float:
    return max(data["cash"], 0.0)


# ── Signal execution (called by scheduler) ────────────────────────────────────

def execute_signals(
    signals: dict[str, int],
    scores: dict[str, float],
    prices: dict[str, float],
    weights: Optional[dict[str, float]] = None,
    regime: str = "bull_caution",
    max_possible: int = 9,
    date: Optional[str] = None,
) -> list[dict]:
    """
    Process daily signals. Auto-buy on signal=1, auto-sell on signal=-1.
    Returns list of trade records for Feishu.
    """
    data  = load()
    today = date or datetime.now().strftime("%Y-%m-%d")
    results: list[dict] = []

    # ── 1. Exits ──────────────────────────────────────────────────────────────
    held_syms = {p["symbol"] for p in data["positions"]}
    for sym, sig in signals.items():
        if sig != -1 or sym not in held_syms:
            continue
        pos        = _get_position(data, sym)
        exit_price = prices.get(sym, 0.0)
        if not pos or exit_price <= 0:
            continue

        pnl     = (exit_price - pos["entry_price"]) * pos["shares"]
        pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
        proceeds = exit_price * pos["shares"]

        data["cash"] += proceeds
        data["positions"].remove(pos)

        record = {
            "date": today, "symbol": sym, "action": "sell",
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "shares": pos["shares"], "notional": round(proceeds, 2),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "score": scores.get(sym, 0), "currency": CURRENCY,
        }
        data["trade_history"].append(record)
        results.append(record)
        logger.info("[hk] SELL %s @ %.2f  pnl=%.0f (%.1f%%)", sym, exit_price, pnl, pnl_pct)

    # ── 2. Entries ────────────────────────────────────────────────────────────
    held_syms = {p["symbol"] for p in data["positions"]}
    new_buys = [s for s, sig in signals.items() if sig == 1 and s not in held_syms]

    if new_buys and _available_cash(data) > 0:
        budget    = _available_cash(data)
        base_frac = _REGIME_FRACTION.get(regime, 0.20)

        if weights:
            sub_w    = {s: weights[s] for s in new_buys if s in weights}
            if not sub_w:
                sub_w = {s: 1 / len(new_buys) for s in new_buys}
            total_w  = sum(sub_w.values())
            raw_w    = {s: sub_w[s] / total_w for s in sub_w}
            new_buys = list(raw_w.keys())
        else:
            raw_totals = {s: max(scores.get(s, 1), 1) for s in new_buys}
            total_w    = sum(raw_totals.values())
            raw_w      = {s: raw_totals[s] / total_w for s in new_buys}

        for sym in new_buys:
            price = prices.get(sym, 0.0)
            if price <= 0:
                continue
            score_factor = max(0.5, min(1.0, scores.get(sym, max_possible) / max_possible))
            dynamic_cap  = TOTAL_CAPITAL * base_frac * score_factor
            notional     = min(budget * raw_w[sym], dynamic_cap)
            if notional < 100:
                continue

            shares = round(notional / price, 2)
            actual_notional = round(shares * price, 2)

            pos_record = {
                "symbol": sym, "entry_date": today,
                "entry_price": price, "shares": shares,
                "notional": actual_notional,
                "score": scores.get(sym, 0), "currency": CURRENCY,
            }
            data["positions"].append(pos_record)
            data["cash"] -= actual_notional

            trade_record = {
                "date": today, "symbol": sym, "action": "buy",
                "entry_price": price, "shares": shares,
                "notional": actual_notional,
                "score": scores.get(sym, 0), "currency": CURRENCY,
            }
            data["trade_history"].append(trade_record)
            results.append(trade_record)
            logger.info("[hk] BUY  %s @ %.2f  notional=%.0f  score=%.1f",
                        sym, price, actual_notional, scores.get(sym, 0))

    save(data)
    return results


# ── Portfolio summary ─────────────────────────────────────────────────────────

def get_summary(prices: Optional[dict[str, float]] = None) -> dict:
    data      = load()
    positions = data["positions"]
    unrealized = 0.0
    pos_with_pnl = []
    for p in positions:
        sym        = p["symbol"]
        cur        = (prices or {}).get(sym, p["entry_price"])
        unreal     = (cur - p["entry_price"]) * p["shares"]
        unreal_pct = (cur / p["entry_price"] - 1) * 100
        unrealized += unreal
        pos_with_pnl.append({**p, "current_price": cur,
                              "unrealized_pnl": round(unreal, 2),
                              "unrealized_pct": round(unreal_pct, 2)})

    realized = sum(
        t.get("pnl", 0) or 0
        for t in data.get("trade_history", [])
        if t.get("action") == "sell"
    )
    invested = sum(p["notional"] for p in positions)
    return {
        "market":         "hk",
        "currency":       CURRENCY,
        "total_capital":  TOTAL_CAPITAL,
        "cash":           round(data["cash"], 2),
        "invested":       round(invested, 2),
        "unrealized_pnl": round(unrealized, 2),
        "realized_pnl":   round(realized, 2),
        "open_count":     len(positions),
        "open_positions": pos_with_pnl,
    }


def get_portfolio_summary(prices: Optional[dict[str, float]] = None) -> dict:
    return get_summary(prices)


# ── Pretty print ──────────────────────────────────────────────────────────────

def format_portfolio(prices: Optional[dict[str, float]] = None) -> str:
    s     = get_summary(prices)
    total = s["total_capital"]
    cash  = s["cash"]
    inv   = s["invested"]
    unr   = s["unrealized_pnl"]
    rlz   = s["realized_pnl"]

    lines = [f"📋 港股虚拟仓  (as of {load()['updated']})"]
    lines.append(
        f"   总资本 HK${total:,.0f}  现金 HK${cash:,.0f}  "
        f"持仓 HK${inv:,.0f}  浮盈 HK${unr:+,.0f}  已实现 HK${rlz:+,.0f}"
    )
    lines.append("")
    if s["open_positions"]:
        for p in sorted(s["open_positions"], key=lambda x: -x["notional"]):
            lines.append(
                f"   {p['symbol']:<12} {p['shares']:.2f}股  "
                f"入场 HK${p['entry_price']:.2f}  现价 HK${p['current_price']:.2f}  "
                f"浮盈 {p['unrealized_pct']:+.1f}%  (score {p['score']:.1f})"
            )
    else:
        lines.append("   （空仓 — 等待买入信号）")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    import sys
    args = sys.argv[1:]
    cmd  = args[0] if args else "show"

    if cmd == "show":
        print(format_portfolio())

    elif cmd == "reset":
        save(_fresh_state())
        print("✅ 港股虚拟仓已重置 — 资本 HK$1,000,000，空仓")
        print(format_portfolio())

    else:
        print(__doc__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
