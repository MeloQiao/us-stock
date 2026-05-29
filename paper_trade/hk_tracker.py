"""
Futu HK Position Tracker

Tracks actual Futu HK positions from a local JSON file and compares them
against daily strategy signals to surface actionable alerts.

Usage
─────
  # Show portfolio summary
  python3 paper_trade/hk_tracker.py show

  # Update a position (after buying/selling in Futu)
  python3 paper_trade/hk_tracker.py update 00700 10 460.00
  python3 paper_trade/hk_tracker.py sell 09618 all
  python3 paper_trade/hk_tracker.py sell 09988 100

  # Update cash balance
  python3 paper_trade/hk_tracker.py cash 50000
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_POSITIONS_FILE = Path(__file__).parent / "hk_positions.json"


# ── Read / Write ──────────────────────────────────────────────────────────────

def load() -> dict:
    if not _POSITIONS_FILE.exists():
        return {"updated": "", "account": {}, "positions": []}
    with open(_POSITIONS_FILE) as f:
        return json.load(f)


def save(data: dict) -> None:
    data["updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(_POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Position operations ───────────────────────────────────────────────────────

def get_positions() -> list[dict]:
    data = load()
    result = []
    for p in data.get("positions", []):
        price    = p.get("current_price", 0) or 0
        shares   = p.get("shares", 0) or 0
        avg_cost = p.get("avg_cost") or price
        mv       = price * shares
        entry_mv = avg_cost * shares
        result.append({
            "symbol":          p["symbol"],
            "qty":             float(shares),
            "avg_entry_price": float(avg_cost),
            "current_price":   float(price),
            "market_value":    float(mv),
            "unrealized_pl":   float(mv - entry_mv),
            "unrealized_plpc": float((price / avg_cost - 1) * 100) if avg_cost else 0.0,
            "side":            "long",
            "in_strategy":     p.get("in_strategy", True),
            "currency":        "HKD",
        })
    return result


def get_account() -> dict:
    data      = load()
    acct      = data.get("account", {})
    positions = get_positions()
    equity    = sum(p["market_value"] for p in positions)
    cash      = acct.get("cash", 0)
    return {
        "cash":            cash,
        "portfolio_value": equity + cash,
        "buying_power":    cash,
        "currency":        "HKD",
        "equity":          equity,
    }


def update_position(symbol: str, shares: float, price: float,
                    avg_cost: Optional[float] = None,
                    in_strategy: bool = True) -> None:
    data      = load()
    positions = data.get("positions", [])
    existing  = next((p for p in positions if p["symbol"] == symbol), None)

    if existing:
        old_shares = existing.get("shares", 0) or 0
        old_cost   = existing.get("avg_cost") or price
        new_shares = old_shares + shares
        if new_shares <= 0:
            positions.remove(existing)
            logger.info("HK Tracker: %s position closed", symbol)
        else:
            new_avg = ((old_shares * old_cost) + (shares * price)) / new_shares
            existing["shares"]        = new_shares
            existing["avg_cost"]      = round(new_avg, 4)
            existing["current_price"] = price
            logger.info("HK Tracker: %s updated → %s shares @ avg HK$%.2f",
                        symbol, new_shares, new_avg)
    else:
        positions.append({
            "symbol":        symbol,
            "shares":        shares,
            "current_price": price,
            "avg_cost":      avg_cost or price,
            "in_strategy":   in_strategy,
            "note":          "",
        })
        logger.info("HK Tracker: %s added → %s shares @ HK$%.2f", symbol, shares, price)

    data["positions"] = positions
    save(data)


def close_position(symbol: str) -> Optional[dict]:
    data      = load()
    positions = data.get("positions", [])
    existing  = next((p for p in positions if p["symbol"] == symbol), None)
    if existing:
        positions.remove(existing)
        data["positions"] = positions
        save(data)
        logger.info("HK Tracker: %s position closed", symbol)
        return {"symbol": symbol, "status": "closed"}
    logger.warning("HK Tracker: %s not found", symbol)
    return None


def update_prices(prices: dict[str, float]) -> None:
    data    = load()
    updated = []
    for p in data.get("positions", []):
        sym = p["symbol"]
        if sym in prices and prices[sym] > 0:
            p["current_price"] = round(prices[sym], 4)
            updated.append(sym)
    if updated:
        save(data)
        logger.debug("HK Tracker: prices refreshed for %s", updated)


def update_cash(cash: float) -> None:
    data = load()
    data.setdefault("account", {})["cash"] = round(cash, 2)
    save(data)


# ── Signal comparison (called by scheduler) ──────────────────────────────────

def compare_signals(
    signals: dict[str, int],
    composite_scores: dict[str, float],
    prices: dict[str, float],
    portfolio_weights: Optional[dict[str, float]] = None,
) -> list[dict]:
    data      = load()
    # Build covered_calls lookup: symbol → cc info
    covered_calls = {cc["symbol"]: cc for cc in data.get("covered_calls", [])}

    positions = {p["symbol"]: p for p in get_positions()}
    actions   = []

    for sym, pos in positions.items():
        if not pos.get("in_strategy", True):
            continue
        sig   = signals.get(sym, 0)
        score = composite_scores.get(sym, 0)
        price = prices.get(sym, pos["current_price"])

        if sig == -1:
            # Check if there's an open covered call on this symbol
            cc = covered_calls.get(sym)
            cc_warning = None
            if cc:
                cc_warning = (
                    f"⚠️ 先买回 {sym} {cc['expiry']} ${cc['strike']} Call"
                    f"（{cc['contracts']}张×{cc['shares_per_contract']}股），再卖股票！"
                )
            actions.append({
                "symbol":               sym,
                "action":               "EXIT",
                "reason":               f"score={score:.1f} → sell signal",
                "shares":               pos["qty"],
                "price":                price,
                "pnl_pct":              pos["unrealized_plpc"],
                "urgency":              "🔴 SELL NOW",
                "covered_call_warning": cc_warning,
            })
        elif sig == 1:
            w         = (portfolio_weights or {}).get(sym, 0)
            acct      = get_account()
            pv        = acct["portfolio_value"]
            target_mv = w * pv if w else 0
            curr_mv   = pos["market_value"]
            if w and curr_mv > target_mv * 1.10:
                actions.append({
                    "symbol":   sym,
                    "action":   "TRIM",
                    "reason":   f"held={curr_mv/pv*100:.1f}% > target={w*100:.1f}%",
                    "trim_hkd": round(curr_mv - target_mv, 0),
                    "price":    price,
                    "urgency":  "🟡 TRIM",
                })
            else:
                actions.append({
                    "symbol":  sym,
                    "action":  "HOLD",
                    "reason":  f"score={score:.1f} still bullish",
                    "shares":  pos["qty"],
                    "pnl_pct": pos["unrealized_plpc"],
                    "urgency": "🟢 HOLD",
                })

    # Check new entries (signal=1, not yet holding)
    # First pass: collect targets with non-zero notional
    enter_targets = []
    acct = get_account()
    for sym, sig in signals.items():
        if sym in positions or sig != 1:
            continue
        score  = composite_scores.get(sym, 0)
        price  = prices.get(sym, 0)
        w      = (portfolio_weights or {}).get(sym, 0)
        target = w * acct["portfolio_value"] if w else 0
        if target <= 0:
            continue   # skip HK$0 targets entirely
        enter_targets.append({"symbol": sym, "score": score,
                               "price": price, "target": target})

    # Second pass: scale down to available cash, filter tiny amounts
    if enter_targets:
        available    = max(acct["cash"], 0)
        total_target = sum(t["target"] for t in enter_targets)
        scale        = min(1.0, available / total_target) if total_target > 0 else 0
        MIN_TRADE    = 200   # below HK$200 not worth showing

        for t in enter_targets:
            notional = round(t["target"] * scale, 0)
            if notional < MIN_TRADE:
                continue
            actions.append({
                "symbol":   t["symbol"],
                "action":   "ENTER",
                "reason":   f"score={t['score']:.1f} buy signal",
                "notional": notional,
                "price":    t["price"],
                "urgency":  "🟢 BUY",
            })

    return actions


def get_portfolio_summary(prices: Optional[dict[str, float]] = None) -> dict:
    if prices:
        update_prices(prices)
    return {
        "account":       get_account(),
        "positions":     get_positions(),
        "recent_orders": [],
    }


# ── Formatted report ─────────────────────────────────────────────────────────

def format_portfolio(prices: Optional[dict[str, float]] = None) -> str:
    if prices:
        update_prices(prices)
    positions = get_positions()
    acct      = get_account()
    lines     = [f"📋 Futu 港股持仓  (as of {load()['updated']})"]
    lines.append(
        f"   总市值 HK${acct['portfolio_value']:,.0f}  "
        f"现金 HK${acct['cash']:,.0f}  "
        f"股票 HK${acct['equity']:,.0f}"
    )
    lines.append("")
    for p in sorted(positions, key=lambda x: -x["market_value"]):
        pnl_str = f"{p['unrealized_plpc']:+.1f}%" if p["avg_entry_price"] else "N/A"
        flag    = "" if p.get("in_strategy", True) else " ⚠️非策略"
        lines.append(
            f"   {p['symbol']:<6} {p['qty']:.0f}股  "
            f"HK${p['current_price']:.2f}  "
            f"市值 HK${p['market_value']:,.0f}  "
            f"盈亏 {pnl_str}{flag}"
        )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    import sys
    args = sys.argv[1:]
    if not args or args[0] == "show":
        print(format_portfolio())
        return

    cmd = args[0]

    if cmd == "update" and len(args) >= 4:
        sym    = args[1]
        shares = float(args[2])
        price  = float(args[3])
        avg    = float(args[4]) if len(args) > 4 else None
        update_position(sym, shares, price, avg_cost=avg)
        print(f"✅ {sym} 更新完成")
        print(format_portfolio())

    elif cmd == "sell" and len(args) >= 3:
        sym = args[1]
        if args[2].lower() == "all":
            close_position(sym)
            print(f"✅ {sym} 已清仓")
        else:
            shares = float(args[2])
            data   = load()
            pos    = next((p for p in data["positions"] if p["symbol"] == sym), None)
            price  = pos["current_price"] if pos else 0
            update_position(sym, -shares, price)
            print(f"✅ {sym} 减仓 {shares} 股")
        print(format_portfolio())

    elif cmd == "cash" and len(args) >= 2:
        update_cash(float(args[1]))
        print(f"✅ 现金更新为 HK${float(args[1]):,.2f}")

    else:
        print(__doc__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
