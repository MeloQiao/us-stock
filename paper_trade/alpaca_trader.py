"""
Alpaca Paper Trading integration.
Executes orders at next-day market open based on strategy signals.
Uses alpaca-py (the official modern SDK).
"""

from __future__ import annotations

import logging
from typing import Optional

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    PAPER_TRADE_POSITION_SIZE,
    LEVERAGED_ETFS,
)

logger = logging.getLogger(__name__)


def _get_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def _get_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


# ── Account ───────────────────────────────────────────────────────────────

def get_account() -> dict:
    """Return account info: cash, portfolio_value, buying_power."""
    try:
        client = _get_trading_client()
        acct = client.get_account()
        return {
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power": float(acct.buying_power),
            "currency": acct.currency,
            "status": acct.status,
        }
    except Exception as e:
        logger.error("Failed to get account: %s", e)
        return {}


def get_positions() -> list[dict]:
    """Return current open positions."""
    try:
        client = _get_trading_client()
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "side": p.side,
            }
            for p in positions
        ]
    except Exception as e:
        logger.error("Failed to get positions: %s", e)
        return []


def get_orders(status: str = "all", limit: int = 50) -> list[dict]:
    """Return recent orders."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = _get_trading_client()
        req = GetOrdersRequest(status=QueryOrderStatus(status), limit=limit)
        orders = client.get_orders(req)
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": float(o.qty or 0),
                "side": str(o.side),
                "type": str(o.type),
                "status": str(o.status),
                "filled_avg_price": float(o.filled_avg_price or 0),
                "submitted_at": str(o.submitted_at),
            }
            for o in orders
        ]
    except Exception as e:
        logger.error("Failed to get orders: %s", e)
        return []


# ── Order execution ───────────────────────────────────────────────────────

def place_market_order(
    symbol: str,
    side: str,  # "buy" or "sell"
    notional: Optional[float] = None,
    qty: Optional[float] = None,
) -> Optional[dict]:
    """
    Place a market order.
    Use notional (USD amount) for fractional shares, or qty for whole shares.
    Orders are submitted with time_in_force=day → execute at next open.
    """
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = _get_trading_client()
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        if notional is not None:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

        order = client.submit_order(req)
        logger.info("Order placed: %s %s %s notional=%s qty=%s", side, symbol, order.id, notional, qty)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": str(order.side),
            "status": str(order.status),
            "notional": notional,
            "qty": qty,
        }
    except Exception as e:
        logger.error("Failed to place order %s %s: %s", side, symbol, e)
        return None


def close_position(symbol: str) -> Optional[dict]:
    """Close the entire position for a symbol."""
    try:
        client = _get_trading_client()
        result = client.close_position(symbol)
        logger.info("Closed position: %s", symbol)
        return {"symbol": symbol, "status": "closed", "order_id": str(result.id)}
    except Exception as e:
        logger.error("Failed to close position %s: %s", symbol, e)
        return None


# ── Signal → Trade ────────────────────────────────────────────────────────

def execute_signals(
    signals: dict[str, int],
    scores: Optional[dict[str, int]] = None,
    weights: Optional[dict[str, float]] = None,
    position_size: float = PAPER_TRADE_POSITION_SIZE,
) -> list[dict]:
    """
    Execute paper trades based on strategy composite signals.

    Parameters
    ----------
    signals : {symbol: signal}  1=buy, -1=sell/exit, 0=hold
    scores  : {symbol: total_score}  used for score-weighted fallback.
    weights : {symbol: fraction}  pre-computed portfolio weights (from optimizer).
              If provided, notional = portfolio_value * weight[symbol].
              If None, falls back to score-proportional then equal-weight.
    """
    account = get_account()
    if not account:
        logger.error("Cannot execute signals: account info unavailable.")
        return []

    portfolio_value = account["portfolio_value"]
    buying_power    = account["buying_power"]
    positions       = {p["symbol"]: p for p in get_positions()}
    results         = []

    # ── 1. Exits first ────────────────────────────────────────────────────
    for symbol, signal in signals.items():
        if signal == -1 and symbol in positions:
            try:
                order = close_position(symbol)
                if order:
                    results.append({**order, "action": "exit"})
            except Exception as e:
                logger.error("Exit error for %s: %s", symbol, e)

    # ── 2. Entries ────────────────────────────────────────────────────────
    new_buys = [sym for sym, sig in signals.items() if sig == 1 and sym not in positions]
    if not new_buys:
        return results

    budget = min(portfolio_value, buying_power)

    # Determine per-symbol notional
    if weights:
        # Use pre-computed optimizer weights (only for symbols in new_buys)
        # Renormalize to the subset that are actual new buys
        sub_w = {s: weights[s] for s in new_buys if s in weights}
        # Log symbols with BUY signal that were filtered out by portfolio optimizer
        filtered_out = [s for s in new_buys if s not in sub_w]
        for s in filtered_out:
            logger.info(
                "[us] %s had BUY signal but was filtered by portfolio optimizer "
                "(corr dedup or sector cap) — no order placed", s
            )
            results.append({"symbol": s, "action": "filtered",
                             "reason": "portfolio_optimizer"})
        if not sub_w:
            sub_w = {s: 1 / len(new_buys) for s in new_buys}
            filtered_out = []
        total_w = sum(sub_w.values())
        notionals = {s: budget * (sub_w[s] / total_w) for s in sub_w}
        new_buys = list(sub_w.keys())  # only attempt orders for non-filtered symbols
    elif scores:
        raw = {s: max(scores.get(s, 1), 1) for s in new_buys}
        total_w = sum(raw.values())
        notionals = {s: budget * (raw[s] / total_w) for s in new_buys}
    else:
        notionals = {s: budget / len(new_buys) for s in new_buys}

    # Fetch current prices for share estimation (best-effort)
    price_map: dict[str, float] = {}
    for symbol in new_buys:
        try:
            import yfinance as yf
            price_map[symbol] = yf.Ticker(symbol).fast_info.last_price or 0.0
        except Exception:
            price_map[symbol] = 0.0

    for symbol in new_buys:
        notional = notionals.get(symbol, 0)
        try:
            if notional < 1:
                logger.warning("Notional too small for %s (%.2f), skipping.", symbol, notional)
                continue
            price      = price_map.get(symbol, 0.0)
            est_shares = round(notional / price, 4) if price > 0 else None
            wt_pct     = (weights or {}).get(symbol, notional / budget) * 100
            logger.info(
                "Buy: %s weight=%.1f%% notional=%.2f est_shares=%s",
                symbol, wt_pct, notional, est_shares,
            )
            order = place_market_order(symbol, "buy", notional=notional)
            if order:
                results.append({
                    **order,
                    "action": "enter_long",
                    "notional": notional,
                    "score": (scores or {}).get(symbol, "—"),
                    "est_shares": est_shares,
                    "ref_price": round(price, 2) if price > 0 else None,
                })
        except Exception as e:
            logger.error("Buy error for %s: %s", symbol, e)

    return results


def get_portfolio_summary() -> dict:
    """Return full portfolio snapshot: account + positions."""
    return {
        "account": get_account(),
        "positions": get_positions(),
        "recent_orders": get_orders(status="all", limit=20),
    }
