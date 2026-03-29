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
    position_size: float = PAPER_TRADE_POSITION_SIZE,
) -> list[dict]:
    """
    Execute paper trades based on strategy composite signals.

    Parameters
    ----------
    signals : {symbol: signal} where signal is 1 (buy), -1 (sell/exit), 0 (hold)
    position_size : fraction of portfolio to allocate per position (default 10%)

    Returns list of executed order results.
    """
    account = get_account()
    if not account:
        logger.error("Cannot execute signals: account info unavailable.")
        return []

    portfolio_value = account["portfolio_value"]
    positions = {p["symbol"]: p for p in get_positions()}
    results = []

    for symbol, signal in signals.items():
        try:
            in_position = symbol in positions

            if signal == 1 and not in_position:
                # Enter long position
                notional = portfolio_value * position_size
                if notional < 1:
                    continue
                order = place_market_order(symbol, "buy", notional=notional)
                if order:
                    results.append({**order, "action": "enter_long"})

            elif signal == -1 and in_position:
                # Exit position
                order = close_position(symbol)
                if order:
                    results.append({**order, "action": "exit"})

            # signal == 0 or already in correct state: do nothing

        except Exception as e:
            logger.error("Signal execution error for %s: %s", symbol, e)

    return results


def get_portfolio_summary() -> dict:
    """Return full portfolio snapshot: account + positions."""
    return {
        "account": get_account(),
        "positions": get_positions(),
        "recent_orders": get_orders(status="all", limit=20),
    }
