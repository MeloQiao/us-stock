"""
Backtest engine: wraps vectorbt to run portfolio simulations.
Signals come from strategy modules (1=long, -1=short/exit, 0=hold).
All strategies trade daily at next-bar open price (realistic execution).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt not installed. Backtest will use simple pandas simulation.")


# ── Simple pandas fallback backtest ───────────────────────────────────────

def _simple_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
) -> dict:
    """
    Minimal long-only backtest when vectorbt is unavailable.
    Returns same keys as vectorbt result for UI compatibility.
    """
    cash = init_cash
    shares = 0.0
    portfolio_values = []
    in_position = False

    for i, date in enumerate(close.index):
        price = close.iloc[i]
        entry = entries.iloc[i]
        exit_ = exits.iloc[i]

        if not in_position and entry:
            shares = (cash * (1 - fees)) / price
            cash = 0.0
            in_position = True
        elif in_position and exit_:
            cash = shares * price * (1 - fees)
            shares = 0.0
            in_position = False

        portfolio_values.append(cash + shares * price)

    final_value = portfolio_values[-1] if portfolio_values else init_cash
    portfolio_series = pd.Series(portfolio_values, index=close.index)

    total_return = (final_value - init_cash) / init_cash
    years = (close.index[-1] - close.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    daily_returns = portfolio_series.pct_change().dropna()
    sharpe = float(daily_returns.mean() / (daily_returns.std() + 1e-9) * np.sqrt(252))

    return {
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "final_value": round(final_value, 2),
        "portfolio_value": portfolio_series,
        "drawdown_series": drawdown,
        "n_trades": int((entries & ~entries.shift(1).fillna(False)).sum()),
    }


# ── Main backtest runner ──────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    signal_series: pd.Series,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    use_next_open: bool = True,
) -> dict[str, Any]:
    """
    Run a backtest for a single symbol using pre-computed signals.

    Parameters
    ----------
    df : OHLCV DataFrame
    signal_series : Series of (1, 0, -1) aligned to df.index
    init_cash : starting portfolio value in USD
    fees : one-way commission rate (0.001 = 0.1%)
    use_next_open : execute at next bar's open (realistic, no lookahead)

    Returns dict with metrics and time-series data for plotting.
    """
    close = df["Close"]
    price = df["Open"].shift(-1).fillna(df["Close"]) if use_next_open else close

    entries = (signal_series == 1) & (signal_series.shift(1).fillna(0) != 1)
    exits = (signal_series != 1) & (signal_series.shift(1).fillna(0) == 1)

    if VBT_AVAILABLE:
        return _vbt_backtest(price, entries, exits, init_cash, fees)
    else:
        return _simple_backtest(close, entries, exits, init_cash, fees)


def _vbt_backtest(
    price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float,
    fees: float,
) -> dict:
    """Run backtest using vectorbt."""
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        freq="D",
    )

    stats = pf.stats()
    portfolio_series = pf.value()
    drawdown_series = pf.drawdown()

    total_return = float(stats.get("Total Return [%]", 0))
    annual_return = float(stats.get("Annualized Return [%]", 0))
    max_drawdown = float(stats.get("Max Drawdown [%]", 0))
    sharpe = float(stats.get("Sharpe Ratio", 0))
    n_trades = int(stats.get("Total Trades", 0))

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(-abs(max_drawdown), 2),
        "sharpe_ratio": round(sharpe, 3),
        "final_value": round(float(portfolio_series.iloc[-1]), 2),
        "portfolio_value": portfolio_series,
        "drawdown_series": drawdown_series,
        "n_trades": n_trades,
        "vbt_portfolio": pf,
    }


def run_benchmark(df: pd.DataFrame, init_cash: float = 10_000.0) -> dict:
    """Buy-and-hold benchmark for comparison."""
    close = df["Close"]
    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    portfolio = init_cash * (close / close.iloc[0])
    rolling_max = portfolio.cummax()
    drawdown = (portfolio - rolling_max) / rolling_max
    daily_rets = portfolio.pct_change().dropna()
    sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252))
    years = (close.index[-1] - close.index[0]).days / 365.25
    annual = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual, 2),
        "max_drawdown": round(float(drawdown.min()) * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "final_value": round(float(portfolio.iloc[-1]), 2),
        "portfolio_value": portfolio,
        "drawdown_series": drawdown,
        "n_trades": 1,
    }


def compare_strategies(
    df: pd.DataFrame,
    strategy_results: dict,
    symbol: str = "",
    init_cash: float = 10_000.0,
) -> pd.DataFrame:
    """
    Run backtests for all strategies and return a comparison DataFrame.

    Parameters
    ----------
    df : OHLCV DataFrame
    strategy_results : output of composite.run_all_strategies()
    symbol : ticker symbol (for display)

    Returns a DataFrame with one row per strategy + benchmark.
    """
    rows = []

    # Benchmark
    bm = run_benchmark(df, init_cash)
    rows.append({
        "strategy": "Buy & Hold",
        "total_return_%": bm["total_return"],
        "annual_return_%": bm["annual_return"],
        "max_drawdown_%": bm["max_drawdown"],
        "sharpe_ratio": bm["sharpe_ratio"],
        "n_trades": bm["n_trades"],
        "final_value": bm["final_value"],
    })

    for name, result in strategy_results.items():
        if name == "composite_score":
            continue  # included separately
        try:
            bt = run_backtest(df, result["signal_series"], init_cash=init_cash)
            rows.append({
                "strategy": name,
                "total_return_%": bt["total_return"],
                "annual_return_%": bt["annual_return"],
                "max_drawdown_%": bt["max_drawdown"],
                "sharpe_ratio": bt["sharpe_ratio"],
                "n_trades": bt["n_trades"],
                "final_value": bt["final_value"],
            })
        except Exception as e:
            logger.warning("Backtest failed for %s / %s: %s", symbol, name, e)

    # Composite
    if "composite_score" in strategy_results:
        try:
            bt = run_backtest(df, strategy_results["composite_score"]["signal_series"], init_cash=init_cash)
            rows.append({
                "strategy": "composite_score",
                "total_return_%": bt["total_return"],
                "annual_return_%": bt["annual_return"],
                "max_drawdown_%": bt["max_drawdown"],
                "sharpe_ratio": bt["sharpe_ratio"],
                "n_trades": bt["n_trades"],
                "final_value": bt["final_value"],
            })
        except Exception as e:
            logger.warning("Composite backtest failed for %s: %s", symbol, e)

    return pd.DataFrame(rows).set_index("strategy")
