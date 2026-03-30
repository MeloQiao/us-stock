"""
Scheduled jobs — one pipeline per market, each firing at that market's close time.

Market   Close time  Timezone
───────  ──────────  ────────────────────
us       16:30       America/New_York
hk       16:00       Asia/Hong_Kong
cn       15:00       Asia/Shanghai

Only markets listed in config.ACTIVE_MARKETS get a scheduled job.
US paper trading (Alpaca) is only wired up for the "us" pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

import pytz

from config import (
    ACTIVE_MARKETS,
    MARKET_WATCHLISTS,
    MARKET_SCHEDULE,
    HISTORY_YEARS,
    FEISHU_WEBHOOK_URL,
    HF_TOKEN,
    HF_DATASET_REPO,
    ALPACA_API_KEY,
)

logger = logging.getLogger(__name__)

Market = Literal["us", "hk", "cn"]


# ════════════════════════════════════════════════════════════════════════
# Core pipeline (market-agnostic)
# ════════════════════════════════════════════════════════════════════════

def run_daily_pipeline(market: Market = "us") -> dict:
    """
    Full daily pipeline for one market:
      1. Fetch OHLCV data
      2. Compute all strategy signals
      3. [us only] Execute Alpaca paper trades
      4. Send Feishu alert
      5. Persist to HF Dataset

    Returns a summary dict.
    """
    logger.info("=== Pipeline [%s] started at %s ===", market, datetime.now().isoformat())
    summary: dict = {
        "market": market,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "symbols": {},
        "errors": [],
    }

    watchlist = MARKET_WATCHLISTS.get(market, {})
    symbols = [s for group in watchlist.values() for s in group]

    if not symbols:
        logger.info("No symbols configured for market [%s], skipping.", market)
        summary["errors"].append(f"No symbols for market: {market}")
        return summary

    # ── 1. Fetch data ─────────────────────────────────────────────────
    from data.fetcher import fetch_multiple, save_to_hf, save_signals_to_hf, cleanup_old_signals_on_hf
    import pandas as pd

    logger.info("Fetching %d symbols for [%s]...", len(symbols), market)
    data = fetch_multiple(symbols, years=HISTORY_YEARS, market=market, force_refresh=True)

    # VIX is only relevant for US market
    vix_df = None
    if market == "us":
        vix_data = fetch_multiple(["^VIX"], years=HISTORY_YEARS, market="us", force_refresh=True)
        vix_df = vix_data.get("^VIX")

    # ── 2. Strategy signals ───────────────────────────────────────────
    from strategies.composite import run_all_strategies
    from strategies import STRATEGY_LABELS

    all_results: dict[str, dict] = {}
    composite_signals: dict[str, int] = {}
    composite_scores: dict[str, int] = {}

    for symbol in symbols:
        if symbol not in data:
            summary["errors"].append(f"No data for {symbol}")
            continue
        try:
            results = run_all_strategies(data[symbol], symbol=symbol, vix_df=vix_df)
            all_results[symbol] = results
            composite_signals[symbol] = results["composite_score"]["signal"]
            composite_scores[symbol] = results["composite_score"].get("total_score", 0)
            summary["symbols"][symbol] = {
                "composite_signal": composite_signals[symbol],
                "total_score": composite_scores[symbol],
            }
            logger.info("[%s] %s → signal: %d score: %d", market, symbol,
                        composite_signals[symbol], composite_scores[symbol])
        except Exception as e:
            logger.error("[%s] Strategy error for %s: %s", market, symbol, e)
            summary["errors"].append(f"Strategy failed {symbol}: {e}")

    # ── 3. Paper trade (US only, Alpaca) ──────────────────────────────
    if market == "us" and ALPACA_API_KEY:
        try:
            from paper_trade.alpaca_trader import execute_signals
            trade_results = execute_signals(composite_signals, scores=composite_scores)
            summary["trades"] = trade_results
            logger.info("[us] Paper trades: %d orders", len(trade_results))
        except Exception as e:
            logger.error("[us] Paper trade failed: %s", e)
            summary["errors"].append(f"Paper trade failed: {e}")

    # ── 4. Feishu alert ───────────────────────────────────────────────
    if FEISHU_WEBHOOK_URL and all_results:
        try:
            from alerts.feishu import send_signal_alert, build_signal_list
            vix_value = None
            if vix_df is not None and not vix_df.empty:
                vix_value = float(vix_df["Close"].iloc[-1])

            signal_list = build_signal_list(all_results, STRATEGY_LABELS)
            ok = send_signal_alert(FEISHU_WEBHOOK_URL, signal_list, vix_value=vix_value)
            summary["feishu_sent"] = ok
        except Exception as e:
            logger.error("[%s] Feishu alert failed: %s", market, e)
            summary["errors"].append(f"Feishu failed: {e}")

    # ── 5. HF Dataset persist ─────────────────────────────────────────
    if HF_TOKEN and HF_DATASET_REPO:
        try:
            for symbol, df in data.items():
                save_to_hf(df, symbol, HF_DATASET_REPO, HF_TOKEN, market=market)

            rows = []
            for symbol, results in all_results.items():
                for strat_name, result in results.items():
                    rows.append({
                        "date": summary["date"],
                        "market": market,
                        "symbol": symbol,
                        "strategy": strat_name,
                        "signal": result["signal"],
                    })
            if rows:
                save_signals_to_hf(pd.DataFrame(rows), HF_DATASET_REPO, HF_TOKEN, market=market)

            deleted = cleanup_old_signals_on_hf(HF_DATASET_REPO, HF_TOKEN, market=market, keep_days=90)
            if deleted:
                logger.info("[%s] Cleaned up %d old signal files.", market, deleted)
            logger.info("[%s] Persisted to HF Dataset.", market)
        except Exception as e:
            logger.error("[%s] HF persist failed: %s", market, e)
            summary["errors"].append(f"HF persist failed: {e}")

    logger.info("=== Pipeline [%s] done. Errors: %d ===", market, len(summary["errors"]))
    return summary


# ── Convenience wrappers so APScheduler can call a plain no-arg function ─

def _run_us():
    return run_daily_pipeline("us")

def _run_hk():
    return run_daily_pipeline("hk")

def _run_cn():
    return run_daily_pipeline("cn")

_MARKET_FUNC = {"us": _run_us, "hk": _run_hk, "cn": _run_cn}


# ════════════════════════════════════════════════════════════════════════
# Scheduler
# ════════════════════════════════════════════════════════════════════════

def start_scheduler():
    """
    Start APScheduler background scheduler.
    Registers one cron job per active market, each in its own local timezone.
    Non-blocking — runs in background thread.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    # Use UTC as the scheduler's base timezone; individual jobs carry their own tz.
    scheduler = BackgroundScheduler(timezone=pytz.utc)

    for market in ACTIVE_MARKETS:
        cfg = MARKET_SCHEDULE[market]
        tz = pytz.timezone(cfg["timezone"])
        func = _MARKET_FUNC[market]

        scheduler.add_job(
            func,
            trigger="cron",
            hour=cfg["hour"],
            minute=cfg["minute"],
            timezone=tz,
            id=f"pipeline_{market}",
            name=f"Daily pipeline [{market}] at {cfg['hour']:02d}:{cfg['minute']:02d} {cfg['timezone']}",
            misfire_grace_time=600,
            replace_existing=True,
        )
        logger.info(
            "Registered job: [%s] at %02d:%02d %s",
            market, cfg["hour"], cfg["minute"], cfg["timezone"],
        )

    scheduler.start()
    logger.info("Scheduler started with %d active market(s): %s",
                len(ACTIVE_MARKETS), ACTIVE_MARKETS)
    return scheduler
