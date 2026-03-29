"""
Scheduled jobs using APScheduler.
Runs daily after US market close (16:30 ET):
  1. Fetch latest OHLCV data for all symbols
  2. Compute all strategy signals
  3. Execute paper trades via Alpaca
  4. Send Feishu alert with today's signals
  5. Persist data to HF Dataset
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytz

from config import (
    ALL_SYMBOLS,
    HISTORY_YEARS,
    FEISHU_WEBHOOK_URL,
    HF_TOKEN,
    HF_DATASET_REPO,
    DAILY_JOB_HOUR,
    DAILY_JOB_MINUTE,
    TIMEZONE,
    ALPACA_API_KEY,
)

logger = logging.getLogger(__name__)


def run_daily_pipeline() -> dict:
    """
    Full daily pipeline: fetch → signal → trade → alert → persist.
    Returns summary dict of results.
    """
    logger.info("=== Daily pipeline started at %s ===", datetime.now().isoformat())
    summary = {"date": datetime.today().strftime("%Y-%m-%d"), "symbols": {}, "errors": []}

    # 1. Fetch data
    from data.fetcher import fetch_multiple, save_to_hf, save_signals_to_hf
    import pandas as pd

    logger.info("Fetching data for %d symbols...", len(ALL_SYMBOLS))
    data = fetch_multiple(ALL_SYMBOLS, years=HISTORY_YEARS, force_refresh=True)
    vix_data = fetch_multiple(["^VIX"], years=HISTORY_YEARS, force_refresh=True)
    vix_df = vix_data.get("^VIX")

    # 2. Compute signals for each symbol
    from strategies.composite import run_all_strategies
    from strategies import STRATEGY_LABELS

    all_results: dict[str, dict] = {}
    composite_signals: dict[str, int] = {}

    for symbol in ALL_SYMBOLS:
        if symbol not in data:
            summary["errors"].append(f"No data for {symbol}")
            continue
        try:
            results = run_all_strategies(data[symbol], symbol=symbol, vix_df=vix_df)
            all_results[symbol] = results
            composite_signals[symbol] = results["composite_score"]["signal"]
            summary["symbols"][symbol] = {
                "composite_signal": composite_signals[symbol],
                "total_score": results["composite_score"].get("total_score", 0),
            }
            logger.info("%s → composite signal: %d", symbol, composite_signals[symbol])
        except Exception as e:
            logger.error("Strategy error for %s: %s", symbol, e)
            summary["errors"].append(f"Strategy failed: {symbol}: {e}")

    # 3. Paper trade execution (only if Alpaca configured)
    if ALPACA_API_KEY:
        try:
            from paper_trade.alpaca_trader import execute_signals
            trade_results = execute_signals(composite_signals)
            summary["trades"] = trade_results
            logger.info("Paper trades executed: %d orders", len(trade_results))
        except Exception as e:
            logger.error("Paper trade execution failed: %s", e)
            summary["errors"].append(f"Paper trade failed: {e}")
    else:
        logger.info("Alpaca not configured, skipping paper trade.")

    # 4. Feishu alert
    if FEISHU_WEBHOOK_URL and all_results:
        try:
            from alerts.feishu import send_signal_alert, build_signal_list
            vix_value = None
            if vix_df is not None and not vix_df.empty:
                vix_value = float(vix_df["Close"].iloc[-1])

            signal_list = build_signal_list(all_results, STRATEGY_LABELS)
            success = send_signal_alert(FEISHU_WEBHOOK_URL, signal_list, vix_value=vix_value)
            summary["feishu_sent"] = success
        except Exception as e:
            logger.error("Feishu alert failed: %s", e)
            summary["errors"].append(f"Feishu failed: {e}")

    # 5. Persist to HF Dataset
    if HF_TOKEN and HF_DATASET_REPO:
        try:
            for symbol, df in data.items():
                save_to_hf(df, symbol, HF_DATASET_REPO, HF_TOKEN)

            # Save signal log
            rows = []
            for symbol, results in all_results.items():
                for strat_name, result in results.items():
                    rows.append({
                        "date": summary["date"],
                        "symbol": symbol,
                        "strategy": strat_name,
                        "signal": result["signal"],
                    })
            if rows:
                signals_df = pd.DataFrame(rows)
                save_signals_to_hf(signals_df, HF_DATASET_REPO, HF_TOKEN)
            logger.info("Data persisted to HF Dataset.")
        except Exception as e:
            logger.error("HF persist failed: %s", e)
            summary["errors"].append(f"HF persist failed: {e}")

    logger.info("=== Daily pipeline complete. Errors: %d ===", len(summary["errors"]))
    return summary


def start_scheduler():
    """
    Start APScheduler with the daily pipeline job.
    Runs at DAILY_JOB_HOUR:DAILY_JOB_MINUTE in TIMEZONE.
    Non-blocking: runs in background thread.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(timezone=pytz.timezone(TIMEZONE))
    scheduler.add_job(
        run_daily_pipeline,
        trigger="cron",
        hour=DAILY_JOB_HOUR,
        minute=DAILY_JOB_MINUTE,
        id="daily_pipeline",
        name="Daily signal + trade pipeline",
        misfire_grace_time=600,  # 10 min grace window
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started. Daily job at %02d:%02d %s",
        DAILY_JOB_HOUR,
        DAILY_JOB_MINUTE,
        TIMEZONE,
    )
    return scheduler
