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
    VIRTUAL_PORTFOLIO_CAPITAL,
    VIRTUAL_PORTFOLIO_CURRENCY,
)

logger = logging.getLogger(__name__)

Market = Literal["us", "hk", "cn"]

# exchange_calendars exchange IDs per market
_EXCHANGE_ID = {"us": "XNYS", "hk": "XHKG", "cn": "XSHG"}


def is_trading_day(market: Market, date: datetime | None = None) -> bool:
    """
    Return True if `date` (defaults to today) is a trading day for `market`.
    Accounts for weekends and public holidays via exchange_calendars.
    Falls back to True if the library is unavailable (fail-open).
    """
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar(_EXCHANGE_ID[market])
        check = (date or datetime.now()).strftime("%Y-%m-%d")
        return cal.is_session(check)
    except Exception as e:
        logger.warning("Trading day check failed for [%s], assuming open: %s", market, e)
        return True


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
    today = datetime.today()
    summary: dict = {
        "market": market,
        "date": today.strftime("%Y-%m-%d"),
        "symbols": {},
        "errors": [],
    }

    if not is_trading_day(market, today):
        logger.info("[%s] Today is not a trading day, pipeline skipped.", market)
        summary["skipped"] = "non-trading day"
        return summary

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
    data = fetch_multiple(symbols, years=HISTORY_YEARS, market=market, force_refresh=False)

    # VIX is only relevant for US market
    vix_df = None
    if market == "us":
        vix_data = fetch_multiple(["^VIX"], years=HISTORY_YEARS, market="us", force_refresh=False)
        vix_df = vix_data.get("^VIX")

    # ── 1b. Regime detection ──────────────────────────────────────────
    from strategies.regime import detect_regime, apply_regime_gate
    regime_info = detect_regime(market=market, price_data=data)
    summary["regime"] = regime_info.get("regime")
    logger.info("[%s] Regime: %s — %s", market, regime_info["regime"], regime_info["reason"])

    # ── 1c-extra. Layer 1 — Crash Shield (rule-based macro protection) ─
    crash_shield_result = {"level": "NONE", "score": 0, "position_multiplier": 1.0}
    if market == "us":
        try:
            from strategies.crash_shield import evaluate_crash_shield
            hyg_data = fetch_multiple(["HYG"], years=HISTORY_YEARS, market="us", force_refresh=False)
            hyg_df_cs = hyg_data.get("HYG")
            spy_cs = data.get("SPY") or data.get(next(iter(data), None))
            if spy_cs is not None:
                crash_shield_result = evaluate_crash_shield(spy_cs, vix_df, hyg_df_cs)
                summary["crash_shield"] = {
                    "level": crash_shield_result["level"],
                    "score": crash_shield_result["score"],
                    "triggered": crash_shield_result.get("triggered", []),
                }
                logger.info("[us] CrashShield: %s (score=%d/4)",
                            crash_shield_result["level"], crash_shield_result["score"])
        except Exception as e:
            logger.warning("[us] CrashShield evaluation failed: %s", e)

    # ── 1c-extra. Layer 2 — ML Regime Classifier ──────────────────────
    ml_prob = 0.5          # neutral default
    ml_multiplier = 1.0
    if market == "us":
        try:
            from strategies.ml_regime import MLRegimeClassifier
            clf = MLRegimeClassifier()
            loaded = clf.load(
                hf_repo=HF_DATASET_REPO if HF_TOKEN else None,
                hf_token=HF_TOKEN if HF_TOKEN else None,
            )
            if loaded:
                spy_cs = data.get("SPY") or data.get(next(iter(data), None))
                hyg_df_ml = hyg_data.get("HYG") if "hyg_data" in dir() else None
                if spy_cs is not None:
                    ml_prob = clf.predict_proba_latest(spy_cs, vix_df, hyg_df_ml)
                    ml_multiplier = clf.to_position_multiplier(ml_prob)
                    summary["ml_regime"] = {
                        "prob": round(ml_prob, 3),
                        "multiplier": round(ml_multiplier, 3),
                    }
                    logger.info("[us] ML Regime: prob=%.3f mult=%.3f", ml_prob, ml_multiplier)
        except Exception as e:
            logger.warning("[us] ML Regime failed: %s", e)

    # ── 1c. Load regime-specific strategy weights (from walk-forward optimizer) ─
    from strategies.walk_forward_optimizer import get_regime_weights
    strategy_weights = get_regime_weights(
        market=market,
        regime=regime_info.get("sub_state", "bull_caution"),
        hf_repo=HF_DATASET_REPO if HF_TOKEN else None,
        hf_token=HF_TOKEN if HF_TOKEN else None,
    )
    summary["strategy_weights_regime"] = regime_info.get("sub_state", "equal")

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
            results = run_all_strategies(
                data[symbol], symbol=symbol, vix_df=vix_df,
                weights=strategy_weights, market=market,
            )
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

    # Apply regime gate — blocks new buys in bear market
    gated_signals = apply_regime_gate(composite_signals, regime_info)

    # ── 2b-extra. Layer 1 gate — Crash Shield blocks new buys in SHIELD ─
    if crash_shield_result["level"] == "SHIELD":
        gated_signals = {
            sym: (0 if sig == 1 else sig)
            for sym, sig in gated_signals.items()
        }
        logger.info("[%s] CrashShield SHIELD: all new buy signals blocked", market)
    elif crash_shield_result["level"] == "CAUTION":
        logger.info("[%s] CrashShield CAUTION: new positions will be halved", market)

    # ── 2b-extra. Layer 3 — Dual Momentum filter ─────────────────────
    dm_result: dict = {}
    dm_multipliers: dict[str, float] = {}
    if market == "us":
        try:
            from strategies.dual_momentum import apply_dual_momentum
            spy_dm = data.get("SPY") or data.get(next(iter(data), None))
            if spy_dm is not None:
                gated_signals, dm_multipliers, dm_result = apply_dual_momentum(
                    gated_signals, spy_dm, data,
                )
                summary["dual_momentum"] = {
                    "abs_momentum_ok": dm_result.get("abs_momentum_ok"),
                    "spy_12m_ret":     dm_result.get("spy_12m_ret"),
                    "crash_protect":   dm_result.get("crash_protect"),
                    "position_scale":  dm_result.get("position_scale"),
                }
                logger.info("[us] DualMomentum: scale=%.2f crash=%s",
                            dm_result.get("position_scale", 1.0),
                            dm_result.get("crash_protect", False))
        except Exception as e:
            logger.warning("[us] DualMomentum failed: %s", e)

    # ── 2b. Portfolio optimization ────────────────────────────────────
    from portfolio.optimizer import optimize_portfolio
    buy_candidates = {
        sym: composite_scores[sym]
        for sym, sig in gated_signals.items()
        if sig == 1
    }
    portfolio_weights: dict[str, float] = {}
    if buy_candidates:
        try:
            portfolio_weights = optimize_portfolio(
                buy_signals=buy_candidates,
                price_data=data,
                market=market,
                method="risk_parity",   # risk_parity is robust; max_sharpe needs scipy
            )
            summary["portfolio_weights"] = portfolio_weights
            logger.info("[%s] Portfolio weights: %s", market,
                        {s: f"{w:.1%}" for s, w in portfolio_weights.items()})
        except Exception as e:
            logger.warning("[%s] Portfolio optimisation failed: %s — using score weights", market, e)

    # ── 2c. Combine 3-layer multipliers into portfolio_weights (US only) ─
    #
    # Combined = CrashShield_mult × ML_mult × DualMomentum_per_symbol_mult
    # Applied as a scaling factor to portfolio_weights before execution.
    # HK/CN only use crash_shield and ml_regime (no SPY-based dual momentum).
    #
    if market == "us" and portfolio_weights:
        cs_pos_mult = crash_shield_result.get("position_multiplier", 1.0)
        for sym in list(portfolio_weights.keys()):
            dm_m = dm_multipliers.get(sym, 1.0)
            combined = cs_pos_mult * ml_multiplier * dm_m
            portfolio_weights[sym] = round(portfolio_weights[sym] * combined, 4)
        summary["combined_multipliers"] = {
            "crash_shield": cs_pos_mult,
            "ml_regime":    round(ml_multiplier, 3),
            "dual_momentum_avg": round(
                sum(dm_multipliers.values()) / max(len(dm_multipliers), 1), 3
            ),
        }
        logger.info(
            "[us] Combined position multipliers: CS=%.2f ML=%.3f DM_avg=%.2f",
            cs_pos_mult, ml_multiplier,
            sum(dm_multipliers.values()) / max(len(dm_multipliers), 1),
        )

    # ── 3. Paper / virtual trade ──────────────────────────────────────
    trade_results: list = []
    portfolio_summary: dict = {}
    prices: dict[str, float] = {
        sym: float(df["Close"].iloc[-1]) for sym, df in data.items() if not df.empty
    }
    # Build price info for Feishu: close + 1-day change %
    price_info: dict[str, dict] = {}
    for sym, df in data.items():
        if df.empty or len(df) < 2:
            continue
        close = float(df["Close"].iloc[-1])
        prev  = float(df["Close"].iloc[-2])
        price_info[sym] = {
            "close": close,
            "change_pct": round((close - prev) / prev * 100, 2) if prev else 0.0,
        }

    if market == "us" and ALPACA_API_KEY:
        try:
            from paper_trade.alpaca_trader import execute_signals, get_portfolio_summary
            trade_results = execute_signals(
                gated_signals,
                scores=composite_scores,
                weights=portfolio_weights if portfolio_weights else None,
            )
            summary["trades"] = trade_results
            portfolio_summary = get_portfolio_summary()
            logger.info("[us] Paper trades: %d orders", len(trade_results))
        except Exception as e:
            logger.error("[us] Paper trade failed: %s", e)
            summary["errors"].append(f"Paper trade failed: {e}")

    elif market in ("hk", "cn") and HF_TOKEN and HF_DATASET_REPO:
        try:
            from paper_trade.virtual_portfolio import VirtualPortfolio
            vp = VirtualPortfolio(
                market=market,
                total_capital=VIRTUAL_PORTFOLIO_CAPITAL[market],
                currency=VIRTUAL_PORTFOLIO_CURRENCY[market],
                hf_repo=HF_DATASET_REPO,
                hf_token=HF_TOKEN,
            )
            trade_results = vp.execute_signals(
                gated_signals,
                scores=composite_scores,
                prices=prices,
                weights=portfolio_weights if portfolio_weights else None,
                regime=regime_info.get("sub_state", "bull_caution"),
                max_possible=9,
            )
            portfolio_summary = vp.get_summary(prices=prices)
            summary["trades"] = trade_results
            logger.info("[%s] Virtual trades: %d orders", market, len(trade_results))
        except Exception as e:
            logger.error("[%s] Virtual portfolio failed: %s", market, e)
            summary["errors"].append(f"Virtual portfolio failed: {e}")

    # ── 4. Feishu alert ───────────────────────────────────────────────
    if FEISHU_WEBHOOK_URL and all_results:
        try:
            from alerts.feishu import send_signal_alert, build_signal_list
            vix_value = None
            if vix_df is not None and not vix_df.empty:
                vix_value = float(vix_df["Close"].iloc[-1])

            signal_list = build_signal_list(all_results, STRATEGY_LABELS)

            # Build extra_info for 3-layer status banners (US only)
            feishu_extra = None
            if market == "us":
                feishu_extra = {
                    "crash_shield": summary.get("crash_shield", {}),
                    "ml_regime":    summary.get("ml_regime", {}),
                    "dual_momentum": summary.get("dual_momentum", {}),
                }

            ok = send_signal_alert(
                FEISHU_WEBHOOK_URL, signal_list,
                vix_value=vix_value,
                trades=trade_results if trade_results else None,
                portfolio_summary=portfolio_summary if portfolio_summary else None,
                market=market,
                regime_info=regime_info,
                price_info=price_info if price_info else None,
                extra_info=feishu_extra,
            )
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

    # ── 5b. Forward-return tracking ───────────────────────────────────
    if HF_TOKEN and HF_DATASET_REPO and all_results:
        try:
            from data.forward_returns import record_today, backfill
            record_today(
                market=market,
                all_results=all_results,
                composite_scores=composite_scores,
                prices=prices,
                regime=regime_info.get("regime", "unknown"),
                date=summary["date"],
                hf_repo=HF_DATASET_REPO,
                hf_token=HF_TOKEN,
            )
            filled = backfill(
                market=market,
                price_data=data,
                hf_repo=HF_DATASET_REPO,
                hf_token=HF_TOKEN,
            )
            summary["forward_return_cells_filled"] = filled
        except Exception as e:
            logger.warning("[%s] Forward-return tracking failed: %s", market, e)
            summary["errors"].append(f"Forward returns failed: {e}")

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
