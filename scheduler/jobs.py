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
    # Symbol-level risk rules
    LEVERAGED_LONG_ETFS,
    LEVERAGED_LONG_MIN_SCORE,
    LEVERAGED_LONG_REQUIRE_SHIELD_NONE,
    LEVERAGED_LONG_MIN_ML_MULT,
    LEVERAGED_LONG_REQUIRE_ABS_MOM,
    SYMBOL_MAX_POSITION,
    SYMBOL_QUALITY_TIER,
    TIER_MAX_POSITION,
    SYMBOL_MIN_SCORE,
    # SGOV cash-substitute
    SGOV_SYMBOL,
    SGOV_TARGET_DEPLOYED,
    SGOV_MAX_ALLOC,
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
    #
    # Direction 1: Graduated position sizing
    # score ≥ 7.5 → 100%  ≥ 6.0 → 80%  ≥ 4.5 → 50%  ≥ 2.5 → 25%  else → 0%
    # Also includes score ≥ 4.5 symbols even when composite signal is HOLD (0),
    # unless blocked by CrashShield SHIELD.
    # Note: Direction 2 (sell_threshold -3 → -5) is handled by config.py.
    #
    def _score_to_fraction(score: float) -> float:
        """Map composite score to graduated position fraction (Direction 1)."""
        if score >= 7.5:   return 1.00
        elif score >= 6.0: return 0.80
        elif score >= 4.5: return 0.50
        elif score >= 2.5: return 0.25
        return 0.0

    shield_active = crash_shield_result.get("level") == "SHIELD"

    from portfolio.optimizer import optimize_portfolio
    buy_candidates: dict[str, float] = {}
    for sym, sig in gated_signals.items():
        raw_score = composite_scores.get(sym, 0.0)
        # Per-symbol minimum score override (e.g. TQQQ≥7.0, MSTR≥7.5, XOM≥6.0)
        sym_min   = SYMBOL_MIN_SCORE.get(sym, 2.5)

        if sig == 1 and raw_score >= sym_min:
            buy_candidates[sym] = raw_score
        elif sig == 1 and raw_score < sym_min:
            logger.info("[%s] %s skipped: score %.1f below sym_min %.1f",
                        market, sym, raw_score, sym_min)
        # Graduated half-entry: score ≥ max(4.5, sym_min) and signal is HOLD
        elif sig == 0 and raw_score >= max(4.5, sym_min) and not shield_active:
            buy_candidates[sym] = raw_score

    # ── Leveraged-long strict four-condition gate (TQQQ / SPXL / SOXL) ───────
    # All four conditions must pass simultaneously; any failure blocks entry.
    if market == "us":
        _shield_ok = (
            not LEVERAGED_LONG_REQUIRE_SHIELD_NONE
            or crash_shield_result.get("level") == "NONE"
        )
        _ml_ok  = ml_multiplier >= LEVERAGED_LONG_MIN_ML_MULT
        _mom_ok = (
            not LEVERAGED_LONG_REQUIRE_ABS_MOM
            or dm_result.get("abs_momentum_ok", True)
        )
        _lev_long_all_ok = _shield_ok and _ml_ok and _mom_ok

        for sym in list(buy_candidates.keys()):
            if sym in LEVERAGED_LONG_ETFS:
                score = buy_candidates[sym]
                if not _lev_long_all_ok or score < LEVERAGED_LONG_MIN_SCORE:
                    logger.info(
                        "[us] Lev-long gate BLOCKED %s: score=%.1f "
                        "shield_ok=%s ml_ok=%s(%.2f) mom_ok=%s",
                        sym, score, _shield_ok, _ml_ok, ml_multiplier, _mom_ok,
                    )
                    del buy_candidates[sym]
                else:
                    logger.info(
                        "[us] Lev-long gate PASSED %s: score=%.1f "
                        "shield=%s ml=%.2f mom=%s",
                        sym, score,
                        crash_shield_result.get("level"), ml_multiplier, _mom_ok,
                    )

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

    # ── 2c. Combine graduated fraction + 3-layer multipliers ─────────────
    #
    # Combined = GraduatedFraction × CrashShield_mult × ML_mult × DM_per_symbol_mult
    # HK/CN only use crash_shield and ml_regime (no SPY-based dual momentum).
    #
    if market == "us" and portfolio_weights:
        cs_pos_mult = crash_shield_result.get("position_multiplier", 1.0)
        grad_fractions: dict[str, float] = {}
        for sym in list(portfolio_weights.keys()):
            raw_score = composite_scores.get(sym, 0.0)
            grad_frac = _score_to_fraction(raw_score)
            dm_m      = dm_multipliers.get(sym, 1.0)
            combined  = grad_frac * cs_pos_mult * ml_multiplier * dm_m
            portfolio_weights[sym] = round(portfolio_weights[sym] * combined, 4)
            grad_fractions[sym] = grad_frac
        summary["combined_multipliers"] = {
            "graduated_fractions": {s: round(f, 2) for s, f in grad_fractions.items()},
            "crash_shield": cs_pos_mult,
            "ml_regime":    round(ml_multiplier, 3),
            "dual_momentum_avg": round(
                sum(dm_multipliers.values()) / max(len(dm_multipliers), 1), 3
            ),
        }
        logger.info(
            "[us] Combined position multipliers: grad=%s CS=%.2f ML=%.3f DM_avg=%.2f",
            {s: f"{f:.0%}" for s, f in grad_fractions.items()},
            cs_pos_mult, ml_multiplier,
            sum(dm_multipliers.values()) / max(len(dm_multipliers), 1),
        )

    # ── 2d. Per-symbol position caps (quality tier + hard overrides) ──────
    #
    # Applied after all multipliers so the final weight never exceeds:
    #   min(TIER_MAX_POSITION[tier], SYMBOL_MAX_POSITION.get(sym, tier_cap))
    #
    # Tier caps:  A→25%  B→20%  C→10%  S→8%
    # Hard caps:  TQQQ→15%  MSTR→8%  XOM→10%  UVXY→5%  …
    # After capping, renormalize if total > 100%.
    if portfolio_weights:
        capped: dict[str, str] = {}
        for sym in list(portfolio_weights.keys()):
            tier     = SYMBOL_QUALITY_TIER.get(sym, "B")
            tier_cap = TIER_MAX_POSITION.get(tier, 0.20)
            sym_cap  = SYMBOL_MAX_POSITION.get(sym, tier_cap)
            cap      = min(tier_cap, sym_cap)
            if portfolio_weights[sym] > cap:
                capped[sym] = f"{portfolio_weights[sym]*100:.1f}%→{cap*100:.0f}%"
                portfolio_weights[sym] = cap
        if capped:
            logger.info("[%s] Position caps applied: %s", market, capped)
            total_w = sum(portfolio_weights.values())
            if total_w > 1.0:
                portfolio_weights = {
                    s: round(w / total_w, 4) for s, w in portfolio_weights.items()
                }
            summary["portfolio_weights"]  = portfolio_weights
            summary["position_caps_hit"]  = capped

    # ── 2e. SGOV cash-substitute — park idle capital in T-bills ──────────
    #
    # After all equity positions are finalised, if the total deployed weight
    # is below SGOV_TARGET_DEPLOYED, fill the gap with SGOV so idle cash
    # earns T-bill yield (~4% annualised) instead of sitting at 0%.
    #
    # SGOV bypasses the normal composite-score pipeline intentionally —
    # T-bill ETFs score near-zero on trend/momentum strategies and would
    # never reach the BUY threshold on their own.
    #
    # Also acts as the natural "safe harbour" when CrashShield fires:
    #   SHIELD active → equity ≈ 0 → SGOV ≈ SGOV_TARGET_DEPLOYED (35%)
    if market == "us":
        equity_total = sum(
            w for sym, w in portfolio_weights.items() if sym != SGOV_SYMBOL
        )
        sgov_alloc = round(
            max(0.0, min(SGOV_MAX_ALLOC, SGOV_TARGET_DEPLOYED - equity_total)), 4
        )
        if sgov_alloc >= 0.01:   # only if meaningful (≥1%)
            portfolio_weights[SGOV_SYMBOL] = sgov_alloc
            gated_signals[SGOV_SYMBOL]     = 1          # synthetic BUY for executor
            composite_scores[SGOV_SYMBOL]  = 5.0        # neutral-positive placeholder
            summary["sgov_alloc"] = sgov_alloc
            logger.info(
                "[us] SGOV cash-sub: equity_total=%.1f%% gap=%.1f%% → SGOV=%.1f%%",
                equity_total * 100,
                (SGOV_TARGET_DEPLOYED - equity_total) * 100,
                sgov_alloc * 100,
            )
        else:
            logger.info(
                "[us] SGOV cash-sub: equity_total=%.1f%% ≥ target %.0f%% — no SGOV needed",
                equity_total * 100, SGOV_TARGET_DEPLOYED * 100,
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

    if market == "us":
        try:
            from paper_trade.rh_tracker import (
                compare_signals, get_portfolio_summary, update_prices,
            )
            # Refresh prices in the local positions file
            update_prices(prices)
            # Compare strategy signals against actual Robinhood positions
            trade_results = compare_signals(
                signals=gated_signals,
                composite_scores=composite_scores,
                prices=prices,
                portfolio_weights=portfolio_weights if portfolio_weights else None,
            )
            portfolio_summary = get_portfolio_summary()
            summary["trades"] = trade_results
            exits   = [t for t in trade_results if t["action"] == "EXIT"]
            entries = [t for t in trade_results if t["action"] == "ENTER"]
            trims   = [t for t in trade_results if t["action"] == "TRIM"]
            logger.info(
                "[us] RH signal check: %d EXIT  %d ENTER  %d TRIM",
                len(exits), len(entries), len(trims),
            )
            for t in exits:
                logger.warning("[us] %s %s — %s", t["urgency"], t["symbol"], t["reason"])
            for t in entries:
                logger.info("[us] %s %s — %s notional=$%s",
                            t["urgency"], t["symbol"], t["reason"],
                            t.get("notional", "?"))
        except Exception as e:
            logger.error("[us] RH tracker failed: %s", e)
            summary["errors"].append(f"RH tracker failed: {e}")

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

    # ── 3b. Options suggestions (US only, advisory) ───────────────────
    options_report: dict = {}
    if market == "us":
        try:
            from paper_trade.options_helper import generate_options_suggestions
            # Build trailing price history from loaded data (last 21 closes)
            price_hist: dict[str, list[float]] = {}
            for sym, df in data.items():
                if "Close" in df.columns and len(df) >= 5:
                    price_hist[sym] = df["Close"].dropna().tail(21).tolist()
            alpaca_positions: list[dict] = []
            if portfolio_summary:
                alpaca_positions = portfolio_summary.get("positions", [])
            options_report = generate_options_suggestions(
                market=market,
                positions=alpaca_positions,
                prices=prices,
                composite_scores=composite_scores,
                signals=gated_signals,
                watch_symbols=symbols,
                price_history=price_hist,
            )
            summary["options_suggestions"] = {
                "n_calls": len(options_report.get("covered_calls", [])),
                "n_puts":  len(options_report.get("cash_puts", [])),
            }
        except Exception as e:
            logger.warning("[us] Options suggestions failed: %s", e)

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
