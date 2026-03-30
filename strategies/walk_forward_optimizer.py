"""
Walk-forward strategy weight optimizer.

Computes regime-specific strategy weights by evaluating each strategy's historical
signal quality (hit rate × avg forward return) per market regime.

Flow
────
1. Fetch 5 years of history for all symbols in the market
2. Label each historical day with a regime (bull_strong / bull_caution / bear)
   using benchmark MA200/MA50 rules — same logic as detect_regime()
3. Run every strategy's signal_series on the full history (no look-ahead bias
   in signal logic — each strategy only uses past data for its signals)
4. For each (strategy, regime), collect all days where signal == BUY (1)
   and look up the 20-day forward return
5. Compute quality score = hit_rate × (1 + avg_return bonus)
6. Normalize → weights in [MIN_WEIGHT, MAX_WEIGHT] with mean = 1.0
7. Save to HF Dataset: weights/{market}/strategy_weights.json

Usage
─────
Run as a standalone script (GitHub Actions weekly or on-demand):

    python -m strategies.walk_forward_optimizer --market us

Or call compute_and_save_weights(market, ...) directly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Tuneable constants ─────────────────────────────────────────────────────────
FORWARD_DAYS  = 20     # evaluate signal quality on 20-trading-day horizon
MIN_WEIGHT    = 0.3    # floor: even weak strategies keep some voice
MAX_WEIGHT    = 3.0    # ceiling: prevent single-strategy dominance
MIN_SIGNALS   = 20     # minimum BUY signals to compute meaningful stats
LOOKBACK_YEARS = 5     # years of history to analyse

# Benchmark per market (same as regime.py)
_BENCHMARK = {"us": "SPY", "hk": "02800", "cn": "510300"}


# ─── Regime labelling ──────────────────────────────────────────────────────────

def _label_regime_series(benchmark_df: pd.DataFrame) -> pd.Series:
    """
    Assign bull_strong / bull_caution / bear to every day in benchmark history.
    Mirrors the rules in strategies/regime.py for consistency.
    """
    close = benchmark_df["Close"].dropna()
    ma200 = close.rolling(200, min_periods=100).mean()
    ma50  = close.rolling(50,  min_periods=25).mean()

    regimes = pd.Series("bear", index=close.index, dtype=str)
    above_200 = close > ma200
    above_50  = close > ma50

    regimes[above_200 & ~above_50] = "bull_caution"
    regimes[above_200 &  above_50] = "bull_strong"
    return regimes


# ─── Forward return helper ─────────────────────────────────────────────────────

def _forward_returns(price_series: pd.Series, days: int = FORWARD_DAYS) -> pd.Series:
    """N-day forward return for each row (shifts close price into the future)."""
    return price_series.shift(-days) / price_series - 1


# ─── Core computation ──────────────────────────────────────────────────────────

def compute_strategy_weights(
    market: str,
    price_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute regime-specific strategy weights from historical signal quality.

    Parameters
    ----------
    market       : "us" | "hk" | "cn"
    price_data   : {symbol: OHLCV DataFrame} for all watchlist symbols
    benchmark_df : OHLCV for the benchmark (SPY / 02800 / 510300)
    vix_df       : VIX OHLCV — passed to vix_timing; auto-fetched if None (US only)

    Returns
    -------
    dict with structure:
        {
          "market": "us",
          "computed_at": "YYYY-MM-DD",
          "data_years": 5,
          "regimes": {
              "bull_strong":  {"golden_cross": 1.8, "supertrend": 2.1, ...},
              "bull_caution": {...},
              "bear":         {...}
          }
        }
    """
    from config import STRATEGY_PARAMS, LEVERAGED_ETFS
    from strategies.trend import golden_cross, supertrend, donchian_channel, ema_adx
    from strategies.momentum import macd_crossover, roc_momentum
    from strategies.mean_reversion import rsi_strategy, bollinger_squeeze
    from strategies.macro import vix_timing

    p = STRATEGY_PARAMS

    # Step 1 — Regime label per day in benchmark history
    regime_series = _label_regime_series(benchmark_df)
    logger.info("[%s] Regime coverage: %s", market,
                regime_series.value_counts().to_dict())

    # Step 2 — Collect per-signal records across all symbols
    records: list[dict] = []

    for symbol, df in price_data.items():
        if df is None or df.empty or len(df) < 252:
            logger.debug("[%s] Skipping %s — insufficient history (%d bars)",
                         market, symbol, len(df) if df is not None else 0)
            continue

        is_leveraged = symbol.upper() in LEVERAGED_ETFS
        close = df["Close"].dropna()
        fwd   = _forward_returns(close, FORWARD_DAYS)

        # Compute each strategy's signal_series on the full history
        strat_signals: dict[str, pd.Series] = {}
        try:
            strat_signals["golden_cross"]     = golden_cross(df, **p["golden_cross"])["signal_series"]
            strat_signals["supertrend"]        = supertrend(df, **p["supertrend"])["signal_series"]
            strat_signals["donchian_channel"]  = donchian_channel(df, **p["donchian"])["signal_series"]
            strat_signals["ema_adx"]           = ema_adx(df, **p["ema_adx"])["signal_series"]
            strat_signals["macd_crossover"]    = macd_crossover(df, **p["macd"])["signal_series"]
            strat_signals["roc_momentum"]      = roc_momentum(df, **p["roc"])["signal_series"]
            if not is_leveraged:
                strat_signals["rsi_strategy"]      = rsi_strategy(df, **p["rsi"])["signal_series"]
                strat_signals["bollinger_squeeze"]  = bollinger_squeeze(df, **p["bollinger"])["signal_series"]
            strat_signals["vix_timing"] = vix_timing(
                df, vix_df=vix_df, **p["vix"]
            )["signal_series"]
        except Exception as e:
            logger.warning("[%s] Strategy compute error for %s: %s", market, symbol, e)
            continue

        # Step 3 — For each BUY signal, record regime + forward return
        for strat_name, sig_series in strat_signals.items():
            aligned = pd.DataFrame({
                "signal":     sig_series.reindex(close.index, fill_value=0),
                "fwd_return": fwd,
                "regime":     regime_series.reindex(close.index, fill_value="unknown"),
            }).dropna()

            buy_rows = aligned[aligned["signal"] == 1]
            for _, row in buy_rows.iterrows():
                if row["regime"] in ("bull_strong", "bull_caution", "bear"):
                    records.append({
                        "symbol":     symbol,
                        "strategy":   strat_name,
                        "regime":     row["regime"],
                        "fwd_return": row["fwd_return"],
                    })

        logger.debug("[%s] %s → %d BUY signal records", market, symbol, len(buy_rows))

    if not records:
        logger.error("[%s] No signal records collected — cannot compute weights", market)
        return {}

    df_rec = pd.DataFrame(records)
    logger.info("[%s] Total BUY-signal records: %d", market, len(df_rec))

    # Step 4 — Compute quality score per (strategy, regime)
    all_strategies = sorted(df_rec["strategy"].unique())
    all_regimes    = ["bull_strong", "bull_caution", "bear"]
    weights_by_regime: dict[str, dict[str, float]] = {}

    for regime in all_regimes:
        regime_df  = df_rec[df_rec["regime"] == regime]
        raw_scores: dict[str, float] = {}

        for strat in all_strategies:
            strat_df = regime_df[regime_df["strategy"] == strat]
            n = len(strat_df)

            if n < MIN_SIGNALS:
                # Not enough history in this regime — neutral weight
                raw_scores[strat] = 1.0
                logger.debug("[%s] %s | %s: only %d signals → neutral weight",
                             market, regime, strat, n)
                continue

            hit_rate   = float((strat_df["fwd_return"] > 0).mean())
            avg_return = float(strat_df["fwd_return"].mean())

            # Quality: positive avg_return boosts score; negative penalises it
            if avg_return > 0:
                score = hit_rate * (1.0 + avg_return * 10)
            else:
                # Even bad strategies get a floor; score decreases with losses
                score = max(hit_rate - 0.5, 0.0) * 0.5 + 0.1

            raw_scores[strat] = max(score, 0.01)

            logger.info(
                "[%s] %-12s | %-18s  hit_rate=%5.1f%%  avg_ret=%+6.2f%%  "
                "score=%.3f  (n=%d)",
                market, regime, strat,
                hit_rate * 100, avg_return * 100, score, n,
            )

        # Normalize so mean weight = 1.0, then clamp to [MIN_WEIGHT, MAX_WEIGHT]
        mean_s = float(np.mean(list(raw_scores.values()))) if raw_scores else 1.0
        normalized = {
            strat: round(
                float(np.clip(score / (mean_s + 1e-9), MIN_WEIGHT, MAX_WEIGHT)), 3
            )
            for strat, score in raw_scores.items()
        }
        weights_by_regime[regime] = normalized
        logger.info("[%s] Weights for %-12s → %s", market, regime, normalized)

    return {
        "market":       market,
        "computed_at":  datetime.today().strftime("%Y-%m-%d"),
        "data_years":   LOOKBACK_YEARS,
        "regimes":      weights_by_regime,
    }


# ─── HF I/O ────────────────────────────────────────────────────────────────────

def _hf_path(market: str) -> str:
    return f"weights/{market}/strategy_weights.json"


def save_weights_to_hf(weights: dict, market: str, hf_repo: str, hf_token: str) -> None:
    """Persist computed weights to HF Dataset."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        content = json.dumps(weights, indent=2, ensure_ascii=False).encode()
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=_hf_path(market),
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=(
                f"Strategy weights [{market}] {weights.get('computed_at', 'unknown')}"
            ),
        )
        logger.info("[%s] Strategy weights saved to HF Dataset", market)
    except Exception as e:
        logger.error("[%s] Failed to save weights to HF: %s", market, e)


def load_weights_from_hf(
    market: str,
    hf_repo: str,
    hf_token: str,
) -> Optional[dict]:
    """Load pre-computed weights from HF Dataset. Returns None if not found."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=_hf_path(market),
            repo_type="dataset",
            token=hf_token,
        )
        with open(local) as f:
            data = json.load(f)
        logger.info("[%s] Loaded strategy weights computed on %s",
                    market, data.get("computed_at", "unknown"))
        return data
    except Exception as e:
        logger.debug("[%s] No pre-computed weights on HF: %s", market, e)
        return None


def get_regime_weights(
    market: str,
    regime: str,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict[str, float]:
    """
    Return strategy weights for a specific market + regime.

    Priority:
      1. HF Dataset pre-computed weights (if hf_repo/hf_token provided)
      2. Equal weights (1.0 for every strategy) — always works

    Parameters
    ----------
    market  : "us" | "hk" | "cn"
    regime  : "bull_strong" | "bull_caution" | "bear" | "unknown"
    """
    from config import STRATEGY_WEIGHTS as _DEFAULTS

    if hf_repo and hf_token:
        data = load_weights_from_hf(market, hf_repo, hf_token)
        if data:
            regime_key = regime if regime in ("bull_strong", "bull_caution", "bear") \
                else "bull_caution"   # fallback for "unknown"
            regime_weights = data.get("regimes", {}).get(regime_key)
            if regime_weights:
                # Start from defaults so any new strategy not yet in HF gets weight 1
                merged = dict(_DEFAULTS)
                merged.update({k: float(v) for k, v in regime_weights.items()})
                logger.info("[%s] Using optimized weights for regime '%s': %s",
                            market, regime, {k: f"{v:.2f}" for k, v in merged.items()})
                return merged

    logger.debug("[%s] Using equal weights (no optimized weights for regime '%s')",
                 market, regime)
    return dict(_DEFAULTS)


# ─── Standalone entry point ────────────────────────────────────────────────────

def compute_and_save_weights(
    market: str,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict:
    """
    Full pipeline: fetch data → compute weights → save to HF.
    Called by GitHub Actions optimize_weights.yml.
    """
    from config import MARKET_WATCHLISTS, HISTORY_YEARS
    from data.fetcher import fetch_multiple

    hf_repo  = hf_repo  or os.getenv("HF_DATASET_REPO", "")
    hf_token = hf_token or os.getenv("HF_TOKEN", "")

    watchlist = MARKET_WATCHLISTS.get(market, {})
    symbols   = [s for group in watchlist.values() for s in group]

    if not symbols:
        logger.warning("[%s] No symbols — skipping weight optimisation", market)
        return {}

    benchmark = _BENCHMARK.get(market, "SPY")
    all_symbols = list(set(symbols + [benchmark]))

    logger.info("[%s] Fetching %d symbols for weight optimisation…", market, len(all_symbols))
    price_data = fetch_multiple(
        all_symbols, years=LOOKBACK_YEARS, market=market, force_refresh=True
    )

    bench_df = price_data.pop(benchmark, None)
    if bench_df is None or bench_df.empty:
        logger.error("[%s] Could not fetch benchmark %s — aborting", market, benchmark)
        return {}

    # VIX for US vix_timing strategy
    vix_df = None
    if market == "us":
        vix_data = fetch_multiple(["^VIX"], years=LOOKBACK_YEARS, market="us", force_refresh=True)
        vix_df = vix_data.get("^VIX")

    logger.info("[%s] Computing strategy weights…", market)
    weights = compute_strategy_weights(
        market=market,
        price_data=price_data,
        benchmark_df=bench_df,
        vix_df=vix_df,
    )

    if not weights:
        logger.error("[%s] Weight computation returned empty result", market)
        return {}

    if hf_repo and hf_token:
        save_weights_to_hf(weights, market, hf_repo, hf_token)
    else:
        logger.warning("[%s] HF credentials not set — weights NOT saved", market)
        print(json.dumps(weights, indent=2))

    return weights


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Walk-forward strategy weight optimizer")
    parser.add_argument("--market", default="all",
                        help="Market to optimize: us | hk | cn | all (default: all)")
    args = parser.parse_args()

    markets = ["us", "hk", "cn"] if args.market == "all" else [args.market]
    success = 0

    for m in markets:
        try:
            result = compute_and_save_weights(m)
            if result:
                success += 1
                logger.info("[%s] Weight optimisation completed ✓", m)
            else:
                logger.error("[%s] Weight optimisation failed", m)
        except Exception as exc:
            logger.error("[%s] Unexpected error: %s", m, exc)

    sys.exit(0 if success == len(markets) else 1)
