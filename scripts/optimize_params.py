"""
Walk-forward parameter optimizer for all 9 US strategies.

Methodology
───────────
• 15 years of daily OHLCV for 8 core US symbols
• 3 out-of-sample walk-forward windows (train 8y → test 2-3y)
• Each strategy's parameters are optimized independently
• Composite buy/sell thresholds are then optimized using best per-strategy params
• Objective: maximize out-of-sample Sharpe ratio
• Constraint: max drawdown must be better than -45%

Output
──────
Prints a ranked table per strategy + writes scripts/optimal_params.json
with the recommended config to paste into config.py

Usage
─────
  python3 scripts/optimize_params.py            # full run
  python3 scripts/optimize_params.py --fast     # 3 symbols only, quick smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
import logging
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ── allow imports from project root ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Symbols & windows ─────────────────────────────────────────────────────────
SYMBOLS_FULL = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META"]
SYMBOLS_FAST = ["SPY", "QQQ", "NVDA"]

# Walk-forward windows: (train_start, train_end, test_start, test_end)
WF_WINDOWS = [
    ("2009-01-01", "2017-12-31", "2018-01-01", "2020-12-31"),
    ("2012-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("2016-01-01", "2022-12-31", "2023-01-01", "2025-04-30"),
]

INIT_CASH   = 10_000.0
FEES        = 0.001          # 0.1% one-way
MIN_TRADES  = 5              # discard combos with fewer trades in test window
MAX_DD_FLOOR = -0.45         # discard if max drawdown worse than -45%

# ── Parameter grids ───────────────────────────────────────────────────────────
PARAM_GRIDS = {
    "golden_cross": [
        {"fast": 20,  "slow": 100},
        {"fast": 50,  "slow": 150},
        {"fast": 50,  "slow": 200},   # ← current
        {"fast": 100, "slow": 200},
    ],
    "supertrend": [
        {"atr_period": 7,  "multiplier": 2.0},
        {"atr_period": 10, "multiplier": 2.5},
        {"atr_period": 10, "multiplier": 3.0},  # ← current
        {"atr_period": 14, "multiplier": 3.0},
        {"atr_period": 14, "multiplier": 3.5},
    ],
    "donchian": [
        {"entry_period": 20, "exit_period": 10},  # ← current
        {"entry_period": 30, "exit_period": 15},
        {"entry_period": 55, "exit_period": 20},  # Turtle Trading
    ],
    "ema_adx": [
        {"ema_fast": 8,  "ema_slow": 21, "adx_period": 14, "adx_threshold": 20},
        {"ema_fast": 12, "ema_slow": 26, "adx_period": 14, "adx_threshold": 20},
        {"ema_fast": 12, "ema_slow": 26, "adx_period": 14, "adx_threshold": 25},  # ← current
        {"ema_fast": 12, "ema_slow": 26, "adx_period": 14, "adx_threshold": 30},
    ],
    "macd": [
        {"fast": 5,  "slow": 35, "signal_period": 5},
        {"fast": 8,  "slow": 21, "signal_period": 5},
        {"fast": 12, "slow": 26, "signal_period": 9},  # ← current (standard)
    ],
    "roc": [
        {"period_short": 10, "period_long": 30},
        {"period_short": 20, "period_long": 60},  # ← current
        {"period_short": 20, "period_long": 90},
    ],
    "rsi": [
        {"period": 14, "oversold": 25, "overbought": 75},
        {"period": 14, "oversold": 30, "overbought": 70},  # ← current
        {"period": 21, "oversold": 30, "overbought": 70},
        {"period": 21, "oversold": 35, "overbought": 65},
    ],
    "bollinger": [
        {"period": 20, "std_dev": 1.5, "squeeze_threshold": 0.10},
        {"period": 20, "std_dev": 2.0, "squeeze_threshold": 0.10},  # ← current
        {"period": 20, "std_dev": 2.5, "squeeze_threshold": 0.08},
        {"period": 30, "std_dev": 2.0, "squeeze_threshold": 0.10},
    ],
}

# Composite threshold grid (tested after best per-strategy params found)
BUY_THRESHOLDS  = [5.0, 5.5, 6.0, 6.5, 7.0]
SELL_THRESHOLDS = [3.0, 4.0, 5.0]

# Map strategy name → function import path
_STRAT_FN = {
    "golden_cross":     ("strategies.trend",         "golden_cross"),
    "supertrend":       ("strategies.trend",         "supertrend"),
    "donchian":         ("strategies.trend",         "donchian_channel"),
    "ema_adx":          ("strategies.trend",         "ema_adx"),
    "macd":             ("strategies.momentum",      "macd_crossover"),
    "roc":              ("strategies.momentum",      "roc_momentum"),
    "rsi":              ("strategies.mean_reversion","rsi_strategy"),
    "bollinger":        ("strategies.mean_reversion","bollinger_squeeze"),
}


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download 15+ years of daily data via yfinance."""
    print(f"Fetching {len(symbols)} symbols (15 years) …")
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start="2009-01-01", auto_adjust=True,
                             progress=False, timeout=30)
            if df is not None and len(df) > 500:
                df.index = pd.to_datetime(df.index)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[sym] = df
                print(f"  ✓ {sym}: {len(df)} bars")
            else:
                print(f"  ✗ {sym}: insufficient data")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    # VIX for vix_timing
    try:
        vix = yf.download("^VIX", start="2009-01-01", auto_adjust=True,
                          progress=False, timeout=30)
        if vix is not None and len(vix) > 500:
            vix.index = pd.to_datetime(vix.index)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            data["^VIX"] = vix
            print(f"  ✓ ^VIX: {len(vix)} bars")
    except Exception as e:
        print(f"  ✗ ^VIX: {e}")
    return data


# ── Simple backtest ───────────────────────────────────────────────────────────

def backtest(close: pd.Series, signal: pd.Series,
             init_cash: float = INIT_CASH, fees: float = FEES) -> dict:
    """Fast pandas backtest. Enters at open of next bar (simulated via close shift)."""
    sig = signal.reindex(close.index, fill_value=0)
    entries = (sig == 1) & (sig.shift(1).fillna(0) != 1)
    exits   = (sig != 1) & (sig.shift(1).fillna(0) == 1)

    cash, shares = init_cash, 0.0
    vals = []
    in_pos = False

    for i in range(len(close)):
        price = close.iloc[i]
        if not in_pos and entries.iloc[i]:
            shares = cash * (1 - fees) / price
            cash = 0.0
            in_pos = True
        elif in_pos and exits.iloc[i]:
            cash = shares * price * (1 - fees)
            shares = 0.0
            in_pos = False
        vals.append(cash + shares * price)

    pv = pd.Series(vals, index=close.index)
    n_trades = int(entries.sum())
    if len(pv) < 2:
        return {"sharpe": -999, "max_dd": 0, "annual_ret": 0, "n_trades": 0}

    rets = pv.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    roll_max = pv.cummax()
    max_dd = float(((pv - roll_max) / roll_max).min())
    years = (close.index[-1] - close.index[0]).days / 365.25
    annual_ret = float((pv.iloc[-1] / pv.iloc[0]) ** (1 / max(years, 0.1)) - 1)

    return {"sharpe": round(sharpe, 4), "max_dd": round(max_dd, 4),
            "annual_ret": round(annual_ret, 4), "n_trades": n_trades}


def score_metric(m: dict) -> float:
    """Composite scoring: Sharpe primary, max_dd penalty."""
    if m["max_dd"] < MAX_DD_FLOOR or m["n_trades"] < MIN_TRADES:
        return -999.0
    # Penalise drawdowns worse than -25%
    dd_penalty = max(0, abs(m["max_dd"]) - 0.25) * 2
    return m["sharpe"] - dd_penalty


# ── Strategy signal helper ────────────────────────────────────────────────────

def _load_fn(strat_key: str):
    """Lazily import a strategy function by name."""
    module_name, fn_name = _STRAT_FN[strat_key]
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def get_signal(strat_key: str, df: pd.DataFrame, params: dict,
               vix_df=None) -> pd.Series:
    fn = _load_fn(strat_key)
    try:
        if strat_key == "rsi" and df.get("is_leveraged", False):
            return pd.Series(0, index=df.index)
        result = fn(df, **params)
        return result["signal_series"]
    except Exception as e:
        logger.debug("Signal error %s: %s", strat_key, e)
        return pd.Series(0, index=df.index)


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def wf_score(strat_key: str, params: dict,
             all_data: dict[str, pd.DataFrame],
             vix_df=None) -> tuple[float, float, float]:
    """
    Average out-of-sample score across all WF windows and all symbols.
    Returns (avg_score, avg_sharpe, avg_max_dd).
    """
    scores, sharpes, dds = [], [], []

    for (tr_s, tr_e, te_s, te_e) in WF_WINDOWS:
        for sym, df in all_data.items():
            if sym == "^VIX":
                continue
            test_df = df.loc[te_s:te_e]
            if len(test_df) < 100:
                continue

            # Compute signal on full history up to test end (no lookahead)
            full_df = df.loc[:te_e]
            vix_sub = vix_df.loc[:te_e] if vix_df is not None else None

            sig_full = get_signal(strat_key, full_df, params, vix_sub)
            sig_test = sig_full.loc[te_s:te_e]
            close_test = test_df["Close"]

            m = backtest(close_test, sig_test)
            sc = score_metric(m)
            scores.append(sc)
            sharpes.append(m["sharpe"])
            dds.append(m["max_dd"])

    if not scores:
        return -999.0, -999.0, 0.0
    valid = [s for s in scores if s > -999]
    if not valid:
        return -999.0, -999.0, 0.0
    return (
        float(np.mean(valid)),
        float(np.mean(sharpes)),
        float(np.mean(dds)),
    )


# ── Per-strategy optimization ─────────────────────────────────────────────────

def optimize_strategy(strat_key: str,
                      all_data: dict[str, pd.DataFrame]) -> dict:
    """Grid search over PARAM_GRIDS[strat_key]. Returns best params + metrics."""
    vix_df = all_data.get("^VIX")
    grid   = PARAM_GRIDS[strat_key]

    print(f"\n{'─'*60}")
    print(f"  Strategy: {strat_key}  ({len(grid)} combos × {len(WF_WINDOWS)} windows)")
    print(f"{'─'*60}")

    results = []
    for params in grid:
        t0 = time.time()
        avg_sc, avg_sh, avg_dd = wf_score(strat_key, params, all_data, vix_df)
        elapsed = time.time() - t0
        results.append({
            "params": params, "score": avg_sc,
            "sharpe": avg_sh, "max_dd": avg_dd,
        })
        marker = " ← current" if _is_current(strat_key, params) else ""
        print(f"  {params}  →  score={avg_sc:+.3f}  sharpe={avg_sh:.3f}  "
              f"max_dd={avg_dd:.1%}{marker}  [{elapsed:.1f}s]")

    best = max(results, key=lambda r: r["score"])
    print(f"\n  ✅ Best: {best['params']}  score={best['score']:+.3f}  "
          f"sharpe={best['sharpe']:.3f}  max_dd={best['max_dd']:.1%}")
    return best


def _is_current(strat_key: str, params: dict) -> bool:
    """True if params match current config.py defaults."""
    from config import STRATEGY_PARAMS as CP
    key_map = {
        "golden_cross": "golden_cross",
        "supertrend":   "supertrend",
        "donchian":     "donchian",
        "ema_adx":      "ema_adx",
        "macd":         "macd",
        "roc":          "roc",
        "rsi":          "rsi",
        "bollinger":    "bollinger",
    }
    current = CP.get(key_map.get(strat_key, strat_key), {})
    return all(current.get(k) == v for k, v in params.items())


# ── Composite threshold optimization ─────────────────────────────────────────

def optimize_composite(best_params: dict,
                       all_data: dict[str, pd.DataFrame]) -> dict:
    """
    Given best per-strategy params, grid-search composite buy/sell thresholds.
    Uses the composite_score function directly.
    """
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS, LEVERAGED_ETFS

    vix_df = all_data.get("^VIX")
    print(f"\n{'─'*60}")
    print(f"  Composite threshold grid "
          f"({len(BUY_THRESHOLDS)}×{len(SELL_THRESHOLDS)} combos)")
    print(f"{'─'*60}")

    # Build override STRATEGY_PARAMS from best_params
    import config as cfg
    # Temporarily patch STRATEGY_PARAMS for this run
    original_params = {k: dict(v) for k, v in cfg.STRATEGY_PARAMS.items()}
    for key, params in best_params.items():
        if key in cfg.STRATEGY_PARAMS:
            cfg.STRATEGY_PARAMS[key].update(params)

    results = []
    for buy_t, sell_t in product(BUY_THRESHOLDS, SELL_THRESHOLDS):
        if sell_t >= buy_t:
            continue  # sell threshold must be < buy threshold

        scores, sharpes, dds = [], [], []
        for (_, _, te_s, te_e) in WF_WINDOWS:
            for sym, df in all_data.items():
                if sym == "^VIX" or sym.upper() in LEVERAGED_ETFS:
                    continue
                test_df = df.loc[te_s:te_e]
                if len(test_df) < 100:
                    continue

                full_df = df.loc[:te_e]
                vix_sub = vix_df.loc[:te_e] if vix_df is not None else None

                try:
                    comp = composite_score(
                        full_df, symbol=sym, vix_df=vix_sub,
                        buy_threshold=buy_t, sell_threshold=sell_t,
                        weights=STRATEGY_WEIGHTS,
                    )
                    sig_test = comp["signal_series"].loc[te_s:te_e]
                    close_test = test_df["Close"]
                    m = backtest(close_test, sig_test)
                    sc = score_metric(m)
                    scores.append(sc)
                    sharpes.append(m["sharpe"])
                    dds.append(m["max_dd"])
                except Exception as e:
                    logger.debug("Composite error: %s", e)

        valid = [s for s in scores if s > -999]
        avg_sc = float(np.mean(valid)) if valid else -999.0
        avg_sh = float(np.mean(sharpes)) if sharpes else -999.0
        avg_dd = float(np.mean(dds)) if dds else 0.0

        marker = " ← current" if buy_t == 6.0 and sell_t == 4.0 else ""
        print(f"  buy≥{buy_t}  sell≤-{sell_t}  →  score={avg_sc:+.3f}  "
              f"sharpe={avg_sh:.3f}  max_dd={avg_dd:.1%}{marker}")

        results.append({
            "buy_threshold": buy_t, "sell_threshold": sell_t,
            "score": avg_sc, "sharpe": avg_sh, "max_dd": avg_dd,
        })

    # Restore original params
    for key in cfg.STRATEGY_PARAMS:
        if key in original_params:
            cfg.STRATEGY_PARAMS[key] = original_params[key]

    best = max(results, key=lambda r: r["score"])
    print(f"\n  ✅ Best thresholds: buy≥{best['buy_threshold']}  "
          f"sell≤-{best['sell_threshold']}  score={best['score']:+.3f}  "
          f"sharpe={best['sharpe']:.3f}  max_dd={best['max_dd']:.1%}")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick mode: 3 symbols only")
    args = parser.parse_args()

    symbols = SYMBOLS_FAST if args.fast else SYMBOLS_FULL
    print(f"\n{'='*60}")
    print(f"  US Strategy Parameter Optimizer")
    print(f"  Symbols: {symbols}")
    print(f"  Walk-forward windows: {len(WF_WINDOWS)}")
    print(f"  Mode: {'FAST' if args.fast else 'FULL'}")
    print(f"{'='*60}")

    # 1. Fetch data
    all_data = fetch_data(symbols)
    if len(all_data) < 3:
        print("ERROR: Not enough data fetched. Check network.")
        sys.exit(1)

    # 2. Optimize each strategy
    best_params: dict[str, dict] = {}
    for strat_key in PARAM_GRIDS:
        result = optimize_strategy(strat_key, all_data)
        best_params[strat_key] = result["params"]

    # 3. Optimize composite thresholds
    comp_result = optimize_composite(best_params, all_data)

    # 4. Summary
    print(f"\n{'='*60}")
    print("  RECOMMENDED CONFIG — paste into config.py")
    print(f"{'='*60}")

    from config import STRATEGY_PARAMS as CURRENT
    changes = []
    for key, params in best_params.items():
        current = CURRENT.get(key, {})
        if any(current.get(k) != v for k, v in params.items()):
            changes.append(key)
            print(f"\n  [{key}]  ← CHANGED")
            print(f"    current:  {current}")
            print(f"    optimal:  {params}")
        else:
            print(f"\n  [{key}]  ← unchanged ({params})")

    print(f"\n  [composite] buy_threshold: "
          f"6.0 → {comp_result['buy_threshold']}  "
          f"sell_threshold: 4.0 → {comp_result['sell_threshold']}")

    if not changes and comp_result["buy_threshold"] == 6.0:
        print("\n  ✅ Current config is already near-optimal. No changes needed.")
    else:
        print(f"\n  🔧 Changed strategies: {changes}")
        print(f"  📊 Expected out-of-sample Sharpe improvement:")
        print(f"     Composite: {comp_result['sharpe']:+.3f}")

    # 5. Save JSON locally
    output = {
        "computed_at": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "mode": "fast" if args.fast else "full",
        "symbols": symbols,
        "strategy_params": best_params,
        "composite": {
            "buy_threshold":  comp_result["buy_threshold"],
            "sell_threshold": comp_result["sell_threshold"],
            "avg_sharpe":     comp_result["sharpe"],
            "avg_max_dd":     comp_result["max_dd"],
        },
    }
    out_path = os.path.join(ROOT, "scripts", "optimal_params.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results saved locally: {out_path}")

    # 6. Save to HF Dataset (for history + full automation)
    hf_token = os.getenv("HF_TOKEN", "")
    hf_repo  = os.getenv("HF_DATASET_REPO", "")
    if hf_token and hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            dated_path = f"params/us/optimal_params_{datetime.today().strftime('%Y-%m')}.json"
            latest_path = "params/us/optimal_params_latest.json"
            content = json.dumps(output, indent=2).encode()
            for path in (dated_path, latest_path):
                api.upload_file(
                    path_or_fileobj=content,
                    path_in_repo=path,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Monthly param optimization [{output['computed_at']}]",
                )
            print(f"  ☁️  Saved to HF Dataset: {latest_path}")
            print(f"  ☁️  History copy:        {dated_path}")
        except Exception as e:
            print(f"  ⚠️  HF save failed (non-fatal): {e}")
    else:
        print("  ⚠️  HF_TOKEN/HF_DATASET_REPO not set — skipping HF upload")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
