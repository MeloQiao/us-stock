"""
15-Year Backtest: 3-Layer Enhanced Strategy — Old vs New Approach

Compares 5 equity curves on SPY (2010 – present):
  0. Buy & Hold SPY               (benchmark)
  1. Old Composite (binary in/out, sell at score ≤ -3)
  2. Old Full 3-Layer (binary + CrashShield + ML + DualMomentum)
  3. New Composite v2 (graduated sizing + sell at score ≤ -5)
  4. New Full 3-Layer v2 (graduated + sell≤-5 + CS + ML + DM)

Direction 1: Graduated position sizing
  score ≥ 7.5 → 100% | ≥ 6.0 → 80% | ≥ 4.5 → 50% | ≥ 2.5 → 25%
  Neutral zone (-5, 2.5): hold current position (no exit noise)

Direction 2: Asymmetric exit threshold
  Enter: score ≥ 6.0 (unchanged)
  Exit:  score ≤ -5.0 (was -3.0) — require stronger bearish signal to exit

Key properties:
  - SPY-centric (single instrument, market timing focus)
  - No lookahead bias: all indicators use past data only
  - ML uses walk-forward OOS predictions (4 non-overlapping windows)
  - Transaction cost: 10 bps per round-trip (realistic for ETFs)
  - Daily rebalancing

Usage
─────
  python3 scripts/backtest_3layer.py
  python3 scripts/backtest_3layer.py --skip-ml   # skip slow XGBoost training
  python3 scripts/backtest_3layer.py --years 10  # shorter history
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(ticker: str, years: int = 15) -> pd.DataFrame:
    import yfinance as yf
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        progress=False, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Vectorized signal computation
# ══════════════════════════════════════════════════════════════════════════════

def get_composite_score_series(spy_df: pd.DataFrame, vix_df: pd.DataFrame | None) -> pd.Series:
    """
    Compute raw weighted composite score for SPY (not thresholded).
    Returns pd.Series[float] — same scale as buy_threshold (6.0) and sell_threshold.
    """
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS, COMPOSITE_BUY_THRESHOLD

    result = composite_score(
        spy_df,
        symbol="SPY",
        vix_df=vix_df,
        buy_threshold=COMPOSITE_BUY_THRESHOLD.get("us", 6.0),
        sell_threshold=3.0,   # only used for signal_series; we use raw score below
        weights=STRATEGY_WEIGHTS,
    )
    return result["indicators"]["Composite_Score"].reindex(spy_df.index, fill_value=0.0)


def build_binary_position(
    score_series: pd.Series,
    buy_threshold: float = 6.0,
    sell_threshold: float = -3.0,
) -> pd.Series:
    """
    Original binary in/out position (v1 approach).

    Enter (1.0) when score >= buy_threshold.
    Exit  (0.0) when score <= sell_threshold.
    Hold current position otherwise (neutral zone).
    """
    pos  = pd.Series(0.0, index=score_series.index)
    curr = 0.0
    for dt, score in score_series.items():
        if score >= buy_threshold:
            curr = 1.0
        elif score <= sell_threshold:
            curr = 0.0
        # else: hold current
        pos[dt] = curr
    return pos


def build_graduated_position(
    score_series: pd.Series,
    sell_threshold: float = -5.0,
) -> pd.Series:
    """
    Direction 1 + 2: Graduated position sizing with asymmetric exit.

    Sizing map (Direction 1):
      score ≥ 7.5 → 100%
      score ≥ 6.0 →  80%
      score ≥ 4.5 →  50%
      score ≥ 2.5 →  25%
      score ≤ sell_threshold → 0%  (Direction 2: default -5.0, was -3.0)
      else (neutral zone) → hold current position unchanged

    The "hold" rule means we do NOT drop out when the score dips temporarily
    into the neutral zone — only a strong bearish signal (≤ -5) fully exits.
    """
    pos  = pd.Series(0.0, index=score_series.index)
    curr = 0.0
    for dt, score in score_series.items():
        if score >= 7.5:
            curr = 1.00
        elif score >= 6.0:
            curr = 0.80
        elif score >= 4.5:
            curr = 0.50
        elif score >= 2.5:
            curr = 0.25
        elif score <= sell_threshold:
            curr = 0.00
        # else (-5 < score < 2.5): hold current position unchanged
        pos[dt] = curr
    return pos


def compute_crash_shield(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    hyg_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Vectorized Crash Shield for full history.
    Returns DataFrame with columns: score, multiplier, level
    """
    spy = spy_df["Close"].dropna()
    idx = spy.index

    # Signal 1: VIX panic — VIX > 25 and 5d rise > 20%
    if vix_df is not None and not vix_df.empty:
        vix = vix_df["Close"].dropna().reindex(idx, method="ffill").fillna(20)
        vix_5d_chg = vix.pct_change(5).fillna(0)
        sig1 = ((vix > 25) & (vix_5d_chg > 0.20)).astype(int)
    else:
        sig1 = pd.Series(0, index=idx)

    # Signal 2: Trend break — price < MA50 and MA50 declining
    ma50 = spy.rolling(50, min_periods=25).mean()
    ma50_10d = ma50.shift(10)
    sig2 = ((spy < ma50) & (ma50 < ma50_10d)).astype(int)

    # Signal 3: Sharp decline — 20d return < -8%
    sig3 = (spy.pct_change(20) < -0.08).astype(int)

    # Signal 4: Credit spread — HYG 20d return < -3% (or SPY 60d < -15% fallback)
    if hyg_df is not None and not hyg_df.empty:
        hyg = hyg_df["Close"].dropna().reindex(idx, method="ffill")
        sig4 = (hyg.pct_change(20) < -0.03).astype(int)
    else:
        sig4 = (spy.pct_change(60) < -0.15).astype(int)

    score = (sig1 + sig2 + sig3 + sig4).fillna(0).astype(int)

    mult  = pd.Series(1.0, index=idx)
    mult[score == 2] = 0.5
    mult[score >= 3] = 0.0   # SHIELD: block new buys entirely

    level = pd.Series("NONE", index=idx)
    level[score == 2] = "CAUTION"
    level[score >= 3] = "SHIELD"

    return pd.DataFrame({"score": score, "multiplier": mult, "level": level})


def compute_ml_probs(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    hyg_df: pd.DataFrame | None,
    target_mode: str = "v3",
) -> pd.Series:
    """
    Walk-forward OOS ML probability predictions (no lookahead bias).
    Uses 4 consecutive OOS windows covering 2015–present.
    Dates before first OOS window default to 0.5 (neutral).

    target_mode : "v3" (Direction 3, correction-aware label, default)
                  "v1" (legacy — pure 20d forward return ≥ 0)
    """
    from strategies.ml_regime import build_features, FEATURE_COLS, _get_model, _fill_missing
    from sklearn.preprocessing import StandardScaler

    df = build_features(spy_df, vix_df, hyg_df, include_target=True, target_mode=target_mode)
    df = df.dropna(subset=["y"])

    prob_series = pd.Series(0.5, index=df.index)

    # 4 expanding-window OOS periods
    windows = [
        ("2009-01-01", "2014-12-31", "2015-01-01", "2017-12-31"),
        ("2009-01-01", "2017-12-31", "2018-01-01", "2020-06-30"),
        ("2009-01-01", "2020-06-30", "2020-07-01", "2022-12-31"),
        ("2009-01-01", "2022-12-31", "2023-01-01", "2099-12-31"),
    ]

    for train_start, train_end, test_start, test_end in windows:
        try:
            train = df.loc[train_start:train_end].dropna()
            test  = df.loc[test_start:test_end].dropna()
            if len(train) < 300 or len(test) < 20:
                print(f"   ⚠️  Skipping window {test_start[:7]}–{test_end[:7]} (not enough data)")
                continue

            X_tr = _fill_missing(train[FEATURE_COLS])
            y_tr = train["y"].values
            X_te = _fill_missing(test[FEATURE_COLS])

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            mdl = _get_model()
            mdl.fit(X_tr_s, y_tr)
            probs = mdl.predict_proba(X_te_s)[:, 1]

            prob_series.loc[test.index] = probs
            n_pos = (probs > 0.55).sum()
            n_neg = (probs < 0.45).sum()
            print(f"   ✅ OOS {test_start[:7]}–{test_end[:7]}: "
                  f"n={len(test):,}  bull={n_pos}  bear={n_neg}  "
                  f"mean_prob={probs.mean():.3f}")
        except Exception as e:
            print(f"   ❌ Window failed: {e}")

    return prob_series


def compute_dual_momentum(spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized Dual Momentum signals.
    Returns DataFrame: abs_ok (bool), crash_protect (bool), scale (float)
    """
    spy = spy_df["Close"].dropna()

    # Absolute momentum: SPY 12M return > 0
    abs_ok = spy.pct_change(252) > 0

    # Vol ratio crash protection: 21-day vol > 2x 126-day vol
    log_ret = np.log(spy / spy.shift(1)).dropna()
    vol_21  = log_ret.rolling(21).std()  * np.sqrt(252)
    vol_126 = log_ret.rolling(126).std() * np.sqrt(252)
    crash   = (vol_21 / (vol_126 + 1e-9)) >= 2.0

    # Position scale
    scale = pd.Series(1.0, index=spy.index)
    scale[~abs_ok.reindex(spy.index, fill_value=False)] = 0.0
    scale[(crash.reindex(spy.index, fill_value=False)) & abs_ok.reindex(spy.index, fill_value=True)] = 0.5

    return pd.DataFrame({
        "abs_ok":        abs_ok.reindex(spy.index, fill_value=False),
        "crash_protect": crash.reindex(spy.index, fill_value=False),
        "scale":         scale,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_equity(
    spy_returns: pd.Series,
    position_fraction: pd.Series,
    cost_bps: float = 10.0,
    label: str = "Strategy",
) -> pd.Series:
    """
    Simulate daily portfolio NAV.

    position_fraction : fraction of capital in SPY each day (0.0 – 1.0)
                        based on previous day's close signal
    cost_bps          : one-way transaction cost in basis points
    """
    pos_prev = 0.0
    nav = [1.0]

    pf = position_fraction.reindex(spy_returns.index).fillna(0.0)

    for i in range(1, len(spy_returns)):
        target = float(pf.iloc[i - 1])          # signal known at close of t-1
        trade  = abs(target - pos_prev)
        tc     = trade * cost_bps / 10_000.0
        ret    = target * float(spy_returns.iloc[i]) - tc
        nav.append(nav[-1] * (1.0 + ret))
        pos_prev = target

    return pd.Series(nav, index=spy_returns.index, name=label)


# ══════════════════════════════════════════════════════════════════════════════
# Performance statistics
# ══════════════════════════════════════════════════════════════════════════════

def performance_stats(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Compute key performance metrics."""
    ret = equity.pct_change().dropna()
    n_years = len(ret) / 252

    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)

    sharpe = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252))

    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    max_dd = float(dd.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    stats = {
        "Total Return":  f"{total_ret*100:+.1f}%",
        "CAGR":          f"{cagr*100:.1f}%",
        "Sharpe":        f"{sharpe:.2f}",
        "Max Drawdown":  f"{max_dd*100:.1f}%",
        "Calmar":        f"{calmar:.2f}",
    }

    if benchmark is not None:
        bret = benchmark.pct_change().dropna()
        # Information ratio
        excess = ret.reindex(bret.index) - bret
        ir = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))
        stats["Info Ratio"] = f"{ir:.2f}"

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",    type=int, default=15, help="Years of history")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML layer (faster)")
    parser.add_argument("--cost",    type=float, default=10.0, help="One-way cost in bps")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart output")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  3-Layer Strategy Backtest — Direction 1 / 2 / 3  ({args.years}yr, {args.cost}bps/trade)")
    print(f"  D1: Graduated sizing  |  D2: Sell≤-5  |  D3: Correction-aware ML")
    print(f"{'='*72}")

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    print("\n📥 Downloading market data...")
    spy_df = _fetch("SPY",  years=args.years)
    print(f"   SPY : {len(spy_df):,} days  ({spy_df.index[0].date()} – {spy_df.index[-1].date()})")

    vix_df, hyg_df = None, None
    try:
        vix_df = _fetch("^VIX", years=args.years)
        print(f"   VIX : {len(vix_df):,} days")
    except Exception as e:
        print(f"   VIX : unavailable ({e})")
    try:
        hyg_df = _fetch("HYG",  years=args.years)
        print(f"   HYG : {len(hyg_df):,} days")
    except Exception as e:
        print(f"   HYG : unavailable ({e})")

    spy_ret    = spy_df["Close"].pct_change()
    common_idx = spy_df.index

    # ── 2. Composite score (raw) ────────────────────────────────────────────
    print("\n📊 Computing composite strategy scores (9 strategies)...")
    raw_scores = get_composite_score_series(spy_df, vix_df)

    # v1: original binary in/out (enter ≥6, exit ≤-3)
    pos_v1 = build_binary_position(raw_scores, buy_threshold=6.0, sell_threshold=-3.0)
    # v2: graduated sizing + asymmetric exit (enter by level, exit ≤-5)
    pos_v2 = build_graduated_position(raw_scores, sell_threshold=-5.0)

    for label, pos in [("v1 Binary (sell≤-3)", pos_v1), ("v2 Graduated (sell≤-5)", pos_v2)]:
        days_in   = int((pos > 0).sum())
        pct_in    = days_in / len(pos) * 100
        avg_frac  = float(pos[pos > 0].mean()) * 100 if (pos > 0).any() else 0.0
        print(f"   {label}: {days_in:,} days invested ({pct_in:.0f}%)  "
              f"avg size when in: {avg_frac:.0f}%")

    # Score distribution for v2
    score_dist = {
        "≥7.5 (100%)": (raw_scores >= 7.5).sum(),
        "6–7.5 (80%)":  ((raw_scores >= 6.0) & (raw_scores < 7.5)).sum(),
        "4.5–6 (50%)":  ((raw_scores >= 4.5) & (raw_scores < 6.0)).sum(),
        "2.5–4.5 (25%)":((raw_scores >= 2.5) & (raw_scores < 4.5)).sum(),
        "neutral":      ((raw_scores > -5.0) & (raw_scores < 2.5)).sum(),
        "≤-5 (exit)":   (raw_scores <= -5.0).sum(),
    }
    dist_str = "  ".join(f"{k}: {v:,}" for k, v in score_dist.items())
    print(f"   Score dist: {dist_str}")

    # ── 3. Crash Shield ────────────────────────────────────────────────────
    print("\n🛡️  Computing Crash Shield (vectorized)...")
    cs = compute_crash_shield(spy_df, vix_df, hyg_df)
    shield_days  = int((cs["level"] == "SHIELD").sum())
    caution_days = int((cs["level"] == "CAUTION").sum())
    print(f"   SHIELD days: {shield_days:,}  |  CAUTION days: {caution_days:,}  "
          f"|  NONE days: {len(cs) - shield_days - caution_days:,}")

    cs_mult = cs["multiplier"].reindex(common_idx, fill_value=1.0)
    pos_v1_cs = (pos_v1 * cs_mult).clip(0, 1)
    pos_v2_cs = (pos_v2 * cs_mult).clip(0, 1)

    # ── 4. ML Regime (two versions: v1 legacy label vs v3 correction-aware) ─
    pos_v1_ml = pos_v1_cs.copy()   # Old 3-Layer: binary position × v1 ML
    pos_v2_ml = pos_v2_cs.copy()   # D1+D2: graduated × v1 ML (Direction 1+2 only)
    pos_v3_ml = pos_v2_cs.copy()   # D1+D2+D3: graduated × v3 ML (all 3 directions)

    if not args.skip_ml:
        from strategies.ml_regime import MLRegimeClassifier

        # ── v1 label (legacy: y=1 if 20d return ≥ 0) ──────────────────────
        print("\n🤖 ML Regime — v1 label (legacy: y=1 if 20d return ≥ 0) ...")
        print("   4 OOS windows, ~1-2 min\n")
        ml_ok_v1 = False
        try:
            ml_probs_v1 = compute_ml_probs(spy_df, vix_df, hyg_df, target_mode="v1")
            ml_mult_v1  = ml_probs_v1.apply(MLRegimeClassifier.to_position_multiplier)
            ml_mult_v1  = ml_mult_v1.reindex(common_idx, fill_value=1.0)
            pos_v1_ml   = (pos_v1_cs * ml_mult_v1).clip(0, 1)
            pos_v2_ml   = (pos_v2_cs * ml_mult_v1).clip(0, 1)
            bull1 = (ml_probs_v1 > 0.55).mean() * 100
            bear1 = (ml_probs_v1 < 0.45).mean() * 100
            print(f"\n   v1 ML: mean={ml_probs_v1.mean():.3f}  "
                  f"bull={bull1:.0f}%  bear={bear1:.0f}%")
            ml_ok_v1 = True
        except Exception as e:
            print(f"\n   ❌ v1 ML failed: {e} → skipping ML")

        # ── v3 label (Direction 3: y=0 if 20d return < -1% OR 10%+ correction)
        print("\n🤖 ML Regime — v3 label (Direction 3: correction-aware) ...")
        print("   4 OOS windows, ~1-2 min\n")
        ml_ok_v3 = False
        try:
            ml_probs_v3 = compute_ml_probs(spy_df, vix_df, hyg_df, target_mode="v3")
            ml_mult_v3  = ml_probs_v3.apply(MLRegimeClassifier.to_position_multiplier)
            ml_mult_v3  = ml_mult_v3.reindex(common_idx, fill_value=1.0)
            pos_v3_ml   = (pos_v2_cs * ml_mult_v3).clip(0, 1)
            bull3 = (ml_probs_v3 > 0.55).mean() * 100
            bear3 = (ml_probs_v3 < 0.45).mean() * 100
            print(f"\n   v3 ML: mean={ml_probs_v3.mean():.3f}  "
                  f"bull={bull3:.0f}%  bear={bear3:.0f}%")
            # Label distribution comparison
            from strategies.ml_regime import build_features
            df_tmp = build_features(spy_df, vix_df, hyg_df, include_target=True,
                                    target_mode="v3").dropna(subset=["y"])
            v3_bear_pct = (df_tmp["y"] == 0).mean() * 100
            df_tmp_v1   = build_features(spy_df, vix_df, hyg_df, include_target=True,
                                         target_mode="v1").dropna(subset=["y"])
            v1_bear_pct = (df_tmp_v1["y"] == 0).mean() * 100
            print(f"\n   Label balance — v1: {v1_bear_pct:.0f}% bearish  "
                  f"v3: {v3_bear_pct:.0f}% bearish  "
                  f"(correction added {v3_bear_pct - v1_bear_pct:.0f}pp more negative samples)")
            ml_ok_v3 = True
        except Exception as e:
            print(f"\n   ❌ v3 ML failed: {e} → v3 uses v1 ML or CS-only")

        if not (ml_ok_v1 or ml_ok_v3):
            args.skip_ml = True

    ml_note = "" if not args.skip_ml else " (ML skipped)"

    # ── 5. Dual Momentum ──────────────────────────────────────────────────
    print("\n📈 Computing Dual Momentum...")
    dm = compute_dual_momentum(spy_df)
    dm_scale = dm["scale"].reindex(common_idx, fill_value=1.0)
    abs_neg_days  = int((~dm["abs_ok"]).sum())
    crash_pr_days = int(dm["crash_protect"].sum())
    print(f"   Absolute momentum blocked: {abs_neg_days:,} days  "
          f"|  Crash protect active: {crash_pr_days:,} days")

    pos_v1_full = (pos_v1_ml * dm_scale).clip(0, 1)   # Old 3-Layer
    pos_v2_full = (pos_v2_ml * dm_scale).clip(0, 1)   # Direction 1+2
    pos_v3_full = (pos_v3_ml * dm_scale).clip(0, 1)   # Direction 1+2+3

    # ── 6. Simulate equity curves ─────────────────────────────────────────
    print("\n📈 Simulating equity curves...")
    bah_eq = pd.Series(
        (spy_df["Close"] / spy_df["Close"].iloc[0]).values,
        index=spy_df.index,
        name="Buy & Hold SPY",
    )

    v3_label = f"4. New 3-Layer v3 (D1+D2+D3{ml_note})"
    EQ_LABELS = [
        ("0. Buy & Hold",                    bah_eq),
        ("1. Old Full 3-Layer",              simulate_equity(spy_ret, pos_v1_full, args.cost, "1. Old Full 3-Layer")),
        (f"2. New 3L v2 (D1+D2{ml_note})",  simulate_equity(spy_ret, pos_v2_full, args.cost, "2. New 3L v2")),
        (v3_label,                           simulate_equity(spy_ret, pos_v3_full, args.cost, "4. New 3L v3")),
    ]
    eq_v1_full  = EQ_LABELS[1][1]
    eq_v2_full  = EQ_LABELS[2][1]
    eq_v3_full  = EQ_LABELS[3][1]

    # ── 7. Performance table ───────────────────────────────────────────────
    print(f"\n{'─'*82}")
    print(f"{'Strategy':<38} {'Total Ret':>10} {'CAGR':>8} {'Sharpe':>8} "
          f"{'Max DD':>10} {'Calmar':>8}")
    print(f"{'─'*82}")
    for label, equity in EQ_LABELS:
        is_bench = label == "0. Buy & Hold"
        s = performance_stats(equity, benchmark=bah_eq if not is_bench else None)
        print(f"{label:<38} {s['Total Return']:>10} {s['CAGR']:>8} "
              f"{s['Sharpe']:>8} {s['Max Drawdown']:>10} {s['Calmar']:>8}")
    print(f"{'─'*82}")

    # Improvement summary
    def _cagr(eq):
        n = len(eq.pct_change().dropna())
        return (float(eq.iloc[-1]) ** (1 / (n / 252)) - 1) * 100
    def _maxdd(eq):
        return float((eq / eq.cummax() - 1).min()) * 100
    def _sharpe(eq):
        r = eq.pct_change().dropna()
        return float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))

    for lbl_old, eq_old, lbl_new, eq_new in [
        ("Old 3L",      eq_v1_full, "v2 (D1+D2)",    eq_v2_full),
        ("v2 (D1+D2)",  eq_v2_full, "v3 (D1+D2+D3)", eq_v3_full),
        ("Old 3L",      eq_v1_full, "v3 (D1+D2+D3)", eq_v3_full),
    ]:
        c_o, c_n = _cagr(eq_old), _cagr(eq_new)
        d_o, d_n = _maxdd(eq_old), _maxdd(eq_new)
        s_o, s_n = _sharpe(eq_old), _sharpe(eq_new)
        print(f"  {lbl_old:<16} → {lbl_new:<16}: "
              f"CAGR {c_o:.1f}%→{c_n:.1f}% ({c_n-c_o:+.1f}pp)  "
              f"MaxDD {d_o:.1f}%→{d_n:.1f}% ({d_n-d_o:+.1f}pp)  "
              f"Sharpe {s_o:.2f}→{s_n:.2f}")

    print(f"\n  ✅ Transaction cost: {args.cost} bps one-way | Rebalance: daily")
    print(f"  ✅ No lookahead bias | ML uses walk-forward OOS predictions only")
    print(f"  ℹ️  v3 ML label: y=0 if next-20d return < -1% OR currently ≥10% below 252d peak")

    # ── 8. Crisis period analysis ─────────────────────────────────────────
    print(f"\n{'─'*82}")
    print("  Crash period analysis (return during drawdown window):")
    print(f"{'─'*82}")
    crisis_periods = [
        ("2020 COVID crash",   "2020-01-17", "2020-03-23"),
        ("2022 Bear market",   "2021-12-31", "2022-10-12"),
        ("2018 Q4 selloff",    "2018-09-28", "2018-12-24"),
        ("2015-16 correction", "2015-07-20", "2016-02-11"),
    ]
    for name, start, end in crisis_periods:
        try:
            cols = []
            for lbl, eq in [("B&H", bah_eq), ("Old3L", eq_v1_full),
                             ("v2", eq_v2_full), ("v3", eq_v3_full)]:
                seg = eq.loc[start:end]
                if len(seg) > 1:
                    r = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
                    cols.append(f"{lbl}: {r:+.1f}%")
            print(f"  {name:<24} " + "  ".join(cols))
        except Exception:
            pass
    print(f"{'─'*82}")

    # ── 9. Chart ──────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n📊 Generating chart...")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.55, 0.25, 0.20],
                shared_xaxes=True,
                subplot_titles=[
                    "Equity Curves (log scale) — All 3 Directions",
                    "Drawdown Comparison",
                    "Composite Score",
                ],
                vertical_spacing=0.06,
            )

            curve_styles = [
                ("0. Buy & Hold",       bah_eq,     "rgba(160,160,160,0.9)", 1.5, "dot"),
                ("1. Old Full 3-Layer", eq_v1_full, "steelblue",             2.0, "solid"),
                ("2. New 3L v2 (D1+D2)", eq_v2_full, "darkorange",          2.0, "solid"),
                (f"3. New 3L v3 (D1+D2+D3{ml_note})", eq_v3_full, "crimson", 2.5, "solid"),
            ]
            for lbl, eq, color, width, dash in curve_styles:
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values, name=lbl,
                    line=dict(color=color, width=width, dash=dash),
                ), row=1, col=1)

            # Drawdown: all three strategy versions
            for lbl, eq, color, fill_color in [
                ("Old DD",  eq_v1_full, "rgba(30,144,255,0.8)",  "rgba(30,144,255,0.15)"),
                ("v2 DD",   eq_v2_full, "rgba(255,140,0,0.8)",   "rgba(255,140,0,0.15)"),
                ("v3 DD",   eq_v3_full, "rgba(220,20,60,0.8)",   "rgba(220,20,60,0.15)"),
            ]:
                roll_max = eq.cummax()
                dd_pct = (eq / roll_max - 1) * 100
                fig.add_trace(go.Scatter(
                    x=dd_pct.index, y=dd_pct.values,
                    name=lbl, fill="tozeroy",
                    line=dict(color=color, width=1),
                    fillcolor=fill_color,
                ), row=2, col=1)

            # Composite score
            score_s = raw_scores.reindex(common_idx, fill_value=0)
            fig.add_trace(go.Scatter(
                x=score_s.index, y=score_s.values,
                name="Score", fill="tozeroy",
                line=dict(color="rgba(100,100,200,0.8)", width=1),
                fillcolor="rgba(100,100,200,0.12)",
            ), row=3, col=1)
            for y_val, color, lbl in [
                (6.0, "green",  "BUY (6)"),
                (2.5, "lime",   "25% (2.5)"),
                (-5.0, "red",   "Exit (-5)"),
            ]:
                fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                              annotation_text=lbl, row=3, col=1)

            fig.update_yaxes(type="log", title="NAV (log)", row=1, col=1)
            fig.update_yaxes(title="Drawdown %", row=2, col=1)
            fig.update_yaxes(title="Score", row=3, col=1)
            fig.update_layout(
                title=(f"3-Layer Strategy — Direction 1/2/3 Comparison  "
                       f"({args.years}yr SPY, {args.cost}bps/trade)"),
                height=960,
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
            )

            out_path = ROOT / "scripts" / "backtest_3layer.html"
            fig.write_html(str(out_path))
            print(f"   ✅ Chart saved: {out_path}")
            print(f"   Open in browser: file://{out_path}")

        except ImportError:
            print("   ⚠️  plotly not available — skipping chart")
        except Exception as e:
            print(f"   ❌ Chart failed: {e}")

    print("\n🎉 Backtest complete!\n")


if __name__ == "__main__":
    main()
