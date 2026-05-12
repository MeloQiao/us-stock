"""
Backtest: Aggressive Enhancement — Dynamic TQQQ + SQQQ Bear Hedge

Compares 5 equity curves on SPY/TQQQ/SQQQ universe (2010–present):
  0. Buy & Hold SPY            (benchmark)
  1. New 3L v2 SPY-only        (current best baseline)
  2. Path 1: v2 + Dynamic TQQQ overlay
  3. Path 2: v2 + SQQQ bear hedge
  4. Path 1+2 combined

Path 1 — Dynamic TQQQ sizing (all 4 gates required):
  Shield=NONE  AND  ML_mult ≥ 0.80  AND  abs_momentum > 0  AND  score ≥ 7.0
    score ≥ 9.0  →  swap 30% of SPY for TQQQ  (effective exposure ≈ 160%)
    score ≥ 7.5  →  swap 22%                   (≈ 146%)
    score ≥ 7.0  →  swap 15%                   (≈ 130%)

Path 2 — SQQQ bear hedge (independent of equity position):
  Shield=SHIELD  →  15% SQQQ  (added on top of cleared equity)
  Shield=CAUTION →   7% SQQQ

Usage
─────
  python3 scripts/backtest_aggressive.py
  python3 scripts/backtest_aggressive.py --skip-ml   # faster (3 conditions, no ML gate)
  python3 scripts/backtest_aggressive.py --years 20  # longer history (TQQQ exists from 2010)
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
# Data
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(ticker: str, years: int) -> pd.DataFrame:
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
# Signal helpers (mirrors backtest_3layer.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_composite_score_series(spy_df: pd.DataFrame, vix_df) -> pd.Series:
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS, COMPOSITE_BUY_THRESHOLD
    res = composite_score(
        spy_df, symbol="SPY", vix_df=vix_df,
        buy_threshold=COMPOSITE_BUY_THRESHOLD.get("us", 6.0),
        sell_threshold=3.0,
        weights=STRATEGY_WEIGHTS,
    )
    return res["indicators"]["Composite_Score"].reindex(spy_df.index, fill_value=0.0)


def build_graduated_position(score_series: pd.Series, sell_threshold: float = -5.0) -> pd.Series:
    pos  = pd.Series(0.0, index=score_series.index)
    curr = 0.0
    for dt, score in score_series.items():
        if   score >= 7.5:              curr = 1.00
        elif score >= 6.0:              curr = 0.80
        elif score >= 4.5:              curr = 0.50
        elif score >= 2.5:              curr = 0.25
        elif score <= sell_threshold:   curr = 0.00
        pos[dt] = curr
    return pos


def compute_crash_shield(spy_df, vix_df=None, hyg_df=None) -> pd.DataFrame:
    spy = spy_df["Close"].dropna()
    idx = spy.index
    # S1: VIX panic
    if vix_df is not None and not vix_df.empty:
        vix  = vix_df["Close"].dropna().reindex(idx, method="ffill").fillna(20)
        sig1 = ((vix > 25) & (vix.pct_change(5).fillna(0) > 0.20)).astype(int)
    else:
        sig1 = pd.Series(0, index=idx)
    # S2: Trend break
    ma50 = spy.rolling(50, min_periods=25).mean()
    sig2 = ((spy < ma50) & (ma50 < ma50.shift(10))).astype(int)
    # S3: Sharp decline
    sig3 = (spy.pct_change(20) < -0.08).astype(int)
    # S4: Credit spread
    if hyg_df is not None and not hyg_df.empty:
        hyg  = hyg_df["Close"].dropna().reindex(idx, method="ffill")
        sig4 = (hyg.pct_change(20) < -0.03).astype(int)
    else:
        sig4 = (spy.pct_change(60) < -0.15).astype(int)

    cs_score = (sig1 + sig2 + sig3 + sig4).fillna(0).astype(int)
    mult  = pd.Series(1.0, index=idx)
    mult[cs_score == 2] = 0.5
    mult[cs_score >= 3] = 0.0
    level = pd.Series("NONE", index=idx, dtype=str)
    level[cs_score == 2] = "CAUTION"
    level[cs_score >= 3] = "SHIELD"
    return pd.DataFrame({"score": cs_score, "multiplier": mult, "level": level})


def compute_ml_probs(spy_df, vix_df, hyg_df) -> pd.Series:
    from strategies.ml_regime import build_features, FEATURE_COLS, _get_model, _fill_missing
    from sklearn.preprocessing import StandardScaler
    df = build_features(spy_df, vix_df, hyg_df, include_target=True, target_mode="v1")
    df = df.dropna(subset=["y"])
    prob_series = pd.Series(0.5, index=df.index)
    windows = [
        ("2009-01-01", "2014-12-31", "2015-01-01", "2017-12-31"),
        ("2009-01-01", "2017-12-31", "2018-01-01", "2020-06-30"),
        ("2009-01-01", "2020-06-30", "2020-07-01", "2022-12-31"),
        ("2009-01-01", "2022-12-31", "2023-01-01", "2099-12-31"),
    ]
    for tr0, tr1, te0, te1 in windows:
        try:
            train = df.loc[tr0:tr1].dropna()
            test  = df.loc[te0:te1].dropna()
            if len(train) < 300 or len(test) < 20:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(_fill_missing(train[FEATURE_COLS]))
            X_te = scaler.transform(_fill_missing(test[FEATURE_COLS]))
            mdl  = _get_model()
            mdl.fit(X_tr, train["y"].values)
            probs = mdl.predict_proba(X_te)[:, 1]
            prob_series.loc[test.index] = probs
            print(f"   ✅ OOS {te0[:7]}–{te1[:7]}: n={len(test):,}  mean_prob={probs.mean():.3f}")
        except Exception as e:
            print(f"   ❌ Window {te0[:7]} failed: {e}")
    return prob_series


def compute_dual_momentum(spy_df) -> pd.DataFrame:
    spy     = spy_df["Close"].dropna()
    abs_ok  = spy.pct_change(252) > 0
    log_ret = np.log(spy / spy.shift(1)).dropna()
    vol_21  = log_ret.rolling(21).std()  * np.sqrt(252)
    vol_126 = log_ret.rolling(126).std() * np.sqrt(252)
    crash   = (vol_21 / (vol_126 + 1e-9)) >= 2.0
    scale   = pd.Series(1.0, index=spy.index)
    scale[~abs_ok.reindex(spy.index, fill_value=False)] = 0.0
    scale[crash.reindex(spy.index, fill_value=False)
          & abs_ok.reindex(spy.index, fill_value=True)] = 0.5
    return pd.DataFrame({
        "abs_ok":  abs_ok.reindex(spy.index, fill_value=False),
        "scale":   scale,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Path 1 — TQQQ dynamic weight
# ══════════════════════════════════════════════════════════════════════════════

def compute_tqqq_weight(
    score_series: pd.Series,
    cs_level: pd.Series,
    ml_mult: pd.Series,
    dm_abs_ok: pd.Series,
    min_score: float = 7.0,
    skip_ml: bool = False,
) -> pd.Series:
    """
    TQQQ fraction of portfolio (replaces equivalent SPY notional).
    All 4 conditions required (3 without ML).

    Dynamic sizing:
      score ≥ 9.0  →  30%
      score ≥ 7.5  →  22%
      score ≥ 7.0  →  15%
    """
    idx    = score_series.index
    cs_ok  = cs_level.reindex(idx, fill_value="NONE") == "NONE"
    dm_ok  = dm_abs_ok.reindex(idx, fill_value=True)
    if skip_ml:
        ml_ok = pd.Series(True, index=idx)
    else:
        ml_ok = ml_mult.reindex(idx, fill_value=1.0) >= 0.80

    gate = cs_ok & ml_ok & dm_ok

    tqqq = pd.Series(0.0, index=idx)
    tqqq[gate & (score_series >= 9.0)]                              = 0.30
    tqqq[gate & (score_series >= 7.5) & (score_series < 9.0)]      = 0.22
    tqqq[gate & (score_series >= min_score) & (score_series < 7.5)] = 0.15
    return tqqq


# ══════════════════════════════════════════════════════════════════════════════
# Path 2 — SQQQ bear-hedge weight
# ══════════════════════════════════════════════════════════════════════════════

def compute_sqqq_weight(
    cs_level: pd.Series,
    on_shield:  float = 0.15,
    on_caution: float = 0.07,
) -> pd.Series:
    """
    SQQQ fraction added independently when Crash Shield fires.
    This is additive: even when equity is cleared to 0, SQQQ can be active.
    """
    sqqq = pd.Series(0.0, index=cs_level.index)
    sqqq[cs_level == "SHIELD"]  = on_shield
    sqqq[cs_level == "CAUTION"] = on_caution
    return sqqq


# ══════════════════════════════════════════════════════════════════════════════
# Multi-asset portfolio simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(
    spy_ret:  pd.Series,
    tqqq_ret: pd.Series,
    sqqq_ret: pd.Series,
    spy_base: pd.Series,   # v2 full position (0–1, already with CS/ML/DM)
    tqqq_w:   pd.Series,   # fraction to swap from SPY → TQQQ
    sqqq_w:   pd.Series,   # fraction to add as SQQQ (independent)
    cost_bps: float = 10.0,
    label:    str   = "Strategy",
) -> pd.Series:
    """
    Weights on each day:
      t_tqqq = min(tqqq_w,  spy_base)          # TQQQ replaces SPY 1:1 notional
      t_sqqq = sqqq_w  (independent short)
      t_spy  = spy_base - t_tqqq               # residual goes to SPY
      cash   = 1 - t_spy - t_tqqq - t_sqqq    # always ≥ 0

    SQQQ is additive: it is NOT sourced from the equity budget.
    When SHIELD fires the equity budget → 0 (already cleared by CS multiplier),
    but SQQQ still activates — providing short profit during the crash.
    """
    idx     = spy_ret.index
    spy_b   = spy_base.reindex(idx).fillna(0.0)
    tqqq_w_ = tqqq_w.reindex(idx).fillna(0.0)
    sqqq_w_ = sqqq_w.reindex(idx).fillna(0.0)
    spy_r   = spy_ret.reindex(idx).fillna(0.0)
    tqqq_r  = tqqq_ret.reindex(idx).fillna(0.0)
    sqqq_r  = sqqq_ret.reindex(idx).fillna(0.0)

    nav  = [1.0]
    prev = {"spy": 0.0, "tqqq": 0.0, "sqqq": 0.0}

    for i in range(1, len(idx)):
        base   = float(spy_b.iloc[i - 1])
        t_tqqq = min(float(tqqq_w_.iloc[i - 1]), base)   # capped by equity budget
        t_spy  = max(0.0, base - t_tqqq)
        t_sqqq = float(sqqq_w_.iloc[i - 1])               # independent (no cap against equity)
        # Hard cap: total allocation ≤ 1.0 (avoid leverage)
        if t_spy + t_tqqq + t_sqqq > 1.0:
            t_sqqq = max(0.0, 1.0 - t_spy - t_tqqq)

        # Transaction costs (one-way per instrument)
        tc = (abs(t_spy  - prev["spy"])  +
              abs(t_tqqq - prev["tqqq"]) +
              abs(t_sqqq - prev["sqqq"])) * cost_bps / 10_000.0

        ret = (t_spy  * float(spy_r.iloc[i]) +
               t_tqqq * float(tqqq_r.iloc[i]) +
               t_sqqq * float(sqqq_r.iloc[i])) - tc

        nav.append(nav[-1] * (1.0 + ret))
        prev = {"spy": t_spy, "tqqq": t_tqqq, "sqqq": t_sqqq}

    return pd.Series(nav, index=idx, name=label)


# ══════════════════════════════════════════════════════════════════════════════
# Performance stats
# ══════════════════════════════════════════════════════════════════════════════

def stats(equity: pd.Series) -> dict:
    ret    = equity.pct_change().dropna()
    n_yrs  = len(ret) / 252
    cagr   = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_yrs) - 1)
    sharpe = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252))
    max_dd = float((equity / equity.cummax() - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    return {
        "CAGR":   f"{cagr * 100:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDD":  f"{max_dd * 100:.1f}%",
        "Calmar": f"{calmar:.2f}",
        "_cagr":  cagr, "_sharpe": sharpe, "_maxdd": max_dd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",   type=int,   default=15)
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML (3-condition gate)")
    parser.add_argument("--cost",    type=float, default=10.0, help="One-way cost bps")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  Aggressive Backtest — Path 1 (TQQQ) + Path 2 (SQQQ Bear Hedge)")
    print(f"  {args.years}yr  |  {args.cost}bps/trade  |  "
          f"{'ML SKIPPED (3-cond gate)' if args.skip_ml else 'Full 4-cond gate with ML'}")
    print(f"{'='*72}")

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    print("\n📥 Downloading data…")
    spy_df  = _fetch("SPY",   args.years)
    tqqq_df = _fetch("TQQQ",  args.years)
    sqqq_df = _fetch("SQQQ",  args.years)
    vix_df, hyg_df = None, None
    try:
        vix_df = _fetch("^VIX", args.years)
    except Exception:
        pass
    try:
        hyg_df = _fetch("HYG",  args.years)
    except Exception:
        pass

    # Align on dates where TQQQ and SQQQ both trade (launched Feb 2010)
    common = spy_df.index.intersection(tqqq_df.index).intersection(sqqq_df.index)
    spy_df  = spy_df.loc[common]
    tqqq_df = tqqq_df.loc[common]
    sqqq_df = sqqq_df.loc[common]
    print(f"   SPY:  {len(spy_df):,} days ({common[0].date()} – {common[-1].date()})")
    print(f"   TQQQ: {len(tqqq_df):,} days")
    print(f"   SQQQ: {len(sqqq_df):,} days")

    spy_ret  = spy_df["Close"].pct_change()
    tqqq_ret = tqqq_df["Close"].pct_change()
    sqqq_ret = sqqq_df["Close"].pct_change()

    # ── 2. Composite score ────────────────────────────────────────────────
    print("\n📊 Computing composite scores…")
    raw_scores = get_composite_score_series(spy_df, vix_df)
    pos_v2     = build_graduated_position(raw_scores, sell_threshold=-5.0)

    # ── 3. Crash Shield ───────────────────────────────────────────────────
    print("\n🛡️  Computing Crash Shield…")
    cs       = compute_crash_shield(spy_df, vix_df, hyg_df)
    cs_level = cs["level"].reindex(common, fill_value="NONE")
    cs_mult  = cs["multiplier"].reindex(common, fill_value=1.0)
    shield_d = int((cs_level == "SHIELD").sum())
    caution_d = int((cs_level == "CAUTION").sum())
    print(f"   SHIELD: {shield_d:,}d  CAUTION: {caution_d:,}d  NONE: {len(common)-shield_d-caution_d:,}d")

    pos_v2_cs = (pos_v2 * cs_mult).clip(0, 1)

    # ── 4. ML + Dual Momentum ─────────────────────────────────────────────
    ml_mult = pd.Series(1.0, index=common)     # neutral default
    dm = compute_dual_momentum(spy_df)
    dm_abs_ok = dm["abs_ok"].reindex(common, fill_value=True)
    dm_scale  = dm["scale"].reindex(common, fill_value=1.0)

    if not args.skip_ml:
        print("\n🤖 Walk-forward ML (v1 label, ~2 min)…")
        try:
            from strategies.ml_regime import MLRegimeClassifier
            ml_probs = compute_ml_probs(spy_df, vix_df, hyg_df)
            ml_mult  = ml_probs.apply(MLRegimeClassifier.to_position_multiplier)
            ml_mult  = ml_mult.reindex(common, fill_value=1.0)
            bull_pct = (ml_probs > 0.55).mean() * 100
            bear_pct = (ml_probs < 0.45).mean() * 100
            print(f"\n   ML: mean={ml_probs.mean():.3f}  bull={bull_pct:.0f}%  bear={bear_pct:.0f}%")
        except Exception as e:
            print(f"   ❌ ML failed: {e} — falling back to neutral multiplier")
            args.skip_ml = True

    # Full v2 (all 3 layers)
    pos_v2_full = (pos_v2_cs * ml_mult * dm_scale).clip(0, 1)

    # ── 5. Path 1: TQQQ weights ───────────────────────────────────────────
    tqqq_w = compute_tqqq_weight(
        raw_scores, cs_level, ml_mult, dm_abs_ok,
        min_score=7.0, skip_ml=args.skip_ml,
    )
    days_30 = int((tqqq_w >= 0.29).sum())
    days_22 = int(((tqqq_w >= 0.21) & (tqqq_w < 0.29)).sum())
    days_15 = int(((tqqq_w >= 0.14) & (tqqq_w < 0.21)).sum())
    print(f"\n   TQQQ active: {int((tqqq_w > 0).sum()):,}d total "
          f"(30%: {days_30:,}d  22%: {days_22:,}d  15%: {days_15:,}d)")

    # ── 6. Path 2: SQQQ weights ───────────────────────────────────────────
    sqqq_w = compute_sqqq_weight(cs_level, on_shield=0.15, on_caution=0.07)
    print(f"   SQQQ active: {int((sqqq_w > 0).sum()):,}d  "
          f"(15% on SHIELD: {shield_d:,}d  7% on CAUTION: {caution_d:,}d)")

    # ── 7. Simulate equity curves ─────────────────────────────────────────
    print("\n📈 Simulating portfolios…")
    _zero = pd.Series(0.0, index=common)

    bah = pd.Series(
        (spy_df["Close"] / spy_df["Close"].iloc[0]).values,
        index=common, name="0. Buy & Hold SPY",
    )
    eq_v2  = simulate_portfolio(spy_ret, tqqq_ret, sqqq_ret, pos_v2_full, _zero, _zero,
                                 args.cost, "1. v2 (current)")
    eq_p1  = simulate_portfolio(spy_ret, tqqq_ret, sqqq_ret, pos_v2_full, tqqq_w, _zero,
                                 args.cost, "2. Path1: +TQQQ")
    eq_p2  = simulate_portfolio(spy_ret, tqqq_ret, sqqq_ret, pos_v2_full, _zero, sqqq_w,
                                 args.cost, "3. Path2: +SQQQ")
    eq_p12 = simulate_portfolio(spy_ret, tqqq_ret, sqqq_ret, pos_v2_full, tqqq_w, sqqq_w,
                                 args.cost, "4. Path1+2")

    EQ = [
        ("0. Buy & Hold SPY",   bah),
        ("1. v2 (current)",     eq_v2),
        ("2. Path 1: +TQQQ",    eq_p1),
        ("3. Path 2: +SQQQ",    eq_p2),
        ("4. Path 1+2",         eq_p12),
    ]

    # ── 8. Performance table ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'Strategy':<28} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8}")
    print(f"{'─'*72}")
    results = {}
    for label, eq in EQ:
        s = stats(eq)
        results[label] = s
        print(f"{label:<28} {s['CAGR']:>8} {s['Sharpe']:>8} {s['MaxDD']:>10} {s['Calmar']:>8}")
    print(f"{'─'*72}")

    # Delta vs v2 baseline
    v2_s = results["1. v2 (current)"]
    print("\n  Delta vs v2 baseline:")
    for label, eq in EQ[2:]:
        s = results[label]
        dc = (s["_cagr"] - v2_s["_cagr"]) * 100
        ds = s["_sharpe"] - v2_s["_sharpe"]
        dd = (s["_maxdd"] - v2_s["_maxdd"]) * 100
        print(f"  {label:<26} CAGR {dc:+.1f}pp  Sharpe {ds:+.2f}  MaxDD {dd:+.1f}pp")

    # ── 9. Crisis periods ─────────────────────────────────────────────────
    crisis = [
        ("2020 COVID crash",     "2020-01-17", "2020-03-23"),
        ("2022 Bear market",     "2021-12-31", "2022-10-12"),
        ("2018 Q4 selloff",      "2018-09-28", "2018-12-24"),
        ("2015-16 correction",   "2015-07-20", "2016-02-11"),
    ]
    print(f"\n{'─'*72}")
    print("  Crisis period returns:")
    print(f"{'─'*72}")
    for name, s, e in crisis:
        parts = []
        for lbl, eq in EQ:
            try:
                seg = eq.loc[s:e]
                if len(seg) > 1:
                    r = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
                    short = lbl.split(":")[0].strip()
                    parts.append(f"{short}: {r:+.1f}%")
            except Exception:
                pass
        print(f"  {name:<22} " + "  ".join(parts))
    print(f"{'─'*72}")

    # ── 10. Allocation breakdown ──────────────────────────────────────────
    print(f"\n  Allocation breakdown (average when active):")
    avg_v2_full  = float(pos_v2_full[pos_v2_full > 0].mean()) * 100 if (pos_v2_full > 0).any() else 0
    avg_tqqq_act = float(tqqq_w[tqqq_w > 0].mean()) * 100 if (tqqq_w > 0).any() else 0
    avg_sqqq_act = float(sqqq_w[sqqq_w > 0].mean()) * 100 if (sqqq_w > 0).any() else 0
    pct_tqqq = float((tqqq_w > 0).mean()) * 100
    pct_sqqq = float((sqqq_w > 0).mean()) * 100
    print(f"  SPY base (v2):  avg={avg_v2_full:.0f}% when invested  "
          f"({float((pos_v2_full > 0).mean())*100:.0f}% of days)")
    print(f"  TQQQ overlay:   avg={avg_tqqq_act:.0f}% when active   "
          f"({pct_tqqq:.0f}% of days)")
    print(f"  SQQQ hedge:     avg={avg_sqqq_act:.0f}% when active   "
          f"({pct_sqqq:.0f}% of days)")
    print(f"  Note: TQQQ replaces SPY notional; SQQQ is additive from cash.")

    ml_note = " (ML skipped)" if args.skip_ml else ""
    print(f"\n  ✅ Transaction cost: {args.cost}bps one-way{ml_note}")
    print(f"  ✅ TQQQ: launched Feb 2010 — full history available")
    print(f"  ✅ No lookahead bias | All signals use prev-day close")

    # ── 11. Chart ─────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n📊 Generating chart…")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.45, 0.20, 0.20, 0.15],
                shared_xaxes=True,
                subplot_titles=[
                    "Equity Curves (log scale)",
                    "Drawdown",
                    "TQQQ / SQQQ Allocation",
                    "Composite Score",
                ],
                vertical_spacing=0.05,
            )

            palette = [
                ("rgba(150,150,150,0.9)", 1.5, "dot"),
                ("steelblue",             2.0, "solid"),
                ("darkorange",            2.5, "solid"),
                ("seagreen",              2.0, "solid"),
                ("crimson",               2.5, "solid"),
            ]
            for (label, eq), (color, width, dash) in zip(EQ, palette):
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values, name=label,
                    line=dict(color=color, width=width, dash=dash),
                ), row=1, col=1)
                dd = (eq / eq.cummax() - 1) * 100
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values, name=f"DD {label[:8]}",
                    line=dict(color=color, width=1), fill="tozeroy",
                    showlegend=False,
                ), row=2, col=1)

            # TQQQ allocation bar
            fig.add_trace(go.Scatter(
                x=tqqq_w.index, y=(tqqq_w * 100).values,
                name="TQQQ %", fill="tozeroy",
                line=dict(color="rgba(255,140,0,0.8)", width=1),
                fillcolor="rgba(255,140,0,0.25)",
            ), row=3, col=1)
            # SQQQ allocation bar
            fig.add_trace(go.Scatter(
                x=sqqq_w.index, y=(sqqq_w * 100).values,
                name="SQQQ %", fill="tozeroy",
                line=dict(color="rgba(220,20,60,0.8)", width=1),
                fillcolor="rgba(220,20,60,0.25)",
            ), row=3, col=1)

            # Score
            score_s = raw_scores.reindex(common, fill_value=0)
            fig.add_trace(go.Scatter(
                x=score_s.index, y=score_s.values,
                name="Score", fill="tozeroy",
                line=dict(color="rgba(100,100,200,0.8)", width=1),
                fillcolor="rgba(100,100,200,0.12)",
            ), row=4, col=1)
            for y_val, color, txt in [(7.0, "green", "7.0"), (-5.0, "red", "-5")]:
                fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                              annotation_text=txt, row=4, col=1)

            # Shade SHIELD periods across all panels
            shield_series = (cs_level == "SHIELD").astype(int)
            starts = shield_series.index[shield_series.diff().fillna(0) == 1].tolist()
            ends   = shield_series.index[shield_series.diff().fillna(0) == -1].tolist()
            if shield_series.iloc[-1] == 1:
                ends.append(shield_series.index[-1])
            for s_dt, e_dt in zip(starts, ends):
                for row in [1, 2, 3, 4]:
                    fig.add_vrect(
                        x0=s_dt, x1=e_dt,
                        fillcolor="rgba(255,0,0,0.07)",
                        line_width=0,
                        row=row, col=1,
                    )

            fig.update_yaxes(type="log", title="NAV", row=1, col=1)
            fig.update_yaxes(title="DD %",    row=2, col=1)
            fig.update_yaxes(title="Alloc %", row=3, col=1)
            fig.update_yaxes(title="Score",   row=4, col=1)
            fig.update_layout(
                title=(f"Aggressive Enhancement — Path 1 (TQQQ) + Path 2 (SQQQ)  "
                       f"({args.years}yr, {args.cost}bps{ml_note})"),
                height=1100,
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
            )

            out = ROOT / "scripts" / "backtest_aggressive.html"
            fig.write_html(str(out))
            print(f"   ✅ Chart: {out}")
            print(f"   Open:  file://{out}")
        except ImportError:
            print("   ⚠️  plotly not available")
        except Exception as e:
            print(f"   ❌ Chart failed: {e}")

    print("\n🎉 Done!\n")


if __name__ == "__main__":
    main()
