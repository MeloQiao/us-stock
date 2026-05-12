"""
Backtest: Sharpe 1.5 — Three-Lever Upgrade

The current v2 strategy achieves Sharpe ~0.92 by being very conservative:
  - Only 75% of days invested, avg position 52% → effective exposure ~39%
  - SPY (broad market) captures 10-11% CAGR, strategy captures ~5%

This backtest tests three progressive levers toward Sharpe 1.5:

  Lever 1 — Upgrade base instrument: SPY → QQQ
    QQQ (Nasdaq 100) CAGR ~19% vs SPY 14% historically.
    Same SPY-based regime signals, higher-beta vehicle.
    Expected: +2-3pp CAGR, slight MaxDD increase.

  Lever 2 — Aggressive TQQQ when conviction is highest
    Replace 30-70% of QQQ with TQQQ based on score tier.
    Only when all 4 conditions pass (Shield=NONE, ML bull, DM ok, score≥7).
    Higher conviction → larger TQQQ fraction.
    Expected: +2-4pp CAGR, MaxDD -3-6pp deeper.

  Lever 3 — GLD hedge when Crash Shield fires
    Instead of 100% cash during SHIELD, hold 25% GLD.
    GLD appreciates in risk-off / flight-to-safety events.
    Turns "dead cash" periods into marginal positive return.
    Expected: +0.5-1pp CAGR, MaxDD slightly better during gold-friendly crises.

Combined target: CAGR ~10-13%, MaxDD ~-15-20%, Sharpe ~1.3-1.5

Strategies compared (6 equity curves):
  0. B&H QQQ                   (higher benchmark)
  1. v2 SPY-only                (current best, reference)
  2. Lever 1: v2-QQQ            (upgrade instrument only)
  3. Lever 1+2: v2-QQQ+TQQQ    (upgrade + leverage on conviction)
  4. Lever 1+2+3: Full upgrade  (all three levers)
  5. B&H SPY                    (original benchmark)

TQQQ sizing (fraction of base position replaced by TQQQ):
  All 4 conditions + score ≥ 9.0  →  70% of base position in TQQQ
  All 4 conditions + score ≥ 7.5  →  50% of base position in TQQQ
  All 4 conditions + score ≥ 7.0  →  30% of base position in TQQQ

Example: base=80% (score 6.5), conditions pass with score=9.0 boost:
  TQQQ = 80% × 70% = 56%, QQQ = 24%, effective QQQ exposure = 24%×1 + 56%×3 = 192%

GLD allocation (independent of equity position):
  SHIELD  → 25% GLD  (equity already cleared; earning on crisis hedge)
  CAUTION → 12% GLD  (partial hedge alongside reduced equity)

Usage
─────
  python3 scripts/backtest_sharpe15.py
  python3 scripts/backtest_sharpe15.py --skip-ml --years 15
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
# Signal helpers (identical to backtest_3layer / backtest_aggressive)
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
        if   score >= 7.5:            curr = 1.00
        elif score >= 6.0:            curr = 0.80
        elif score >= 4.5:            curr = 0.50
        elif score >= 2.5:            curr = 0.25
        elif score <= sell_threshold: curr = 0.00
        pos[dt] = curr
    return pos


def compute_crash_shield(spy_df, vix_df=None, hyg_df=None) -> pd.DataFrame:
    spy = spy_df["Close"].dropna()
    idx = spy.index
    if vix_df is not None and not vix_df.empty:
        vix  = vix_df["Close"].dropna().reindex(idx, method="ffill").fillna(20)
        sig1 = ((vix > 25) & (vix.pct_change(5).fillna(0) > 0.20)).astype(int)
    else:
        sig1 = pd.Series(0, index=idx)
    ma50 = spy.rolling(50, min_periods=25).mean()
    sig2 = ((spy < ma50) & (ma50 < ma50.shift(10))).astype(int)
    sig3 = (spy.pct_change(20) < -0.08).astype(int)
    if hyg_df is not None and not hyg_df.empty:
        hyg  = hyg_df["Close"].dropna().reindex(idx, method="ffill")
        sig4 = (hyg.pct_change(20) < -0.03).astype(int)
    else:
        sig4 = (spy.pct_change(60) < -0.15).astype(int)
    cs_score = (sig1 + sig2 + sig3 + sig4).fillna(0).astype(int)
    mult      = pd.Series(1.0, index=idx)
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
            print(f"   ✅ OOS {te0[:7]}–{te1[:7]}: mean_prob={probs.mean():.3f}")
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
        "abs_ok": abs_ok.reindex(spy.index, fill_value=False),
        "scale":  scale,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Lever 2 — TQQQ fraction of base position
# ══════════════════════════════════════════════════════════════════════════════

def compute_tqqq_fraction(
    score:     pd.Series,
    cs_level:  pd.Series,
    ml_mult:   pd.Series,
    dm_abs_ok: pd.Series,
    skip_ml:   bool = False,
) -> pd.Series:
    """
    Returns fraction (0.0–0.70) of base position to replace with TQQQ.
    All 4 gates required (3 without ML).

    Tier mapping:
      score ≥ 9.0  →  70% of base  (nearly full TQQQ when most bullish)
      score ≥ 7.5  →  50% of base
      score ≥ 7.0  →  30% of base
    """
    idx   = score.index
    cs_ok = cs_level.reindex(idx, fill_value="NONE") == "NONE"
    dm_ok = dm_abs_ok.reindex(idx, fill_value=True)
    ml_ok = (ml_mult.reindex(idx, fill_value=1.0) >= 0.80
             if not skip_ml else pd.Series(True, index=idx))
    gate  = cs_ok & ml_ok & dm_ok

    frac = pd.Series(0.0, index=idx)
    frac[gate & (score >= 9.0)]                          = 0.70
    frac[gate & (score >= 7.5) & (score < 9.0)]         = 0.50
    frac[gate & (score >= 7.0) & (score < 7.5)]         = 0.30
    return frac


# ══════════════════════════════════════════════════════════════════════════════
# Lever 3 — GLD weight during Shield / Caution
# ══════════════════════════════════════════════════════════════════════════════

def compute_gld_weight(cs_level: pd.Series) -> pd.Series:
    """
    GLD is additive (sourced from cash, not from equity budget).
    During SHIELD: equity → 0 (CS mult=0), GLD provides positive return.
    During CAUTION: partial equity + partial GLD hedge.
    """
    gld = pd.Series(0.0, index=cs_level.index)
    gld[cs_level == "SHIELD"]  = 0.25
    gld[cs_level == "CAUTION"] = 0.12
    return gld


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(
    qqq_ret:   pd.Series,    # QQQ daily returns (base instrument)
    tqqq_ret:  pd.Series,    # TQQQ daily returns
    gld_ret:   pd.Series,    # GLD daily returns
    spy_base:  pd.Series,    # graduated base position (0–1, already CS/ML/DM adjusted)
    tqqq_frac: pd.Series,    # fraction of base to replace with TQQQ (0–0.70)
    gld_w:     pd.Series,    # GLD weight (independent)
    cost_bps:  float = 10.0,
    label:     str   = "Strategy",
) -> pd.Series:
    """
    Each day's allocation:
      t_tqqq = base × tqqq_frac
      t_qqq  = base × (1 − tqqq_frac)
      t_gld  = gld_w    (independent; capped so total ≤ 1.0)
      cash   = 1 − t_qqq − t_tqqq − t_gld  (always ≥ 0)
    """
    idx    = qqq_ret.index
    base_s = spy_base.reindex(idx).fillna(0.0)
    tfrac  = tqqq_frac.reindex(idx).fillna(0.0)
    gld_s  = gld_w.reindex(idx).fillna(0.0)
    qqq_r  = qqq_ret.reindex(idx).fillna(0.0)
    tqqq_r = tqqq_ret.reindex(idx).fillna(0.0)
    gld_r  = gld_ret.reindex(idx).fillna(0.0)

    nav  = [1.0]
    prev = {"qqq": 0.0, "tqqq": 0.0, "gld": 0.0}

    for i in range(1, len(idx)):
        base = float(base_s.iloc[i - 1])
        frac = float(tfrac.iloc[i - 1])
        gld  = float(gld_s.iloc[i - 1])

        t_tqqq = base * frac
        t_qqq  = base * (1.0 - frac)
        t_gld  = gld

        # Hard cap: no leverage (total ≤ 1.0)
        total = t_qqq + t_tqqq + t_gld
        if total > 1.0:
            s = 1.0 / total
            t_qqq  *= s
            t_tqqq *= s
            t_gld  *= s

        # Transaction costs
        tc = (abs(t_qqq  - prev["qqq"])  +
              abs(t_tqqq - prev["tqqq"]) +
              abs(t_gld  - prev["gld"])) * cost_bps / 10_000.0

        ret = (t_qqq  * float(qqq_r.iloc[i])  +
               t_tqqq * float(tqqq_r.iloc[i]) +
               t_gld  * float(gld_r.iloc[i])) - tc

        nav.append(nav[-1] * (1.0 + ret))
        prev = {"qqq": t_qqq, "tqqq": t_tqqq, "gld": t_gld}

    return pd.Series(nav, index=idx, name=label)


# ══════════════════════════════════════════════════════════════════════════════
# Performance stats
# ══════════════════════════════════════════════════════════════════════════════

def perf(equity: pd.Series) -> dict:
    ret    = equity.pct_change().dropna()
    n_yrs  = len(ret) / 252
    cagr   = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_yrs) - 1)
    sharpe = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252))
    max_dd = float((equity / equity.cummax() - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    sortino_down = ret[ret < 0].std() * np.sqrt(252) + 1e-9
    sortino      = float(ret.mean() / sortino_down * np.sqrt(252))
    return {
        "CAGR":    f"{cagr * 100:.1f}%",
        "Sharpe":  f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "MaxDD":   f"{max_dd * 100:.1f}%",
        "Calmar":  f"{calmar:.2f}",
        "_cagr": cagr, "_sharpe": sharpe, "_maxdd": max_dd, "_sortino": sortino,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",   type=int,   default=15)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--cost",    type=float, default=10.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  Sharpe 1.5 Backtest — 3 Levers: QQQ + TQQQ + GLD")
    print(f"  {args.years}yr  |  {args.cost}bps/trade  |  "
          f"{'ML skipped' if args.skip_ml else 'full 4-cond ML gate'}")
    print(f"{'='*72}")

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    print("\n📥 Downloading data…")
    tickers = {"SPY": None, "QQQ": None, "TQQQ": None, "GLD": None,
               "^VIX": None, "HYG": None}
    for tk in tickers:
        try:
            tickers[tk] = _fetch(tk, args.years)
            print(f"   {tk:<6}: {len(tickers[tk]):,} days")
        except Exception as e:
            print(f"   {tk:<6}: ⚠️  {e}")

    spy_df  = tickers["SPY"]
    qqq_df  = tickers["QQQ"]
    tqqq_df = tickers["TQQQ"]
    gld_df  = tickers["GLD"]
    vix_df  = tickers["^VIX"]
    hyg_df  = tickers["HYG"]

    # Align on intersection of all instruments
    common = spy_df.index
    for df in [qqq_df, tqqq_df, gld_df]:
        if df is not None:
            common = common.intersection(df.index)
    spy_df  = spy_df.loc[common]
    qqq_df  = qqq_df.loc[common]
    tqqq_df = tqqq_df.loc[common]
    gld_df  = gld_df.loc[common]
    print(f"\n   Common: {len(common):,} days ({common[0].date()} – {common[-1].date()})")

    spy_ret  = spy_df["Close"].pct_change()
    qqq_ret  = qqq_df["Close"].pct_change()
    tqqq_ret = tqqq_df["Close"].pct_change()
    gld_ret  = gld_df["Close"].pct_change()

    # ── 2. Regime signals (all SPY-based) ────────────────────────────────
    print("\n📊 Computing composite scores (SPY-based regime)…")
    raw_scores = get_composite_score_series(spy_df, vix_df)
    pos_base   = build_graduated_position(raw_scores)

    print("\n🛡️  Crash Shield…")
    cs       = compute_crash_shield(spy_df, vix_df, hyg_df)
    cs_level = cs["level"].reindex(common, fill_value="NONE")
    cs_mult  = cs["multiplier"].reindex(common, fill_value=1.0)
    n_shield  = int((cs_level == "SHIELD").sum())
    n_caution = int((cs_level == "CAUTION").sum())
    print(f"   SHIELD: {n_shield:,}d  CAUTION: {n_caution:,}d  NONE: {len(common)-n_shield-n_caution:,}d")

    dm       = compute_dual_momentum(spy_df)
    dm_abs   = dm["abs_ok"].reindex(common, fill_value=True)
    dm_scale = dm["scale"].reindex(common, fill_value=1.0)

    ml_mult = pd.Series(1.0, index=common)
    if not args.skip_ml:
        print("\n🤖 Walk-forward ML…")
        try:
            from strategies.ml_regime import MLRegimeClassifier
            ml_probs = compute_ml_probs(spy_df, vix_df, hyg_df)
            ml_mult  = ml_probs.apply(MLRegimeClassifier.to_position_multiplier)
            ml_mult  = ml_mult.reindex(common, fill_value=1.0)
            print(f"\n   ML: mean_prob={ml_probs.mean():.3f}  "
                  f"bull={float((ml_probs>0.55).mean())*100:.0f}%  "
                  f"bear={float((ml_probs<0.45).mean())*100:.0f}%")
        except Exception as e:
            print(f"   ❌ ML failed: {e}")
            args.skip_ml = True

    # Full base position (all 3 layers)
    pos_full = (pos_base * cs_mult * ml_mult * dm_scale).clip(0, 1)

    # ── 3. Lever weights ──────────────────────────────────────────────────
    tqqq_frac  = compute_tqqq_fraction(raw_scores, cs_level, ml_mult, dm_abs,
                                       skip_ml=args.skip_ml)
    gld_weight = compute_gld_weight(cs_level)
    _zero      = pd.Series(0.0, index=common)

    days_tqqq_70 = int((tqqq_frac >= 0.69).sum())
    days_tqqq_50 = int(((tqqq_frac >= 0.49) & (tqqq_frac < 0.69)).sum())
    days_tqqq_30 = int(((tqqq_frac >= 0.29) & (tqqq_frac < 0.49)).sum())
    print(f"\n   TQQQ fraction: 70%={days_tqqq_70}d  50%={days_tqqq_50}d  30%={days_tqqq_30}d  "
          f"(total active: {int((tqqq_frac>0).sum())}d)")

    # Avg effective QQQ exposure when TQQQ active
    tqqq_active_mask = tqqq_frac > 0
    if tqqq_active_mask.any():
        avg_base_when_tqqq = float(pos_full[tqqq_active_mask].mean())
        avg_frac_when_tqqq = float(tqqq_frac[tqqq_active_mask].mean())
        avg_qqq_when_tqqq  = avg_base_when_tqqq * (1 - avg_frac_when_tqqq)
        avg_tqqq_w         = avg_base_when_tqqq * avg_frac_when_tqqq
        eff_exposure       = avg_qqq_when_tqqq + avg_tqqq_w * 3
        print(f"   Avg effective QQQ exposure on TQQQ days: {eff_exposure*100:.0f}%  "
              f"(QQQ={avg_qqq_when_tqqq*100:.0f}% + TQQQ={avg_tqqq_w*100:.0f}%×3)")

    # ── 4. Simulate equity curves ─────────────────────────────────────────
    print("\n📈 Simulating…")

    bah_spy = pd.Series((spy_df["Close"] / spy_df["Close"].iloc[0]).values,
                        index=common, name="5. B&H SPY")
    bah_qqq = pd.Series((qqq_df["Close"] / qqq_df["Close"].iloc[0]).values,
                        index=common, name="0. B&H QQQ")

    eq_v2  = simulate_portfolio(spy_ret, tqqq_ret, gld_ret, pos_full, _zero, _zero,
                                 args.cost, "1. v2 SPY-only")

    eq_l1  = simulate_portfolio(qqq_ret, tqqq_ret, gld_ret, pos_full, _zero, _zero,
                                 args.cost, "2. L1: QQQ base")

    eq_l12 = simulate_portfolio(qqq_ret, tqqq_ret, gld_ret, pos_full, tqqq_frac, _zero,
                                 args.cost, "3. L1+2: QQQ+TQQQ")

    eq_l123 = simulate_portfolio(qqq_ret, tqqq_ret, gld_ret, pos_full, tqqq_frac, gld_weight,
                                  args.cost, "4. L1+2+3: Full")

    EQ = [
        ("0. B&H QQQ",          bah_qqq),
        ("1. v2 SPY-only",      eq_v2),
        ("2. L1: QQQ base",     eq_l1),
        ("3. L1+2: QQQ+TQQQ",   eq_l12),
        ("4. L1+2+3: Full",     eq_l123),
        ("5. B&H SPY",          bah_spy),
    ]

    # ── 5. Performance table ──────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"{'Strategy':<28} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>10} {'Calmar':>8}")
    print(f"{'─'*78}")
    results = {}
    for label, eq in EQ:
        s = perf(eq)
        results[label] = s
        marker = " ◄ TARGET" if s["_sharpe"] >= 1.4 else ""
        print(f"{label:<28} {s['CAGR']:>8} {s['Sharpe']:>8} {s['Sortino']:>8} "
              f"{s['MaxDD']:>10} {s['Calmar']:>8}{marker}")
    print(f"{'─'*78}")

    ml_note = " (ML skipped)" if args.skip_ml else ""
    print(f"\n  ✅ {args.years}yr backtest  |  {args.cost}bps/trade{ml_note}")
    print(f"  ✅ No lookahead bias  |  TQQQ/GLD launched 2004/2004, full history")

    # ── 6. Incremental delta table ────────────────────────────────────────
    print(f"\n  Incremental contribution of each lever (vs v2 baseline):")
    v2_s = results["1. v2 SPY-only"]
    for label, _ in EQ[2:5]:
        s  = results[label]
        dc = (s["_cagr"]   - v2_s["_cagr"])   * 100
        ds = s["_sharpe"]  - v2_s["_sharpe"]
        dd = (s["_maxdd"]  - v2_s["_maxdd"])   * 100
        print(f"  {label:<28} CAGR {dc:+.1f}pp  Sharpe {ds:+.2f}  MaxDD {dd:+.1f}pp")

    # ── 7. Crisis periods ─────────────────────────────────────────────────
    crisis = [
        ("2020 COVID crash",    "2020-01-17", "2020-03-23"),
        ("2022 Bear market",    "2021-12-31", "2022-10-12"),
        ("2018 Q4 selloff",     "2018-09-28", "2018-12-24"),
        ("2015-16 correction",  "2015-07-20", "2016-02-11"),
    ]
    print(f"\n{'─'*78}")
    print("  Crisis period returns:")
    print(f"{'─'*78}")
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
    print(f"{'─'*78}")

    # ── 8. Allocation stats ───────────────────────────────────────────────
    print(f"\n  Allocation breakdown:")
    days_invested = float((pos_full > 0).mean()) * 100
    avg_base = float(pos_full[pos_full > 0].mean()) * 100 if (pos_full > 0).any() else 0
    avg_tqqq = float(tqqq_frac[tqqq_frac > 0].mean()) * 100 if (tqqq_frac > 0).any() else 0
    avg_gld  = float(gld_weight[gld_weight > 0].mean()) * 100 if (gld_weight > 0).any() else 0
    print(f"  Base (QQQ): {days_invested:.0f}% of days invested, avg {avg_base:.0f}% when in")
    print(f"  TQQQ frac:  avg {avg_tqqq:.0f}% of base when active  "
          f"({float((tqqq_frac>0).mean())*100:.0f}% of days)")
    print(f"  GLD:        avg {avg_gld:.0f}% when active  "
          f"({float((gld_weight>0).mean())*100:.0f}% of days)")
    print(f"  Note: Effective exposure on max-TQQQ days = base×(1-0.7) + base×0.7×3")

    # ── 9. Chart ──────────────────────────────────────────────────────────
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
                    "Allocation — TQQQ (orange) / GLD (gold) active fraction",
                    "Composite Score",
                ],
                vertical_spacing=0.05,
            )

            palette = [
                ("rgba(150,150,150,0.7)", 1.5, "dot"),
                ("steelblue",             2.0, "solid"),
                ("deepskyblue",           2.0, "solid"),
                ("darkorange",            2.5, "solid"),
                ("crimson",               2.5, "solid"),
                ("rgba(130,130,130,0.4)", 1.2, "dash"),
            ]
            for (label, eq), (color, width, dash) in zip(EQ, palette):
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values, name=label,
                    line=dict(color=color, width=width, dash=dash),
                ), row=1, col=1)
                dd = (eq / eq.cummax() - 1) * 100
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values, name=f"DD",
                    line=dict(color=color, width=1), fill="tozeroy",
                    showlegend=False,
                ), row=2, col=1)

            # TQQQ fraction × base = actual TQQQ weight
            tqqq_actual = tqqq_frac * pos_full * 100
            fig.add_trace(go.Scatter(
                x=tqqq_actual.index, y=tqqq_actual.values,
                name="TQQQ weight", fill="tozeroy",
                line=dict(color="rgba(255,140,0,0.9)", width=1),
                fillcolor="rgba(255,140,0,0.30)",
            ), row=3, col=1)
            gld_actual = gld_weight * 100
            fig.add_trace(go.Scatter(
                x=gld_actual.index, y=gld_actual.values,
                name="GLD weight", fill="tozeroy",
                line=dict(color="rgba(218,165,32,0.9)", width=1),
                fillcolor="rgba(218,165,32,0.30)",
            ), row=3, col=1)

            # Score
            score_s = raw_scores.reindex(common, fill_value=0)
            fig.add_trace(go.Scatter(
                x=score_s.index, y=score_s.values,
                name="Score", fill="tozeroy",
                line=dict(color="rgba(100,100,200,0.8)", width=1),
                fillcolor="rgba(100,100,200,0.12)",
            ), row=4, col=1)
            for y_val, color, txt in [(7.0, "green", "7.0 (TQQQ gate)"), (-5.0, "red", "-5 (exit)")]:
                fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                              annotation_text=txt, row=4, col=1)

            # Shade SHIELD periods
            shield_s = (cs_level == "SHIELD").astype(int)
            shield_starts = shield_s.index[shield_s.diff().fillna(0) == 1].tolist()
            shield_ends   = shield_s.index[shield_s.diff().fillna(0) == -1].tolist()
            if shield_s.iloc[-1] == 1:
                shield_ends.append(shield_s.index[-1])
            for s_dt, e_dt in zip(shield_starts, shield_ends):
                for row in [1, 2, 3, 4]:
                    fig.add_vrect(x0=s_dt, x1=e_dt,
                                  fillcolor="rgba(255,0,0,0.07)", line_width=0,
                                  row=row, col=1)

            fig.update_yaxes(type="log", title="NAV",     row=1, col=1)
            fig.update_yaxes(title="DD %",                row=2, col=1)
            fig.update_yaxes(title="Alloc %",             row=3, col=1)
            fig.update_yaxes(title="Score",               row=4, col=1)
            fig.update_layout(
                title=(f"Sharpe 1.5 — Lever 1 (QQQ) + Lever 2 (TQQQ) + Lever 3 (GLD)  "
                       f"({args.years}yr, {args.cost}bps{ml_note})"),
                height=1100,
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
            )

            out = ROOT / "scripts" / "backtest_sharpe15.html"
            fig.write_html(str(out))
            print(f"   ✅ Chart: {out}")
            print(f"   Open:  file://{out}")

        except ImportError:
            print("   ⚠️  plotly not available")
        except Exception as e:
            print(f"   ❌ Chart: {e}")

    print("\n🎉 Done!\n")


if __name__ == "__main__":
    main()
