"""
15-Year Backtest: 3-Layer Enhanced Strategy vs Buy & Hold

Simulates and compares 5 equity curves on SPY (2010 – present):
  0. Buy & Hold SPY               (benchmark)
  1. Composite Signal only        (9 strategies, binary in/out)
  2. + Crash Shield overlay       (reduce/block positions in macro stress)
  3. + ML Regime multiplier       (walk-forward XGBoost prob → sizing)
  4. Full 3-Layer (+ Dual Momentum, absolute + crash protection)

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

def compute_composite_signal(spy_df: pd.DataFrame, vix_df: pd.DataFrame | None) -> pd.Series:
    """
    Compute vectorized composite buy/sell signal for SPY.
    Returns pd.Series[int]: 1=BUY, 0=HOLD, -1=SELL
    """
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS, COMPOSITE_BUY_THRESHOLD

    result = composite_score(
        spy_df,
        symbol="SPY",
        vix_df=vix_df,
        buy_threshold=COMPOSITE_BUY_THRESHOLD.get("us", 6.0),
        sell_threshold=3.0,
        weights=STRATEGY_WEIGHTS,
    )
    return result["signal_series"].reindex(spy_df.index, fill_value=0)


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
) -> pd.Series:
    """
    Walk-forward OOS ML probability predictions (no lookahead bias).
    Uses 4 consecutive OOS windows covering 2015–present.
    Dates before first OOS window default to 0.5 (neutral).
    """
    from strategies.ml_regime import build_features, FEATURE_COLS, _get_model, _fill_missing
    from sklearn.preprocessing import StandardScaler

    df = build_features(spy_df, vix_df, hyg_df, include_target=True)
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

    print(f"\n{'='*62}")
    print(f"  3-Layer Strategy Backtest  ({args.years}yr, cost={args.cost}bps/trade)")
    print(f"{'='*62}")

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

    spy_ret = spy_df["Close"].pct_change()
    common_idx = spy_df.index

    # ── 2. Composite signal ────────────────────────────────────────────────
    print("\n📊 Computing composite strategy signals (9 strategies)...")
    comp_sig = compute_composite_signal(spy_df, vix_df)
    # Convert to position: 1→long, 0/−1→flat (simple binary)
    # We hold as long as signal ≥ 0, exit only on -1
    in_pos = pd.Series(False, index=common_idx)
    pos = False
    for dt in common_idx:
        s = int(comp_sig.get(dt, 0))
        if s == 1:
            pos = True
        elif s == -1:
            pos = False
        in_pos[dt] = pos
    pos_composite = in_pos.astype(float)

    n_in  = int(pos_composite.sum())
    n_out = int((~in_pos).sum())
    buys  = int((comp_sig == 1).sum())
    sells = int((comp_sig == -1).sum())
    pct_invested = n_in / len(pos_composite) * 100
    print(f"   BUY signals: {buys}  |  SELL signals: {sells}  "
          f"|  Days invested: {n_in:,} ({pct_invested:.0f}%)")

    # ── 3. Crash Shield ────────────────────────────────────────────────────
    print("\n🛡️  Computing Crash Shield (vectorized)...")
    cs = compute_crash_shield(spy_df, vix_df, hyg_df)
    shield_days   = int((cs["level"] == "SHIELD").sum())
    caution_days  = int((cs["level"] == "CAUTION").sum())
    print(f"   SHIELD days: {shield_days:,}  |  CAUTION days: {caution_days:,}  "
          f"|  NONE days: {len(cs) - shield_days - caution_days:,}")

    # Crash Shield position: block buys on SHIELD days, halve on CAUTION
    cs_mult = cs["multiplier"].reindex(common_idx, fill_value=1.0)
    pos_cs = (pos_composite * cs_mult).clip(0, 1)

    # ── 4. ML Regime ───────────────────────────────────────────────────────
    pos_ml = pos_cs.copy()   # default: same as crash shield if ML skipped

    if not args.skip_ml:
        print("\n🤖 Computing ML Regime walk-forward predictions...")
        print("   (Training XGBoost on ~3,750 rows × 4 windows — ~1-2 min)\n")
        try:
            ml_probs = compute_ml_probs(spy_df, vix_df, hyg_df)
            from strategies.ml_regime import MLRegimeClassifier
            ml_mult_series = ml_probs.apply(MLRegimeClassifier.to_position_multiplier)
            ml_mult_series = ml_mult_series.reindex(common_idx, fill_value=1.0)

            # ML multiplier applied on top of crash shield layer
            pos_ml = (pos_cs * ml_mult_series).clip(0, 1)

            bull_pct = (ml_probs > 0.55).mean() * 100
            bear_pct = (ml_probs < 0.45).mean() * 100
            print(f"\n   ML prob stats: mean={ml_probs.mean():.3f}  "
                  f"bull={bull_pct:.0f}%  bear={bear_pct:.0f}%")
        except Exception as e:
            print(f"\n   ❌ ML layer failed: {e}")
            print("   → Falling back to Crash Shield only for Layer 2+3")
            args.skip_ml = True

    # ── 5. Dual Momentum ──────────────────────────────────────────────────
    print("\n📈 Computing Dual Momentum...")
    dm = compute_dual_momentum(spy_df)
    dm_scale = dm["scale"].reindex(common_idx, fill_value=1.0)
    abs_neg_days  = int((~dm["abs_ok"]).sum())
    crash_pr_days = int(dm["crash_protect"].sum())
    print(f"   Absolute momentum blocked: {abs_neg_days:,} days  "
          f"|  Crash protect active: {crash_pr_days:,} days")

    # Full 3-layer position
    pos_3layer = (pos_ml * dm_scale).clip(0, 1)

    # ── 6. Simulate equity curves ─────────────────────────────────────────
    print("\n📈 Simulating equity curves...")
    bah    = (spy_df["Close"] / spy_df["Close"].iloc[0]).rename("Buy & Hold SPY")
    eq_comp   = simulate_equity(spy_ret, pos_composite, cost_bps=args.cost, label="1. Composite only")
    eq_cs     = simulate_equity(spy_ret, pos_cs,        cost_bps=args.cost, label="2. + Crash Shield")
    eq_ml     = simulate_equity(spy_ret, pos_ml,        cost_bps=args.cost, label="3. + ML Regime")
    eq_full   = simulate_equity(spy_ret, pos_3layer,    cost_bps=args.cost, label="4. Full 3-Layer")

    # ── 7. Performance table ───────────────────────────────────────────────
    bah_eq = pd.Series(bah.values, index=bah.index)

    print(f"\n{'─'*78}")
    print(f"{'Strategy':<28} {'Total Ret':>10} {'CAGR':>8} {'Sharpe':>8} "
          f"{'Max DD':>10} {'Calmar':>8}")
    print(f"{'─'*78}")

    for label, equity in [
        ("Buy & Hold SPY", bah_eq),
        ("1. Composite only", eq_comp),
        ("2. + Crash Shield", eq_cs),
        ("3. + ML Regime" + (" (=2)" if args.skip_ml else ""), eq_ml),
        ("4. Full 3-Layer", eq_full),
    ]:
        s = performance_stats(equity, benchmark=bah_eq if label != "Buy & Hold SPY" else None)
        row = (f"{label:<28} {s['Total Return']:>10} {s['CAGR']:>8} "
               f"{s['Sharpe']:>8} {s['Max Drawdown']:>10} {s['Calmar']:>8}")
        if label != "Buy & Hold SPY":
            row += f"  IR={s.get('Info Ratio','—'):>5}"
        print(row)

    print(f"{'─'*78}")
    print(f"\n  ✅ Transaction cost: {args.cost} bps one-way | Rebalance: daily")
    print(f"  ✅ No lookahead bias | ML uses walk-forward OOS predictions only")

    # ── 8. Key crisis periods ─────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("  Crash period analysis:")
    print(f"{'─'*78}")
    crisis_periods = [
        ("2020 COVID crash",   "2020-01-17", "2020-03-23"),
        ("2022 Bear market",   "2021-12-31", "2022-10-12"),
        ("2018 Q4 selloff",    "2018-09-28", "2018-12-24"),
    ]
    for name, start, end in crisis_periods:
        try:
            rows = []
            for lbl, eq in [("B&H", bah_eq), ("Composite", eq_comp),
                             ("CrashShield", eq_cs), ("Full 3L", eq_full)]:
                s = eq.loc[start:end]
                if len(s) > 1:
                    dd_pct = (s.iloc[-1] / s.iloc[0] - 1) * 100
                    rows.append(f"{lbl}: {dd_pct:+.1f}%")
            print(f"  {name:<22} " + "  ".join(rows))
        except Exception:
            pass
    print(f"{'─'*78}")

    # ── 9. Chart ──────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n📊 Generating chart...")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.2, 0.2],
                shared_xaxes=True,
                subplot_titles=["Equity Curves (log scale)", "Drawdown", "Crash Shield Score"],
                vertical_spacing=0.06,
            )

            colors = {
                "Buy & Hold SPY":    "rgba(180,180,180,0.8)",
                "1. Composite only": "steelblue",
                "2. + Crash Shield": "orange",
                "3. + ML Regime":    "mediumpurple",
                "4. Full 3-Layer":   "crimson",
            }
            widths = {
                "Buy & Hold SPY": 1.5,
                "1. Composite only": 1.5,
                "2. + Crash Shield": 1.5,
                "3. + ML Regime":    1.5,
                "4. Full 3-Layer":   2.5,
            }

            for lbl, eq in [
                ("Buy & Hold SPY", bah_eq),
                ("1. Composite only", eq_comp),
                ("2. + Crash Shield", eq_cs),
                ("3. + ML Regime" + (" (=2)" if args.skip_ml else ""), eq_ml),
                ("4. Full 3-Layer", eq_full),
            ]:
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values,
                    name=lbl,
                    line=dict(color=colors.get(lbl, "gray"),
                              width=widths.get(lbl, 1.5)),
                ), row=1, col=1)

            # Drawdown for full 3-layer and B&H
            for lbl, eq, color in [
                ("B&H DD",     bah_eq,  "rgba(180,180,180,0.5)"),
                ("3-Layer DD", eq_full, "rgba(220,20,60,0.6)"),
            ]:
                roll_max = eq.cummax()
                dd = (eq / roll_max - 1) * 100
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values,
                    name=lbl, fill="tozeroy",
                    line=dict(color=color, width=1),
                    fillcolor=color,
                ), row=2, col=1)

            # Crash shield score
            cs_score = cs["score"].reindex(common_idx, fill_value=0)
            fig.add_trace(go.Scatter(
                x=cs_score.index, y=cs_score.values,
                name="CS Score", fill="tozeroy",
                line=dict(color="rgba(255,165,0,0.8)", width=1),
                fillcolor="rgba(255,165,0,0.3)",
            ), row=3, col=1)
            # Threshold line at 3 (SHIELD)
            fig.add_hline(y=3, line_dash="dash", line_color="red",
                          annotation_text="SHIELD", row=3, col=1)
            fig.add_hline(y=2, line_dash="dot", line_color="orange",
                          annotation_text="CAUTION", row=3, col=1)

            fig.update_yaxes(type="log", title="NAV (log)", row=1, col=1)
            fig.update_yaxes(title="Drawdown %", row=2, col=1)
            fig.update_yaxes(title="Score", range=[0, 4.5], row=3, col=1)
            fig.update_layout(
                title=f"3-Layer Strategy Backtest  ({args.years}yr, SPY universe)",
                height=900,
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
