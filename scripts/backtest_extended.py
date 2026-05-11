"""
Extended 20-Year Backtest: 3-Layer Strategy across Indices + Watchlist Stocks

For each instrument, compares:
  ① Buy & Hold
  ② Full 3-Layer strategy (Composite Signal + Crash Shield + ML + Dual Momentum)

Market universe (with 20yr data from 2005):
  Indices:  SPY  QQQ  DIA  IWM
  Tech:     AAPL MSFT GOOGL AMZN NVDA META
  Other:    JPM  XOM  GLD  TLT  TSLA

Key crash periods tested:
  - 2008 Financial Crisis  (SPY -56%)
  - 2018 Q4 selloff        (SPY -19%)
  - 2020 COVID crash       (SPY -34%)
  - 2022 Bear market       (SPY -25%)

Usage
─────
  python3 scripts/backtest_extended.py
  python3 scripts/backtest_extended.py --years 20    # full 20yr (default)
  python3 scripts/backtest_extended.py --years 10    # quick test
  python3 scripts/backtest_extended.py --skip-ml     # skip ML layer
  python3 scripts/backtest_extended.py --symbols SPY,QQQ,NVDA  # specific symbols
"""

from __future__ import annotations

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── Universe definition ────────────────────────────────────────────────────────
UNIVERSE: dict[str, str] = {
    # 宽基指数 ETF
    "SPY":   "标普500",
    "QQQ":   "纳斯达克100",
    "DIA":   "道琼斯30",
    "IWM":   "罗素2000",
    # 科技巨头（都有20年以上数据）
    "AAPL":  "苹果",
    "MSFT":  "微软",
    "GOOGL": "谷歌",    # IPO 2004
    "AMZN":  "亚马逊",
    "META":  "Meta",    # IPO 2012 — shorter history
    "NVDA":  "英伟达",
    # 金融 / 能源 / 避险
    "JPM":   "摩根大通",
    "XOM":   "埃克森美孚",
    "GLD":   "黄金ETF",  # since 2004
    "TLT":   "20年美债",
    # 高波动
    "TSLA":  "特斯拉",   # IPO 2010
}

# Crash periods: (label, start, end)
CRASH_PERIODS = [
    ("2008金融危机",    "2007-10-09", "2009-03-09"),
    ("2018 Q4杀跌",    "2018-09-28", "2018-12-24"),
    ("2020 COVID暴跌", "2020-02-19", "2020-03-23"),
    ("2022熊市",       "2021-12-31", "2022-10-12"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers (reused from backtest_3layer)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(ticker: str, start: str = "2005-01-01") -> pd.DataFrame | None:
    import yfinance as yf
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def _composite_signal_series(df: pd.DataFrame, vix_df: pd.DataFrame | None,
                               symbol: str) -> pd.Series:
    """Vectorized composite signal for a single symbol."""
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS, COMPOSITE_BUY_THRESHOLD
    result = composite_score(
        df, symbol=symbol, vix_df=vix_df,
        buy_threshold=COMPOSITE_BUY_THRESHOLD.get("us", 6.0),
        sell_threshold=3.0,
        weights=STRATEGY_WEIGHTS,
    )
    return result["signal_series"].reindex(df.index, fill_value=0)


def _crash_shield_mult(spy_df: pd.DataFrame,
                        vix_df: pd.DataFrame | None,
                        hyg_df: pd.DataFrame | None) -> pd.Series:
    """Vectorized Crash Shield multiplier (SPY-based, applies to all stocks)."""
    spy = spy_df["Close"].dropna()
    idx = spy.index

    # VIX panic
    if vix_df is not None and not vix_df.empty:
        vix = vix_df["Close"].dropna().reindex(idx, method="ffill").fillna(20)
        sig1 = ((vix > 25) & (vix.pct_change(5).fillna(0) > 0.20)).astype(int)
    else:
        sig1 = pd.Series(0, index=idx)

    # Trend break
    ma50 = spy.rolling(50, min_periods=25).mean()
    sig2 = ((spy < ma50) & (ma50 < ma50.shift(10))).astype(int)

    # Sharp decline
    sig3 = (spy.pct_change(20) < -0.08).astype(int)

    # Credit spread
    if hyg_df is not None and not hyg_df.empty:
        hyg = hyg_df["Close"].dropna().reindex(idx, method="ffill")
        sig4 = (hyg.pct_change(20) < -0.03).astype(int)
    else:
        sig4 = (spy.pct_change(60) < -0.15).astype(int)

    score = (sig1 + sig2 + sig3 + sig4).fillna(0).astype(int)
    mult  = pd.Series(1.0, index=idx)
    mult[score == 2] = 0.5
    mult[score >= 3] = 0.0
    return mult


def _ml_mult_series(spy_df: pd.DataFrame,
                    vix_df: pd.DataFrame | None,
                    hyg_df: pd.DataFrame | None) -> pd.Series:
    """Walk-forward OOS ML probabilities → position multipliers."""
    from strategies.ml_regime import (
        build_features, FEATURE_COLS, _get_model, _fill_missing, MLRegimeClassifier
    )
    from sklearn.preprocessing import StandardScaler

    df = build_features(spy_df, vix_df, hyg_df, include_target=True)
    df = df.dropna(subset=["y"])
    prob_series = pd.Series(0.5, index=df.index)

    windows = [
        ("2005-01-01", "2012-12-31", "2013-01-01", "2015-12-31"),
        ("2005-01-01", "2015-12-31", "2016-01-01", "2018-12-31"),
        ("2005-01-01", "2018-12-31", "2019-01-01", "2021-12-31"),
        ("2005-01-01", "2021-12-31", "2022-01-01", "2099-12-31"),
    ]
    for train_start, train_end, test_start, test_end in windows:
        try:
            train = df.loc[train_start:train_end].dropna()
            test  = df.loc[test_start:test_end].dropna()
            if len(train) < 400 or len(test) < 30:
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
            print(f"   ✅ OOS {test_start[:7]}–{test_end[:7]}: "
                  f"n={len(test):,}  mean_prob={probs.mean():.3f}")
        except Exception as e:
            print(f"   ⚠️  Window {test_start[:7]} failed: {e}")

    return prob_series.apply(MLRegimeClassifier.to_position_multiplier)


def _dm_scale(spy_df: pd.DataFrame) -> pd.Series:
    """Dual momentum scale (absolute momentum + crash protection), SPY-based."""
    spy = spy_df["Close"].dropna()
    abs_ok = (spy.pct_change(252) > 0).reindex(spy.index, fill_value=False)
    log_ret = np.log(spy / spy.shift(1)).dropna()
    vol_21  = log_ret.rolling(21).std()  * np.sqrt(252)
    vol_126 = log_ret.rolling(126).std() * np.sqrt(252)
    crash   = ((vol_21 / (vol_126 + 1e-9)) >= 2.0).reindex(spy.index, fill_value=False)

    scale = pd.Series(1.0, index=spy.index)
    scale[~abs_ok] = 0.0
    scale[crash & abs_ok] = 0.5
    return scale


def _sim(returns: pd.Series, pos_frac: pd.Series,
          cost_bps: float = 10.0) -> pd.Series:
    """Simulate equity NAV."""
    pf  = pos_frac.reindex(returns.index).fillna(0.0)
    nav = [1.0]
    cur = 0.0
    for i in range(1, len(returns)):
        tgt  = float(pf.iloc[i - 1])
        cost = abs(tgt - cur) * cost_bps / 10_000
        nav.append(nav[-1] * (1.0 + tgt * float(returns.iloc[i]) - cost))
        cur = tgt
    return pd.Series(nav, index=returns.index)


def _stats(eq: pd.Series) -> dict:
    ret     = eq.pct_change().dropna()
    n_years = len(ret) / 252
    total   = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr    = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 0.1)) - 1)
    sharpe  = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252))
    max_dd  = float((eq / eq.cummax() - 1).min())
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0.0
    return dict(total=total, cagr=cagr, sharpe=sharpe, max_dd=max_dd, calmar=calmar)


def _crash_ret(eq: pd.Series, start: str, end: str) -> float | None:
    try:
        s = eq.loc[start:end]
        if len(s) < 2:
            return None
        return float(s.iloc[-1] / s.iloc[0] - 1)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",    type=int,   default=20,      help="History in years")
    parser.add_argument("--start",    type=str,   default="2005-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--skip-ml",  action="store_true", help="Skip ML layer")
    parser.add_argument("--cost",     type=float, default=10.0,    help="One-way cost bps")
    parser.add_argument("--symbols",  type=str,   default="",      help="Comma-separated override")
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    start_date = args.start
    symbols_override = [s.strip() for s in args.symbols.split(",") if s.strip()]
    universe = (
        {s: UNIVERSE.get(s, s) for s in symbols_override}
        if symbols_override else UNIVERSE
    )

    print(f"\n{'='*70}")
    print(f"  Extended 20-Year Backtest  (from {start_date}, cost={args.cost}bps)")
    print(f"  Universe: {', '.join(universe.keys())}")
    print(f"{'='*70}")

    # ── 1. Fetch macro data (SPY / VIX / HYG) ─────────────────────────────
    print("\n📥 Fetching macro data (SPY / VIX / HYG)...")
    spy_df  = _fetch("SPY",  start=start_date)
    vix_df  = _fetch("^VIX", start=start_date)
    hyg_df  = _fetch("HYG",  start=start_date)
    if spy_df is None:
        print("❌ SPY data unavailable — abort"); return
    print(f"   SPY: {len(spy_df):,} days  ({spy_df.index[0].date()} – {spy_df.index[-1].date()})")

    # ── 2. Pre-compute market-wide signals (SPY-based, apply to all stocks) ─
    print("\n🛡️  Computing Crash Shield...")
    cs_mult = _crash_shield_mult(spy_df, vix_df, hyg_df)
    shield_d = int((cs_mult == 0.0).sum())
    caution_d= int((cs_mult == 0.5).sum())
    print(f"   SHIELD: {shield_d} days  |  CAUTION: {caution_d} days")

    print("\n📈 Computing Dual Momentum (SPY-based)...")
    dm_scale_s = _dm_scale(spy_df)
    abs_blocked = int((dm_scale_s == 0.0).sum())
    print(f"   Absolute momentum blocked: {abs_blocked} days")

    ml_mult_s = pd.Series(1.0, index=spy_df.index)   # default: no ML effect
    if not args.skip_ml:
        print("\n🤖 Computing ML Regime (walk-forward, 4 OOS windows)...")
        try:
            ml_mult_s = _ml_mult_series(spy_df, vix_df, hyg_df)
            ml_mult_s = ml_mult_s.reindex(spy_df.index, fill_value=1.0)
            print(f"   ML multiplier mean={ml_mult_s.mean():.3f}  "
                  f"min={ml_mult_s.min():.3f}  max={ml_mult_s.max():.3f}")
        except Exception as e:
            print(f"   ⚠️  ML failed: {e} — using neutral multiplier")

    # Combined market-level multiplier
    combined_mkt = (cs_mult * ml_mult_s * dm_scale_s).reindex(spy_df.index, fill_value=1.0)

    # ── 3. Per-symbol backtest ─────────────────────────────────────────────
    results: dict[str, dict] = {}

    for sym, label in universe.items():
        print(f"\n  [{sym}] {label}...")
        df = _fetch(sym, start=start_date) if sym != "SPY" else spy_df
        if df is None or len(df) < 300:
            print(f"   ⚠️  Insufficient data — skip")
            continue

        # Align to SPY index intersection
        common = df.index.intersection(spy_df.index)
        df_c  = df.reindex(common)
        ret   = df_c["Close"].pct_change()

        # Composite signal
        try:
            sig = _composite_signal_series(df_c, vix_df, symbol=sym)
        except Exception as e:
            print(f"   ⚠️  Signal error: {e} — skip")
            continue

        # Build position series: hold while signal >= 0, exit on -1
        in_pos = pd.Series(False, index=common)
        pos = False
        for dt in common:
            s = int(sig.get(dt, 0))
            if s == 1:   pos = True
            elif s == -1: pos = False
            in_pos[dt] = pos
        pos_composite = in_pos.astype(float)

        # Apply combined market-level multiplier
        mkt_mult = combined_mkt.reindex(common, fill_value=1.0)
        pos_full  = (pos_composite * mkt_mult).clip(0, 1)

        # Simulate
        bah   = (df_c["Close"] / df_c["Close"].iloc[0])
        eq_bh = pd.Series(bah.values, index=common)
        eq_s  = _sim(ret, pos_composite, cost_bps=args.cost)
        eq_f  = _sim(ret, pos_full,      cost_bps=args.cost)

        days_in   = int(pos_composite.sum())
        pct_in    = days_in / max(len(common), 1) * 100
        print(f"   Days: {len(common):,}  |  Invested: {days_in:,} ({pct_in:.0f}%)")

        results[sym] = {
            "label":   label,
            "eq_bh":   eq_bh,
            "eq_strat": eq_s,
            "eq_full":  eq_f,
            "stats_bh":    _stats(eq_bh),
            "stats_strat": _stats(eq_s),
            "stats_full":  _stats(eq_f),
            "n_days":  len(common),
            "pct_in":  pct_in,
        }

    if not results:
        print("\n❌ No results — check data availability"); return

    # ── 4. Print performance table ─────────────────────────────────────────
    print(f"\n\n{'═'*92}")
    print(f"  Performance Summary — B&H vs Full 3-Layer Strategy  (from {start_date})")
    print(f"{'═'*92}")
    print(f"  {'Symbol':<8} {'Name':<12}  "
          f"{'BH Total':>9} {'BH CAGR':>8} {'BH MaxDD':>9} │ "
          f"{'S Total':>9} {'S CAGR':>8} {'S MaxDD':>9} {'S Sharpe':>9} {'S Calmar':>8}")
    print(f"  {'─'*8} {'─'*12}  {'─'*9} {'─'*8} {'─'*9} ┼ "
          f"{'─'*9} {'─'*8} {'─'*9} {'─'*9} {'─'*8}")

    for sym, r in results.items():
        bh = r["stats_bh"]
        sf = r["stats_full"]
        # Max DD improvement indicator
        dd_imp = sf["max_dd"] - bh["max_dd"]
        dd_mark = "✅" if dd_imp > 0.05 else ("⚡" if dd_imp > 0 else "")
        print(
            f"  {sym:<8} {r['label'][:12]:<12}  "
            f"{bh['total']*100:>+8.0f}% {bh['cagr']*100:>7.1f}% {bh['max_dd']*100:>8.1f}% │ "
            f"{sf['total']*100:>+8.0f}% {sf['cagr']*100:>7.1f}% {sf['max_dd']*100:>8.1f}% "
            f"{sf['sharpe']:>9.2f} {sf['calmar']:>8.2f}  {dd_mark}"
        )

    print(f"\n  ✅ = MaxDD reduced by >5%  ⚡ = MaxDD slightly improved")
    print(f"  Cost: {args.cost}bps/trade | ML: {'yes' if not args.skip_ml else 'skip'}")

    # ── 5. Crash period table ──────────────────────────────────────────────
    print(f"\n\n{'═'*92}")
    print(f"  Crisis Period Protection Analysis")
    print(f"{'═'*92}")
    header = f"  {'Symbol':<8}"
    for name, *_ in CRASH_PERIODS:
        header += f"  {name[:10]:>12}BH  {name[:10]:>12}3L"
    print(header[:160])
    print("  " + "─" * 88)

    for sym, r in results.items():
        row = f"  {sym:<8}"
        for name, cs, ce in CRASH_PERIODS:
            bh_r = _crash_ret(r["eq_bh"],    cs, ce)
            fl_r = _crash_ret(r["eq_full"],   cs, ce)
            if bh_r is None:
                row += f"  {'N/A':>14}  {'N/A':>14}"
            else:
                bh_str = f"{bh_r*100:+.0f}%"
                fl_str = f"{fl_r*100:+.0f}%" if fl_r is not None else "N/A"
                protect = "🛡" if fl_r is not None and fl_r > bh_r + 0.05 else " "
                row += f"  {bh_str:>14}  {fl_str:>13}{protect}"
        print(row)

    print(f"\n  🛡 = Strategy reduced loss by >5% vs B&H during crash period")

    # ── 6. Compact crash summary ───────────────────────────────────────────
    print(f"\n\n{'═'*92}")
    print(f"  2008 Financial Crisis Highlight (worst drawdown in 20yr)")
    print(f"{'═'*92}")
    cs_2008, ce_2008 = "2007-10-09", "2009-03-09"
    print(f"  {'Symbol':<8}  {'B&H':>8}  {'3-Layer':>8}  {'Saved':>8}  Comment")
    print("  " + "─" * 65)
    for sym, r in results.items():
        bh_r = _crash_ret(r["eq_bh"],  cs_2008, ce_2008)
        fl_r = _crash_ret(r["eq_full"], cs_2008, ce_2008)
        if bh_r is None or fl_r is None:
            print(f"  {sym:<8}  {'N/A':>8}  {'N/A':>8}  {'—':>8}  (no 2008 data)")
            continue
        saved = fl_r - bh_r
        emoji = "✅✅" if saved > 0.20 else ("✅" if saved > 0.10 else "⚡")
        print(f"  {sym:<8}  {bh_r*100:>+7.1f}%  {fl_r*100:>+7.1f}%  {saved*100:>+7.1f}%  {emoji}")

    # ── 7. Chart ──────────────────────────────────────────────────────────
    if not args.no_plot:
        _make_chart(results, start_date)


def _make_chart(results: dict, start_date: str):
    print("\n\n📊 Generating charts...")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # ── Chart 1: Index ETFs ────────────────────────────────────────────
        index_syms = [s for s in ["SPY", "QQQ", "DIA", "IWM"] if s in results]
        if index_syms:
            fig1 = make_subplots(
                rows=2, cols=2 if len(index_syms) > 2 else 1,
                subplot_titles=[f"{s} — {results[s]['label']}" for s in index_syms],
                vertical_spacing=0.12, horizontal_spacing=0.08,
            )
            positions = [(1,1),(1,2),(2,1),(2,2)]
            for i, sym in enumerate(index_syms):
                r   = results[sym]
                row = positions[i][0]
                col = positions[i][1]
                fig1.add_trace(go.Scatter(
                    x=r["eq_bh"].index, y=r["eq_bh"].values,
                    name=f"{sym} B&H", line=dict(color="gray", width=1.5, dash="dot"),
                ), row=row, col=col)
                fig1.add_trace(go.Scatter(
                    x=r["eq_full"].index, y=r["eq_full"].values,
                    name=f"{sym} 3-Layer", line=dict(color="crimson", width=2.0),
                ), row=row, col=col)

            fig1.update_yaxes(type="log")
            fig1.update_layout(
                title=f"指数ETF — B&H vs 3-Layer策略 (from {start_date})",
                height=700, showlegend=True,
                hovermode="x unified",
            )
            path1 = ROOT / "scripts" / "backtest_indices.html"
            fig1.write_html(str(path1))
            print(f"   ✅ Index chart: file://{path1}")

        # ── Chart 2: Individual stocks ─────────────────────────────────────
        stock_syms = [s for s in results if s not in ["SPY","QQQ","DIA","IWM","GLD","TLT"]]
        if stock_syms:
            n = len(stock_syms)
            cols = 3
            rows_n = (n + cols - 1) // cols
            fig2 = make_subplots(
                rows=rows_n, cols=cols,
                subplot_titles=[f"{s} {results[s]['label']}" for s in stock_syms],
                vertical_spacing=0.10, horizontal_spacing=0.06,
            )
            for i, sym in enumerate(stock_syms):
                r   = results[sym]
                row = i // cols + 1
                col = i  % cols + 1
                fig2.add_trace(go.Scatter(
                    x=r["eq_bh"].index, y=r["eq_bh"].values,
                    name=f"{sym} B&H", line=dict(color="rgba(150,150,150,0.6)", width=1.2),
                    legendgroup=sym, showlegend=(i == 0),
                ), row=row, col=col)
                fig2.add_trace(go.Scatter(
                    x=r["eq_full"].index, y=r["eq_full"].values,
                    name=f"{sym} 3L", line=dict(color="steelblue", width=1.8),
                    legendgroup=sym, showlegend=(i == 0),
                ), row=row, col=col)

            fig2.update_yaxes(type="log")
            fig2.update_layout(
                title=f"个股 — B&H vs 3-Layer策略 (from {start_date})",
                height=max(500, rows_n * 280),
                hovermode="x unified",
            )
            path2 = ROOT / "scripts" / "backtest_stocks.html"
            fig2.write_html(str(path2))
            print(f"   ✅ Stocks chart: file://{path2}")

        # ── Chart 3: Crash period comparison (SPY, QQQ, NVDA during 2008) ──
        crisis_syms = ["SPY", "QQQ", "NVDA", "AAPL", "JPM", "XOM"]
        crisis_syms = [s for s in crisis_syms if s in results]
        fig3 = make_subplots(
            rows=len(CRASH_PERIODS), cols=1,
            subplot_titles=[f"崩溃期 {n}" for n, *_ in CRASH_PERIODS],
            shared_xaxes=False, vertical_spacing=0.08,
        )
        row_colors = ["steelblue","orange","mediumpurple","crimson"]
        for ri, (name, cs, ce) in enumerate(CRASH_PERIODS, 1):
            plotted = 0
            for sym in crisis_syms:
                if sym not in results:
                    continue
                r   = results[sym]
                try:
                    seg_bh = r["eq_bh"].loc[cs:ce]
                    seg_fl = r["eq_full"].loc[cs:ce]
                    if len(seg_bh) < 5:
                        continue
                    # Normalize to 100 at start of period
                    norm_bh = seg_bh / seg_bh.iloc[0] * 100
                    norm_fl = seg_fl / seg_fl.iloc[0] * 100
                    c = row_colors[plotted % len(row_colors)]
                    fig3.add_trace(go.Scatter(
                        x=norm_bh.index, y=norm_bh.values,
                        name=f"{sym} B&H",
                        line=dict(color=c, width=1.2, dash="dot"),
                        legendgroup=f"{sym}",
                        showlegend=(ri == 1),
                    ), row=ri, col=1)
                    fig3.add_trace(go.Scatter(
                        x=norm_fl.index, y=norm_fl.values,
                        name=f"{sym} 3L",
                        line=dict(color=c, width=2.0),
                        legendgroup=f"{sym}",
                        showlegend=(ri == 1),
                    ), row=ri, col=1)
                    plotted += 1
                except Exception:
                    pass

        fig3.update_layout(
            title="各次危机中 B&H vs 3-Layer保护效果对比 (基期=100)",
            height=1000, hovermode="x unified",
        )
        path3 = ROOT / "scripts" / "backtest_crashes.html"
        fig3.write_html(str(path3))
        print(f"   ✅ Crash chart: file://{path3}")

    except ImportError:
        print("   ⚠️  plotly not available — skip charts")
    except Exception as e:
        print(f"   ❌ Chart error: {e}")

    print("\n🎉 Extended backtest complete!\n")


if __name__ == "__main__":
    main()
