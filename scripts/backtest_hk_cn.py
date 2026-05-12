"""
HK & CN Backtest  (2012-01-01 – today)

Compares per symbol:
  ① Buy & Hold
  ② Old Composite  (binary: HK≥7.0 / CN≥6.0,  SELL ≤ -3)
  ③ New Composite  (graduated ≥2.5→25% … ≥7.5→100%, SELL ≤ -5, regime gate)

Note: No 3-Layer (Crash Shield / ML / Dual Momentum) — those are SPY-based US-only.
      HK/CN protection comes from composite score + regime gate only.

Usage
─────
  python3 scripts/backtest_hk_cn.py             # all symbols
  python3 scripts/backtest_hk_cn.py --market hk # HK only
  python3 scripts/backtest_hk_cn.py --market cn # CN only
"""

from __future__ import annotations
import argparse, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Universe
# ══════════════════════════════════════════════════════════════════════════════

HK_UNIVERSE: dict[str, str] = {
    # ETF
    "2800.HK":  "盈富基金(HSI)",
    "3032.HK":  "华夏恒生科技",
    "3067.HK":  "iShares恒科ETF",
    "3188.HK":  "华夏纳斯达克ETF",
    # 科技互联网
    "0700.HK":  "腾讯",
    "9988.HK":  "阿里巴巴",
    "3690.HK":  "美团",
    "9618.HK":  "京东",
    "9999.HK":  "网易",
    "1024.HK":  "快手",
    # 半导体/硬件
    "0981.HK":  "中芯国际",
    "2382.HK":  "舜宇光学",
    # 新能源
    "1211.HK":  "比亚迪H",
    "9866.HK":  "蔚来",
    "2015.HK":  "理想汽车",
}

CN_UNIVERSE: dict[str, str] = {
    # 宽基 ETF
    "510300.SS": "沪深300ETF",
    "159915.SZ": "创业板ETF",
    "588000.SS": "科创50ETF",
    "512480.SS": "半导体ETF",
    "510500.SS": "中证500ETF",
    # 消费/金融
    "600519.SS": "贵州茅台",
    "600036.SS": "招商银行",
    "000333.SZ": "美的集团",
    # 新能源/半导体
    "300750.SZ": "宁德时代",
    "002594.SZ": "比亚迪A",
    "688981.SS": "中芯国际A",
    "300760.SZ": "迈瑞医疗",
    "601012.SS": "隆基绿能",
    "002415.SZ": "海康威视",
}

# Benchmark for regime gate
HK_BENCHMARK  = "2800.HK"   # HSI ETF
CN_BENCHMARK  = "510300.SS" # CSI300 ETF

# Crisis periods relevant to HK/CN
CRASH_PERIODS = [
    ("2015股灾",     "2015-06-12", "2016-02-11"),
    ("2018贸易战",   "2018-01-26", "2018-10-31"),
    ("2020 COVID",  "2020-01-20", "2020-03-19"),
    ("2021监管风暴", "2021-02-17", "2022-03-15"),
    ("2022熊市",    "2022-01-01", "2022-10-31"),
]

START    = "2012-01-01"
COST_BPS = 15.0   # slightly higher than US (HK≈10–15bps, CN≈15bps with stamp duty)


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(ticker: str, start: str = START) -> pd.DataFrame | None:
    import yfinance as yf
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        return None
    df = df[list(needed)].dropna(subset=["Close"])
    return df if len(df) > 120 else None


def _composite_scores(df: pd.DataFrame, symbol: str,
                       buy_thresh: float, market: str) -> pd.Series:
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS
    result = composite_score(
        df, symbol=symbol, vix_df=None,   # no VIX for HK/CN
        buy_threshold=buy_thresh,
        sell_threshold=3.0,
        weights=STRATEGY_WEIGHTS,
    )
    return result["indicators"]["Composite_Score"].reindex(df.index, fill_value=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Position builders
# ══════════════════════════════════════════════════════════════════════════════

def _binary_pos(score: pd.Series, buy_thresh: float,
                sell_thresh: float = -3.0,
                regime: pd.Series | None = None) -> pd.Series:
    """Old binary in/out with optional regime gate."""
    pos  = pd.Series(0.0, index=score.index)
    curr = 0.0
    for dt, s in score.items():
        if regime is not None and not regime.get(dt, True):
            curr = 0.0          # regime gate: price < MA200 → no new buys
        elif s >= buy_thresh:
            curr = 1.0
        elif s <= sell_thresh:
            curr = 0.0
        pos[dt] = curr
    return pos


def _graduated_pos(score: pd.Series,
                   sell_thresh: float = -5.0,
                   regime: pd.Series | None = None) -> pd.Series:
    """
    Direction 1+2 graduated sizing with asymmetric exit + regime gate.
    ≥7.5→100%  ≥6.0→80%  ≥4.5→50%  ≥2.5→25%  ≤sell_thresh→0%  else hold.
    Regime gate: when local index < MA200, cap new entries (don't ADD to position).
    """
    pos  = pd.Series(0.0, index=score.index)
    curr = 0.0
    for dt, s in score.items():
        in_bear = regime is not None and not regime.get(dt, True)
        if s <= sell_thresh:
            curr = 0.0
        elif not in_bear:        # normal: graduated sizing
            if   s >= 7.5: curr = 1.00
            elif s >= 6.0: curr = 0.80
            elif s >= 4.5: curr = 0.50
            elif s >= 2.5: curr = 0.25
            # else: hold current in neutral zone
        else:                    # bear regime: don't increase position
            if s >= 7.5: curr = min(curr, 1.00)  # only keep existing, don't add
            elif s <= sell_thresh: curr = 0.0
            # else: hold whatever we have, don't add
        pos[dt] = curr
    return pos


def _regime_series(benchmark_df: pd.DataFrame | None) -> pd.Series | None:
    """True = OK to buy, False = bear regime (price < MA200)."""
    if benchmark_df is None:
        return None
    price = benchmark_df["Close"].dropna()
    ma200 = price.rolling(200, min_periods=100).mean()
    return (price > ma200).reindex(price.index)


def _equity_curve(price: pd.Series, pos: pd.Series,
                  cost_bps: float = COST_BPS) -> pd.Series:
    ret   = price.pct_change().fillna(0)
    pos_s = pos.reindex(price.index, method="ffill").fillna(0).shift(1).fillna(0)
    trades = pos_s.diff().abs().fillna(0)
    cost   = trades * cost_bps / 10_000
    strat  = pos_s * ret - cost
    return (1 + strat).cumprod()


# ══════════════════════════════════════════════════════════════════════════════
# Analytics
# ══════════════════════════════════════════════════════════════════════════════

def _stats(eq: pd.Series) -> dict:
    if len(eq) < 60:
        return dict(cagr=0, maxdd=0, sharpe=0, calmar=0, total=0, pct_in=0)
    years  = len(eq) / 252
    cagr   = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd     = (eq / eq.cummax() - 1)
    maxdd  = dd.min()
    ret    = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0
    total  = eq.iloc[-1] / eq.iloc[0] - 1
    return dict(cagr=cagr, maxdd=maxdd, sharpe=sharpe, calmar=calmar, total=total)


def _crash_ret(eq: pd.Series, start: str, end: str) -> float | None:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    sl   = eq.loc[(eq.index >= s) & (eq.index <= e)]
    if len(sl) < 2:
        return None
    return sl.iloc[-1] / sl.iloc[0] - 1


# ══════════════════════════════════════════════════════════════════════════════
# Per-symbol runner
# ══════════════════════════════════════════════════════════════════════════════

def run_symbol(ticker: str, name: str, market: str,
               buy_thresh: float, regime: pd.Series | None) -> dict | None:
    df = _fetch(ticker)
    if df is None:
        print(f"  ⚠ {ticker} ({name}): no data, skip")
        return None

    price = df["Close"].dropna()
    score = _composite_scores(df, ticker, buy_thresh, market)

    # align
    score = score.reindex(price.index, method="ffill").fillna(0)
    reg   = regime.reindex(price.index, method="ffill") if regime is not None else None

    pos_old = _binary_pos(score, buy_thresh, sell_thresh=-3.0, regime=reg)
    pos_new = _graduated_pos(score, sell_thresh=-5.0, regime=reg)

    eq_bh  = price / price.iloc[0]
    eq_old = _equity_curve(price, pos_old)
    eq_new = _equity_curve(price, pos_new)

    pct_in_old = (pos_old > 0).mean()
    pct_in_new = (pos_new > 0).mean()
    avg_frac   = pos_new.mean()

    return dict(
        ticker=ticker, name=name,
        eq_bh=eq_bh, eq_old=eq_old, eq_new=eq_new,
        stats_bh =_stats(eq_bh),
        stats_old=_stats(eq_old),
        stats_new=_stats(eq_new),
        pct_in_old=pct_in_old, pct_in_new=pct_in_new, avg_frac=avg_frac,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

W = dict(ticker=8, name=12, bh_c=8, bh_d=8,
         old_c=8, old_d=8, old_s=9,
         new_c=8, new_d=8, new_s=9,
         dc=8, dd=8)


def _pf(v, pct=True, decimals=1):
    if v is None: return "  N/A "
    s = f"{v*100:+.{decimals}f}%" if pct else f"{v:.2f}"
    return s


def print_table(results: list[dict], market_label: str, buy_thresh: float):
    sep = "═" * 118
    print(f"\n{sep}")
    print(f"  {market_label}  |  Old: binary BUY≥{buy_thresh:.0f} SELL≤-3"
          f"  |  New: graduated BUY≥2.5 SELL≤-5 + regime gate")
    print(f"  from {START}  |  cost={COST_BPS}bps")
    print(sep)
    hdr = (f"  {'Ticker':<8} {'名称':<12}  {'B&H':>7} {'B&H DD':>8}"
           f" │ {'OldCAGR':>7} {'OldDD':>7} {'OldSR':>6}"
           f" │ {'NewCAGR':>7} {'NewDD':>7} {'NewSR':>6}"
           f"  {'ΔCAGR':>7} {'ΔDD':>7}")
    print(hdr)
    print("  " + "─" * 114)

    for r in results:
        if r is None: continue
        bh, old, new = r["stats_bh"], r["stats_old"], r["stats_new"]
        dcagr = new["cagr"] - old["cagr"]
        ddd   = abs(new["maxdd"]) - abs(old["maxdd"])   # negative = improvement
        flag  = "↑✅" if dcagr > 0.005 else ("≈✅" if abs(dcagr) <= 0.005 else "↓⚠")
        dd_flag = "✅" if ddd < -0.02 else ("≈" if abs(ddd) <= 0.02 else "⚠")

        print(
            f"  {r['ticker']:<8} {r['name']:<12}"
            f"  {_pf(bh['cagr']):>7} {_pf(bh['maxdd']):>8}"
            f" │ {_pf(old['cagr']):>7} {_pf(old['maxdd']):>8} {old['sharpe']:>6.2f}"
            f" │ {_pf(new['cagr']):>7} {_pf(new['maxdd']):>8} {new['sharpe']:>6.2f}"
            f"  {_pf(dcagr):>7} {_pf(-ddd):>7}  {flag}{dd_flag}"
        )
    print()
    print(f"  ↑ CAGR improved  ✅ MaxDD reduced >2pp  ⚠ MaxDD worse")


def print_crash_table(results: list[dict], market_label: str):
    valid = [r for r in results if r is not None]
    if not valid: return

    sep = "═" * 100
    print(f"\n{sep}")
    print(f"  危机期间保护 — {market_label}")
    print(sep)

    hdr = f"  {'Ticker':<8} {'名称':<12}"
    for label, *_ in CRASH_PERIODS:
        hdr += f"  {label[:6]:>8}  {'Old':>5}  {'New':>5}"
    print(hdr)
    print("  " + "─" * 96)

    for r in valid:
        row = f"  {r['ticker']:<8} {r['name']:<12}"
        for label, s, e in CRASH_PERIODS:
            bh  = _crash_ret(r["eq_bh"],  s, e)
            old = _crash_ret(r["eq_old"], s, e)
            new = _crash_ret(r["eq_new"], s, e)
            bhs  = f"{bh*100:+.0f}%"  if bh  is not None else " N/A"
            olds = f"{old*100:+.0f}%" if old is not None else " N/A"
            news = f"{new*100:+.0f}%" if new is not None else " N/A"
            row += f"  {bhs:>8}  {olds:>5}  {news:>5}"
        print(row)


def make_charts(hk_results, cn_results):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("  plotly not installed, skipping charts")
        return

    def add_traces(fig, results, row, col, label):
        colors_bh  = "#aaa"
        colors_old = "#e07b00"
        colors_new = "#2196F3"
        shown = {"B&H": False, "Old": False, "New": False}
        for r in results:
            if r is None: continue
            nm = r["name"]
            for eq, key, clr in [
                (r["eq_bh"],  "B&H", colors_bh),
                (r["eq_old"], "Old", colors_old),
                (r["eq_new"], "New", colors_new),
            ]:
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values,
                    name=f"{key} ({nm})",
                    line=dict(color=clr, width=1.2),
                    legendgroup=key,
                    showlegend=not shown[key],
                ), row=row, col=col)
                shown[key] = True

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["港股 HK", "A股 CN"],
        shared_yaxes=False,
    )
    add_traces(fig, hk_results, 1, 1, "HK")
    add_traces(fig, cn_results, 1, 2, "CN")

    fig.update_layout(
        title="HK & CN Backtest — Old vs New Composite Strategy",
        height=600, width=1400,
        template="plotly_dark",
        hovermode="x unified",
    )
    out = ROOT / "scripts" / "backtest_hk_cn.html"
    pio.write_html(fig, str(out))
    print(f"\n   ✅ Chart: file://{out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["hk", "cn", "all"], default="all")
    args = parser.parse_args()

    run_hk = args.market in ("hk", "all")
    run_cn = args.market in ("cn", "all")

    hk_results: list = []
    cn_results: list = []

    # ── HK ────────────────────────────────────────────────────────────────
    if run_hk:
        print("\n" + "═" * 60)
        print("  📥 Fetching HK benchmark (2800.HK) for regime gate …")
        hk_bench_df = _fetch(HK_BENCHMARK)
        hk_regime   = _regime_series(hk_bench_df)
        print(f"     Bear-regime days: {(~hk_regime).sum() if hk_regime is not None else 'N/A'}")

        print("\n  📊 Running HK symbols …")
        for ticker, name in HK_UNIVERSE.items():
            print(f"     [{ticker}] {name} …", end=" ", flush=True)
            r = run_symbol(ticker, name, market="hk",
                           buy_thresh=7.0, regime=hk_regime)
            if r:
                print(f"B&H {r['stats_bh']['cagr']*100:+.1f}%  "
                      f"Old {r['stats_old']['cagr']*100:+.1f}%  "
                      f"New {r['stats_new']['cagr']*100:+.1f}%")
            hk_results.append(r)

        print_table(hk_results, "港股 HK", buy_thresh=7.0)
        print_crash_table(hk_results, "港股 HK")

    # ── CN ────────────────────────────────────────────────────────────────
    if run_cn:
        print("\n" + "═" * 60)
        print("  📥 Fetching CN benchmark (510300.SS) for regime gate …")
        cn_bench_df = _fetch(CN_BENCHMARK)
        cn_regime   = _regime_series(cn_bench_df)
        print(f"     Bear-regime days: {(~cn_regime).sum() if cn_regime is not None else 'N/A'}")

        print("\n  📊 Running CN symbols …")
        for ticker, name in CN_UNIVERSE.items():
            print(f"     [{ticker}] {name} …", end=" ", flush=True)
            r = run_symbol(ticker, name, market="cn",
                           buy_thresh=6.0, regime=cn_regime)
            if r:
                print(f"B&H {r['stats_bh']['cagr']*100:+.1f}%  "
                      f"Old {r['stats_old']['cagr']*100:+.1f}%  "
                      f"New {r['stats_new']['cagr']*100:+.1f}%")
            cn_results.append(r)

        print_table(cn_results, "A股 CN", buy_thresh=6.0)
        print_crash_table(cn_results, "A股 CN")

    # ── Charts ─────────────────────────────────────────────────────────────
    if run_hk or run_cn:
        print("\n  📊 Generating charts …")
        make_charts(hk_results if run_hk else [],
                    cn_results if run_cn else [])

    print("\n🎉 HK/CN backtest complete!\n")


if __name__ == "__main__":
    main()
