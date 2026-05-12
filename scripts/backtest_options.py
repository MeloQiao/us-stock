"""
Backtest: Options Overlay — Covered Call / Protective Put / Collar
────────────────────────────────────────────────────────────────────────

期权知识速成 (Options Primer)
═══════════════════════════════

【Call Option 看涨期权】
  买方：花钱购买"权利"——在到期日前以约定价格(行权价)买入股票
  卖方：收取权利金(Premium)，承担义务——若买方行权，必须按约定价格卖出股票

  例子：QQQ 当前价 $480
    你卖出 1 个月期、行权价 $504(+5%)的 Call
    收取权利金 $7.20（即 1.5% of $480）
    一个月后：
      QQQ < $504 → 买方不行权，你赚 $7.20 权利金
      QQQ > $504 → 买方行权，你以 $504 卖出（错过 $504 以上的涨幅），但依然赚了权利金

  Covered Call（备兑看涨）= 持有股票 + 卖出 Call
    用途：让持仓"生利息"，牺牲超额涨幅换取稳定收入

【Put Option 看跌期权】
  买方：花钱购买"权利"——在到期日前以约定价格(行权价)卖出股票
  卖方：收取权利金，承担义务——若买方行权，必须按约定价格买入股票

  例子：QQQ $480，买入行权价 $456(-5%)的 Put，付权利金 $7.20
    一个月后：
      QQQ > $456 → 不行权，损失 $7.20 权利金（买了保险没用上）
      QQQ < $456 → 行权，以 $456 卖出，损失被锁定在 5% 以内

  Protective Put（保护性看跌）= 持有股票 + 买入 Put
    用途：购买"下跌保险"，限制最大损失

【Collar 领口策略】= Covered Call + Protective Put
  同时卖出 OTM Call（收权利金）+ 买入 OTM Put（付权利金）
  如果 Call 权利金 ≈ Put 权利金 → Zero-cost Collar（零成本）
  效果：上涨锁定在 +5%，下跌锁定在 -5%，波动范围大幅压缩 → Sharpe 潜在提升

关键参数说明
────────────
  OTM (Out-of-the-money): 虚值期权。行权价高于当前价（Call）或低于当前价（Put）
  ATM (At-the-money): 平值期权。行权价 ≈ 当前价
  Premium: 权利金。买方支付给卖方的费用
  IV (Implied Volatility): 隐含波动率。越高→期权越贵→卖方收入越高
  Delta: 期权价格对标的价格变动的敏感度。5% OTM Call delta ≈ 0.25

适合我们策略的期权逻辑
──────────────────────
  现有策略（Crash Shield）已经做到：
    信号 ≤ -5 → 清仓 ≈ 免费的"虚拟 Put"（等待期持现金不付费用）

  期权能额外贡献的地方：
    ① 持仓期间 Covered Call → 每月额外 1-2% 权利金收入
    ② 等待期（现金） Covered Put → 卖出下方 Put 收权利金，等于"打折买入"
    ③ Collar → 压缩持仓波动区间，降低 MaxDD，可能改善 Sharpe

  期权的潜在风险：
    ① Covered Call 在牛市动能强时损失上涨空间（与动量策略冲突）
    ② Protective Put 成本 > 收益（策略已有 Crash Shield 保护）
    ③ 权利金溢出（IV > RV）：卖方通常长期赚钱，买方通常长期亏权利金

本回测基于 Black-Scholes 定价 + 历史波动率模拟
  IV = 近30日实现波动率 × 1.25（含 VRP 波动风险溢价）
  Period = 月度（22 个交易日）
  Strike = 当前价 × (1 ± moneyness)
  Base strategy = L1+2 (QQQ + TQQQ overlay，来自 backtest_sharpe15.py)

Usage
─────
  python3 scripts/backtest_options.py
  python3 scripts/backtest_options.py --skip-ml --call-otm 0.03  # tight 3% cap
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

try:
    from scipy.stats import norm as _norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("⚠️  scipy not found — using approximation for Black-Scholes")


# ══════════════════════════════════════════════════════════════════════════════
# Black-Scholes pricing
# ══════════════════════════════════════════════════════════════════════════════

def _ncdf(x: float) -> float:
    """Normal CDF — scipy if available, else polynomial approximation."""
    if _HAS_SCIPY:
        return float(_norm.cdf(x))
    # Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    p = 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2) * poly
    return p if x >= 0 else 1.0 - p


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price.  Returns 0 if T<=0."""
    if T <= 1e-8 or sigma <= 1e-6:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * _ncdf(d1) - K * np.exp(-r * T) * _ncdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price.  Returns 0 if T<=0."""
    if T <= 1e-8 or sigma <= 1e-6:
        return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers (same as backtest_sharpe15.py)
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


def get_composite_score_series(spy_df, vix_df):
    from strategies.composite import composite_score
    from config import STRATEGY_WEIGHTS, COMPOSITE_BUY_THRESHOLD
    res = composite_score(
        spy_df, symbol="SPY", vix_df=vix_df,
        buy_threshold=COMPOSITE_BUY_THRESHOLD.get("us", 6.0),
        sell_threshold=3.0, weights=STRATEGY_WEIGHTS,
    )
    return res["indicators"]["Composite_Score"].reindex(spy_df.index, fill_value=0.0)


def build_graduated_position(score_series, sell_threshold=-5.0):
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


def compute_crash_shield(spy_df, vix_df=None, hyg_df=None):
    spy = spy_df["Close"].dropna()
    idx = spy.index
    sig1 = pd.Series(0, index=idx)
    if vix_df is not None and not vix_df.empty:
        vix  = vix_df["Close"].dropna().reindex(idx, method="ffill").fillna(20)
        sig1 = ((vix > 25) & (vix.pct_change(5).fillna(0) > 0.20)).astype(int)
    ma50 = spy.rolling(50, min_periods=25).mean()
    sig2 = ((spy < ma50) & (ma50 < ma50.shift(10))).astype(int)
    sig3 = (spy.pct_change(20) < -0.08).astype(int)
    sig4 = pd.Series(0, index=idx)
    if hyg_df is not None and not hyg_df.empty:
        hyg  = hyg_df["Close"].dropna().reindex(idx, method="ffill")
        sig4 = (hyg.pct_change(20) < -0.03).astype(int)
    else:
        sig4 = (spy.pct_change(60) < -0.15).astype(int)
    cs   = (sig1 + sig2 + sig3 + sig4).fillna(0).astype(int)
    mult = pd.Series(1.0, index=idx)
    mult[cs == 2] = 0.5
    mult[cs >= 3] = 0.0
    level = pd.Series("NONE", index=idx, dtype=str)
    level[cs == 2] = "CAUTION"
    level[cs >= 3] = "SHIELD"
    return pd.DataFrame({"multiplier": mult, "level": level})


def compute_ml_and_dm(spy_df, vix_df, hyg_df, skip_ml=False):
    from strategies.ml_regime import build_features, FEATURE_COLS, _get_model, _fill_missing, MLRegimeClassifier
    from sklearn.preprocessing import StandardScaler

    dm_spy  = spy_df["Close"].dropna()
    abs_ok  = dm_spy.pct_change(252) > 0
    lr      = np.log(dm_spy / dm_spy.shift(1)).dropna()
    crash   = (lr.rolling(21).std() / (lr.rolling(126).std() + 1e-9)) >= 2.0
    dm_sc   = pd.Series(1.0, index=dm_spy.index)
    dm_sc[~abs_ok.reindex(dm_spy.index, fill_value=False)] = 0.0
    dm_sc[crash.reindex(dm_spy.index, fill_value=False)
          & abs_ok.reindex(dm_spy.index, fill_value=True)] = 0.5
    dm_abs  = abs_ok.reindex(dm_spy.index, fill_value=False)

    ml_mult = pd.Series(1.0, index=spy_df.index)
    if not skip_ml:
        try:
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
                train = df.loc[tr0:tr1].dropna()
                test  = df.loc[te0:te1].dropna()
                if len(train) < 300 or len(test) < 20:
                    continue
                scl   = StandardScaler()
                X_tr  = scl.fit_transform(_fill_missing(train[FEATURE_COLS]))
                X_te  = scl.transform(_fill_missing(test[FEATURE_COLS]))
                mdl   = _get_model()
                mdl.fit(X_tr, train["y"].values)
                probs = mdl.predict_proba(X_te)[:, 1]
                prob_series.loc[test.index] = probs
                print(f"   ✅ OOS {te0[:7]}–{te1[:7]}: mean_prob={probs.mean():.3f}")
            ml_mult = prob_series.apply(MLRegimeClassifier.to_position_multiplier)
            ml_mult = ml_mult.reindex(spy_df.index, fill_value=1.0)
        except Exception as e:
            print(f"   ❌ ML failed ({e}) — using neutral")
    return ml_mult, dm_sc.reindex(spy_df.index, fill_value=1.0), dm_abs.reindex(spy_df.index, fill_value=True)


# ══════════════════════════════════════════════════════════════════════════════
# Monthly options simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_with_options(
    qqq_prices:  pd.Series,    # QQQ daily close prices
    tqqq_ret:    pd.Series,    # TQQQ daily returns
    pos_full:    pd.Series,    # daily graduated position (0–1)
    tqqq_frac:   pd.Series,    # fraction of base in TQQQ (0–0.5)
    call_otm:    float = 0.05, # call strike = S × (1 + call_otm)
    put_otm:     float = 0.05, # put strike  = S × (1 - put_otm)
    use_call:    bool  = False,
    use_put:     bool  = False,
    iv_premium:  float = 1.25, # IV = RV × iv_premium (vol risk premium)
    rf:          float = 0.04, # annual risk-free rate
    cost_bps:    float = 10.0,
    label:       str   = "Strategy",
) -> pd.Series:
    """
    Monthly options overlay on QQQ+TQQQ base strategy.

    Options are settled on the last trading day of each calendar month.
    At the start of each month we observe:
      - Current position fraction
      - Trailing 21-day realized vol (annualized)
      - QQQ price (for strike calculation)
    At month end we observe the actual return and compute option payoff.

    Options are scaled by the QQQ fraction of the base position
    (TQQQ is excluded from option simulation — too complex).
    """
    T  = 22 / 252   # ~1 month in years
    rf_monthly = (1 + rf) ** (22 / 252) - 1

    # Build daily returns
    qqq_ret    = qqq_prices.pct_change()
    log_ret    = np.log(qqq_prices / qqq_prices.shift(1))
    rv_21      = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252)

    # Identify month-start indices (first trading day of each month)
    month_groups  = qqq_prices.groupby(qqq_prices.index.to_period("M"))
    month_starts  = [g.index[0]  for _, g in month_groups]
    month_ends    = [g.index[-1] for _, g in month_groups]

    # Build a mask: which daily returns belong to which month
    # and compute option adjustments per month
    month_adj = pd.Series(0.0, index=qqq_prices.index)  # daily option PnL

    for ms, me in zip(month_starts, month_ends):
        try:
            pos_at_start    = float(pos_full.loc[ms])
            tqqq_f_at_start = float(tqqq_frac.loc[ms])
            qqq_f_at_start  = pos_at_start * (1.0 - tqqq_f_at_start)  # QQQ fraction

            if qqq_f_at_start < 0.01:    # no meaningful QQQ exposure
                continue

            S   = float(qqq_prices.loc[ms])
            rv  = float(rv_21.loc[ms]) if not np.isnan(rv_21.loc[ms]) else 0.20
            iv  = max(rv * iv_premium, 0.12)   # floor 12% IV

            # Black-Scholes premiums
            K_call = S * (1.0 + call_otm)
            K_put  = S * (1.0 - put_otm)
            call_prem = bs_call(S, K_call, T, rf, iv) / S if use_call else 0.0
            put_prem  = bs_put(S, K_put,  T, rf, iv) / S if use_put  else 0.0

            # Month return
            S_end = float(qqq_prices.loc[me])
            month_stock_ret = (S_end - S) / S

            # Option payoff (from position holder's perspective)
            if use_call and use_put:
                # Collar: sell call, buy put
                net_prem = call_prem - put_prem
                if month_stock_ret >= call_otm:
                    stock_return_capped = call_otm
                elif month_stock_ret <= -put_otm:
                    stock_return_capped = -put_otm
                else:
                    stock_return_capped = month_stock_ret
                # Delta = capped return vs actual + net premium
                option_adj_pct = (stock_return_capped - month_stock_ret) + net_prem
            elif use_call:
                # Covered call: sell call, collect premium
                if month_stock_ret >= call_otm:
                    stock_return_capped = call_otm
                else:
                    stock_return_capped = month_stock_ret
                option_adj_pct = (stock_return_capped - month_stock_ret) + call_prem
            elif use_put:
                # Protective put: buy put, pay premium
                if month_stock_ret <= -put_otm:
                    stock_return_capped = -put_otm
                else:
                    stock_return_capped = month_stock_ret
                option_adj_pct = (stock_return_capped - month_stock_ret) - put_prem
            else:
                option_adj_pct = 0.0

            # Scale by QQQ exposure (option only applies to QQQ portion)
            portfolio_option_adj = option_adj_pct * qqq_f_at_start

            # Distribute the adjustment to the LAST day of the month
            month_adj.loc[me] += portfolio_option_adj

        except Exception:
            pass

    # Simulate daily NAV with options adjustment
    nav       = [1.0]
    prev_spy  = 0.0
    prev_tqqq = 0.0

    for i in range(1, len(qqq_prices.index)):
        dt   = qqq_prices.index[i]
        base = float(pos_full.iloc[i - 1])
        tf   = float(tqqq_frac.iloc[i - 1])
        t_tqqq = base * tf
        t_qqq  = base * (1.0 - tf)

        tc = (abs(t_qqq  - prev_spy) +
              abs(t_tqqq - prev_tqqq)) * cost_bps / 10_000.0

        ret = (t_qqq  * float(qqq_ret.iloc[i])  +
               t_tqqq * float(tqqq_ret.iloc[i]) +
               float(month_adj.iloc[i])) - tc

        nav.append(nav[-1] * (1.0 + ret))
        prev_spy  = t_qqq
        prev_tqqq = t_tqqq

    return pd.Series(nav, index=qqq_prices.index, name=label)


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
    return {
        "CAGR":  f"{cagr*100:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDD": f"{max_dd*100:.1f}%",
        "Calmar": f"{calmar:.2f}",
        "_cagr": cagr, "_sharpe": sharpe, "_maxdd": max_dd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",    type=int,   default=15)
    parser.add_argument("--skip-ml",  action="store_true")
    parser.add_argument("--cost",     type=float, default=10.0)
    parser.add_argument("--call-otm", type=float, default=0.05, help="Call OTM %")
    parser.add_argument("--put-otm",  type=float, default=0.05, help="Put OTM %")
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  Options Overlay Backtest — Covered Call / Put / Collar")
    print(f"  {args.years}yr | {args.cost}bps | Call {args.call_otm*100:.0f}% OTM | "
          f"Put {args.put_otm*100:.0f}% OTM | {'skip ML' if args.skip_ml else 'full ML'}")
    print(f"{'='*72}")

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    print("\n📥 Downloading data…")
    spy_df = tqqq_df = qqq_df = vix_df = hyg_df = None
    for tk, key in [("SPY","spy"),("QQQ","qqq"),("TQQQ","tqqq"),("^VIX","vix"),("HYG","hyg")]:
        try:
            df = _fetch(tk, args.years)
            if   key == "spy":  spy_df  = df
            elif key == "qqq":  qqq_df  = df
            elif key == "tqqq": tqqq_df = df
            elif key == "vix":  vix_df  = df
            elif key == "hyg":  hyg_df  = df
            print(f"   {tk:<6}: {len(df):,} days")
        except Exception as e:
            print(f"   {tk:<6}: ⚠️  {e}")

    common = spy_df.index
    for df in [qqq_df, tqqq_df]:
        if df is not None:
            common = common.intersection(df.index)
    spy_df  = spy_df.loc[common]
    qqq_df  = qqq_df.loc[common]
    tqqq_df = tqqq_df.loc[common]
    print(f"\n   Common: {len(common):,} days ({common[0].date()} – {common[-1].date()})")

    qqq_prices = qqq_df["Close"]
    tqqq_ret   = tqqq_df["Close"].pct_change()

    # ── 2. Regime signals ─────────────────────────────────────────────────
    print("\n📊 Computing composite scores…")
    raw_scores = get_composite_score_series(spy_df, vix_df)
    pos_base   = build_graduated_position(raw_scores)

    cs       = compute_crash_shield(spy_df, vix_df, hyg_df)
    cs_mult  = cs["multiplier"].reindex(common, fill_value=1.0)
    cs_level = cs["level"].reindex(common, fill_value="NONE")

    if not args.skip_ml:
        print("\n🤖 Walk-forward ML…")
    ml_mult, dm_scale, dm_abs = compute_ml_and_dm(spy_df, vix_df, hyg_df, args.skip_ml)
    ml_mult  = ml_mult.reindex(common, fill_value=1.0)
    dm_scale = dm_scale.reindex(common, fill_value=1.0)
    dm_abs   = dm_abs.reindex(common, fill_value=True)

    pos_full = (pos_base * cs_mult * ml_mult * dm_scale).clip(0, 1)

    # TQQQ fraction (same as backtest_sharpe15.py Lever 2)
    cs_ok  = cs_level == "NONE"
    ml_ok  = ml_mult >= 0.80 if not args.skip_ml else pd.Series(True, index=common)
    dm_ok  = dm_abs
    gate   = cs_ok & ml_ok & dm_ok
    tqqq_frac = pd.Series(0.0, index=common)
    tqqq_frac[gate & (raw_scores >= 9.0)]                       = 0.50
    tqqq_frac[gate & (raw_scores >= 7.5) & (raw_scores < 9.0)] = 0.50
    tqqq_frac[gate & (raw_scores >= 7.0) & (raw_scores < 7.5)] = 0.30
    # (capped at 50% so options simulation retains ≥50% QQQ exposure)

    # ── 3. Options premium analytics (educational) ────────────────────────
    log_ret    = np.log(qqq_prices / qqq_prices.shift(1))
    rv_21d     = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    iv_series  = (rv_21d * 1.25).clip(0.12, 1.0)
    T_month    = 22 / 252

    call_prem_series = pd.Series(index=common, dtype=float)
    put_prem_series  = pd.Series(index=common, dtype=float)
    for dt in common:
        S  = float(qqq_prices.loc[dt])
        iv = float(iv_series.loc[dt]) if not np.isnan(iv_series.loc[dt]) else 0.20
        call_prem_series[dt] = bs_call(S, S*(1+args.call_otm), T_month, 0.04, iv) / S
        put_prem_series[dt]  = bs_put(S,  S*(1-args.put_otm),  T_month, 0.04, iv) / S

    avg_iv   = float(iv_series.dropna().mean()) * 100
    avg_call = float(call_prem_series.dropna().mean()) * 100
    avg_put  = float(put_prem_series.dropna().mean())  * 100
    avg_net  = avg_call - avg_put
    print(f"\n  📊 Options analytics (15yr average):")
    print(f"     Avg IV (realized×1.25): {avg_iv:.1f}%")
    print(f"     Avg monthly call premium ({args.call_otm*100:.0f}% OTM): {avg_call:.2f}% of QQQ")
    print(f"     Avg monthly put premium  ({args.put_otm*100:.0f}% OTM): {avg_put:.2f}% of QQQ")
    print(f"     Net collar premium (call-put):             {avg_net:+.2f}%")
    ann_call = avg_call * 12
    ann_put  = avg_put  * 12
    print(f"     Annualized gross call income: {ann_call:.1f}%/yr  "
          f"(×75% invested = {ann_call*0.75:.1f}%/yr to portfolio)")
    print(f"     Annualized gross put cost:    {ann_put:.1f}%/yr  "
          f"(×75% invested = {ann_put*0.75:.1f}%/yr drag)")

    # ── 4. Simulate equity curves ─────────────────────────────────────────
    print("\n📈 Simulating…")
    bah_qqq = pd.Series((qqq_prices / qqq_prices.iloc[0]).values,
                        index=common, name="0. B&H QQQ")

    _z = pd.Series(0.0, index=common)

    eq_base = simulate_with_options(
        qqq_prices, tqqq_ret, pos_full, tqqq_frac,
        use_call=False, use_put=False, cost_bps=args.cost,
        label="1. L1+2 base (no options)",
    )
    eq_cc_5 = simulate_with_options(
        qqq_prices, tqqq_ret, pos_full, tqqq_frac,
        call_otm=args.call_otm, use_call=True, use_put=False,
        cost_bps=args.cost, label=f"2. +Covered Call {args.call_otm*100:.0f}%",
    )
    eq_cc_3 = simulate_with_options(
        qqq_prices, tqqq_ret, pos_full, tqqq_frac,
        call_otm=0.03, use_call=True, use_put=False,
        cost_bps=args.cost, label="3. +Covered Call 3% (tight)",
    )
    eq_pp = simulate_with_options(
        qqq_prices, tqqq_ret, pos_full, tqqq_frac,
        put_otm=args.put_otm, use_call=False, use_put=True,
        cost_bps=args.cost, label=f"4. +Protective Put {args.put_otm*100:.0f}%",
    )
    eq_collar = simulate_with_options(
        qqq_prices, tqqq_ret, pos_full, tqqq_frac,
        call_otm=args.call_otm, put_otm=args.put_otm,
        use_call=True, use_put=True,
        cost_bps=args.cost, label=f"5. Collar {args.call_otm*100:.0f}%/{args.put_otm*100:.0f}%",
    )

    EQ = [
        ("0. B&H QQQ",                  bah_qqq),
        ("1. L1+2 base",                eq_base),
        (f"2. CC {args.call_otm*100:.0f}% OTM",   eq_cc_5),
        ("3. CC 3% OTM (tight)",        eq_cc_3),
        (f"4. Put {args.put_otm*100:.0f}% OTM",   eq_pp),
        (f"5. Collar {args.call_otm*100:.0f}%/{args.put_otm*100:.0f}%", eq_collar),
    ]

    # ── 5. Performance table ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'Strategy':<32} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8}")
    print(f"{'─'*72}")
    results = {}
    for label, eq in EQ:
        s = results[label] = perf(eq)
        print(f"{label:<32} {s['CAGR']:>8} {s['Sharpe']:>8} {s['MaxDD']:>10} {s['Calmar']:>8}")
    print(f"{'─'*72}")

    # Delta vs base
    base_s = results["1. L1+2 base"]
    print(f"\n  Delta vs L1+2 base:")
    for label, eq in EQ[2:]:
        s  = results[label]
        dc = (s["_cagr"]   - base_s["_cagr"])   * 100
        ds = s["_sharpe"]  - base_s["_sharpe"]
        dd = (s["_maxdd"]  - base_s["_maxdd"])   * 100
        print(f"  {label:<32} CAGR {dc:+.1f}pp  Sharpe {ds:+.2f}  MaxDD {dd:+.1f}pp")

    # ── 6. Crisis analysis ────────────────────────────────────────────────
    crisis = [
        ("2020 COVID crash",   "2020-01-17", "2020-03-23"),
        ("2022 Bear market",   "2021-12-31", "2022-10-12"),
        ("2021 bull run",      "2020-12-31", "2021-12-31"),   # check upside capture
        ("2023-24 AI rally",   "2022-12-31", "2024-12-31"),   # check upside capture
    ]
    print(f"\n{'─'*72}")
    print("  Key period returns (including 2 bull periods to check upside capture):")
    print(f"{'─'*72}")
    for name, s, e in crisis:
        parts = []
        for lbl, eq in EQ:
            try:
                seg = eq.loc[s:e]
                if len(seg) > 1:
                    r = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
                    short = lbl[:8].strip()
                    parts.append(f"{short}: {r:+.1f}%")
            except Exception:
                pass
        print(f"  {name:<22} " + "  ".join(parts))
    print(f"{'─'*72}")

    # ── 7. Educational summary ────────────────────────────────────────────
    print(f"""
  ══ 期权策略效果解读 ══════════════════════════════════════════════

  Covered Call 作用原理：
    每月收取权利金 (avg {avg_call:.2f}%/月)，但上涨空间被锁定在 {args.call_otm*100:.0f}%/月
    年化权利金收入 ≈ {ann_call:.1f}%，但动量强时（月涨>5%）损失超额涨幅
    → 对"动量策略"可能适得其反：正是买入时最看好，却限制了涨幅

  Protective Put 作用原理：
    每月支付保险费 (avg {avg_put:.2f}%/月 = {ann_put:.1f}%/年)
    当月跌超 {args.put_otm*100:.0f}% 时才发挥保护
    本策略已有 Crash Shield（跌到一定程度→清仓持现金）
    → 两层保护重叠，Put 成本难以被补偿

  Collar 作用原理：
    卖 Call 收权利金 - 买 Put 付权利金 = 净权利金 {avg_net:+.2f}%/月
    把月度收益区间锁定在 [-{args.put_otm*100:.0f}%, +{args.call_otm*100:.0f}%]
    波动率下降 → Sharpe 可能改善，但 CAGR 通常减少

  真正适合我们策略的期权用法（回测未覆盖）：
    ★ 现金等待期（position=0）卖 Put：
       信号弱，等待买入时机 → 卖下方 Put（行权价 = 目标买入价）
       赚权利金 且 若真跌到目标价自动买入
       这才是利用期权增收而不损失上涨空间的正确姿势
  """)

    # ── 8. Chart ──────────────────────────────────────────────────────────
    if not args.no_plot:
        print("📊 Generating chart…")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.55, 0.25, 0.20],
                shared_xaxes=True,
                subplot_titles=["Equity Curves (log)", "Drawdown", "IV & Option Premiums"],
                vertical_spacing=0.06,
            )
            palette = [
                ("rgba(150,150,150,0.7)", 1.5, "dot"),
                ("steelblue",             2.5, "solid"),
                ("darkorange",            2.0, "solid"),
                ("chocolate",             2.0, "dash"),
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
                    x=dd.index, y=dd.values,
                    line=dict(color=color, width=1), fill="tozeroy",
                    showlegend=False,
                ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=iv_series.index, y=(iv_series * 100).values,
                name="IV %", fill="tozeroy",
                line=dict(color="purple", width=1), fillcolor="rgba(128,0,128,0.12)",
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=call_prem_series.index, y=(call_prem_series * 100).values,
                name="Call prem %", line=dict(color="orange", width=1),
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=put_prem_series.index, y=(put_prem_series * 100).values,
                name="Put prem %", line=dict(color="green", width=1),
            ), row=3, col=1)

            fig.update_yaxes(type="log", title="NAV",    row=1, col=1)
            fig.update_yaxes(title="DD %",               row=2, col=1)
            fig.update_yaxes(title="IV / Premium %",     row=3, col=1)
            ml_note = " (ML skipped)" if args.skip_ml else ""
            fig.update_layout(
                title=(f"Options Overlay — CC / Put / Collar  "
                       f"({args.years}yr, {args.call_otm*100:.0f}%/{args.put_otm*100:.0f}% OTM{ml_note})"),
                height=960, legend=dict(x=0.01, y=0.99), hovermode="x unified",
            )
            out = ROOT / "scripts" / "backtest_options.html"
            fig.write_html(str(out))
            print(f"   ✅ Chart: {out}")
            print(f"   Open:  file://{out}")
        except Exception as e:
            print(f"   ❌ Chart: {e}")

    print("\n🎉 Done!\n")


if __name__ == "__main__":
    main()
