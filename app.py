"""
Streamlit UI — Multi-Market Quantitative Monitoring Platform
Tabs:
  1. 📊 盯盘总览    — US + HK + CN signal dashboard (3 sections)
  2. 📈 策略回测    — backtest a single strategy (market → symbol picker)
  3. 🔁 策略对比    — compare all strategies (market → symbol picker)
  4. 💼 持仓总览    — Alpaca (US) + virtual portfolios (HK, CN)
  5. ⚙️  设置        — manual pipeline trigger + config status
"""

import logging
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ACTIVE_MARKETS,
    ALPACA_API_KEY,
    FEISHU_WEBHOOK_URL,
    HF_DATASET_REPO,
    HF_TOKEN,
    HISTORY_YEARS,
    MARKET_WATCHLISTS,
    STRATEGY_PARAMS,
    SYMBOL_NAMES,
    UI_REFRESH_SECONDS,
    VIRTUAL_PORTFOLIO_CAPITAL,
    VIRTUAL_PORTFOLIO_CURRENCY,
    WATCHLIST,
)
from strategies import STRATEGY_FUNCTIONS, STRATEGY_LABELS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_MARKET_LABEL = {"us": "🇺🇸 美股", "hk": "🇭🇰 港股", "cn": "🇨🇳 A股"}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="量化盯盘平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────
if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None


# ── Shared helpers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=UI_REFRESH_SECONDS)
def get_quotes(symbols: list[str], market: str = "us") -> list[dict]:
    """
    During market hours: fetch live quotes from yfinance / akshare.
    During closed hours: read last-close price from HF Dataset OHLCV cache
    — much faster, no live API calls needed.
    """
    from utils.market_hours import is_market_open

    if is_market_open(market):
        from data.fetcher import fetch_quotes
        return fetch_quotes(symbols, market=market)

    # ── Market closed: use last close from HF-cached OHLCV ──────────────
    results = []
    for sym in symbols:
        try:
            df = get_history(sym, years=1, market=market)
            if df is None or df.empty:
                raise ValueError("empty")
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else last
            change = last["Close"] - prev["Close"]
            change_pct = change / prev["Close"] * 100 if prev["Close"] else 0.0
            results.append({
                "symbol": sym,
                "price": round(float(last["Close"]), 2),
                "change": round(float(change), 2),
                "change_pct": round(float(change_pct), 2),
                "volume": float(last.get("Volume", 0)),
                "timestamp": str(df.index[-1].date()),
            })
        except Exception:
            results.append({
                "symbol": sym, "price": None, "change": None,
                "change_pct": None, "volume": None, "timestamp": None,
            })
    return results


@st.cache_data(ttl=300)
def get_history(symbol: str, years: int = HISTORY_YEARS, market: str = "us") -> pd.DataFrame:
    from data.fetcher import fetch_history
    return fetch_history(symbol, years=years, market=market)


@st.cache_data(ttl=300)
def get_vix() -> pd.DataFrame:
    from data.fetcher import fetch_history
    return fetch_history("^VIX", years=HISTORY_YEARS, market="us")


@st.cache_data(ttl=300)
def compute_signals(symbol: str, market: str = "us") -> dict:
    """Full signal computation — used by backtest tabs."""
    from strategies.composite import run_all_strategies
    df = get_history(symbol, market=market)
    vix_df = get_vix() if market == "us" else None
    return run_all_strategies(df, symbol=symbol, vix_df=vix_df)


@st.cache_data(ttl=300)
def load_hf_signals(market: str) -> dict:
    """Load today's pre-computed signals from HF Dataset for a market."""
    if not (HF_TOKEN and HF_DATASET_REPO):
        return {}
    try:
        from data.fetcher import load_today_signals_from_hf
        return load_today_signals_from_hf(
            market=market, hf_repo=HF_DATASET_REPO, hf_token=HF_TOKEN
        ) or {}
    except Exception:
        return {}


def signal_badge(signal: int) -> str:
    return {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}.get(signal, "—")


def sym_display(symbol: str) -> str:
    """Return 'CODE - 名称' if name exists, else just 'CODE'."""
    name = SYMBOL_NAMES.get(symbol, "")
    return f"{symbol} - {name}" if name else symbol


def _candlestick_fig(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=symbol,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
    )
    return fig


def _render_market_section(market: str) -> None:
    """Render one market's signal table + heatmap inside Tab1."""
    symbols = [s for g in MARKET_WATCHLISTS.get(market, {}).values() for s in g]
    if not symbols:
        st.caption("该市场暂无标的")
        return

    # Load signals
    signals = load_hf_signals(market)
    source_label = "HF Dataset ✅" if signals else "—"

    if not signals:
        st.caption("⏳ 暂无今日信号（daily pipeline 尚未运行）")

    # Quotes — live during market hours, cached close price when closed
    from utils.market_hours import is_market_open
    market_status = "📈 盘中" if is_market_open(market) else "🔒 闭市 (上次收盘价)"
    with st.spinner(f"加载 {_MARKET_LABEL[market]} 行情..."):
        quotes = get_quotes(symbols, market=market)

    quotes_df = pd.DataFrame(quotes)
    if quotes_df.empty or "price" not in quotes_df.columns:
        st.caption("行情数据暂不可用")
        return

    quotes_df = quotes_df[quotes_df["price"].notna()].copy()
    quotes_df["名称"] = quotes_df["symbol"].map(lambda s: SYMBOL_NAMES.get(s, ""))

    signal_col, score_col = [], []
    for sym in quotes_df["symbol"]:
        sym_sigs = signals.get(sym, {})
        comp = sym_sigs.get("composite_score", 0)
        total = sum(v for v in sym_sigs.values() if v == 1) - sum(1 for v in sym_sigs.values() if v == -1)
        max_s = max(len(sym_sigs) - 1, 1)
        signal_col.append(signal_badge(comp))
        score_col.append(f"{total}/{max_s}" if sym_sigs else "—")

    quotes_df["综合信号"] = signal_col
    quotes_df["评分"] = score_col

    display_cols = [c for c in ["symbol", "名称", "price", "change_pct", "综合信号", "评分"] if c in quotes_df.columns]
    display_df = quotes_df[display_cols].rename(columns={
        "symbol": "代码", "price": "价格", "change_pct": "涨跌%"
    })

    def color_change(val):
        try:
            return "color: #26a69a" if float(val) >= 0 else "color: #ef5350"
        except Exception:
            return ""

    st.caption(f"信号来源: {source_label} · {market_status} · {len(quotes_df)} 个标的")
    fmt = {"价格": "{:.2f}", "涨跌%": "{:.2f}"}
    st.dataframe(
        display_df.style
            .format(fmt, na_rep="—")
            .applymap(color_change, subset=["涨跌%"]),
        use_container_width=True,
        hide_index=True,
    )

    # Signal heatmap
    if signals:
        matrix_data = {}
        for sym in symbols:
            sym_sigs = signals.get(sym, {})
            if sym_sigs:
                label = SYMBOL_NAMES.get(sym, sym)
                matrix_data[label] = {
                    STRATEGY_LABELS.get(k, k): v
                    for k, v in sym_sigs.items()
                    if k != "composite_score"
                }
        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data).T
            fig_heat = go.Figure(data=go.Heatmap(
                z=matrix_df.values,
                x=matrix_df.columns.tolist(),
                y=matrix_df.index.tolist(),
                colorscale=[[0, "#ef5350"], [0.5, "#424242"], [1, "#26a69a"]],
                zmin=-1, zmax=1,
                text=matrix_df.values,
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                height=max(250, len(matrix_data) * 22),
                margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ── Tabs ───────────────────────────────────────────────────────────────────
st.title("📊 量化盯盘平台")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 盯盘总览", "📈 策略回测", "🔁 策略对比", "💼 持仓总览", "⚙️ 设置"
])


# ══════════════════════════════════════════════════════════════════════════
# Tab 1: Dashboard — all markets
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    col_refresh, col_time = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 刷新"):
            st.cache_data.clear()
            st.rerun()
    with col_time:
        st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for mkt in ACTIVE_MARKETS:
        with st.expander(f"{_MARKET_LABEL[mkt]}", expanded=(mkt == "us")):
            _render_market_section(mkt)


# ══════════════════════════════════════════════════════════════════════════
# Tab 2: Strategy Backtest
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("单策略回测")

    col_mkt, col_sym, col_strat, col_yr = st.columns([1, 2, 2, 1])
    with col_mkt:
        bt_market = st.selectbox(
            "市场", ACTIVE_MARKETS,
            format_func=lambda m: _MARKET_LABEL[m],
            key="bt_market",
        )
    with col_sym:
        bt_symbols = [s for g in MARKET_WATCHLISTS.get(bt_market, {}).values() for s in g]
        bt_symbol = st.selectbox(
            "标的",
            bt_symbols,
            format_func=sym_display,
            key="bt_symbol",
        )
    with col_strat:
        bt_strategy = st.selectbox(
            "策略",
            list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k],
            key="bt_strategy",
        )
    with col_yr:
        bt_years = st.slider("年数", 1, 10, 5, key="bt_years")

    if st.button("▶ 运行回测", key="run_bt"):
        with st.spinner("运行回测..."):
            try:
                df = get_history(bt_symbol, years=bt_years, market=bt_market)
                vix_df = get_vix() if bt_market == "us" else None

                if bt_strategy == "vix_timing":
                    result = STRATEGY_FUNCTIONS[bt_strategy](df, vix_df=vix_df)
                elif bt_strategy == "composite_score":
                    from strategies.composite import composite_score
                    result = composite_score(df, symbol=bt_symbol, vix_df=vix_df)
                else:
                    result = STRATEGY_FUNCTIONS[bt_strategy](df)

                from backtest.engine import run_backtest, run_benchmark
                bt_result = run_backtest(df, result["signal_series"])
                bm_result = run_benchmark(df)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("总收益", f"{bt_result['total_return']:.1f}%",
                          f"vs B&H {bm_result['total_return']:.1f}%")
                m2.metric("年化收益", f"{bt_result['annual_return']:.1f}%",
                          f"vs {bm_result['annual_return']:.1f}%")
                m3.metric("最大回撤", f"{bt_result['max_drawdown']:.1f}%",
                          f"vs {bm_result['max_drawdown']:.1f}%")
                m4.metric("夏普比率", f"{bt_result['sharpe_ratio']:.2f}",
                          f"vs {bm_result['sharpe_ratio']:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bt_result["portfolio_value"].index,
                    y=bt_result["portfolio_value"],
                    name=STRATEGY_LABELS.get(bt_strategy, bt_strategy),
                    line=dict(color="#26a69a"),
                ))
                fig.add_trace(go.Scatter(
                    x=bm_result["portfolio_value"].index,
                    y=bm_result["portfolio_value"],
                    name="Buy & Hold",
                    line=dict(color="#888", dash="dash"),
                ))
                title_name = sym_display(bt_symbol)
                fig.update_layout(
                    title=f"{title_name} · {STRATEGY_LABELS.get(bt_strategy, bt_strategy)}",
                    height=400, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="Portfolio Value",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("K线 + 指标")
                recent_df = df.tail(252)
                candle_fig = _candlestick_fig(recent_df, bt_symbol)
                indicators = result.get("indicators", {})
                _skip = {"RSI", "ADX", "MACD", "Signal", "Histogram", "ROC20", "ROC60",
                         "BB_Width", "VIX", "Fear_Level", "Greed_Level",
                         "Composite_Score", "Buy_Threshold", "Sell_Threshold",
                         "Oversold", "Overbought"}
                for name, series in indicators.items():
                    if name not in _skip:
                        candle_fig.add_trace(go.Scatter(
                            x=series.tail(252).index, y=series.tail(252),
                            name=name, line=dict(width=1),
                        ))
                st.plotly_chart(candle_fig, use_container_width=True)

            except Exception as e:
                st.error(f"回测失败: {e}")
                logger.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# Tab 3: Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("全策略对比")

    col_mkt3, col_sym3 = st.columns([1, 3])
    with col_mkt3:
        comp_market = st.selectbox(
            "市场", ACTIVE_MARKETS,
            format_func=lambda m: _MARKET_LABEL[m],
            key="comp_market",
        )
    with col_sym3:
        comp_symbols = [s for g in MARKET_WATCHLISTS.get(comp_market, {}).values() for s in g]
        comp_symbol = st.selectbox(
            "标的",
            comp_symbols,
            format_func=sym_display,
            key="comp_symbol",
        )

    if st.button("▶ 运行全量对比", key="run_comp"):
        with st.spinner("运行所有策略回测（约需10秒）..."):
            try:
                df = get_history(comp_symbol, market=comp_market)
                results = compute_signals(comp_symbol, market=comp_market)

                from backtest.engine import compare_strategies
                comparison_df = compare_strategies(df, results, symbol=comp_symbol)

                def highlight_best(s):
                    return ["background-color: #1b5e20" if v == s.max() else "" for v in s]

                def highlight_worst(s):
                    return ["background-color: #b71c1c" if v == s.min() else "" for v in s]

                renamed = comparison_df.rename(columns={
                    "total_return_%": "总收益%",
                    "annual_return_%": "年化%",
                    "max_drawdown_%": "最大回撤%",
                    "sharpe_ratio": "夏普",
                    "n_trades": "交易次数",
                    "final_value": "终值",
                })
                renamed.index = [STRATEGY_LABELS.get(i, i) for i in renamed.index]

                st.dataframe(
                    renamed.style
                        .apply(highlight_best, subset=["年化%", "夏普"])
                        .apply(highlight_worst, subset=["最大回撤%"]),
                    use_container_width=True,
                )

                fig_bar = go.Figure(go.Bar(
                    x=renamed.index,
                    y=renamed["年化%"],
                    marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in renamed["年化%"]],
                ))
                title_name = sym_display(comp_symbol)
                fig_bar.update_layout(
                    title=f"{title_name} — 各策略年化收益对比",
                    height=350, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=80),
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"对比失败: {e}")
                logger.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# Tab 4: Portfolio — Alpaca (US) + Virtual (HK, CN)
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("持仓总览")

    if st.button("🔄 刷新持仓", key="refresh_positions"):
        st.cache_data.clear()
        st.rerun()

    def color_pl(val):
        try:
            return "color: #26a69a" if float(val) >= 0 else "color: #ef5350"
        except Exception:
            return ""

    # ── US: Alpaca ─────────────────────────────────────────────────────
    with st.expander("🇺🇸 美股 — Alpaca 纸账户", expanded=True):
        if not ALPACA_API_KEY:
            st.warning("未配置 Alpaca API Key")
        else:
            try:
                from paper_trade.alpaca_trader import get_portfolio_summary
                portfolio = get_portfolio_summary()
                acct = portfolio.get("account", {})
                if acct:
                    a1, a2, a3 = st.columns(3)
                    a1.metric("总值", f"${acct.get('portfolio_value', 0):,.2f}")
                    a2.metric("现金", f"${acct.get('cash', 0):,.2f}")
                    a3.metric("可用买力", f"${acct.get('buying_power', 0):,.2f}")

                positions = portfolio.get("positions", [])
                if positions:
                    pos_df = pd.DataFrame(positions)
                    pos_df["unrealized_plpc"] = pos_df["unrealized_plpc"].round(2)
                    st.dataframe(
                        pos_df.style.applymap(color_pl, subset=["unrealized_pl", "unrealized_plpc"]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("当前无持仓")

                orders = portfolio.get("recent_orders", [])
                if orders:
                    st.caption("最近订单")
                    st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Alpaca 获取失败: {e}")

    # ── HK + CN: Virtual Portfolios ────────────────────────────────────
    for mkt in [m for m in ACTIVE_MARKETS if m in ("hk", "cn")]:
        currency = VIRTUAL_PORTFOLIO_CURRENCY.get(mkt, "")
        cap = VIRTUAL_PORTFOLIO_CAPITAL.get(mkt, 0)
        with st.expander(f"{_MARKET_LABEL[mkt]} — 虚拟账户 (总资金 {currency}{cap:,.0f})", expanded=True):
            if not (HF_TOKEN and HF_DATASET_REPO):
                st.warning("未配置 HF Token / Dataset Repo，无法读取虚拟持仓")
            else:
                try:
                    from paper_trade.virtual_portfolio import VirtualPortfolio
                    symbols_mkt = [s for g in MARKET_WATCHLISTS.get(mkt, {}).values() for s in g]
                    # Get latest prices for P&L
                    raw_quotes = get_quotes(symbols_mkt, market=mkt)
                    prices = {q["symbol"]: q["price"] for q in raw_quotes if q.get("price")}

                    vp = VirtualPortfolio(
                        market=mkt,
                        total_capital=cap,
                        currency=currency,
                        hf_repo=HF_DATASET_REPO,
                        hf_token=HF_TOKEN,
                    )
                    summary = vp.get_summary(prices=prices)

                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("总资金", f"{currency}{summary['total_capital']:,.0f}")
                    v2.metric("已投入", f"{currency}{summary['invested']:,.0f}")
                    v3.metric("可用余额", f"{currency}{summary['available']:,.0f}")
                    v4.metric("未实现盈亏", f"{currency}{summary['unrealized_pnl']:+,.0f}")

                    open_pos = summary.get("open_positions", [])
                    if open_pos:
                        st.caption(f"持仓 ({len(open_pos)} 个标的)")
                        pos_df = pd.DataFrame(open_pos)
                        # Add name column
                        pos_df["名称"] = pos_df["symbol"].map(lambda s: SYMBOL_NAMES.get(s, ""))
                        display_pos_cols = [c for c in ["symbol", "名称", "entry_date", "entry_price",
                                                         "shares", "notional", "current_price",
                                                         "unrealized_pnl", "unrealized_pct"] if c in pos_df.columns]
                        pos_df_show = pos_df[display_pos_cols].rename(columns={
                            "symbol": "代码", "entry_date": "买入日",
                            "entry_price": "成本价", "shares": "股数",
                            "notional": "成本金额", "current_price": "现价",
                            "unrealized_pnl": "浮盈亏", "unrealized_pct": "盈亏%",
                        })
                        pnl_cols = [c for c in ["浮盈亏", "盈亏%"] if c in pos_df_show.columns]
                        if pnl_cols:
                            st.dataframe(
                                pos_df_show.style.applymap(color_pl, subset=pnl_cols),
                                use_container_width=True, hide_index=True,
                            )
                        else:
                            st.dataframe(pos_df_show, use_container_width=True, hide_index=True)
                    else:
                        st.info("当前无持仓")

                    history = vp.get_trade_history(limit=20)
                    if history:
                        st.caption("最近交易记录")
                        hist_df = pd.DataFrame(history)
                        hist_df["名称"] = hist_df["symbol"].map(lambda s: SYMBOL_NAMES.get(s, ""))
                        display_hist_cols = [c for c in ["date", "symbol", "名称", "action",
                                                          "price", "shares", "notional", "pnl"] if c in hist_df.columns]
                        hist_show = hist_df[display_hist_cols].rename(columns={
                            "date": "日期", "symbol": "代码", "action": "操作",
                            "price": "价格", "shares": "股数", "notional": "金额", "pnl": "盈亏",
                        })
                        pnl_cols_h = [c for c in ["盈亏"] if c in hist_show.columns]
                        if pnl_cols_h:
                            st.dataframe(
                                hist_show.style.applymap(color_pl, subset=pnl_cols_h),
                                use_container_width=True, hide_index=True,
                            )
                        else:
                            st.dataframe(hist_show, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"{_MARKET_LABEL[mkt]} 虚拟账户读取失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Tab 5: Settings
# ══════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("配置状态")

    c1, c2, c3 = st.columns(3)
    c1.metric("Alpaca", "✅ 已配置" if ALPACA_API_KEY else "❌ 未配置")
    c2.metric("飞书 Webhook", "✅ 已配置" if FEISHU_WEBHOOK_URL else "❌ 未配置")
    c3.metric("HF Dataset", "✅ 已配置" if HF_DATASET_REPO else "❌ 未配置")

    st.divider()
    st.subheader("手动触发 Pipeline")

    col_a, col_b, col_c = st.columns(3)
    for col, mkt in zip([col_a, col_b, col_c], ["us", "hk", "cn"]):
        with col:
            if st.button(f"▶ {_MARKET_LABEL[mkt]}", key=f"run_{mkt}"):
                with st.spinner(f"运行 {_MARKET_LABEL[mkt]} pipeline..."):
                    try:
                        from scheduler.jobs import run_daily_pipeline
                        result = run_daily_pipeline(mkt)
                        st.success(f"完成！{len(result['symbols'])} 个标的，{len(result['errors'])} 个错误")
                        if result["errors"]:
                            for err in result["errors"]:
                                st.warning(err)
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"失败: {e}")

    st.divider()
    if st.button("📤 发送测试飞书消息", key="test_feishu"):
        if not FEISHU_WEBHOOK_URL:
            st.error("未配置 FEISHU_WEBHOOK_URL")
        else:
            from alerts.feishu import send_text_message
            ok = send_text_message(FEISHU_WEBHOOK_URL, "🤖 量化盯盘平台连接测试 ✅")
            st.success("发送成功！") if ok else st.error("发送失败")

    st.divider()
    st.subheader("关注标的")
    for mkt in ACTIVE_MARKETS:
        wl = MARKET_WATCHLISTS.get(mkt, {})
        st.caption(f"**{_MARKET_LABEL[mkt]}**")
        etfs = wl.get("etf", [])
        stocks = wl.get("stocks", [])
        if etfs:
            st.write("ETF:", ", ".join(f"{s}({SYMBOL_NAMES.get(s, '')})" for s in etfs))
        if stocks:
            st.write("个股:", ", ".join(f"{s}({SYMBOL_NAMES.get(s, '')})" for s in stocks))
