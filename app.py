"""
Streamlit UI — US Stock Quantitative Monitoring Platform
Tabs:
  1. 📊 盯盘总览    — real-time quotes + composite signal dashboard
  2. 📈 策略回测    — backtest a single strategy on a selected symbol
  3. 🔁 策略对比    — compare all strategies for a symbol
  4. 💼 纸交易持仓  — Alpaca paper trading portfolio
  5. ⚙️  设置        — manual pipeline trigger + config status
"""

import logging
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import os

from config import (
    ALL_SYMBOLS,
    ALPACA_API_KEY,
    FEISHU_WEBHOOK_URL,
    HF_DATASET_REPO,
    HF_TOKEN,
    HISTORY_YEARS,
    STRATEGY_PARAMS,
    UI_REFRESH_SECONDS,
    WATCHLIST,
)
from strategies import STRATEGY_FUNCTIONS, STRATEGY_LABELS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Stock Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────────
if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None
if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = False


# ── Helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=UI_REFRESH_SECONDS)
def get_quotes(symbols: list[str]) -> list[dict]:
    from data.fetcher import fetch_quotes
    return fetch_quotes(symbols)


@st.cache_data(ttl=300)
def get_history(symbol: str, years: int = HISTORY_YEARS) -> pd.DataFrame:
    from data.fetcher import fetch_history
    return fetch_history(symbol, years=years)


@st.cache_data(ttl=300)
def get_vix() -> pd.DataFrame:
    from data.fetcher import fetch_history
    return fetch_history("^VIX", years=HISTORY_YEARS)


@st.cache_data(ttl=300)
def compute_signals(symbol: str) -> dict:
    """Full signal computation — used by backtest tabs (always fresh)."""
    from strategies.composite import run_all_strategies
    df = get_history(symbol)
    vix_df = get_vix()
    return run_all_strategies(df, symbol=symbol, vix_df=vix_df)


@st.cache_data(ttl=300)
def get_dashboard_signals(market: str = "us") -> dict[str, dict[str, int]]:
    """
    Load today's pre-computed signals from HF Dataset for the dashboard.
    Falls back to computing fresh for US symbols only (fast enough for Streamlit).
    Returns {symbol: {strategy: signal_int}}.
    """
    from data.fetcher import load_today_signals_from_hf
    from config import MARKET_WATCHLISTS
    if HF_TOKEN and HF_DATASET_REPO:
        signals = load_today_signals_from_hf(
            market=market, hf_repo=HF_DATASET_REPO, hf_token=HF_TOKEN
        )
        if signals:
            return signals
    # Fallback: compute on-the-fly for this market's symbols only
    watchlist = MARKET_WATCHLISTS.get(market, {})
    market_symbols = [s for group in watchlist.values() for s in group]
    result = {}
    for sym in market_symbols:
        try:
            r = compute_signals(sym)
            result[sym] = {k: v["signal"] for k, v in r.items()}
        except Exception:
            pass
    return result


def signal_badge(signal: int) -> str:
    return {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}.get(signal, "—")


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


# ── Tabs ──────────────────────────────────────────────────────────────────
st.title("📊 美股量化盯盘平台")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 盯盘总览", "📈 策略回测", "🔁 策略对比", "💼 纸交易持仓", "⚙️ 设置"
])


# ════════════════════════════════════════════════════════════════════════
# Tab 1: Dashboard
# ════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("实时行情 + 综合信号")

    col_refresh, col_time = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 刷新"):
            st.cache_data.clear()
            st.rerun()
    with col_time:
        st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  自动 {UI_REFRESH_SECONDS}s 刷新")

    # Market selector for dashboard
    selected_market = st.selectbox(
        "市场", ["us", "hk", "cn"],
        format_func=lambda m: {"us": "🇺🇸 美股", "hk": "🇭🇰 港股", "cn": "🇨🇳 A股"}[m],
        key="dash_market",
    )

    from config import MARKET_WATCHLISTS
    market_symbols = [s for g in MARKET_WATCHLISTS.get(selected_market, {}).values() for s in g]

    # Try HF Dataset first; fall back only if necessary
    dashboard_signals: dict = {}
    signal_source = "实时计算"
    if HF_TOKEN and HF_DATASET_REPO:
        with st.spinner("📡 读取 HF Dataset 信号..."):
            from data.fetcher import load_today_signals_from_hf
            dashboard_signals = load_today_signals_from_hf(
                market=selected_market, hf_repo=HF_DATASET_REPO, hf_token=HF_TOKEN
            ) or {}
        if dashboard_signals:
            signal_source = "HF Dataset ✅"

    if not dashboard_signals:
        st.info("⏳ HF Dataset 中暂无今日信号（daily pipeline 尚未运行），正在实时计算中，请稍等约 2 分钟...")
        with st.spinner(f"实时计算 {len(market_symbols)} 个标的信号..."):
            progress = st.progress(0)
            for i, sym in enumerate(market_symbols):
                try:
                    r = compute_signals(sym)
                    dashboard_signals[sym] = {k: v["signal"] for k, v in r.items()}
                except Exception:
                    pass
                progress.progress((i + 1) / len(market_symbols))
            progress.empty()

    st.caption(f"信号来源: {signal_source}")

    with st.spinner("加载行情..."):
        quotes = get_quotes(market_symbols)

    quotes_df = pd.DataFrame(quotes)
    if not quotes_df.empty and "price" in quotes_df.columns:
        quotes_df = quotes_df[quotes_df["price"].notna()]

        signal_col = []
        score_col = []
        for sym in quotes_df["symbol"]:
            try:
                sym_signals = dashboard_signals.get(sym, {})
                comp_signal = sym_signals.get("composite_score", 0)
                # score = count of bullish signals
                total = sum(v for v in sym_signals.values() if v == 1) - sum(1 for v in sym_signals.values() if v == -1)
                max_s = len(sym_signals) - 1  # exclude composite itself
                signal_col.append(signal_badge(comp_signal))
                score_col.append(f"{total}/{max_s}" if max_s else "—")
            except Exception:
                signal_col.append("—")
                score_col.append("—")

        quotes_df["综合信号"] = signal_col
        quotes_df["评分"] = score_col

        # Style change column
        def color_change(val):
            try:
                v = float(val)
                color = "#26a69a" if v >= 0 else "#ef5350"
                return f"color: {color}"
            except Exception:
                return ""

        from config import SYMBOL_NAMES
        quotes_df["名称"] = quotes_df["symbol"].map(lambda s: SYMBOL_NAMES.get(s, ""))
        display_cols = ["symbol", "名称", "price", "change", "change_pct", "综合信号", "评分"]
        display_df = quotes_df[[c for c in display_cols if c in quotes_df.columns]].rename(columns={
            "symbol": "标的", "名称": "名称", "price": "价格", "change": "变动", "change_pct": "涨跌%"
        })
        st.dataframe(
            display_df.style.applymap(color_change, subset=["涨跌%"]),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Signal heatmap: symbol × strategy
    st.subheader("信号矩阵")
    matrix_data = {}
    for sym in ALL_SYMBOLS:
        sym_signals = dashboard_signals.get(sym, {})
        if sym_signals:
            matrix_data[sym] = {
                STRATEGY_LABELS.get(k, k): v
                for k, v in sym_signals.items()
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
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Auto-refresh
    time.sleep(0.1)
    st.caption(f"⏱ 页面将在 {UI_REFRESH_SECONDS}s 后自动刷新")


# ════════════════════════════════════════════════════════════════════════
# Tab 2: Strategy Backtest
# ════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("单策略回测")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_symbol = st.selectbox("选择标的", ALL_SYMBOLS, key="bt_symbol")
    with col2:
        bt_strategy = st.selectbox(
            "选择策略",
            list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k],
            key="bt_strategy",
        )
    with col3:
        bt_years = st.slider("回测年数", 1, 10, 5, key="bt_years")

    if st.button("▶ 运行回测", key="run_bt"):
        with st.spinner("运行回测..."):
            try:
                df = get_history(bt_symbol, years=bt_years)
                vix_df = get_vix()

                # Run strategy
                if bt_strategy == "vix_timing":
                    result = STRATEGY_FUNCTIONS[bt_strategy](df, vix_df=vix_df)
                elif bt_strategy == "composite_score":
                    from strategies.composite import composite_score
                    result = composite_score(df, symbol=bt_symbol, vix_df=vix_df)
                else:
                    params = STRATEGY_PARAMS.get(bt_strategy.replace("_strategy", "").replace("_crossover", "").replace("_momentum", "").replace("_squeeze", "").replace("_channel", ""), {})
                    result = STRATEGY_FUNCTIONS[bt_strategy](df)

                # Run backtest
                from backtest.engine import run_backtest, run_benchmark
                bt_result = run_backtest(df, result["signal_series"])
                bm_result = run_benchmark(df)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("总收益", f"{bt_result['total_return']:.1f}%", f"vs 买持 {bm_result['total_return']:.1f}%")
                m2.metric("年化收益", f"{bt_result['annual_return']:.1f}%", f"vs {bm_result['annual_return']:.1f}%")
                m3.metric("最大回撤", f"{bt_result['max_drawdown']:.1f}%", f"vs {bm_result['max_drawdown']:.1f}%")
                m4.metric("夏普比率", f"{bt_result['sharpe_ratio']:.2f}", f"vs {bm_result['sharpe_ratio']:.2f}")

                # Portfolio value chart
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
                fig.update_layout(
                    title=f"{bt_symbol} · {STRATEGY_LABELS.get(bt_strategy, bt_strategy)}",
                    height=400, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="Portfolio Value ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Candlestick + indicators
                st.subheader("K线 + 指标")
                recent_df = df.tail(252)  # last year
                candle_fig = _candlestick_fig(recent_df, bt_symbol)

                indicators = result.get("indicators", {})
                price_indicators = {k: v for k, v in indicators.items()
                                    if k not in ("RSI", "ADX", "MACD", "Signal", "Histogram",
                                                 "ROC20", "ROC60", "BB_Width", "VIX",
                                                 "Fear_Level", "Greed_Level", "Composite_Score",
                                                 "Buy_Threshold", "Sell_Threshold",
                                                 "Oversold", "Overbought")}
                for name, series in price_indicators.items():
                    candle_fig.add_trace(go.Scatter(
                        x=series.tail(252).index, y=series.tail(252),
                        name=name, line=dict(width=1),
                    ))
                st.plotly_chart(candle_fig, use_container_width=True)

            except Exception as e:
                st.error(f"回测失败: {e}")
                logger.exception(e)


# ════════════════════════════════════════════════════════════════════════
# Tab 3: Strategy Comparison
# ════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("全策略对比")

    comp_symbol = st.selectbox("选择标的", ALL_SYMBOLS, key="comp_symbol")

    if st.button("▶ 运行全量对比", key="run_comp"):
        with st.spinner("运行所有策略回测（约需10秒）..."):
            try:
                df = get_history(comp_symbol)
                vix_df = get_vix()
                results = compute_signals(comp_symbol)

                from backtest.engine import compare_strategies
                comparison_df = compare_strategies(df, results, symbol=comp_symbol)

                # Highlight best
                def highlight_best(s):
                    is_max = s == s.max()
                    return ["background-color: #1b5e20" if v else "" for v in is_max]

                def highlight_worst(s):
                    is_min = s == s.min()
                    return ["background-color: #b71c1c" if v else "" for v in is_min]

                renamed = comparison_df.rename(columns={
                    "total_return_%": "总收益%",
                    "annual_return_%": "年化%",
                    "max_drawdown_%": "最大回撤%",
                    "sharpe_ratio": "夏普",
                    "n_trades": "交易次数",
                    "final_value": "终值($)",
                })
                renamed.index = [STRATEGY_LABELS.get(i, i) for i in renamed.index]

                st.dataframe(
                    renamed.style
                        .apply(highlight_best, subset=["年化%", "夏普"])
                        .apply(highlight_worst, subset=["最大回撤%"]),
                    use_container_width=True,
                )

                # Annual return bar chart
                fig_bar = go.Figure(go.Bar(
                    x=renamed.index,
                    y=renamed["年化%"],
                    marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in renamed["年化%"]],
                ))
                fig_bar.update_layout(
                    title=f"{comp_symbol} — 各策略年化收益对比",
                    height=350, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=80),
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"对比失败: {e}")
                logger.exception(e)


# ════════════════════════════════════════════════════════════════════════
# Tab 4: Paper Trading Portfolio
# ════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Alpaca 纸交易持仓")

    if not ALPACA_API_KEY:
        st.warning("未配置 Alpaca API Key。请在 .env 文件中设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY。")
    else:
        if st.button("🔄 刷新持仓", key="refresh_positions"):
            st.cache_data.clear()

        try:
            from paper_trade.alpaca_trader import get_portfolio_summary
            portfolio = get_portfolio_summary()

            acct = portfolio.get("account", {})
            if acct:
                a1, a2, a3 = st.columns(3)
                a1.metric("投资组合总值", f"${acct.get('portfolio_value', 0):,.2f}")
                a2.metric("现金", f"${acct.get('cash', 0):,.2f}")
                a3.metric("可用买入力", f"${acct.get('buying_power', 0):,.2f}")

            positions = portfolio.get("positions", [])
            if positions:
                pos_df = pd.DataFrame(positions)
                pos_df["unrealized_plpc"] = pos_df["unrealized_plpc"].round(2)

                def color_pl(val):
                    try:
                        return "color: #26a69a" if float(val) >= 0 else "color: #ef5350"
                    except Exception:
                        return ""

                st.dataframe(
                    pos_df.style.applymap(color_pl, subset=["unrealized_pl", "unrealized_plpc"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("当前无持仓。")

            st.divider()
            st.subheader("最近订单")
            orders = portfolio.get("recent_orders", [])
            if orders:
                st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)
            else:
                st.info("暂无历史订单。")

        except Exception as e:
            st.error(f"获取持仓失败: {e}")


# ════════════════════════════════════════════════════════════════════════
# Tab 5: Settings
# ════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("配置状态")

    c1, c2, c3 = st.columns(3)
    c1.metric("Alpaca", "✅ 已配置" if ALPACA_API_KEY else "❌ 未配置")
    c2.metric("飞书 Webhook", "✅ 已配置" if FEISHU_WEBHOOK_URL else "❌ 未配置")
    c3.metric("HF Dataset", "✅ 已配置" if HF_DATASET_REPO else "❌ 未配置")

    st.divider()
    st.subheader("手动触发")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("▶ 立即运行每日流水线", key="run_pipeline"):
            with st.spinner("运行中，约需30秒..."):
                try:
                    from scheduler.jobs import run_daily_pipeline
                    result = run_daily_pipeline("us")
                    st.success(f"完成！处理 {len(result['symbols'])} 个标的，{len(result['errors'])} 个错误")
                    if result["errors"]:
                        for err in result["errors"]:
                            st.warning(err)
                    st.json(result["symbols"])
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"流水线失败: {e}")

    with col_b:
        if st.button("📤 发送测试飞书消息", key="test_feishu"):
            if not FEISHU_WEBHOOK_URL:
                st.error("未配置 FEISHU_WEBHOOK_URL")
            else:
                from alerts.feishu import send_text_message
                ok = send_text_message(FEISHU_WEBHOOK_URL, "🤖 US Stock Monitor 连接测试 ✅")
                st.success("发送成功！") if ok else st.error("发送失败，检查 webhook URL")

    st.divider()
    st.subheader("关注标的")
    st.write("**ETF:**", ", ".join(WATCHLIST["etf"]))
    st.write("**个股:**", ", ".join(WATCHLIST["stocks"]))
    st.caption("⏱ 每日流水线由 GitHub Actions 在 21:30 UTC 自动触发")
