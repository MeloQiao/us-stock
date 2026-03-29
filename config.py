import os
from dotenv import load_dotenv

load_dotenv()

# ── Alpaca ──────────────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Feishu ───────────────────────────────────────────────────────────────
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")

# ── Hugging Face ─────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")

# ── Market-specific watchlists ────────────────────────────────────────────
# US: ready. HK / CN: placeholders, populate when those markets are enabled.
MARKET_WATCHLISTS: dict[str, dict[str, list[str]]] = {
    "us": {
        "etf":    ["QQQ", "TQQQ", "SQQQ", "SPY", "UVXY"],
        "stocks": ["NVDA", "TSLA", "AAPL", "META", "MSTR"],
    },
    "hk": {
        # yfinance format — 5-digit code WITHOUT .HK suffix (auto-appended by fetcher)
        "etf": [
            "02800",  # 盈富基金 (Tracker Fund / HSI) — SPY equiv
            "03032",  # 南方纳斯达克100 ETF          — QQQ equiv
            "03067",  # iShares 恒生科技 ETF           — HK tech index
            "03188",  # 华夏沪深300 ETF (H shares)    — CSI 300 in HK
            "07226",  # 南方2x做多恒生科技             — TQQQ equiv (LEVERAGED)
            "07500",  # 三星2x做多恒生指数             — TQQQ-HSI (LEVERAGED)
        ],
        "stocks": [
            "00700",  # 腾讯    — AAPL (stable trend, huge ecosystem)
            "09988",  # 阿里巴巴 — TSLA (high volatility, recovery play)
            "03690",  # 美团    — META (strong domestic trend)
            "09618",  # 京东    — mid-beta e-commerce
            "09999",  # 网易    — stable gaming + education
            "01024",  # 快手    — high-beta short video (social media)
            "00981",  # 中芯国际H — NVDA proxy (China's leading foundry)
            "01211",  # 比亚迪H  — TSLA proxy (world's largest EV maker)
            "09866",  # 蔚来    — high-beta EV (premium segment)
            "02015",  # 理想汽车 — strong trend EV
            "02382",  # 舜宇光学 — AI optics hardware
            "09626",  # 哔哩哔哩 — high-beta content platform
        ],
    },
    "cn": {
        # akshare format — 6-digit code only, no exchange prefix
        "etf": [
            "510300",  # 华泰沪深300ETF    — SPY equiv (benchmark)
            "159915",  # 易方达创业板ETF   — QQQ equiv (China growth/tech)
            "588000",  # 华夏科创50ETF     — STAR Market (China's Nasdaq)
            "512480",  # 国联半导体ETF     — SOXX equiv (AI hardware theme)
            "515070",  # 华夏AI ETF        — pure AI sector
            "510500",  # 南方中证500ETF    — mid-cap (IWM equiv)
            "159741",  # 国泰纳斯达克100   — US tech exposure via A-shares
            "159745",  # 新能源ETF         — EV + battery theme
        ],
        "stocks": [
            "600519",  # 贵州茅台  — AAPL (most stable trend in A-shares)
            "300750",  # 宁德时代  — NVDA (battery + EV momentum king)
            "002594",  # 比亚迪A   — TSLA proxy (EV)
            "688981",  # 中芯国际A — NVDA proxy (China's leading foundry)
            "688111",  # 金山办公  — AI software (China's Microsoft Office)
            "603501",  # 韦尔股份  — AI image sensor (camera supply chain)
            "688041",  # 海光信息  — domestic compute / AI infra (high beta)
            "300760",  # 迈瑞医疗  — stable medical devices
            "601012",  # 隆基绿能  — solar energy trend
            "002415",  # 海康威视  — AI vision / surveillance
            "600036",  # 招商银行  — China's best retail bank (stable)
            "000333",  # 美的集团  — stable consumer appliances
        ],
    },
}

# Active markets — toggle here when ready to enable HK / CN
ACTIVE_MARKETS: list[str] = ["us"]

# Convenience: combined list for the active markets
ALL_SYMBOLS: list[str] = [
    s
    for mkt in ACTIVE_MARKETS
    for group in MARKET_WATCHLISTS[mkt].values()
    for s in group
]

# Legacy alias (used by strategies)
WATCHLIST = MARKET_WATCHLISTS["us"]

# Leveraged/inverse ETFs — only use trend strategies, no mean reversion
# Includes HK 2x ETFs (07226 / 07500)
LEVERAGED_ETFS = {"TQQQ", "SQQQ", "UVXY", "07226", "07500"}

# ── Strategy parameters ───────────────────────────────────────────────────
STRATEGY_PARAMS = {
    "golden_cross": {"fast": 50, "slow": 200},
    "supertrend": {"atr_period": 10, "multiplier": 3.0},
    "donchian": {"entry_period": 20, "exit_period": 10},
    "ema_adx": {"ema_fast": 12, "ema_slow": 26, "adx_period": 14, "adx_threshold": 25},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "roc": {"period_short": 20, "period_long": 60},
    "rsi": {"period": 14, "oversold": 30, "overbought": 70},
    "bollinger": {"period": 20, "std_dev": 2.0, "squeeze_threshold": 0.1},
    "vix": {"fear_threshold": 30, "greed_threshold": 15},
    "composite": {"buy_threshold": 6, "sell_threshold": 4},
}

# ── Data settings ─────────────────────────────────────────────────────────
HISTORY_YEARS = 10        # years of historical data for backtest
DATA_INTERVAL = "1d"      # daily candles
UI_REFRESH_SECONDS = 30   # Streamlit auto-refresh interval

# ── Per-market scheduler config ───────────────────────────────────────────
# Each entry: close time (local) + IANA timezone
MARKET_SCHEDULE: dict[str, dict] = {
    "us": {"hour": 16, "minute": 30, "timezone": "America/New_York"},
    "hk": {"hour": 16, "minute":  0, "timezone": "Asia/Hong_Kong"},
    "cn": {"hour": 15, "minute":  0, "timezone": "Asia/Shanghai"},
}

# Legacy aliases kept for backward-compat
DAILY_JOB_HOUR   = MARKET_SCHEDULE["us"]["hour"]
DAILY_JOB_MINUTE = MARKET_SCHEDULE["us"]["minute"]
TIMEZONE         = MARKET_SCHEDULE["us"]["timezone"]

# ── Paper trade ───────────────────────────────────────────────────────────
PAPER_TRADE_POSITION_SIZE = 0.1   # 10% of portfolio per signal
