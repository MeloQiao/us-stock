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
        # yfinance format — 5-digit code (leading zeros) WITHOUT .HK suffix;
        # the fetcher appends .HK automatically.
        # e.g. "00700" = Tencent, "09988" = Alibaba, "02800" = Tracker Fund ETF
        "etf":    [],
        "stocks": [],
    },
    "cn": {
        # akshare format — 6-digit code only, no exchange prefix.
        # e.g. "510300" = CSI 300 ETF, "600519" = Moutai, "000858" = Wuliangye
        "etf":    [],
        "stocks": [],
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
LEVERAGED_ETFS = {"TQQQ", "SQQQ", "UVXY"}

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
