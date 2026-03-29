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

# ── Watchlist ─────────────────────────────────────────────────────────────
WATCHLIST = {
    "etf": ["QQQ", "TQQQ", "SQQQ", "SPY", "UVXY"],
    "stocks": ["NVDA", "TSLA", "AAPL", "META", "MSTR"],
}
ALL_SYMBOLS = WATCHLIST["etf"] + WATCHLIST["stocks"]

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

# ── Scheduler (US Eastern time) ───────────────────────────────────────────
# Run after market close: 16:30 ET
DAILY_JOB_HOUR = 16
DAILY_JOB_MINUTE = 30
TIMEZONE = "America/New_York"

# ── Paper trade ───────────────────────────────────────────────────────────
PAPER_TRADE_POSITION_SIZE = 0.1   # 10% of portfolio per signal
