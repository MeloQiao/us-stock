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
HF_TOKEN       = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")

# ── Market-specific watchlists ────────────────────────────────────────────
# US: ready. HK / CN: placeholders, populate when those markets are enabled.
MARKET_WATCHLISTS: dict[str, dict[str, list[str]]] = {
    "us": {
        "etf": [
            # ── 宽基指数 ──────────────────────────────────────────────────
            "SPY",   # 标普500           — 基准
            "QQQ",   # 纳指100           — 科技主战场
            "IWM",   # 罗素2000小盘股    — 风险偏好风向标
            "DIA",   # 道琼斯工业        — 传统蓝筹基准
            # ── 行业主题 ─────────────────────────────────────────────────
            "XLK",   # 科技板块ETF       — 科技精选
            "SMH",   # VanEck半导体ETF   — 芯片行业最佳代理
            "XLE",   # 能源板块ETF       — 油气整体敞口
            "ICLN",  # 全球清洁能源ETF   — 新能源主题
            "ARKK",  # ARK Innovation    — 颠覆性创新高beta
            "GLD",   # 黄金              — 宏观避险
            "TLT",   # 20年期美债        — 宏观利率方向（战术配置）
            "SGOV",  # 美国T-Bill ETF    — 现金替代，空仓时停泊资金（年化~4%）
            "IBIT",  # 贝莱德比特币ETF   — BTC现货合规代理
            # ── 杠杆/反向 (仅趋势策略) ────────────────────────────────────
            "TQQQ",  # 3x 做多纳指       — 趋势强时主攻
            "SQQQ",  # 3x 做空纳指       — 对冲/熊市
            "SPXL",  # 3x 做多标普       — 趋势强时进攻
            "SOXL",  # 3x 做多费城半导体  — AI+芯片主题杠杆
            "SOXS",  # 3x 做空费城半导体  — 芯片回调对冲
            # ── 波动率 ───────────────────────────────────────────────────
            "UVXY",  # VIX 衍生品        — 极端避险 (LEVERAGED)
        ],
        "stocks": [
            # ── 科技巨头 ──────────────────────────────────────────────────
            "AAPL",  # 苹果      — 最稳定趋势，均线策略效果好
            "MSFT",  # 微软      — AI+云计算，趋势清晰
            "GOOGL", # 谷歌      — 广告+AI+云，稳健动量
            "AMZN",  # 亚马逊    — 电商+云，趋势跟踪
            "META",  # Meta      — 社交+AI，趋势走势清晰
            "ORCL",  # 甲骨文    — 云数据库+AI基础设施
            # ── AI 芯片硬件 ───────────────────────────────────────────────
            "NVDA",  # 英伟达    — AI主线，动量策略效果极好
            "AMD",   # AMD       — NVDA替代，高beta芯片
            "AVGO",  # 博通      — AI网络芯片+自定义AI ASIC
            "MU",    # 美光科技  — HBM内存，AI训练不可或缺
            "ASML",  # 阿斯麦    — EUV光刻垄断，芯片产业链上游
            "ARM",   # ARM控股   — 芯片IP授权，AI端侧主导架构
            "MRVL",  # 迈威尔    — 数据中心自定义芯片
            "SMCI",  # 超微电脑  — AI服务器，极高beta
            "PLTR",  # Palantir  — AI软件，机构青睐
            "CRWD",  # CrowdStrike — 网络安全，强趋势
            # ── 能源 (传统+清洁+核电) ────────────────────────────────────
            "XOM",   # 埃克森美孚 — 油气龙头，宏观原油代理
            "CVX",   # 雪佛龙    — 油气，XOM对标
            "CEG",   # 星座能源  — 核电，AI数据中心供电主题
            "VST",   # 维斯塔能源 — 电力，数据中心用电增长受益
            "NEE",   # 新时代能源 — 清洁能源+电网，最大可再生能源商
            # ── 高波动/主题 ───────────────────────────────────────────────
            "TSLA",  # 特斯拉    — 高波动，趋势+动量适合
            "MSTR",  # MicroStrategy — 比特币代理，极高beta
            "COIN",  # Coinbase  — 加密货币代理，跟随BTC周期
            "RKLB",  # Rocket Lab — 商业航天，高beta小盘
            # ── 稳健蓝筹 ─────────────────────────────────────────────────
            "JPM",   # 摩根大通  — 金融龙头，利率方向指标
            "BRK-B", # 伯克希尔B — 价值投资基准
            "V",     # Visa      — 支付网络，稳定复利
        ],
    },
    "hk": {
        # yfinance format — 5-digit code WITHOUT .HK suffix (auto-appended by fetcher)
        "etf": [
            "02800",  # 盈富基金 (Tracker Fund / HSI) — SPY equiv
            "03032",  # 南方纳斯达克100 ETF          — QQQ equiv
            "03067",  # iShares 恒生科技 ETF           — HK tech index
            "03188",  # 华夏沪深300 ETF (H shares)    — CSI 300 in HK
            "07226",  # 南方2x做多恒生科技             — TQQQ equiv (LEVERAGED)
            "07552",  # 南方2x做空恒生科技             — SQQQ equiv (LEVERAGED)
            "07500",  # 三星2x做多恒生指数             — TQQQ-HSI (LEVERAGED)
        ],
        "stocks": [
            # ── 科技互联网 ─────────────────────────────────────────────────
            "00700",  # 腾讯    — 生态系统最稳定，趋势策略效果好
            "09988",  # 阿里巴巴 — 回调后修复，动量特征改善
            "03690",  # 美团    — 本地生活龙头，趋势清晰
            "09618",  # 京东    — 电商+物流，中等 beta
            "09999",  # 网易    — 稳定游戏+教育，低波动
            "09961",  # 携程    — 亚太出行龙头，复苏趋势动量好
            # ── 半导体/硬件 ────────────────────────────────────────────────
            "00981",  # 中芯国际H — 中国晶圆代工龙头，AI主线
            "02382",  # 舜宇光学 — AI光学硬件，趋势跟踪效果好
            # ── 新能源/汽车 ────────────────────────────────────────────────
            "01211",  # 比亚迪H  — 全球EV销量第一，趋势策略适合
            # ── 金融/保险 ──────────────────────────────────────────────────
            "00388",  # 港交所   — 垄断金融基础设施，趋势稳定，流动性极好
            "01299",  # 友邦保险 — 亚太区保险龙头，中长期趋势清晰
            # ── 内容平台 ───────────────────────────────────────────────────
            "09626",  # 哔哩哔哩 — 高 beta 内容平台
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

# ── Symbol display names (for HK / CN where codes aren't self-explanatory) ──
SYMBOL_NAMES: dict[str, str] = {
    # US ETFs
    "SPY": "标普500",
    "QQQ": "纳指100",
    "IWM": "罗素2000",
    "DIA": "道指ETF",
    "XLK": "科技板块",
    "SMH": "半导体ETF",
    "XLE": "能源板块",
    "ICLN": "清洁能源",
    "ARKK": "ARK创新",
    "GLD": "黄金",
    "TLT":  "20年美债",
    "SGOV": "T-Bill现金替代",
    "IBIT": "比特币ETF",
    "TQQQ": "纳指3x多",
    "SQQQ": "纳指3x空",
    "SPXL": "标普3x多",
    "SOXL": "半导体3x多",
    "SOXS": "半导体3x空",
    "UVXY": "VIX衍生品",
    # US Stocks
    "AAPL": "苹果",
    "MSFT": "微软",
    "GOOGL": "谷歌",
    "AMZN": "亚马逊",
    "META": "Meta",
    "ORCL": "甲骨文",
    "NVDA": "英伟达",
    "AMD": "AMD",
    "AVGO": "博通",
    "MU": "美光科技",
    "ASML": "阿斯麦",
    "ARM": "ARM控股",
    "MRVL": "迈威尔",
    "SMCI": "超微电脑",
    "PLTR": "Palantir",
    "CRWD": "CrowdStrike",
    "XOM": "埃克森美孚",
    "CVX": "雪佛龙",
    "CEG": "星座能源",
    "VST": "维斯塔能源",
    "NEE": "新时代能源",
    "TSLA": "特斯拉",
    "MSTR": "MicroStrategy",
    "COIN": "Coinbase",
    "RKLB": "Rocket Lab",
    "JPM": "摩根大通",
    "BRK-B": "伯克希尔B",
    "V": "Visa",
    # HK ETFs
    "02800": "盈富基金",
    "03032": "南方纳斯达克100",
    "03067": "恒生科技ETF",
    "03188": "华夏沪深300",
    "07226": "南方2x恒生科技",
    "07552": "南方2x做空恒生科技",
    "07500": "三星2x恒生指数",
    # HK Stocks
    "00700": "腾讯",
    "09988": "阿里巴巴",
    "03690": "美团",
    "09618": "京东",
    "09999": "网易",
    "09961": "携程",
    "00981": "中芯国际H",
    "02382": "舜宇光学",
    "01211": "比亚迪H",
    "00388": "港交所",
    "01299": "友邦保险",
    "09626": "哔哩哔哩",
    # CN ETFs
    "510300": "沪深300ETF",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
    "512480": "半导体ETF",
    "515070": "AI ETF",
    "510500": "中证500ETF",
    "159741": "纳斯达克100ETF",
    "159745": "新能源ETF",
    # CN Stocks
    "600519": "贵州茅台",
    "300750": "宁德时代",
    "002594": "比亚迪A",
    "688981": "中芯国际A",
    "688111": "金山办公",
    "603501": "韦尔股份",
    "688041": "海光信息",
    "300760": "迈瑞医疗",
    "601012": "隆基绿能",
    "002415": "海康威视",
    "600036": "招商银行",
    "000333": "美的集团",
}

# Active markets — us always on; hk/cn enabled with virtual portfolio
ACTIVE_MARKETS: list[str] = ["us", "hk", "cn"]

# ── Virtual portfolio (HK / CN — no paper trading API) ───────────────────
# Capital in local currency. Positions tracked in HF Dataset.
VIRTUAL_PORTFOLIO_CAPITAL: dict[str, float] = {
    "hk": 1_000_000,  # HKD 100万
    "cn": 1_000_000,  # CNY 100万
}
VIRTUAL_PORTFOLIO_CURRENCY: dict[str, str] = {
    "hk": "HKD",
    "cn": "CNY",
}

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
LEVERAGED_ETFS = {"TQQQ", "SQQQ", "SPXL", "SOXL", "SOXS", "UVXY", "07226", "07552", "07500"}

# Inverse/short ETFs: VIX signal is flipped — high VIX = BUY, low VIX = SELL
INVERSE_ETFS = {"SQQQ", "SOXS", "07552"}

# 3x leveraged LONG ETFs — subject to strict entry conditions (see below)
LEVERAGED_LONG_ETFS: set[str] = {"TQQQ", "SPXL", "SOXL", "07226", "07500"}

# ── Per-symbol hard position cap (fraction of total portfolio) ────────────
# Symbols not listed use the tier cap from TIER_MAX_POSITION.
# Applied AFTER portfolio optimisation + 3-layer multipliers.
SYMBOL_MAX_POSITION: dict[str, float] = {
    # 3x leveraged long — strict cap regardless of signal strength
    "TQQQ": 0.15,  "SPXL": 0.15,  "SOXL": 0.15,
    "07226": 0.10, "07500": 0.10,
    # Inverse / short — tactical only, small size
    "SQQQ": 0.10,  "SOXS": 0.10,  "07552": 0.10,
    # Extreme volatility / VIX derivative
    "UVXY": 0.05,
    # Speculative / crypto proxy / extreme beta
    "MSTR": 0.08,  "COIN": 0.08,  "SMCI": 0.08,  "RKLB": 0.08,
    # Poor strategy-signal fit (energy sector underperforms in backtest)
    "XOM":  0.10,  "CVX":  0.10,  "XLE":  0.10,
}

# ── Symbol quality tier ───────────────────────────────────────────────────
# Based on 20-yr backtest strategy CAGR and signal consistency.
#   A  high signal quality  → up to 25% per position
#   B  normal quality       → up to 20%
#   C  poor signal fit      → capped at 10%
#   S  speculative          → capped at 8%
SYMBOL_QUALITY_TIER: dict[str, str] = {
    # Tier A — strategy CAGR > 9%, clean trend signals
    "NVDA": "A", "TSLA": "A", "AAPL": "A",
    "QQQ":  "A", "SPY":  "A", "SMH":  "A",
    # Tier B — decent signal, strategy CAGR 4–9%
    "MSFT": "B", "GOOGL": "B", "AMZN": "B", "META": "B",
    "JPM":  "B", "AVGO":  "B", "AMD":  "B",
    "PLTR": "B", "CRWD":  "B", "GLD":  "B",
    "TLT":  "B", "IWM":   "B", "DIA":  "B",
    "IBIT": "B", "CEG":   "B", "VST":  "B",
    # HK new additions
    "00388": "B",  # 港交所 — 稳定金融基础设施
    "01299": "B",  # 友邦保险 — 清晰趋势
    "09961": "B",  # 携程 — 动量特征好
    # Tier C — strategy consistently underperforms, signal unreliable
    "XOM":  "C", "CVX":  "C", "XLE":  "C",
    "ICLN": "C", "ARKK": "C",
    # Tier S — speculative, max 8%, needs very high composite score
    "MSTR": "S", "COIN": "S", "SMCI": "S", "RKLB": "S", "UVXY": "S",
    # Cash substitute — bypasses normal signal pipeline, injected by jobs.py
    "SGOV": "CASH",
}

# Maximum portfolio fraction by quality tier
TIER_MAX_POSITION: dict[str, float] = {
    "A":    0.25,   # up to 25% of portfolio
    "B":    0.20,   # up to 20%
    "C":    0.10,   # capped at 10%
    "S":    0.08,   # capped at  8%
    "CASH": 0.40,   # SGOV: up to 40% (cash substitute, not a risk position)
}

# ── Per-symbol minimum composite score to enter ───────────────────────────
# Overrides the global graduated floor (2.5) for specific symbols.
# Symbols not listed use the graduated floor: ≥2.5 → 25%, ≥4.5 → 50% …
SYMBOL_MIN_SCORE: dict[str, float] = {
    # 3x leveraged long — require strong consensus before entry
    "TQQQ": 7.0, "SPXL": 7.0, "SOXL": 7.0, "07226": 7.0, "07500": 7.0,
    # Speculative — very high bar, full-conviction only
    "MSTR": 7.5, "COIN": 7.5, "SMCI": 7.5,
    "RKLB": 7.0, "UVXY": 7.0,
    # Poor signal fit — no graduated half-entry, require standard BUY signal
    "XOM": 6.0, "CVX": 6.0, "XLE": 6.0, "ICLN": 6.0, "ARKK": 6.0,
}

# ── Leveraged-long strict entry conditions ────────────────────────────────
# TQQQ / SPXL / SOXL require ALL four conditions simultaneously.
# If any fails the symbol is removed from buy_candidates for that day.
LEVERAGED_LONG_MIN_SCORE:            float = 7.0   # vs standard 6.0
LEVERAGED_LONG_REQUIRE_SHIELD_NONE:  bool  = True  # CrashShield must be NONE (not CAUTION)
LEVERAGED_LONG_MIN_ML_MULT:          float = 0.80  # ML clearly bullish (≥80% position multiplier)
LEVERAGED_LONG_REQUIRE_ABS_MOM:      bool  = True  # SPY 12-month return must be positive

# ── Strategy weights (used by composite scoring) ─────────────────────────
# Each strategy contributes its weight to the composite score.
# Default: all equal (1). Increase a weight to give that strategy more influence.
# Max possible score = sum of weights for applicable strategies.
STRATEGY_WEIGHTS: dict[str, float] = {
    "golden_cross":      2.0,  # trend: 50/200 MA cross  ← best performer, boosted
    "supertrend":        1.0,  # trend: ATR-based
    "donchian_channel":  1.0,  # trend: breakout
    "ema_adx":           0.5,  # trend: EMA + ADX filter ← poor performer, halved
    "macd_crossover":    1.0,  # momentum: MACD signal line
    "roc_momentum":      1.0,  # momentum: rate of change
    "rsi_strategy":      1.0,  # mean reversion (skipped for leveraged)
    "bollinger_squeeze": 1.0,  # mean reversion (skipped for leveraged)
    "vix_timing":        0.3,  # macro: VIX gate         ← failed this period, reduced
}

# Per-market composite BUY threshold (absolute weighted score)
# HK raised to 7 — signal quality was poor, require stronger consensus
COMPOSITE_BUY_THRESHOLD: dict[str, float] = {
    "us": 6.0,
    "hk": 7.0,
    "cn": 6.0,
}

# ── Strategy parameters ───────────────────────────────────────────────────
STRATEGY_PARAMS = {
    "golden_cross": {"fast": 20, "slow": 100},
    "supertrend": {"atr_period": 10, "multiplier": 2.5},
    "donchian": {"entry_period": 20, "exit_period": 10},
    "ema_adx": {"ema_fast": 8, "ema_slow": 21, "adx_period": 14, "adx_threshold": 20},
    "macd": {"fast": 12, "slow": 26, "signal_period": 9},
    "roc": {"period_short": 20, "period_long": 60},
    "rsi": {"period": 14, "oversold": 25, "overbought": 75},
    "bollinger": {"period": 20, "std_dev": 1.5, "squeeze_threshold": 0.1},
    "vix": {"fear_threshold": 30, "greed_threshold": 15},
    "composite": {"buy_threshold": 6, "sell_threshold": 5.0},
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

# ── SGOV cash-substitute settings ────────────────────────────────────────
# When equity positions are under-invested, idle capital is parked in SGOV
# (iShares 0-3 Month T-Bill ETF) to earn T-bill yield (~4% annualised).
#
# Logic (applied after all equity positions are finalised):
#   sgov_alloc = clamp(SGOV_TARGET_DEPLOYED - equity_total, 0, SGOV_MAX_ALLOC)
#
# Examples (SGOV_TARGET_DEPLOYED=0.35, SGOV_MAX_ALLOC=0.40):
#   equity_total=0.05  → SGOV=0.30   (light signal day, mostly T-bills)
#   equity_total=0.20  → SGOV=0.15   (moderate positions)
#   equity_total=0.35  → SGOV=0.00   (fully invested, no cash sub needed)
#   SHIELD active (equity≈0) → SGOV=0.35  (crash mode: park everything in T-bills)
SGOV_SYMBOL:           str   = "SGOV"
SGOV_TARGET_DEPLOYED:  float = 0.35   # aim to have at least 35% deployed at all times
SGOV_MAX_ALLOC:        float = 0.40   # never exceed 40% in SGOV alone
