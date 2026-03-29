from .trend import golden_cross, supertrend, donchian_channel, ema_adx
from .momentum import macd_crossover, roc_momentum
from .mean_reversion import rsi_strategy, bollinger_squeeze
from .macro import vix_timing
from .composite import composite_score

ALL_STRATEGIES = [
    "golden_cross",
    "supertrend",
    "donchian_channel",
    "ema_adx",
    "macd_crossover",
    "roc_momentum",
    "rsi_strategy",
    "bollinger_squeeze",
    "vix_timing",
    "composite_score",
]

STRATEGY_FUNCTIONS = {
    "golden_cross": golden_cross,
    "supertrend": supertrend,
    "donchian_channel": donchian_channel,
    "ema_adx": ema_adx,
    "macd_crossover": macd_crossover,
    "roc_momentum": roc_momentum,
    "rsi_strategy": rsi_strategy,
    "bollinger_squeeze": bollinger_squeeze,
    "vix_timing": vix_timing,
    "composite_score": composite_score,
}

STRATEGY_LABELS = {
    "golden_cross": "黄金/死亡交叉 (MA50/200)",
    "supertrend": "SuperTrend (ATR)",
    "donchian_channel": "海龟突破 (Donchian)",
    "ema_adx": "EMA + ADX 过滤",
    "macd_crossover": "MACD 信号线交叉",
    "roc_momentum": "ROC 动量排名",
    "rsi_strategy": "RSI 超买超卖",
    "bollinger_squeeze": "布林带收缩突破",
    "vix_timing": "VIX 宏观择时",
    "composite_score": "多策略综合评分",
}
