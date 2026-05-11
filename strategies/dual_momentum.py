"""
Layer 3: Dual Momentum + Momentum Crash Protection

Implements Gary Antonacci's dual momentum concept adapted for our stock universe,
plus a vol-based momentum crash protection overlay.

Two components:
  A) Absolute Momentum Filter
     Only allow long positions when SPY 12-month return > 0.
     If the overall market trend is down, go defensive (reduce all positions).

  B) Cross-Sectional (Relative) Momentum
     Within buy-signal stocks, rank by 6-month return.
     Prefer top-tercile; reduce weight on bottom-tercile.

  C) Momentum Crash Protection
     When short-term (1m) realised vol > 2× medium-term (6m) realised vol,
     we're likely in a momentum crash environment — reduce positions by 50%.

Usage
─────
  from strategies.dual_momentum import DualMomentumFilter
  dm = DualMomentumFilter()
  result = dm.evaluate(spy_df, candidate_symbols_df_dict)
  # result["abs_momentum_ok"]  : bool
  # result["position_scale"]   : float  (0.0–1.0 applied to all positions)
  # result["symbol_ranks"]     : dict[symbol, percentile 0-1]
  # result["symbol_multiplier"]: dict[symbol, multiplier 0.5-1.2]
  # result["crash_protect"]    : bool
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DualMomentumFilter:
    """
    Dual momentum + crash protection signal generator.

    Parameters
    ----------
    abs_lookback     : months for absolute momentum check (default 12)
    cs_lookback      : months for cross-sectional ranking (default 6)
    crash_vol_ratio  : vol_1m / vol_6m ratio threshold to trigger crash protect (default 2.0)
    crash_scale      : position scale when crash protect triggered (default 0.5)
    top_bonus        : multiplier for top-tercile momentum stocks (default 1.15)
    bottom_penalty   : multiplier for bottom-tercile momentum stocks (default 0.6)
    """

    def __init__(
        self,
        abs_lookback: int = 252,      # ~12 months trading days
        cs_lookback:  int = 126,      # ~6 months trading days
        crash_vol_ratio: float = 2.0,
        crash_scale:     float = 0.5,
        top_bonus:       float = 1.15,
        bottom_penalty:  float = 0.60,
    ):
        self.abs_lookback     = abs_lookback
        self.cs_lookback      = cs_lookback
        self.crash_vol_ratio  = crash_vol_ratio
        self.crash_scale      = crash_scale
        self.top_bonus        = top_bonus
        self.bottom_penalty   = bottom_penalty

    # ── Public interface ───────────────────────────────────────────────────

    def evaluate(
        self,
        spy_df: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
    ) -> dict:
        """
        Evaluate dual momentum + crash protection on the current data snapshot.

        Parameters
        ----------
        spy_df      : SPY OHLCV DataFrame (must have 252+ days of history)
        price_data  : {symbol: OHLCV DataFrame} for all candidate symbols

        Returns
        -------
        {
            "abs_momentum_ok"  : bool,
            "position_scale"   : float,          # 0.0 – 1.0
            "crash_protect"    : bool,
            "crash_vol_ratio"  : float,
            "spy_12m_ret"      : float,
            "symbol_ranks"     : {sym: percentile float 0-1},
            "symbol_multiplier": {sym: float 0.5-1.2},
            "label"            : str,
        }
        """
        spy_close = spy_df["Close"].dropna()

        # ── A. Absolute momentum ────────────────────────────────────────────
        abs_ok, spy_12m = self._absolute_momentum(spy_close)

        # ── B. Crash protection ────────────────────────────────────────────
        crash_protect, vol_ratio = self._crash_protection(spy_close)

        # ── C. Cross-sectional momentum ────────────────────────────────────
        symbol_ranks, symbol_mult = self._cross_sectional(price_data)

        # ── Aggregate position scale ───────────────────────────────────────
        position_scale = 1.0
        labels = []

        if not abs_ok:
            position_scale *= 0.0   # block all new longs — market in downtrend
            labels.append("绝对动量为负(SPY 12M<0) — 屏蔽做多")
        elif crash_protect:
            position_scale *= self.crash_scale
            labels.append(f"动量崩溃保护触发(Vol比={vol_ratio:.2f}x) — 仓位×{self.crash_scale}")

        if abs_ok and not labels:
            labels.append(f"SPY 12M={spy_12m*100:+.1f}% 动量正常")

        logger.info(
            "DualMomentum: abs_ok=%s crash=%s scale=%.2f symbols=%d",
            abs_ok, crash_protect, position_scale, len(symbol_ranks),
        )

        return {
            "abs_momentum_ok":   abs_ok,
            "position_scale":    position_scale,
            "crash_protect":     crash_protect,
            "crash_vol_ratio":   round(vol_ratio, 3),
            "spy_12m_ret":       round(spy_12m, 4),
            "symbol_ranks":      symbol_ranks,
            "symbol_multiplier": symbol_mult,
            "label":             " | ".join(labels),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _absolute_momentum(self, spy_close: pd.Series) -> tuple[bool, float]:
        """
        Return (ok, 12m_return).
        ok = True  → market 12m return positive, longs allowed
        ok = False → market in long-term downtrend, block new longs
        """
        if len(spy_close) < self.abs_lookback + 5:
            return True, 0.0   # insufficient history → stay neutral (allow)
        ret_12m = float(spy_close.iloc[-1] / spy_close.iloc[-(self.abs_lookback + 1)] - 1)
        return ret_12m > 0, ret_12m

    def _crash_protection(self, spy_close: pd.Series) -> tuple[bool, float]:
        """
        Return (crash_protect, vol_ratio).
        If 21-day realised vol > crash_vol_ratio × 126-day realised vol → crash mode.
        """
        if len(spy_close) < 130:
            return False, 1.0
        log_ret = np.log(spy_close / spy_close.shift(1)).dropna()
        vol_1m  = float(log_ret.iloc[-21:].std()  * np.sqrt(252))
        vol_6m  = float(log_ret.iloc[-126:].std() * np.sqrt(252))
        ratio   = vol_1m / (vol_6m + 1e-9)
        return ratio >= self.crash_vol_ratio, ratio

    def _cross_sectional(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Rank all symbols by 6-month return.  Return (ranks_dict, multipliers_dict).
        """
        returns: dict[str, float] = {}
        for sym, df in price_data.items():
            if df is None or df.empty:
                continue
            closes = df["Close"].dropna()
            if len(closes) < self.cs_lookback + 5:
                continue
            ret = float(closes.iloc[-1] / closes.iloc[-(self.cs_lookback + 1)] - 1)
            returns[sym] = ret

        if not returns:
            return {}, {}

        vals  = list(returns.values())
        syms  = list(returns.keys())
        ranks = pd.Series(vals, index=syms).rank(pct=True)   # 0–1 percentile

        ranks_dict = {s: round(float(ranks[s]), 3) for s in syms}
        mult_dict: dict[str, float] = {}
        for sym in syms:
            pct = ranks_dict[sym]
            if pct >= 0.67:           # top tercile
                mult_dict[sym] = self.top_bonus
            elif pct <= 0.33:         # bottom tercile
                mult_dict[sym] = self.bottom_penalty
            else:                     # middle tercile
                mult_dict[sym] = 1.0

        return ranks_dict, mult_dict


def apply_dual_momentum(
    buy_signals: dict[str, int],
    spy_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    dm_filter: Optional[DualMomentumFilter] = None,
) -> tuple[dict[str, int], dict[str, float], dict]:
    """
    Convenience wrapper: apply dual momentum to a dict of composite signals.

    Parameters
    ----------
    buy_signals  : {symbol: signal (1=buy, 0=hold, -1=sell)}
    spy_df       : SPY OHLCV
    price_data   : {symbol: OHLCV DataFrame}
    dm_filter    : DualMomentumFilter instance (creates default if None)

    Returns
    -------
    (adjusted_signals, per_symbol_multipliers, dm_result_dict)

    adjusted_signals: same as buy_signals but BUY signals → HOLD if abs_momentum fails
    per_symbol_multipliers: {sym: float} — applied to position sizing
    dm_result_dict: full evaluate() output
    """
    if dm_filter is None:
        dm_filter = DualMomentumFilter()

    dm = dm_filter.evaluate(spy_df, price_data)
    position_scale  = dm["position_scale"]
    sym_mult        = dm["symbol_multiplier"]

    adj_signals: dict[str, int] = {}
    for sym, sig in buy_signals.items():
        if sig == 1 and position_scale == 0.0:
            adj_signals[sym] = 0   # block new buy
        else:
            adj_signals[sym] = sig

    # Combine global scale × per-symbol multiplier
    final_mult: dict[str, float] = {}
    for sym in buy_signals:
        base = sym_mult.get(sym, 1.0)
        final_mult[sym] = round(position_scale * base, 3)

    return adj_signals, final_mult, dm
