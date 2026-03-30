"""
Portfolio-level optimizer.

Pipeline
────────
  buy_signals (symbol → score)
       │
       ▼
  1. Correlation deduplication
     Remove the lower-scored symbol from pairs with corr > threshold
       │
       ▼
  2. Weight calculation
     score_weighted  : weight ∝ composite score  (default, always works)
     risk_parity     : weight ∝ 1/volatility      (requires ≥2 symbols + history)
     max_sharpe      : mean-variance optimization  (requires scipy + ≥2 symbols)
       │
       ▼
  3. Sector concentration cap
     No single sector > MAX_SECTOR_WEIGHT (default 65%)
       │
       ▼
  4. Single-position cap
     No single symbol > MAX_SINGLE_POSITION (default 30%)
       │
       ▼
  Output: {symbol: normalized_weight}   (sums to 1.0)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────
MAX_SECTOR_WEIGHT     = 0.65   # no single sector > 65% of portfolio
MAX_SINGLE_POSITION   = 0.30   # no single stock  > 30%
CORR_DEDUP_THRESHOLD  = 0.85   # drop the lower-scored of highly-correlated pairs


# ── Sector maps (symbol → sector) ────────────────────────────────────────────

_US_SECTORS: dict[str, str] = {
    # broad market
    "SPY": "broad", "IWM": "broad", "SPXL": "broad",
    # tech / semiconductor
    "QQQ": "tech", "TQQQ": "tech",
    "NVDA": "tech", "AMD": "tech", "SMCI": "tech", "AVGO": "tech",
    "TSM": "tech", "MRVL": "tech", "AMAT": "tech", "KLAC": "tech", "LRCX": "tech",
    "SOXL": "tech",
    # macro / volatility
    "UVXY": "volatility", "SQQQ": "volatility", "SOXS": "volatility",
    # energy
    "XLE": "energy", "OXY": "energy",
    # consumer / other
    "AMZN": "consumer", "META": "consumer",
    "MSFT": "tech", "AAPL": "tech", "GOOGL": "tech",
}

_HK_SECTORS: dict[str, str] = {
    "02800": "broad", "03188": "broad", "07226": "broad", "07500": "broad",
    "03032": "tech", "03067": "tech",
    "00700": "tech", "09988": "tech", "03690": "tech", "09618": "tech",
    "09999": "tech", "01024": "tech", "09626": "tech", "00981": "tech", "02382": "tech",
    "01211": "ev", "09866": "ev", "02015": "ev",
    "07552": "inverse",
}

_CN_SECTORS: dict[str, str] = {
    "510300": "broad", "159915": "broad", "588000": "broad", "510500": "broad",
    "512480": "tech", "515070": "tech", "159741": "tech",
    "300750": "tech", "688981": "tech", "688111": "tech",
    "603501": "tech", "688041": "tech",
    "601012": "energy", "159745": "energy",
    "600519": "consumer", "000333": "consumer", "300760": "consumer",
    "600036": "finance",
    "002594": "ev",
}

_SECTOR_MAP = {"us": _US_SECTORS, "hk": _HK_SECTORS, "cn": _CN_SECTORS}


# ── Main entry point ──────────────────────────────────────────────────────────

def optimize_portfolio(
    buy_signals: dict[str, float],          # {symbol: composite_score}
    price_data: dict[str, pd.DataFrame],    # {symbol: OHLCV DataFrame}
    market: str = "us",
    method: str = "score_weighted",         # "score_weighted" | "risk_parity" | "max_sharpe"
) -> dict[str, float]:
    """
    Compute optimal position weights for a set of buy signals.

    Parameters
    ----------
    buy_signals : symbols with signal == 1, mapped to their composite score
    price_data  : full OHLCV history per symbol (for correlation / vol estimates)
    market      : "us" | "hk" | "cn"
    method      : weighting method

    Returns
    -------
    {symbol: weight}  — weights sum to 1.0; empty if no valid signals.
    """
    if not buy_signals:
        return {}

    symbols = list(buy_signals.keys())
    logger.info("[%s] Portfolio optimizer: %d candidates, method=%s", market, len(symbols), method)

    # Step 1 — correlation deduplication
    symbols = _dedup_correlated(symbols, buy_signals, price_data)
    if not symbols:
        return {}
    logger.info("[%s] After dedup: %d symbols", market, len(symbols))

    # Step 2 — compute weights
    if method == "max_sharpe" and len(symbols) >= 2:
        weights = _max_sharpe(symbols, price_data)
    elif method == "risk_parity" and len(symbols) >= 2:
        weights = _risk_parity(symbols, price_data)
    else:
        weights = _score_weighted(symbols, buy_signals)

    # Step 3 — sector cap
    weights = _apply_sector_cap(weights, market)

    # Step 4 — single-position cap
    weights = _apply_position_cap(weights)

    # Renormalize
    total = sum(weights.values())
    if total <= 0:
        return {s: 1 / len(symbols) for s in symbols}
    weights = {s: w / total for s, w in weights.items()}

    for sym, w in weights.items():
        logger.info("[%s]   %s → weight %.1f%%", market, sym, w * 100)

    return weights


def portfolio_stats(
    weights: dict[str, float],
    price_data: dict[str, pd.DataFrame],
    lookback: int = 60,
) -> dict:
    """Estimate annualised return and volatility for a given weight set."""
    try:
        returns = {}
        for s, w in weights.items():
            if s in price_data and not price_data[s].empty:
                r = price_data[s]["Close"].pct_change().dropna().iloc[-lookback:]
                returns[s] = r * w
        if not returns:
            return {}
        port_ret = pd.concat(returns.values(), axis=1).sum(axis=1)
        ann_vol  = float(port_ret.std() * (252 ** 0.5))
        ann_ret  = float(port_ret.mean() * 252)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
        return {"ann_return": round(ann_ret, 4),
                "ann_vol": round(ann_vol, 4),
                "sharpe": round(sharpe, 3)}
    except Exception as e:
        logger.warning("portfolio_stats failed: %s", e)
        return {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _score_weighted(symbols: list[str], scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(scores.get(s, 1), 1) for s in symbols)
    return {s: max(scores.get(s, 1), 1) / total for s in symbols}


def _risk_parity(symbols: list[str], price_data: dict, lookback: int = 60) -> dict[str, float]:
    try:
        vols = {}
        for s in symbols:
            if s in price_data and not price_data[s].empty:
                r = price_data[s]["Close"].pct_change().dropna().iloc[-lookback:]
                vols[s] = float(r.std()) if len(r) >= 5 else 0.02
            else:
                vols[s] = 0.02
        inv_vol = {s: 1.0 / (v + 1e-9) for s, v in vols.items()}
        total = sum(inv_vol.values())
        return {s: iv / total for s, iv in inv_vol.items()}
    except Exception as e:
        logger.warning("risk_parity failed: %s — falling back to score_weighted", e)
        return {s: 1 / len(symbols) for s in symbols}


def _max_sharpe(symbols: list[str], price_data: dict, lookback: int = 120) -> dict[str, float]:
    try:
        from scipy.optimize import minimize

        ret_df = _returns_matrix(symbols, price_data, lookback)
        if ret_df is None or len(ret_df) < 20:
            raise ValueError("Insufficient data for max-Sharpe optimisation")

        mu  = ret_df.mean().values * 252
        cov = ret_df.cov().values  * 252
        n   = len(symbols)

        def neg_sharpe(w):
            pr = np.dot(w, mu)
            pv = np.sqrt(np.dot(w, np.dot(cov, w)))
            return -pr / (pv + 1e-9)

        bounds      = [(0.05, MAX_SINGLE_POSITION)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        w0          = np.ones(n) / n

        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-9})

        if res.success:
            return {s: float(w) for s, w in zip(symbols, res.x)}
        raise ValueError(f"Optimiser did not converge: {res.message}")

    except Exception as e:
        logger.warning("max_sharpe failed: %s — falling back to risk_parity", e)
        return _risk_parity(symbols, price_data)


def _dedup_correlated(
    symbols: list[str],
    scores: dict[str, float],
    price_data: dict,
    lookback: int = 60,
) -> list[str]:
    """Remove the lower-scored symbol from pairs with |correlation| > threshold."""
    if len(symbols) < 2:
        return symbols

    ret_df = _returns_matrix(symbols, price_data, lookback)
    if ret_df is None or ret_df.empty:
        return symbols

    corr = ret_df.corr()
    to_remove: set[str] = set()
    syms = list(ret_df.columns)

    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            if a in to_remove or b in to_remove:
                continue
            if abs(corr.loc[a, b]) > CORR_DEDUP_THRESHOLD:
                loser = b if scores.get(a, 0) >= scores.get(b, 0) else a
                winner = a if loser == b else b
                to_remove.add(loser)
                logger.info(
                    "Corr dedup: removed %s (corr=%.2f with %s, score %s vs %s)",
                    loser, corr.loc[a, b], winner,
                    scores.get(loser, "?"), scores.get(winner, "?"),
                )

    return [s for s in symbols if s not in to_remove]


def _apply_sector_cap(weights: dict[str, float], market: str) -> dict[str, float]:
    sector_map = _SECTOR_MAP.get(market, {})
    if not sector_map:
        return weights

    def _sector(sym):
        return sector_map.get(sym, "other")

    # Aggregate by sector
    sector_total: dict[str, float] = {}
    for s, w in weights.items():
        sec = _sector(s)
        sector_total[sec] = sector_total.get(sec, 0.0) + w

    adjusted = dict(weights)
    for sec, sw in sector_total.items():
        if sw > MAX_SECTOR_WEIGHT:
            scale = MAX_SECTOR_WEIGHT / sw
            for s in adjusted:
                if _sector(s) == sec:
                    adjusted[s] *= scale
            logger.info("Sector cap [%s]: %s scaled by %.2f", market, sec, scale)

    return adjusted


def _apply_position_cap(weights: dict[str, float]) -> dict[str, float]:
    return {s: min(w, MAX_SINGLE_POSITION) for s, w in weights.items()}


def _returns_matrix(
    symbols: list[str],
    price_data: dict,
    lookback: int,
) -> Optional[pd.DataFrame]:
    """Build an aligned daily-returns DataFrame for the given symbols."""
    cols = {}
    for s in symbols:
        if s in price_data and not price_data[s].empty:
            r = price_data[s]["Close"].pct_change().dropna().iloc[-lookback:]
            if len(r) >= 5:
                cols[s] = r
    if not cols:
        return None
    return pd.DataFrame(cols).dropna()
