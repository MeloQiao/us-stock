"""
Data fetcher: yfinance for historical OHLCV, with optional HF Dataset caching.
All data is daily candles. Cache is refreshed once per day after market close.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Local in-memory cache (survives within one Spaces session) ────────────
_cache: dict[str, pd.DataFrame] = {}


def fetch_history(
    symbol: str,
    years: int = 10,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for `symbol` going back `years` years.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex (UTC-normalized).
    """
    cache_key = f"{symbol}_{years}y"

    if not force_refresh and cache_key in _cache:
        logger.debug("Cache hit: %s", cache_key)
        return _cache[cache_key]

    end = datetime.today()
    start = end - timedelta(days=years * 365 + 5)  # +5 buffer for weekends

    logger.info("Downloading %s from yfinance (%d years)...", symbol, years)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    # Normalize columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip tz for simplicity
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)

    _cache[cache_key] = df
    return df


def fetch_multiple(
    symbols: list[str],
    years: int = 10,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple symbols. Returns {symbol: DataFrame}."""
    result = {}
    for sym in symbols:
        try:
            result[sym] = fetch_history(sym, years=years, force_refresh=force_refresh)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", sym, e)
    return result


def fetch_latest_quote(symbol: str) -> dict:
    """
    Fetch the latest available quote snapshot for a symbol.
    Returns dict with keys: symbol, price, change, change_pct, volume, timestamp.
    Uses yfinance fast_info for speed.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = info.last_price
        prev_close = info.previous_close or price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": info.three_month_average_volume,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logger.warning("Failed to fetch quote for %s: %s", symbol, e)
        return {"symbol": symbol, "price": None, "change": None, "change_pct": None, "volume": None, "timestamp": None}


def fetch_quotes(symbols: list[str]) -> list[dict]:
    """Fetch latest quotes for multiple symbols."""
    return [fetch_latest_quote(s) for s in symbols]


def clear_cache():
    """Clear in-memory cache to force fresh download."""
    _cache.clear()
    logger.info("Data cache cleared.")


# ── HF Dataset persistence ────────────────────────────────────────────────

def save_to_hf(df: pd.DataFrame, symbol: str, hf_repo: str, hf_token: str) -> bool:
    """
    Save a symbol's OHLCV DataFrame to a Hugging Face Dataset repo as parquet.
    Path in repo: data/{symbol}.parquet
    Returns True on success.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        buf = io.BytesIO()
        df.reset_index().to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=f"data/{symbol}.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Update {symbol} {datetime.today().strftime('%Y-%m-%d')}",
        )
        logger.info("Saved %s to HF Dataset: %s", symbol, hf_repo)
        return True
    except Exception as e:
        logger.error("HF save failed for %s: %s", symbol, e)
        return False


def load_from_hf(symbol: str, hf_repo: str, hf_token: str) -> Optional[pd.DataFrame]:
    """
    Load a symbol's OHLCV DataFrame from HF Dataset repo.
    Returns None if not found.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=hf_repo,
            filename=f"data/{symbol}.parquet",
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        logger.info("Loaded %s from HF Dataset.", symbol)
        return df
    except Exception as e:
        logger.debug("HF load failed for %s: %s", symbol, e)
        return None


def save_signals_to_hf(signals_df: pd.DataFrame, hf_repo: str, hf_token: str) -> bool:
    """
    Append today's strategy signals to signals_log.parquet in HF Dataset.
    signals_df columns: date, symbol, strategy, signal
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        buf = io.BytesIO()
        signals_df.to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=f"signals/signals_{datetime.today().strftime('%Y%m%d')}.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Signals {datetime.today().strftime('%Y-%m-%d')}",
        )
        return True
    except Exception as e:
        logger.error("HF signals save failed: %s", e)
        return False
