"""
Data fetcher — multi-market OHLCV.

market="us"  →  yfinance (US tickers as-is, e.g. "QQQ")
market="hk"  →  yfinance (.HK suffix auto-appended, e.g. "00700" → "00700.HK")
market="cn"  →  akshare  (6-digit A-share / ETF code, e.g. "510300", "600519")

All outputs share the same normalised DataFrame schema:
    Index : DatetimeIndex, tz-naive, daily frequency
    Cols  : Open, High, Low, Close, Volume  (all float)
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

Market = Literal["us", "hk", "cn"]

# ── In-memory cache ───────────────────────────────────────────────────────
_cache: dict[str, pd.DataFrame] = {}


# ════════════════════════════════════════════════════════════════════════
# Internal backends
# ════════════════════════════════════════════════════════════════════════

def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _is_cn_etf(symbol: str) -> bool:
    """Heuristic: A-share ETFs start with 51/15/16/18/56/58 etc. (fund codes)."""
    return symbol[:2] in {"51", "15", "16", "18", "56", "58", "12", "11"}


def _fetch_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch A-share / CN ETF history via akshare."""
    import akshare as ak

    # akshare date format: YYYYMMDD
    s = start.replace("-", "")
    e = end.replace("-", "")

    if _is_cn_etf(symbol):
        df = ak.fund_etf_hist_em(
            symbol=symbol, period="daily",
            start_date=s, end_date=e, adjust="qfq",
        )
    else:
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=s, end_date=e, adjust="qfq",
        )

    if df is None or df.empty:
        raise ValueError(f"akshare returned no data for {symbol}")

    # Column mapping (akshare returns Chinese headers)
    col_map = {
        "日期": "Date", "开盘": "Open", "最高": "High",
        "最低": "Low", "收盘": "Close", "成交量": "Volume",
    }
    df = df.rename(columns=col_map)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    return df


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════

def fetch_history(
    symbol: str,
    years: int = 10,
    market: Market = "us",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV for `symbol` going back `years` years.

    Parameters
    ----------
    symbol : ticker string in market-native format
        us → "QQQ", "NVDA", "^VIX"
        hk → "00700", "02800"  (no .HK suffix needed)
        cn → "510300", "600519"
    market : "us" | "hk" | "cn"
    """
    cache_key = f"{market}:{symbol}:{years}y"
    if not force_refresh and cache_key in _cache:
        logger.debug("Cache hit: %s", cache_key)
        return _cache[cache_key]

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=years * 365 + 5)
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    logger.info("Fetching %s [%s] via %s (%d yr)...", symbol, market, market, years)

    if market == "cn":
        df = _fetch_akshare(symbol, start, end)
    elif market == "hk":
        yf_sym = symbol if symbol.upper().endswith(".HK") else f"{symbol}.HK"
        df = _fetch_yfinance(yf_sym, start, end)
    else:  # us
        df = _fetch_yfinance(symbol, start, end)

    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    df = df.astype({"Open": float, "High": float, "Low": float,
                    "Close": float, "Volume": float})

    _cache[cache_key] = df
    return df


def fetch_multiple(
    symbols: list[str],
    years: int = 10,
    market: Market = "us",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple symbols for the same market. Returns {symbol: DataFrame}."""
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            result[sym] = fetch_history(sym, years=years, market=market,
                                        force_refresh=force_refresh)
        except Exception as e:
            logger.warning("Failed to fetch %s [%s]: %s", sym, market, e)
    return result


# ── Real-time quote snapshots ─────────────────────────────────────────────

def _quote_yfinance(symbol: str) -> dict:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    price = info.last_price
    prev = info.previous_close or price
    change = price - prev
    pct = (change / prev * 100) if prev else 0.0
    return {
        "symbol": symbol,
        "price": round(price, 3),
        "change": round(change, 3),
        "change_pct": round(pct, 2),
        "volume": info.three_month_average_volume,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _quote_akshare_cn(symbol: str) -> dict:
    """Real-time A-share quote via akshare."""
    import akshare as ak
    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            raise ValueError(f"Symbol {symbol} not found in A-share spot data")
        row = row.iloc[0]
        return {
            "symbol": symbol,
            "price": float(row["最新价"]),
            "change": float(row["涨跌额"]),
            "change_pct": float(row["涨跌幅"]),
            "volume": float(row["成交量"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logger.warning("akshare quote failed for %s: %s", symbol, e)
        return _empty_quote(symbol)


def _quote_akshare_hk(symbol: str) -> dict:
    """Real-time HK stock quote via akshare."""
    import akshare as ak
    try:
        df = ak.stock_hk_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            raise ValueError(f"Symbol {symbol} not found in HK spot data")
        row = row.iloc[0]
        return {
            "symbol": symbol,
            "price": float(row["最新价"]),
            "change": float(row["涨跌额"]),
            "change_pct": float(row["涨跌幅"]),
            "volume": float(row["成交量"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logger.warning("akshare HK quote failed for %s: %s", symbol, e)
        # Fallback to yfinance .HK
        try:
            yf_sym = symbol if symbol.upper().endswith(".HK") else f"{symbol}.HK"
            return _quote_yfinance(yf_sym)
        except Exception:
            return _empty_quote(symbol)


def _empty_quote(symbol: str) -> dict:
    return {"symbol": symbol, "price": None, "change": None,
            "change_pct": None, "volume": None, "timestamp": None}


def fetch_latest_quote(symbol: str, market: Market = "us") -> dict:
    """Fetch the latest real-time quote snapshot for a single symbol."""
    try:
        if market == "cn":
            return _quote_akshare_cn(symbol)
        elif market == "hk":
            return _quote_akshare_hk(symbol)
        else:
            return _quote_yfinance(symbol)
    except Exception as e:
        logger.warning("Quote failed %s [%s]: %s", symbol, market, e)
        return _empty_quote(symbol)


def fetch_quotes(symbols: list[str], market: Market = "us") -> list[dict]:
    """Fetch real-time quotes for multiple symbols."""
    return [fetch_latest_quote(s, market=market) for s in symbols]


def clear_cache():
    _cache.clear()
    logger.info("Data cache cleared.")


# ── HF Dataset persistence ────────────────────────────────────────────────

def save_to_hf(df: pd.DataFrame, symbol: str, hf_repo: str, hf_token: str,
               market: Market = "us") -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        buf = io.BytesIO()
        df.reset_index().to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=f"data/{market}/{symbol}.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Update {market}/{symbol} {datetime.today().strftime('%Y-%m-%d')}",
        )
        logger.info("Saved %s [%s] to HF.", symbol, market)
        return True
    except Exception as e:
        logger.error("HF save failed %s [%s]: %s", symbol, market, e)
        return False


def load_from_hf(symbol: str, hf_repo: str, hf_token: str,
                 market: Market = "us") -> Optional[pd.DataFrame]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=hf_repo,
            filename=f"data/{market}/{symbol}.parquet",
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")
    except Exception as e:
        logger.debug("HF load failed %s [%s]: %s", symbol, market, e)
        return None


def save_signals_to_hf(signals_df: pd.DataFrame, hf_repo: str, hf_token: str,
                        market: Market = "us") -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        buf = io.BytesIO()
        signals_df.to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=f"signals/{market}/signals_{datetime.today().strftime('%Y%m%d')}.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Signals {market} {datetime.today().strftime('%Y-%m-%d')}",
        )
        return True
    except Exception as e:
        logger.error("HF signals save failed [%s]: %s", market, e)
        return False
