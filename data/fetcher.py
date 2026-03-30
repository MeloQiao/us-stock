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


def _cn_to_yf_ticker(symbol: str) -> str:
    """Map 6-digit CN code to yfinance ticker.
    Shanghai: 6xx, 5xx, 9xx → .SS
    Shenzhen: 0xx, 1xx, 2xx, 3xx, 7xx → .SZ
    """
    return f"{symbol}.SS" if symbol[0] in "569" else f"{symbol}.SZ"


def _fetch_cn_via_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: fetch CN stock/ETF via yfinance (.SS/.SZ suffix)."""
    import yfinance as yf

    yf_sym = _cn_to_yf_ticker(symbol)
    df = yf.download(yf_sym, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for {yf_sym}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def _fetch_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch A-share / CN ETF history.
    Primary: akshare (richer data, no suffix needed).
    Fallback: yfinance (.SS/.SZ) — used when akshare is unreachable
    (e.g. GitHub Actions US runners blocked by Chinese data providers).
    """
    try:
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
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    except Exception as ak_err:
        logger.warning("akshare failed for %s (%s), falling back to yfinance.", symbol, ak_err)
        return _fetch_cn_via_yfinance(symbol, start, end)


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════

def _hf_data_is_fresh(df: pd.DataFrame, market: str = "us") -> bool:
    """
    Return True if the HF-cached DataFrame already contains today's close.

    We compare the last bar's date against 'today' in the market's local timezone.
    Since pipelines run after close, the latest bar should equal today's date.
    Falls back to a 1-day tolerance for timezone edge cases.
    """
    if df is None or df.empty:
        return False
    tz_map = {"us": "America/New_York", "hk": "Asia/Hong_Kong", "cn": "Asia/Shanghai"}
    import pytz
    tz = pytz.timezone(tz_map.get(market, "UTC"))
    today_local = datetime.now(tz).date()
    last_date = df.index[-1]
    last_day = last_date.date() if hasattr(last_date, "date") else last_date
    return last_day >= today_local


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
    # Auto-detect market from symbol format if caller forgot to specify
    if market == "us" and symbol.isdigit():
        if len(symbol) == 6:
            logger.warning("CN-looking code '%s' passed with market='us' — auto-routing to cn", symbol)
            market = "cn"
        elif len(symbol) == 5:
            logger.warning("HK-looking code '%s' passed with market='us' — auto-routing to hk", symbol)
            market = "hk"

    cache_key = f"{market}:{symbol}:{years}y"
    if not force_refresh and cache_key in _cache:
        logger.debug("Cache hit: %s", cache_key)
        return _cache[cache_key]

    # ── Try HF Dataset first ──────────────────────────────────────────
    import os
    hf_token = os.getenv("HF_TOKEN", "")
    hf_repo  = os.getenv("HF_DATASET_REPO", "")
    if not force_refresh and hf_token and hf_repo:
        cached = load_from_hf(symbol, hf_repo, hf_token, market=market)
        if _hf_data_is_fresh(cached, market=market):
            logger.info("HF Dataset hit: %s [%s]", symbol, market)
            _cache[cache_key] = cached
            return cached

    # ── Fallback: fetch from source ───────────────────────────────────
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=years * 365 + 5)
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    logger.info("Fetching %s [%s] from source (%d yr)...", symbol, market, years)

    if market == "cn":
        df = _fetch_akshare(symbol, start, end)
    elif market == "hk":
        if symbol.upper().endswith(".HK"):
            yf_sym = symbol
        else:
            # Config stores 5-digit codes (e.g. "00700"); yfinance wants 4-digit (e.g. "0700.HK")
            yf_sym = symbol.lstrip("0").zfill(4) + ".HK"
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
            if symbol.upper().endswith(".HK"):
                yf_sym = symbol
            else:
                yf_sym = symbol.lstrip("0").zfill(4) + ".HK"
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
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=False)
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
    """Fetch OHLCV parquet directly from HF HTTP API (no local disk cache)."""
    import requests as _requests

    url = (
        f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
        f"/data/{market}/{symbol}.parquet"
    )
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        resp = _requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")
    except Exception as e:
        logger.debug("HF load failed %s [%s]: %s", symbol, market, e)
        return None


def load_today_signals_from_hf(
    market: Market = "us",
    hf_repo: str = "",
    hf_token: str = "",
    date: Optional[str] = None,
) -> dict[str, dict[str, int]]:
    """
    Load today's pre-computed signals from HF Dataset.
    Always fetches directly from HF HTTP API (bypasses hf_hub_download local
    disk cache, which can return stale 'not found' results in long-running
    Streamlit containers).

    Returns {symbol: {strategy_name: signal_int}} or {} if not found.
    """
    import requests as _requests

    date = date or datetime.today().strftime("%Y%m%d")
    url = (
        f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
        f"/signals/{market}/signals_{date}.parquet"
    )
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        resp = _requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            logger.debug("No HF signals for today [%s] (404): %s", market, url)
            return {}
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        result: dict[str, dict[str, int]] = {}
        for _, row in df.iterrows():
            sym = row["symbol"]
            strat = row["strategy"]
            result.setdefault(sym, {})[strat] = int(row["signal"])
        logger.info("Loaded today's signals [%s] from HF Dataset (%d rows).", market, len(df))
        return result
    except Exception as e:
        logger.warning("Failed to load HF signals [%s]: %s", market, e)
        return {}


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


def cleanup_old_signals_on_hf(
    hf_repo: str,
    hf_token: str,
    market: Market = "us",
    keep_days: int = 90,
) -> int:
    """
    Delete signal files older than keep_days from HF Dataset.
    Returns number of files deleted.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset", token=hf_token)
        prefix = f"signals/{market}/signals_"
        cutoff = (datetime.today() - timedelta(days=keep_days)).strftime("%Y%m%d")
        deleted = 0
        for f in files:
            if f.startswith(prefix):
                date_str = f.replace(prefix, "").replace(".parquet", "")
                if date_str < cutoff:
                    api.delete_file(
                        path_in_repo=f,
                        repo_id=hf_repo,
                        repo_type="dataset",
                        commit_message=f"Cleanup old signals: {f}",
                    )
                    deleted += 1
                    logger.info("Deleted old signal file: %s", f)
        return deleted
    except Exception as e:
        logger.warning("HF signal cleanup failed [%s]: %s", market, e)
        return 0
