"""
Forward-return tracker — the foundation of strategy quality measurement.

Flow
────
Day T  (pipeline runs after close):
  record_today()  →  writes signal + price_at_signal to HF history

Day T+N (next pipeline run, any day):
  backfill()  →  scans rows where return_Nd is still null,
                 fills them in using today's available price data

Storage: HF Dataset
  forward_returns/{market}/history.parquet

Schema
──────
  date             : signal date (YYYY-MM-DD)
  market           : "us" | "hk" | "cn"
  symbol           : ticker
  strategy         : strategy name
  signal           : -1 / 0 / 1 (composite signal)
  composite_score  : int
  regime           : "bull" | "bear" | "neutral" | "unknown"
  price_at_signal  : close price on signal date
  return_1d        : float or null
  return_5d        : float or null
  return_10d       : float or null
  return_20d       : float or null
  price_1d         : float or null
  price_5d         : float or null
  price_10d        : float or null
  price_20d        : float or null
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FORWARD_WINDOWS = [1, 5, 10, 20]

_COLS = [
    "date", "market", "symbol", "strategy", "signal", "composite_score", "regime",
    "price_at_signal",
    "return_1d",  "return_5d",  "return_10d",  "return_20d",
    "price_1d",   "price_5d",   "price_10d",   "price_20d",
]


# ─── HF I/O ────────────────────────────────────────────────────────────────

def _hf_path(market: str) -> str:
    return f"forward_returns/{market}/history.parquet"


def _load(market: str, hf_repo: str, hf_token: str) -> pd.DataFrame:
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=hf_repo, filename=_hf_path(market),
            repo_type="dataset", token=hf_token,
        )
        return pd.read_parquet(local)
    except Exception as e:
        logger.debug("No existing forward-return history for [%s]: %s", market, e)
        return pd.DataFrame(columns=_COLS)


def _save(df: pd.DataFrame, market: str, hf_repo: str, hf_token: str) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=_hf_path(market),
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Forward returns update [{market}] {datetime.today().strftime('%Y-%m-%d')}",
        )
        logger.info("Saved forward-return history [%s]: %d rows", market, len(df))
    except Exception as e:
        logger.error("Failed to save forward returns [%s]: %s", market, e)


# ─── Public API ────────────────────────────────────────────────────────────

def record_today(
    market: str,
    all_results: dict[str, dict],       # {symbol: {strategy: result_dict}}
    composite_scores: dict[str, int],    # {symbol: score}
    prices: dict[str, float],           # {symbol: last_close}
    regime: str = "unknown",
    date: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> None:
    """
    Record today's signals + prices so we can compute forward returns later.
    Skips symbols that already have a row for today (idempotent).
    """
    if not hf_repo or not hf_token:
        return

    date = date or datetime.today().strftime("%Y-%m-%d")
    rows = []

    for symbol, strategies in all_results.items():
        composite = strategies.get("composite_score", {})
        sig   = composite.get("signal", 0)
        score = composite_scores.get(symbol, 0)
        price = prices.get(symbol)
        if price is None:
            continue

        row = {
            "date": date, "market": market, "symbol": symbol,
            "strategy": "composite_score", "signal": sig,
            "composite_score": score, "regime": regime,
            "price_at_signal": price,
        }
        for w in FORWARD_WINDOWS:
            row[f"return_{w}d"] = None
            row[f"price_{w}d"]  = None
        rows.append(row)

    if not rows:
        return

    existing = _load(market, hf_repo, hf_token)

    # Deduplicate: don't re-add rows for (date, symbol) that already exist
    if not existing.empty:
        new_df = pd.DataFrame(rows)
        key = ["date", "symbol"]
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="first")
    else:
        merged = pd.DataFrame(rows, columns=_COLS)

    _save(merged, market, hf_repo, hf_token)
    logger.info("[%s] Recorded %d signal seeds for forward-return tracking.", market, len(rows))


def backfill(
    market: str,
    price_data: dict,   # {symbol: pd.DataFrame with DatetimeIndex + Close column}
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> int:
    """
    Fill in any null forward-return cells using historical price data we already have.
    Runs automatically each pipeline cycle.

    Returns number of cells filled.
    """
    if not hf_repo or not hf_token:
        return 0

    df = _load(market, hf_repo, hf_token)
    if df is None or df.empty:
        return 0

    today = datetime.today().date()
    filled = 0

    for idx, row in df.iterrows():
        symbol = row["symbol"]
        if symbol not in price_data:
            continue

        close = price_data[symbol]["Close"].dropna()
        signal_date = pd.to_datetime(row["date"]).date()

        for w in FORWARD_WINDOWS:
            ret_col   = f"return_{w}d"
            price_col = f"price_{w}d"

            if pd.notna(df.at[idx, ret_col]):
                continue   # already filled

            target_date = signal_date + timedelta(days=w)
            if target_date > today:
                continue   # future — can't fill yet

            # Find the first trading day on or after target_date
            candidates = close.index[close.index.date >= target_date]
            if len(candidates) == 0:
                continue

            fwd_price  = float(close[candidates[0]])
            base_price = row["price_at_signal"]
            if not base_price or base_price <= 0:
                continue

            df.at[idx, price_col] = round(fwd_price, 6)
            df.at[idx, ret_col]   = round((fwd_price - base_price) / base_price, 6)
            filled += 1

    if filled:
        _save(df, market, hf_repo, hf_token)
        logger.info("[%s] Backfilled %d forward-return cell(s).", market, filled)

    return filled
