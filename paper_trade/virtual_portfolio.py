"""
Virtual portfolio tracker for markets without a paper trading API (HK, CN).

Positions and trade history are persisted to HF Dataset:
  portfolio/{market}/open_positions.parquet
  portfolio/{market}/trade_history.parquet

Open positions schema:
  symbol | entry_date | entry_price | shares | notional | score | currency

Trade history schema:
  date | symbol | action | price | shares | notional | score | pnl | currency
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_OPEN_COLS = ["symbol", "entry_date", "entry_price", "shares", "notional", "score", "currency"]
_HIST_COLS = ["date", "symbol", "action", "price", "shares", "notional", "score", "pnl", "currency"]


class VirtualPortfolio:
    """
    Score-weighted virtual portfolio for a single market.

    Parameters
    ----------
    market        : "hk" | "cn"
    total_capital : starting capital in local currency
    currency      : display label, e.g. "HKD" or "CNY"
    hf_repo       : HF Dataset repo id
    hf_token      : HF access token
    """

    def __init__(
        self,
        market: str,
        total_capital: float,
        currency: str,
        hf_repo: str,
        hf_token: str,
        max_position_fraction: float = 0.25,
    ):
        self.market = market
        self.total_capital = total_capital
        self.currency = currency
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.max_position_fraction = max_position_fraction  # static fallback cap

        self._open: pd.DataFrame = self._load_open()
        self._history: pd.DataFrame = self._load_history()

    # ── HF I/O ────────────────────────────────────────────────────────────

    def _hf_path(self, name: str) -> str:
        return f"portfolio/{self.market}/{name}.parquet"

    def _load_df(self, name: str, columns: list[str]) -> pd.DataFrame:
        try:
            import requests
            url = (
                f"https://huggingface.co/datasets/{self.hf_repo}"
                f"/resolve/main/{self._hf_path(name)}"
            )
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return pd.read_parquet(io.BytesIO(resp.content))
        except Exception as e:
            logger.debug("Virtual portfolio %s load failed [%s]: %s", name, self.market, e)
            return pd.DataFrame(columns=columns)

    def _save_df(self, df: pd.DataFrame, name: str) -> None:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.hf_repo, repo_type="dataset", exist_ok=True, private=False)
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            api.upload_file(
                path_or_fileobj=buf,
                path_in_repo=self._hf_path(name),
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=f"Virtual portfolio update [{self.market}] {datetime.today().strftime('%Y-%m-%d')}",
            )
        except Exception as e:
            logger.error("Virtual portfolio %s save failed [%s]: %s", name, self.market, e)

    def _load_open(self) -> pd.DataFrame:
        return self._load_df("open_positions", _OPEN_COLS)

    def _load_history(self) -> pd.DataFrame:
        return self._load_df("trade_history", _HIST_COLS)

    # ── Portfolio state ────────────────────────────────────────────────────

    @property
    def open_symbols(self) -> set[str]:
        return set(self._open["symbol"].tolist()) if not self._open.empty else set()

    @property
    def invested_notional(self) -> float:
        return float(self._open["notional"].sum()) if not self._open.empty else 0.0

    @property
    def available_capital(self) -> float:
        return max(self.total_capital - self.invested_notional, 0.0)

    # ── Execute signals ────────────────────────────────────────────────────

    # Regime → base max-position fraction (tunable after live-data accumulates)
    _REGIME_BASE_FRACTION: dict[str, float] = {
        "bull_strong":  0.30,   # confident trend — allow bigger positions
        "bull_caution": 0.20,   # mixed signals — stay moderate
        "bear":         0.15,   # defensive — inverse ETFs only, keep small
    }

    def execute_signals(
        self,
        signals: dict[str, int],
        scores: dict[str, int],
        prices: dict[str, float],
        weights: Optional[dict[str, float]] = None,
        date: Optional[str] = None,
        regime: str = "bull_caution",
        max_possible: int = 9,
    ) -> list[dict]:
        """
        Process signals for this market. Updates internal state and persists.

        Parameters
        ----------
        signals     : {symbol: 1 / -1 / 0}
        scores      : {symbol: composite_score}
        prices      : {symbol: last_close_price}  in local currency
        regime      : current market regime ("bull_strong"|"bull_caution"|"bear")
        max_possible: max composite score denominator (default 9)
        date        : trade date string (YYYY-MM-DD), defaults to today

        Position cap per symbol:
          base_fraction = _REGIME_BASE_FRACTION[regime]   (e.g. 0.20)
          score_factor  = clamp(score / max_possible, 0.5, 1.0)
          cap           = total_capital * base_fraction * score_factor

        Returns list of trade records for Feishu notification.
        """
        date = date or datetime.today().strftime("%Y-%m-%d")
        results: list[dict] = []

        # ── 1. Exits ──────────────────────────────────────────────────────
        for symbol, signal in signals.items():
            if signal != -1 or symbol not in self.open_symbols:
                continue
            row = self._open[self._open["symbol"] == symbol].iloc[0]
            exit_price = prices.get(symbol, 0.0)
            entry_price = float(row["entry_price"])
            shares = float(row["shares"])
            entry_notional = float(row["notional"])
            pnl = (exit_price - entry_price) * shares if exit_price > 0 else None
            pnl_pct = (pnl / entry_notional * 100) if (pnl is not None and entry_notional > 0) else None

            trade = {
                "date": date, "symbol": symbol, "action": "sell",
                "price": exit_price, "shares": shares,
                "notional": round(exit_price * shares, 2) if exit_price > 0 else entry_notional,
                "score": int(row["score"]),
                "pnl": round(pnl, 2) if pnl is not None else None,
                "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                "currency": self.currency,
            }
            results.append({**trade, "entry_price": entry_price, "entry_date": str(row["entry_date"])})

            # Remove from open, append to history
            self._open = self._open[self._open["symbol"] != symbol].reset_index(drop=True)
            hist_row = {k: trade.get(k) for k in _HIST_COLS}
            self._history = pd.concat(
                [self._history, pd.DataFrame([hist_row])], ignore_index=True
            )
            logger.info("[%s] Exit %s pnl=%.2f (%s)", self.market, symbol,
                        pnl or 0, self.currency)

        # ── 2. Entries (score-weighted, from available capital) ───────────
        new_buys = [
            sym for sym, sig in signals.items()
            if sig == 1 and sym not in self.open_symbols
        ]

        if new_buys and self.available_capital > 0:
            budget = self.available_capital

            if weights:
                sub_w = {s: weights[s] for s in new_buys if s in weights}
                # Log any BUY signals that were filtered out by the portfolio optimizer
                filtered_out = [s for s in new_buys if s not in sub_w]
                for s in filtered_out:
                    logger.info(
                        "[%s] %s had BUY signal but was filtered by portfolio optimizer "
                        "(corr dedup or sector cap) — no order placed", self.market, s
                    )
                    results.append({
                        "symbol": s, "action": "filtered",
                        "reason": "portfolio_optimizer",
                        "currency": self.currency,
                    })
                if not sub_w:
                    # All symbols filtered — fall back to equal weight
                    sub_w = {s: 1 / len(new_buys) for s in new_buys}
                    filtered_out = []  # reset since we're now using all
                    raw_weights = sub_w
                else:
                    total_w = sum(sub_w.values())
                    raw_weights = {s: sub_w[s] / total_w for s in sub_w}
                new_buys = list(raw_weights.keys())
            else:
                raw_weights_raw = {sym: max(scores.get(sym, 1), 1) for sym in new_buys}
                total_w = sum(raw_weights_raw.values())
                raw_weights = {s: raw_weights_raw[s] / total_w for s in new_buys}

            base_fraction = self._REGIME_BASE_FRACTION.get(regime, self.max_position_fraction)

            for symbol in new_buys:
                sym_score = scores.get(symbol, max_possible)
                score_factor = max(0.5, min(1.0, sym_score / max_possible))
                dynamic_cap = self.total_capital * base_fraction * score_factor
                notional = min(budget * raw_weights[symbol], dynamic_cap)
                price = prices.get(symbol, 0.0)
                if notional < 1 or price <= 0:
                    logger.warning("[%s] Skip %s notional=%.2f price=%.4f",
                                   self.market, symbol, notional, price)
                    continue

                shares = round(notional / price, 4)
                score = scores.get(symbol, raw_weights[symbol])
                trade = {
                    "date": date, "symbol": symbol, "action": "buy",
                    "price": price, "shares": shares,
                    "notional": round(notional, 2),
                    "score": score, "pnl": None,
                    "currency": self.currency,
                }
                results.append({**trade, "est_shares": shares, "ref_price": price})

                open_row = {
                    "symbol": symbol, "entry_date": date,
                    "entry_price": price, "shares": shares,
                    "notional": round(notional, 2),
                    "score": score, "currency": self.currency,
                }
                self._open = pd.concat(
                    [self._open, pd.DataFrame([open_row])], ignore_index=True
                )
                hist_row = {k: trade.get(k) for k in _HIST_COLS}
                self._history = pd.concat(
                    [self._history, pd.DataFrame([hist_row])], ignore_index=True
                )
                logger.info("[%s] Buy %s shares=%.4f notional=%.2f score=%d",
                            self.market, symbol, shares, notional, score)

        # ── 3. Persist ────────────────────────────────────────────────────
        if results:
            self._save_df(self._open, "open_positions")
            self._save_df(self._history, "trade_history")

        return results

    # ── Read-only helpers (for dashboard) ─────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        """Return current open positions with live P&L if prices available."""
        return self._open.to_dict(orient="records")

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        """Return recent trade history."""
        if self._history.empty:
            return []
        return self._history.tail(limit).to_dict(orient="records")

    def get_summary(self, prices: Optional[dict[str, float]] = None) -> dict:
        """
        Return portfolio summary with unrealized P&L.
        prices: latest prices for open positions (optional).
        """
        open_positions = self.get_open_positions()
        unrealized_pnl = 0.0
        if prices:
            for pos in open_positions:
                sym = pos["symbol"]
                if sym in prices and prices[sym] > 0:
                    pos["current_price"] = prices[sym]
                    pos["unrealized_pnl"] = round(
                        (prices[sym] - pos["entry_price"]) * pos["shares"], 2
                    )
                    pos["unrealized_pct"] = round(
                        (prices[sym] - pos["entry_price"]) / pos["entry_price"] * 100, 2
                    )
                    unrealized_pnl += pos["unrealized_pnl"]

        realized_pnl = float(
            self._history[self._history["action"] == "sell"]["pnl"].sum()
        ) if not self._history.empty else 0.0

        return {
            "market": self.market,
            "currency": self.currency,
            "total_capital": self.total_capital,
            "invested": round(self.invested_notional, 2),
            "available": round(self.available_capital, 2),
            "open_count": len(open_positions),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "open_positions": open_positions,
        }
