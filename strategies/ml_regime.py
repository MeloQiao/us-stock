"""
Layer 2: ML Regime Classifier — XGBoost-based market state predictor

Predicts P(SPY 20-day forward return > 0) from macro + technical features.
Used as an additional position-sizing multiplier layered on top of the
rule-based Crash Shield (Layer 1).

Feature groups:
  SPY   — price/MA ratios, multi-timeframe momentum, realized vol ratio,
           drawdown from 252d high, RSI
  VIX   — level, ratio vs 20d MA, 1-year percentile, 5d change
  HYG   — 20d return + vs MA50 (credit spread proxy)

Training:
  Walk-forward: 3 non-overlapping out-of-sample windows
    Window 1: train 2009-2016  / test 2017-2018
    Window 2: train 2009-2019  / test 2020-2021
    Window 3: train 2009-2021  / test 2022-2025
  Target: binary — SPY 20d forward return ≥ 0
  Model:  XGBoost (falls back to sklearn GradientBoosting if xgboost missing)

Usage
─────
  from strategies.ml_regime import MLRegimeClassifier
  clf = MLRegimeClassifier()
  clf.load()                                           # from HF or local
  prob = clf.predict_proba_latest(spy_df, vix_df, hyg_df)
  multiplier = clf.to_position_multiplier(prob)        # 0.3 – 1.0
"""

from __future__ import annotations

import io
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODEL_FILENAME = "ml_regime_model.pkl"
_LOCAL_MODEL_DIR = Path(__file__).parent.parent / "models"
_HF_MODEL_PATH   = "models/ml_regime_model.pkl"   # path inside HF dataset repo


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period // 2).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period // 2).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def build_features(
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    hyg_df: Optional[pd.DataFrame] = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Build the feature matrix (and optionally the target column).

    All features use only past data — no lookahead bias.
    The target column `y` is spy.pct_change(20).shift(-20), which means
    rows where `y` is NaN (last 20 trading days) must be dropped before training.

    Parameters
    ----------
    spy_df         : SPY (or benchmark) OHLCV DataFrame
    vix_df         : VIX OHLCV (optional)
    hyg_df         : HYG OHLCV (optional)
    include_target : whether to compute the forward-return target column

    Returns
    -------
    DataFrame with columns: [feature columns...] + optionally ["y"]
    """
    spy = spy_df["Close"].dropna()
    idx = spy.index
    f   = pd.DataFrame(index=idx)

    # ── SPY MA ratios ──────────────────────────────────────────────────────
    ma20  = spy.rolling(20,  min_periods=10).mean()
    ma50  = spy.rolling(50,  min_periods=25).mean()
    ma200 = spy.rolling(200, min_periods=100).mean()

    f["price_to_ma20"]  = spy / ma20  - 1
    f["price_to_ma50"]  = spy / ma50  - 1
    f["price_to_ma200"] = spy / ma200 - 1
    f["ma50_to_ma200"]  = ma50 / (ma200 + 1e-9) - 1

    # ── SPY momentum (multi-timeframe) ─────────────────────────────────────
    f["ret_1m"]  = spy.pct_change(21)
    f["ret_3m"]  = spy.pct_change(63)
    f["ret_6m"]  = spy.pct_change(126)
    f["ret_12m"] = spy.pct_change(252)

    # ── Drawdown from 1-year high ──────────────────────────────────────────
    roll_max      = spy.rolling(252, min_periods=50).max()
    f["drawdown"] = spy / (roll_max + 1e-9) - 1

    # ── Realized vol + vol-ratio (crash protection signal) ─────────────────
    log_ret    = np.log(spy / spy.shift(1))
    vol_21     = log_ret.rolling(21).std() * np.sqrt(252)
    vol_63     = log_ret.rolling(63).std() * np.sqrt(252)
    f["vol_21"]    = vol_21
    f["vol_63"]    = vol_63
    f["vol_ratio"] = vol_21 / (vol_63 + 1e-9)   # > 1.5 → vol spike / crash risk

    # ── RSI ───────────────────────────────────────────────────────────────
    f["rsi_14"] = _rsi(spy, 14) / 100.0   # normalise to 0-1

    # ── VIX features ──────────────────────────────────────────────────────
    if vix_df is not None and not vix_df.empty:
        vix       = vix_df["Close"].dropna().reindex(idx, method="ffill")
        vix_ma20  = vix.rolling(20, min_periods=10).mean()
        vix_pct   = vix.rolling(252, min_periods=50).rank(pct=True)

        f["vix"]            = vix / 100.0
        f["vix_vs_ma20"]    = vix / (vix_ma20 + 1e-9) - 1
        f["vix_percentile"] = vix_pct
        f["vix_ret_5d"]     = vix.pct_change(5)
    else:
        for col in ("vix", "vix_vs_ma20", "vix_percentile", "vix_ret_5d"):
            f[col] = np.nan

    # ── HYG credit-spread proxy ────────────────────────────────────────────
    if hyg_df is not None and not hyg_df.empty:
        hyg       = hyg_df["Close"].dropna().reindex(idx, method="ffill")
        hyg_ma50  = hyg.rolling(50, min_periods=25).mean()
        f["hyg_ret_20d"] = hyg.pct_change(20)
        f["hyg_vs_ma50"] = hyg / (hyg_ma50 + 1e-9) - 1
    else:
        f["hyg_ret_20d"] = np.nan
        f["hyg_vs_ma50"] = np.nan

    # ── Target (20-day forward return, binary) ─────────────────────────────
    if include_target:
        fwd = spy.pct_change(20).shift(-20)
        f["y"] = (fwd >= 0).astype(int)

    return f


FEATURE_COLS = [
    "price_to_ma20", "price_to_ma50", "price_to_ma200", "ma50_to_ma200",
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "drawdown",
    "vol_21", "vol_63", "vol_ratio",
    "rsi_14",
    "vix", "vix_vs_ma20", "vix_percentile", "vix_ret_5d",
    "hyg_ret_20d", "hyg_vs_ma50",
]


# ══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_model():
    """
    Return a fitted-ready gradient boosting classifier.
    Preference order: XGBoost → LightGBM → sklearn GradientBoosting.
    Each candidate is tested with a tiny fit() to catch runtime dylib errors
    (e.g. XGBoost on macOS without libomp) before committing to it.
    """
    import numpy as np

    def _quick_test(mdl):
        """Return True if mdl.fit() works on a 4-sample toy dataset."""
        try:
            X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
            y = np.array([0, 1, 0, 1])
            mdl.fit(X, y)
            return True
        except Exception:
            return False

    try:
        from xgboost import XGBClassifier
        mdl = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
        if _quick_test(mdl):
            # Return a fresh unfitted instance (quick test consumed the previous one)
            logger.info("ML Regime: using XGBoost")
            return XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                reg_alpha=0.1, reg_lambda=1.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42, n_jobs=1,
            )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        mdl = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1, verbose=-1,
        )
        if _quick_test(mdl):
            logger.info("ML Regime: using LightGBM")
            return LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1, verbose=-1,
            )
    except Exception:
        pass

    from sklearn.ensemble import GradientBoostingClassifier
    logger.info("ML Regime: using sklearn GradientBoosting (xgboost/lgbm not available)")
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )


def _fill_missing(X: pd.DataFrame, fallback_medians: dict | None = None) -> pd.DataFrame:
    """
    Fill NaNs with column median.
    If X has only one row (inference), column-wise median = NaN for NaN cols.
    Use fallback_medians (from training distribution) in that case.
    """
    result = X.fillna(X.median())
    if fallback_medians and result.isnull().any(axis=None):
        # Single-row or all-NaN column: use training median
        fb = pd.Series(fallback_medians)
        result = result.fillna(fb)
    # Final safety: replace any remaining NaN with 0
    return result.fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward training
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_train(
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    hyg_df: Optional[pd.DataFrame] = None,
) -> tuple:
    """
    Walk-forward training.  Returns (fitted_model, oos_accuracy, oos_df).

    Training windows (non-overlapping OOS):
      W1: train 2009-2016 / OOS 2017-2018
      W2: train 2009-2019 / OOS 2020-2021
      W3: train 2009-2021 / OOS 2022-2025
    Final model: retrained on full history (all data up to today – 20d).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, roc_auc_score

    df = build_features(spy_df, vix_df, hyg_df, include_target=True)
    df = df.dropna(subset=["y"])                # drop last 20 days (target NaN)
    df = df.dropna(subset=FEATURE_COLS, how="all")

    windows = [
        ("2009-01-01", "2016-12-31", "2017-01-01", "2018-12-31"),
        ("2009-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
        ("2009-01-01", "2021-12-31", "2022-01-01", "2025-12-31"),
    ]

    oos_results = []
    for train_start, train_end, test_start, test_end in windows:
        train = df.loc[train_start:train_end].dropna()
        test  = df.loc[test_start:test_end].dropna()
        if len(train) < 200 or len(test) < 50:
            logger.warning("Skipping WF window %s–%s (too few rows)", train_start, test_end)
            continue

        X_train = _fill_missing(train[FEATURE_COLS])
        y_train = train["y"].values
        X_test  = _fill_missing(test[FEATURE_COLS])
        y_test  = test["y"].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        mdl = _get_model()
        mdl.fit(X_tr_s, y_train)

        preds  = mdl.predict(X_te_s)
        probas = mdl.predict_proba(X_te_s)[:, 1]
        acc    = float(accuracy_score(y_test, preds))
        auc    = float(roc_auc_score(y_test, probas))

        logger.info(
            "WF window %s—%s  acc=%.3f  auc=%.3f",
            test_start, test_end, acc, auc,
        )
        oos_results.append({"test_start": test_start, "test_end": test_end,
                             "accuracy": acc, "auc": auc,
                             "n_test": len(test)})

    # ── Final model: train on ALL history ─────────────────────────────────
    full = df.dropna()
    if len(full) < 300:
        raise ValueError(f"Not enough training data: {len(full)} rows after dropna")

    X_full = _fill_missing(full[FEATURE_COLS])
    y_full = full["y"].values

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_full)

    final_mdl = _get_model()
    final_mdl.fit(X_s, y_full)

    avg_acc = float(np.mean([r["accuracy"] for r in oos_results])) if oos_results else 0.0
    avg_auc = float(np.mean([r["auc"]      for r in oos_results])) if oos_results else 0.0
    logger.info("Final model trained on %d rows. OOS avg acc=%.3f  auc=%.3f",
                len(full), avg_acc, avg_auc)

    # Store training medians so inference works even with short history
    train_medians = X_full.median().to_dict()

    pipeline = {"scaler": scaler, "model": final_mdl, "medians": train_medians}
    meta = {
        "oos_windows": oos_results,
        "avg_accuracy": avg_acc,
        "avg_auc": avg_auc,
        "n_train": len(full),
        "feature_cols": FEATURE_COLS,
    }
    return pipeline, meta, oos_results


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class MLRegimeClassifier:
    """
    Encapsulates the trained pipeline (StandardScaler + XGBoost/LGBM/GB).

    Pipeline lifecycle
    ──────────────────
    Training (GitHub Actions, monthly):
      clf = MLRegimeClassifier()
      clf.train(spy_df, vix_df, hyg_df)
      clf.save_local()
      clf.upload_to_hf(hf_repo, hf_token)

    Inference (daily pipeline):
      clf = MLRegimeClassifier()
      clf.load(hf_repo=..., hf_token=...)   # tries HF first, then local
      prob = clf.predict_proba_latest(spy_df, vix_df, hyg_df)
      mult = clf.to_position_multiplier(prob)
    """

    def __init__(self):
        self._pipeline: dict | None = None   # {"scaler": ..., "model": ...}
        self._meta: dict = {}
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        hyg_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Train walk-forward model. Returns meta dict."""
        pipeline, meta, _ = walk_forward_train(spy_df, vix_df, hyg_df)
        self._pipeline = pipeline
        self._meta     = meta
        self._trained  = True
        return meta

    # ── Persistence ───────────────────────────────────────────────────────

    def save_local(self, path: Optional[Path] = None) -> Path:
        """Save model + meta to local disk."""
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        save_dir = path or _LOCAL_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / _MODEL_FILENAME
        with open(out, "wb") as f:
            pickle.dump({"pipeline": self._pipeline, "meta": self._meta}, f)
        logger.info("ML Regime model saved to %s", out)
        return out

    def load_local(self, path: Optional[Path] = None) -> bool:
        """Try to load from local disk. Returns True on success."""
        src = (path or _LOCAL_MODEL_DIR) / _MODEL_FILENAME
        if not src.exists():
            return False
        try:
            with open(src, "rb") as f:
                obj = pickle.load(f)
            self._pipeline = obj["pipeline"]
            self._meta     = obj.get("meta", {})
            self._trained  = True
            logger.info("ML Regime: loaded from local %s", src)
            return True
        except Exception as e:
            logger.warning("ML Regime: local load failed: %s", e)
            return False

    def upload_to_hf(self, hf_repo: str, hf_token: str) -> bool:
        """Upload model pickle to HF Dataset."""
        if not self._trained:
            raise RuntimeError("Model not trained.")
        try:
            import requests
            buf = io.BytesIO()
            pickle.dump({"pipeline": self._pipeline, "meta": self._meta}, buf)
            buf.seek(0)

            api_url = (
                f"https://huggingface.co/api/datasets/{hf_repo}/upload"
                f"/{_HF_MODEL_PATH}"
            )
            headers = {"Authorization": f"Bearer {hf_token}"}
            resp = requests.put(
                f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{_HF_MODEL_PATH}",
                headers=headers,
                data=buf.read(),
            )
            # HF uses the Inference API / hub API for uploads; use huggingface_hub instead
            self._upload_via_hub(hf_repo, hf_token)
            return True
        except Exception as e:
            logger.error("ML Regime: HF upload via requests failed: %s — trying hub", e)
            return self._upload_via_hub(hf_repo, hf_token)

    def _upload_via_hub(self, hf_repo: str, hf_token: str) -> bool:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            local_path = _LOCAL_MODEL_DIR / _MODEL_FILENAME
            if not local_path.exists():
                self.save_local()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=_HF_MODEL_PATH,
                repo_id=hf_repo,
                repo_type="dataset",
                commit_message="Auto: update ML regime model",
            )
            logger.info("ML Regime: uploaded to HF Dataset %s/%s", hf_repo, _HF_MODEL_PATH)
            return True
        except Exception as e:
            logger.error("ML Regime: HF hub upload failed: %s", e)
            return False

    def load_from_hf(self, hf_repo: str, hf_token: str) -> bool:
        """Download model from HF Dataset and load it. Returns True on success."""
        try:
            import requests
            url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{_HF_MODEL_PATH}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                logger.info("ML Regime: HF model not found (status %d)", r.status_code)
                return False
            obj = pickle.loads(r.content)
            self._pipeline = obj["pipeline"]
            self._meta     = obj.get("meta", {})
            self._trained  = True
            logger.info("ML Regime: loaded from HF Dataset")
            return True
        except Exception as e:
            logger.warning("ML Regime: HF load failed: %s", e)
            return False

    def load(
        self,
        hf_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> bool:
        """
        Load model — tries HF first (if credentials supplied), then local.
        Returns True if a model was loaded successfully.
        """
        if hf_repo and hf_token:
            if self.load_from_hf(hf_repo, hf_token):
                return True
        if self.load_local():
            return True
        logger.warning("ML Regime: no model found (HF or local). Will return neutral probability.")
        return False

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba_latest(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        hyg_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Predict P(market positive next 20 days) using the latest available row.

        Returns
        -------
        float in [0.0, 1.0]
          0.0 = very bearish, 0.5 = neutral, 1.0 = very bullish
          Returns 0.5 (neutral) if model not loaded or features invalid.
        """
        if not self._trained or self._pipeline is None:
            logger.debug("ML Regime: model not loaded — returning neutral 0.5")
            return 0.5

        try:
            df = build_features(spy_df, vix_df, hyg_df, include_target=False)
            latest = df.iloc[-1:][FEATURE_COLS]

            # Use training-distribution medians for NaN imputation
            # (avoids NaN when history is shorter than longest lookback, e.g. 252d)
            stored_medians = self._pipeline.get("medians")
            latest = _fill_missing(latest, fallback_medians=stored_medians)

            if latest.isnull().any(axis=None):
                logger.warning("ML Regime: still has NaN after imputation — returning 0.5")
                return 0.5

            scaler = self._pipeline["scaler"]
            model  = self._pipeline["model"]
            X_s    = scaler.transform(latest.values)
            prob   = float(model.predict_proba(X_s)[0, 1])
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as e:
            logger.error("ML Regime: predict_proba_latest failed: %s", e)
            return 0.5

    @staticmethod
    def to_position_multiplier(
        prob: float,
        min_mult: float = 0.25,
        max_mult: float = 1.0,
        bull_threshold: float = 0.55,
        bear_threshold: float = 0.45,
    ) -> float:
        """
        Convert ML probability to a position size multiplier.

        Design principle: the ML layer acts as a "risk-off" tool only.
          - Neutral / bullish zone (prob ≥ 0.45): multiplier = 1.0 (no change)
          - Bearish zone (prob < 0.45):           scale down linearly to min_mult
            prob = 0.45 → 1.0
            prob = 0.00 → min_mult (0.25)

        This ensures that when the model has no strong view (prob ~0.5) we do
        NOT inadvertently reduce position sizes.  The multiplier only kicks in
        when the model is genuinely bearish.

        Returns
        -------
        float in [min_mult, 1.0]
        """
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob >= bear_threshold:
            return max_mult                                    # neutral / bullish → no change
        else:
            # Linear: bear_threshold → 1.0, 0.0 → min_mult
            t = (bear_threshold - prob) / bear_threshold       # 0 at threshold, 1 at prob=0
            t = min(t, 1.0)
            return round(max_mult - (max_mult - min_mult) * t, 3)

    def describe(self) -> dict:
        """Return a human-readable summary of the loaded model."""
        if not self._trained:
            return {"status": "not loaded"}
        return {
            "status": "loaded",
            "avg_accuracy":  self._meta.get("avg_accuracy"),
            "avg_auc":       self._meta.get("avg_auc"),
            "n_train":       self._meta.get("n_train"),
            "oos_windows":   self._meta.get("oos_windows", []),
            "feature_count": len(FEATURE_COLS),
        }
