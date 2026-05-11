"""
Train the ML Regime Classifier (Layer 2) on historical SPY/VIX/HYG data.

Walk-forward training on 15+ years of daily data:
  - Features: SPY MA ratios, momentum, vol, VIX, HYG
  - Target:   Binary (SPY 20d forward return ≥ 0)
  - Model:    XGBoost (or LightGBM / sklearn fallback)

Outputs:
  - Local:  models/ml_regime_model.pkl
  - HF:     {HF_DATASET_REPO}/models/ml_regime_model.pkl
  - JSON:   models/ml_regime_meta.json  (OOS accuracy, AUC, etc.)

Usage
─────
  # Full training (default)
  python3 scripts/train_ml_regime.py

  # Quick test (1 symbol, fewer features)
  python3 scripts/train_ml_regime.py --fast

  # Dry run — train but don't upload to HF
  python3 scripts/train_ml_regime.py --no-upload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yfinance as yf

from strategies.ml_regime import MLRegimeClassifier


def fetch(ticker: str, years: int = 15) -> "pd.DataFrame":
    import pandas as pd
    end   = datetime.today()
    start = pd.Timestamp(end) - pd.DateOffset(years=years)
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    df.index = pd.to_datetime(df.index)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train ML Regime Classifier")
    parser.add_argument("--fast",      action="store_true", help="Quick mode (10yr, skip VIX/HYG)")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--years",     type=int, default=15, help="Years of history to use (default 15)")
    args = parser.parse_args()

    hf_repo  = os.getenv("HF_DATASET_REPO", "")
    hf_token = os.getenv("HF_TOKEN", "")

    years = 10 if args.fast else args.years

    print(f"\n{'='*60}")
    print(f"  ML Regime Classifier — Training")
    print(f"  Mode: {'fast' if args.fast else 'full'}  |  History: {years}yr")
    print(f"  Upload to HF: {not args.no_upload and bool(hf_repo and hf_token)}")
    print(f"{'='*60}\n")

    # ── Fetch data ─────────────────────────────────────────────────────────
    print("📥 Fetching SPY historical data...")
    spy_df = fetch("SPY", years=years)
    print(f"   SPY: {len(spy_df)} rows  ({spy_df.index[0].date()} – {spy_df.index[-1].date()})")

    vix_df = None
    hyg_df = None

    if not args.fast:
        print("📥 Fetching VIX data...")
        try:
            vix_df = fetch("^VIX", years=years)
            print(f"   VIX: {len(vix_df)} rows")
        except Exception as e:
            print(f"   ⚠️  VIX fetch failed: {e} — training without VIX features")

        print("📥 Fetching HYG data...")
        try:
            hyg_df = fetch("HYG", years=years)
            print(f"   HYG: {len(hyg_df)} rows")
        except Exception as e:
            print(f"   ⚠️  HYG fetch failed: {e} — training without HYG features")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n🏋️  Training walk-forward model...")
    clf  = MLRegimeClassifier()
    meta = clf.train(spy_df, vix_df, hyg_df)

    print(f"\n📊 Walk-forward results:")
    for w in meta.get("oos_windows", []):
        print(
            f"   OOS {w['test_start'][:7]}–{w['test_end'][:7]}: "
            f"acc={w['accuracy']:.3f}  AUC={w['auc']:.3f}  n={w['n_test']}"
        )
    print(f"\n   ✅ Avg OOS accuracy: {meta['avg_accuracy']:.3f}")
    print(f"   ✅ Avg OOS AUC:      {meta['avg_auc']:.3f}")
    print(f"   ✅ Final training n:  {meta['n_train']} rows\n")

    # ── Save local ─────────────────────────────────────────────────────────
    model_path = clf.save_local()
    print(f"💾 Model saved: {model_path}")

    meta_out = {
        **meta,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": "fast" if args.fast else "full",
        "years": years,
    }
    meta_path = ROOT / "models" / "ml_regime_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"💾 Meta saved: {meta_path}")

    # ── Upload to HF ───────────────────────────────────────────────────────
    if not args.no_upload:
        if hf_repo and hf_token:
            print(f"\n☁️  Uploading to HF Dataset ({hf_repo})...")
            ok = clf._upload_via_hub(hf_repo, hf_token)
            if ok:
                print("   ✅ Model uploaded successfully")
                # Also upload meta JSON
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.upload_file(
                        path_or_fileobj=str(meta_path),
                        path_in_repo="models/ml_regime_meta.json",
                        repo_id=hf_repo,
                        repo_type="dataset",
                        commit_message="Auto: update ML regime meta",
                    )
                    print("   ✅ Meta JSON uploaded")
                except Exception as e:
                    print(f"   ⚠️  Meta upload failed: {e}")
            else:
                print("   ❌ Upload failed — model saved locally only")
        else:
            print("\n⚠️  HF_TOKEN / HF_DATASET_REPO not set — skipping upload")

    print(f"\n🎉 Done! Model ready for inference.\n")
    return meta_out


if __name__ == "__main__":
    main()
