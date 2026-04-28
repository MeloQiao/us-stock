"""
Reset virtual portfolio for HK and CN markets.
Uploads empty open_positions + trade_history parquet files to HF Dataset.

Usage:
  export HF_TOKEN=hf_xxx
  export HF_DATASET_REPO=your-username/your-dataset-repo
  python3 scripts/reset_virtual_portfolio.py
"""

import io
import os
import sys

import pandas as pd
from huggingface_hub import HfApi

HF_TOKEN       = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")

if not HF_TOKEN or not HF_DATASET_REPO:
    print("ERROR: HF_TOKEN and HF_DATASET_REPO must be set as environment variables.")
    sys.exit(1)

_OPEN_COLS = ["symbol", "entry_date", "entry_price", "shares", "notional", "score", "currency"]
_HIST_COLS = ["date", "symbol", "action", "price", "shares", "notional", "score", "pnl", "currency"]

api = HfApi(token=HF_TOKEN)

for market in ("hk", "cn"):
    for name, cols in (("open_positions", _OPEN_COLS), ("trade_history", _HIST_COLS)):
        path_in_repo = f"portfolio/{market}/{name}.parquet"
        empty_df = pd.DataFrame(columns=cols)
        buf = io.BytesIO()
        empty_df.to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Reset virtual portfolio [{market}] — fresh start",
        )
        print(f"✓ Reset {path_in_repo}")

print("\nDone. Both HK and CN virtual portfolios have been cleared.")
