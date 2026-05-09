"""
Apply the results from optimize_params.py back into config.py.

Usage (after downloading optimal_params.json from GitHub Actions artifacts):
  python3 scripts/apply_optimal_params.py [--dry-run]

With --dry-run: just print what would change, don't write.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "scripts", "optimal_params.json")
CFG_PATH  = os.path.join(ROOT, "config.py")


def load_results() -> dict:
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: {JSON_PATH} not found. Run optimize_params.py first.")
        sys.exit(1)
    with open(JSON_PATH) as f:
        return json.load(f)


def update_strategy_params(cfg_text: str, new_params: dict) -> tuple[str, list[str]]:
    """
    Replace values inside STRATEGY_PARAMS dict in config.py.
    Returns (updated_text, list_of_change_descriptions).
    """
    changes = []

    # Key mapping: optimize_params key → config.py STRATEGY_PARAMS key
    KEY_MAP = {
        "golden_cross": "golden_cross",
        "supertrend":   "supertrend",
        "donchian":     "donchian",
        "ema_adx":      "ema_adx",
        "macd":         "macd",
        "roc":          "roc",
        "rsi":          "rsi",
        "bollinger":    "bollinger",
    }

    for opt_key, cfg_key in KEY_MAP.items():
        if opt_key not in new_params:
            continue
        new_vals = new_params[opt_key]

        for param_name, new_val in new_vals.items():
            # Find the config key block and update the specific param
            # Pattern: inside the cfg_key block, find "param_name": old_val
            pattern = (
                r'(\"' + re.escape(cfg_key) + r'\":\s*\{[^}]*?'
                r'\"' + re.escape(param_name) + r'\":\s*)([0-9.]+)'
            )
            def replacer(m, nv=new_val):
                old = m.group(2)
                if float(old) == nv:
                    return m.group(0)
                changes.append(
                    f"  STRATEGY_PARAMS['{cfg_key}']['{param_name}']: {old} → {nv}"
                )
                return m.group(1) + str(nv)

            cfg_text = re.sub(pattern, replacer, cfg_text, flags=re.DOTALL)

    return cfg_text, changes


def update_composite_threshold(cfg_text: str, buy_t: float, sell_t: float) -> tuple[str, list[str]]:
    """Update composite buy/sell threshold in STRATEGY_PARAMS."""
    changes = []

    for key, new_val in [("buy_threshold", buy_t), ("sell_threshold", sell_t)]:
        pattern = r'(\"composite\":\s*\{[^}]*?\"' + re.escape(key) + r'\":\s*)([0-9.]+)'
        def replacer(m, nv=new_val, k=key):
            old = m.group(2)
            if float(old) == nv:
                return m.group(0)
            changes.append(f"  STRATEGY_PARAMS['composite']['{k}']: {old} → {nv}")
            return m.group(1) + str(nv)
        cfg_text = re.sub(pattern, replacer, cfg_text, flags=re.DOTALL)

    return cfg_text, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = load_results()
    print(f"\nApplying results from: {data.get('computed_at', 'unknown')}")
    print(f"Mode: {data.get('mode', 'unknown')}  |  Symbols: {data.get('symbols', [])}\n")

    with open(CFG_PATH) as f:
        cfg_text = f.read()

    all_changes = []

    # Update strategy params
    cfg_text, ch = update_strategy_params(cfg_text, data.get("strategy_params", {}))
    all_changes.extend(ch)

    # Update composite thresholds
    comp = data.get("composite", {})
    cfg_text, ch = update_composite_threshold(
        cfg_text,
        comp.get("buy_threshold", 6.0),
        comp.get("sell_threshold", 4.0),
    )
    all_changes.extend(ch)

    if not all_changes:
        print("✅ No changes needed — config.py already matches optimal params.")
        return

    print("Changes to apply:")
    for c in all_changes:
        print(c)

    if args.dry_run:
        print("\n[DRY RUN] config.py not modified.")
        return

    # Backup
    backup = CFG_PATH + ".bak"
    with open(backup, "w") as f:
        with open(CFG_PATH) as orig:
            f.write(orig.read())

    with open(CFG_PATH, "w") as f:
        f.write(cfg_text)

    print(f"\n✅ config.py updated ({len(all_changes)} changes).")
    print(f"   Backup saved to config.py.bak")
    print(f"\nNext steps:")
    print(f"  1. Review the diff:  git diff config.py")
    print(f"  2. Commit:           git add config.py && git commit -m 'Monthly param update'")
    print(f"  3. Push:             git push origin main")


if __name__ == "__main__":
    main()
