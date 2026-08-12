#!/usr/bin/env python3
"""
Phase 5: head-to-head with Wang et al. 2025 TFT+TimesNet baseline.

Requires Wang's official predictions on our eval split. If missing, the script
falls back to comparing our leak-free run against the published numbers
(29.02% WAPE, +2.58% WPE from their Table 2). In both cases it emits the
comparison CSV the paper's Table `tab:full_results` / `tab:bridge` will cite.

How to produce Wang-side predictions (documented here for reproducibility):
    1. Isolated env:
         conda create -n frn_baseline python=3.8 -y
         conda activate frn_baseline
         pip install -r freshretailnet_repos/frn-50k-baseline/requirements.txt
    2. Run latent demand recovery (TimesNet):
         cd freshretailnet_repos/frn-50k-baseline/latent_demand_recovery
         python exp/app.py --model TimesNet --data_path <path_to_train.parquet>
         # produces demand.parquet with recovered series
    3. Run TFT forecasting on recovered data:
         cd ../demand_forecasting/TFT
         python trainTFT.py --demand
         python predictTFT.py --demand > /tmp/tft_preds.csv
    4. Re-run this script with --wang_preds /tmp/tft_preds.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from src.metrics import wape, wpe, mae, rmse


WANG_PUBLISHED = {
    'WAPE_pct': 29.02,
    'WPE_pct': 2.58,
    'source': 'Wang et al. 2025 Table 2 (TFT + TimesNet recovery)',
}


def _read_wang_predictions(path: Path) -> pd.DataFrame:
    """Load Wang-style predictions from a CSV and normalize prediction column naming."""
    df = pd.read_csv(path)
    if {'sp', 'dt'}.issubset(df.columns) is False:
        raise ValueError(f"Wang predictions CSV must contain columns ['sp', 'dt']; got {list(df.columns)}")

    pred_col = next((c for c in [
        'pred_tft_timesnet',
        'prediction',
        'y_pred',
    ] if c in df.columns), None)
    if pred_col is None:
        raise ValueError(
            "Could not find prediction column in Wang CSV. "
            "Expected one of: pred_tft_timesnet, prediction, y_pred."
        )

    return df[['sp', 'dt', pred_col]].rename(columns={pred_col: 'pred_wang'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', default='output/benchmark/predictions.csv',
                    help='Our leak-free predictions CSV.')
    ap.add_argument('--wang_preds', default=None,
                    help='Optional: Wang TFT+TimesNet predictions CSV with '
                         '(sp, dt, pred_tft_timesnet) columns, aligned to our eval.')
    ap.add_argument('--out', default='output/benchmark')
    args = ap.parse_args()

    ours_path = SCRIPT_DIR / args.ours
    if not ours_path.exists():
        print(f"[error] Our predictions file missing: {ours_path}")
        return 1
    ours = pd.read_csv(ours_path)
    ours['dt'] = pd.to_datetime(ours['dt'])

    out = []
    y = ours['y_true'].values
    ours_pred = ours['pred_ensemble'].values
    ours_row = {
        'model': 'Our Ensemble (leak-free)',
        'n_obs': len(ours),
        'n_sp': int(ours['sp'].nunique()),
        'MAE': mae(y, ours_pred),
        'RMSE': rmse(y, ours_pred),
        'WAPE_pct': wape(y, ours_pred),
        'WPE_pct': wpe(y, ours_pred),
        'source': 'This work (leak-free pipeline)',
    }
    out.append(ours_row)

    if args.wang_preds:
        wp = SCRIPT_DIR / args.wang_preds
        if not wp.exists():
            print(f"[warn] Wang preds file not found: {wp}; falling back to published numbers")
        else:
            wang = _read_wang_predictions(wp)
            wang['dt'] = pd.to_datetime(wang['dt'])
            # Align by (sp, dt)
            merged = ours.merge(wang[['sp', 'dt', 'pred_wang']],
                                on=['sp', 'dt'], how='inner')
            if len(merged) == 0:
                print("[warn] no overlap between our eval and Wang's preds on (sp,dt)")
            else:
                y2 = merged['y_true'].values
                wp_arr = merged['pred_wang'].values
                out.append({
                    'model': 'Wang TFT+TimesNet (re-run)',
                    'n_obs': len(merged),
                    'n_sp': int(merged['sp'].nunique()),
                    'MAE': mae(y2, wp_arr),
                    'RMSE': rmse(y2, wp_arr),
                    'WAPE_pct': wape(y2, wp_arr),
                    'WPE_pct': wpe(y2, wp_arr),
                    'source': 'Wang et al. official baseline, rerun on our eval split',
                })
    if len(out) == 1:
        out.append({
            'model': 'Wang TFT+TimesNet (published)',
            'n_obs': None, 'n_sp': None,
            'MAE': None, 'RMSE': None,
            'WAPE_pct': WANG_PUBLISHED['WAPE_pct'],
            'WPE_pct': WANG_PUBLISHED['WPE_pct'],
            'source': WANG_PUBLISHED['source'],
        })
        print("[info] Wang not rerun; using published numbers. Paper MUST caveat this.")

    df = pd.DataFrame(out)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'wang_vs_ours_head_to_head.csv'
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(df.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
