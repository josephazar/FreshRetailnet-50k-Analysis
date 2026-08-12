#!/usr/bin/env python3
"""
Statistical inference for the benchmark uses per-SP paired bootstrap intervals
and a fixed Holm family of 4 model pairs x 2 metrics = 8 tests.

Reads benchmark_per_sp_predictions.csv (produced by benchmark_comparison.py or
ablation.py); performs per-SP clustered bootstrap CIs and paired Wilcoxon tests
(LightGBM-Minimal vs. each alternative) on MAE and WAPE.
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

from src.metrics import (
    wape,
    wpe,
    mae,
    cluster_bootstrap_ci,
    paired_wilcoxon,
    holm_bonferroni,
    per_sp_metrics,
)


REFERENCE_MODEL = 'LightGBM-Minimal'
ALT_MODELS = ['Naive Seasonal', 'SSA', 'XGBoost-Minimal', 'CatBoost-Minimal']
METRICS = ['mae', 'wape']
FIXED_FAMILY_SIZE = len(ALT_MODELS) * len(METRICS)  # 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='output/benchmark/benchmark_per_sp_predictions.csv',
                    help='Path to per-SP predictions CSV (from benchmark_comparison)')
    ap.add_argument('--out', default='output/benchmark')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = SCRIPT_DIR / args.preds
    if not preds_path.exists():
        # No silent fallback to other run directories: significance tables
        # must come from the predictions file the caller named.
        print(f"[error] preds file not found: {preds_path}")
        return 1

    df = pd.read_csv(preds_path)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Identify prediction columns (pred_* prefix by convention)
    pred_cols = {
        'LightGBM-Minimal': 'pred_lightgbm_minimal',
        'XGBoost-Minimal': 'pred_xgboost_minimal',
        'CatBoost-Minimal': 'pred_catboost_minimal',
        'Naive Seasonal': 'pred_naive_seasonal',
        'SSA': 'pred_ssa',
    }
    pred_cols = {k: v for k, v in pred_cols.items() if v in df.columns}
    if REFERENCE_MODEL not in pred_cols:
        print(f"[error] reference model {REFERENCE_MODEL} not in predictions")
        return 1
    ref_col = pred_cols[REFERENCE_MODEL]
    y = df['actual'].values if 'actual' in df.columns else df['y_true'].values
    sp = df['sp'].values

    # Bootstrap CIs per model per metric
    boot_rows = []
    metric_fns = {'mae': mae, 'wape': wape, 'wpe': wpe}
    for model_name, pcol in pred_cols.items():
        p = df[pcol].values
        for mname, fn in metric_fns.items():
            point, lo, hi = cluster_bootstrap_ci(
                y, p, sp, fn, n_boot=args.n_boot, alpha=0.05, seed=args.seed,
            )
            boot_rows.append({
                'model': model_name,
                'metric': mname,
                'point': point,
                'ci_low': lo,
                'ci_high': hi,
            })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / 'bootstrap_ci.csv', index=False)

    # Per-SP metrics for paired Wilcoxon
    sp_ref = per_sp_metrics(y, df[ref_col].values, sp)
    wilcoxon_rows = []
    p_values = []
    for alt_name in ALT_MODELS:
        if alt_name not in pred_cols:
            # Fixed family: a missing test remains NaN and still counts.
            for mname in METRICS:
                p_values.append(float('nan'))
                wilcoxon_rows.append({
                    'alt_model': alt_name, 'metric': mname,
                    'p_raw': float('nan'), 'r_biserial': 0.0,
                    'note': 'model not in predictions CSV',
                })
            continue
        alt_col = pred_cols[alt_name]
        sp_alt = per_sp_metrics(y, df[alt_col].values, sp)
        merged = sp_ref[['sp'] + METRICS].merge(
            sp_alt[['sp'] + METRICS], on='sp', suffixes=('_ref', '_alt')
        )
        for mname in METRICS:
            r = merged[f'{mname}_ref'].values
            a = merged[f'{mname}_alt'].values
            p, rb = paired_wilcoxon(r, a, alternative='less')
            p_values.append(p)
            wilcoxon_rows.append({
                'alt_model': alt_name, 'metric': mname,
                'p_raw': p, 'r_biserial': rb,
                'note': '',
            })

    # Holm-Bonferroni with the fixed family size
    adjusted = holm_bonferroni(p_values, family_size=FIXED_FAMILY_SIZE)
    for row, adj in zip(wilcoxon_rows, adjusted):
        row['p_holm'] = adj
    wx_df = pd.DataFrame(wilcoxon_rows)
    wx_df.to_csv(out_dir / 'wilcoxon_holm.csv', index=False)

    summary = {
        'reference_model': REFERENCE_MODEL,
        'fixed_family_size': FIXED_FAMILY_SIZE,
        'n_boot': args.n_boot,
        'n_sps': int(len(np.unique(sp))),
        'n_rows': int(len(df)),
        'bootstrap': out_dir / 'bootstrap_ci.csv',
        'wilcoxon': out_dir / 'wilcoxon_holm.csv',
    }
    summary_path = out_dir / 'statistical_significance_summary.json'
    summary_path.write_text(json.dumps({k: str(v) for k, v in summary.items()}, indent=2))
    print(f"\nFixed family size: {FIXED_FAMILY_SIZE}")
    print(f"Bootstrap rows: {len(boot_df)}")
    print(f"Wilcoxon rows: {len(wx_df)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
