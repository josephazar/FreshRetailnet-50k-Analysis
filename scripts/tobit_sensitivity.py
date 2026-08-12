#!/usr/bin/env python3
"""
Phase 4b: sensitivity of heuristic-IMR blending weight w in {0.0, 0.25, 0.5, 0.75, 1.0}.

Also runs the real Tobit MLE as an additional row.

Uses a stratified 2000-SP subsample (same as recovery_comparison.py) for
computational tractability. Trains a small LightGBM on each recovered series
and reports downstream WAPE, WPE, profit.

Outputs: output/benchmark/tobit_sensitivity.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from src.feature_builder import FeatureBuilder, predict_eval_recursive
from src.recovery import heuristic_imr, tobit_mle
from src.inventory import DEFAULT_INV_CFG, compute_policies, residual_quantile_from_cv
from src.metrics import wape, wpe, mae
from pipeline_leakfree import load_data, train_cv, retrain_final, set_all_seeds


def run_one(train, ev, method: str, blend: float | None, out_tag: str) -> dict:
    t0 = time.time()
    if method == 'heuristic_imr':
        tr = heuristic_imr(train.copy(), blend=blend)
    elif method == 'tobit_mle':
        tr = tobit_mle(train.copy())
    else:
        raise ValueError(method)
    uplift = (tr['dem_rec'].mean() / tr['sale_amount'].mean() - 1) * 100

    fb = FeatureBuilder(n_clusters=min(25, max(2, tr['sp'].nunique() // 10)),
                         seed=42).fit(tr)
    tf = fb.transform(tr)
    fc = fb.feature_cols(tf)

    _, cv_residuals, best_iters, _ = train_cv(tf, fc, cv_folds=2, val_size=7)
    finals = retrain_final(tf, fc, best_iters)
    preds, ev_sorted = predict_eval_recursive(fb, finals, fc, tr, ev)
    ens = np.mean([preds['mae'], preds['huber'], preds['q50']], axis=0)

    y = ev_sorted['sale_amount'].values
    sp = ev_sorted['sp'].values

    # Compute inventory with OOS residuals
    cr = DEFAULT_INV_CFG['p'] / (DEFAULT_INV_CFG['p'] + DEFAULT_INV_CFG['w'])
    sp_rq = residual_quantile_from_cv(cv_residuals, cr)
    preds_for_inv = {'ensemble': ens, 'q10': preds['q10'],
                     'q50': preds['q50'], 'q90': preds['q90']}
    rdf, _ = compute_policies(preds_for_inv, sp, tr, sp_rq,
                               DEFAULT_INV_CFG, actuals_for_eval=y)
    emp_row = rdf[rdf['Policy'] == 'Empirical Newsvendor'].iloc[0]

    return {
        'config': out_tag,
        'method': method,
        'blend': blend if blend is not None else float('nan'),
        'uplift_pct': float(uplift),
        'wape': wape(y, ens),
        'wpe': wpe(y, ens),
        'mae': mae(y, ens),
        'profit': float(emp_row['Avg Profit']),
        'fill_rate': float(emp_row['Fill Rate']),
        'time_s': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=2000)
    ap.add_argument('--out', default='output/benchmark')
    args = ap.parse_args()

    set_all_seeds(42)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_sp}-SP stratified subsample...")
    train, ev = load_data(n_sp=args.n_sp)
    print(f"  Train: {train.shape}, Eval: {ev.shape}")

    rows = []
    for blend in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"\n--- heuristic_imr blend={blend} ---")
        rows.append(run_one(train, ev, 'heuristic_imr', blend,
                             f'heuristic_imr_w={blend}'))
        print(f"  wape={rows[-1]['wape']:.2f} wpe={rows[-1]['wpe']:+.2f} "
              f"profit={rows[-1]['profit']:.4f}")

    print("\n--- tobit_mle ---")
    rows.append(run_one(train, ev, 'tobit_mle', None, 'tobit_mle'))
    print(f"  wape={rows[-1]['wape']:.2f} wpe={rows[-1]['wpe']:+.2f} "
          f"profit={rows[-1]['profit']:.4f}")

    df = pd.DataFrame(rows)
    out_path = out_dir / 'tobit_sensitivity.csv'
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(df[['config', 'uplift_pct', 'wape', 'wpe', 'profit', 'time_s']].to_string(index=False))


if __name__ == '__main__':
    main()
