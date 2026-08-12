"""Hyperparameter robustness grid under the common validation protocol.

Purpose: show that the pipeline's LightGBM configuration sits on a flat
region of the loss surface, i.e. the headline results are not an artifact of
a lucky hyperparameter draw. This study provides RATIONALE for the reported
configuration; it never re-tunes it and never touches the held-out test set.

The grid runs under the same fold-causal machinery as
scripts/multistep_validation.py:
  - recovery and FeatureBuilder refit per fold on data up to the fold's
    train_end only (amendment A1);
  - boosting iterations chosen on an inner temporal split (last 7 days of
    the fold-training window), refit on the full fold-training window at
    fixed rounds; the outer validation fold never participates in training
    (amendment A2);
  - true recursive 7-day multi-step validation via predict_eval_recursive,
    the same code path as the final test evaluation (amendment A3).

Design
------
- Grid: learning_rate in {0.02, 0.03, 0.05} x num_leaves in {127, 255, 511}
  (9 configurations; the paper's configuration is lr=0.03, leaves=255).
- Single MAE-objective model per configuration (the grid varies capacity and
  step size, not the ensemble composition).
- Metric: pooled validation WAPE/WPE over folds 2-4 (fold 1 is excluded for
  consistency with the E2 selection pool).

Run:
    ./venv/bin/python scripts/hyperparam_grid.py --n_sp 1500

Outputs (output/hyperparameter_grid/):
    hyperparam_grid_results.csv   one row per configuration
    hyperparam_grid_metadata.json provenance (git SHA, args, runtime)
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_leakfree import (  # noqa: E402
    LGBM_BASE_PARAMS,
    LGBM_OBJECTIVES,
    apply_recovery,
    load_data,
    set_all_seeds,
)
from src.feature_builder import FeatureBuilder, predict_eval_recursive  # noqa: E402
from src.metrics import wape, wpe  # noqa: E402
from scripts.multistep_validation import (  # noqa: E402
    INNER_VAL_DAYS,
    PROTOCOL_NAME,
    _fold_specs,
    _weights,
)

LEARNING_RATES = [0.02, 0.03, 0.05]
NUM_LEAVES = [127, 255, 511]
PAPER_CFG = (0.03, 255)
POOLED_FOLDS = [2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=1500)
    ap.add_argument('--out', default='output/hyperparameter_grid')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[1/3] Load (n_sp={args.n_sp}); recovery/features are per-fold")
    train, _ev = load_data(n_sp=args.n_sp)
    dates = sorted(train['dt'].unique())
    warmup_date, specs = _fold_specs(dates)
    n_clusters = min(50, max(2, train['sp'].nunique() // 10))

    grid = list(itertools.product(LEARNING_RATES, NUM_LEAVES))
    acc = {cfg: {'y': [], 'p': [], 'per_fold': {}, 'runtime': 0.0}
           for cfg in grid}

    pooled_specs = [s for s in specs if s['fold'] in POOLED_FOLDS]
    print(f"[2/3] {len(grid)} configurations x {len(pooled_specs)} folds")
    for s in pooled_specs:
        f, T, val_days = s['fold'], s['train_end'], s['val_days']
        print(f"  fold {f}: origin {pd.Timestamp(T).date()}")
        rec_f = apply_recovery(train[train['dt'] <= T].copy(),
                               'heuristic_imr')
        fb = FeatureBuilder(n_clusters=n_clusters, seed=args.seed).fit(rec_f)
        feat = fb.transform(rec_f)
        fcols = fb.feature_cols(feat)
        val_raw = train[train['dt'].isin(val_days)][
            [c for c in train.columns if c not in ('cens', 'dem_rec')]].copy()

        days_f = [d for d in dates if warmup_date < d <= T]
        inner_val_days = days_f[-INNER_VAL_DAYS:]
        inner_T = days_f[-INNER_VAL_DAYS - 1]
        outer_mask = (feat['dt'] > warmup_date) & (feat['dt'] <= T)
        in_tr_mask = (feat['dt'] > warmup_date) & (feat['dt'] <= inner_T)
        in_vl_mask = feat['dt'].isin(inner_val_days)
        # one design per fold, shared by all 9 configurations
        X_in = feat.loc[in_tr_mask, fcols].values
        y_in = feat.loc[in_tr_mask, 'sale_amount'].values
        w_in = _weights(feat, in_tr_mask)
        X_iv = feat.loc[in_vl_mask, fcols].values
        y_iv = feat.loc[in_vl_mask, 'sale_amount'].values
        X_out = feat.loc[outer_mask, fcols].values
        y_out = feat.loc[outer_mask, 'sale_amount'].values
        w_out = _weights(feat, outer_mask)

        for lr, leaves in grid:
            t0 = time.time()
            params = {**LGBM_BASE_PARAMS, **LGBM_OBJECTIVES['mae'],
                      'learning_rate': lr, 'num_leaves': leaves}
            d_in = lgb.Dataset(X_in, label=y_in, weight=w_in)
            d_iv = lgb.Dataset(X_iv, label=y_iv, reference=d_in)
            inner = lgb.train(params, d_in, num_boost_round=1500,
                              valid_sets=[d_iv],
                              callbacks=[lgb.early_stopping(50),
                                         lgb.log_evaluation(0)])
            nround = max(50, inner.best_iteration)
            mdl = lgb.train(params,
                            lgb.Dataset(X_out, label=y_out, weight=w_out),
                            num_boost_round=nround)
            preds, val_sorted = predict_eval_recursive(
                fb, {'mae': mdl}, fcols, rec_f, val_raw)
            y = val_sorted['sale_amount'].values.astype('float64')
            p = preds['mae']
            a = acc[(lr, leaves)]
            a['y'].append(y)
            a['p'].append(p)
            a['per_fold'][f] = wape(y, p)
            a['runtime'] += time.time() - t0

    print("[3/3] Results")
    rows = []
    for (lr, leaves), a in acc.items():
        y_all = np.concatenate(a['y'])
        p_all = np.concatenate(a['p'])
        rows.append({'learning_rate': lr, 'num_leaves': leaves,
                     'is_paper_config': int((lr, leaves) == PAPER_CFG),
                     'pooled_val_wape': wape(y_all, p_all),
                     'pooled_val_wpe': wpe(y_all, p_all),
                     **{f'wape_fold{f}': a['per_fold'][f]
                        for f in POOLED_FOLDS},
                     'runtime_s': round(a['runtime'], 1)})
    df = pd.DataFrame(rows).sort_values('pooled_val_wape')
    df.to_csv(out_dir / 'hyperparam_grid_results.csv', index=False)
    print(df.to_string(index=False))

    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                         capture_output=True, text=True).stdout.strip()
    meta = {'git_sha': sha, 'args': vars(args),
            'protocol': PROTOCOL_NAME,
            'purpose': ('hyperparameter robustness / rationale only; '
                        'validation-only; the paper configuration is not '
                        're-tuned and the held-out test set is never used'),
            'grid': {'learning_rate': LEARNING_RATES,
                     'num_leaves': NUM_LEAVES},
            'paper_config': {'learning_rate': PAPER_CFG[0],
                             'num_leaves': PAPER_CFG[1]},
            'pooled_folds': POOLED_FOLDS,
            'runtime_s_total': round(time.time() - t_start, 1)}
    (out_dir / 'hyperparam_grid_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"Done in {meta['runtime_s_total']:.0f}s -> {out_dir}/")


if __name__ == '__main__':
    main()
