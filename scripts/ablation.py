#!/usr/bin/env python3
"""
Phase 6: ablation study decomposing (recovery / features / ensemble / policy)
contributions. Six pruned configs:

  A: no recovery + minimal features + single MAE + Naive
  B: heuristic_imr + minimal + single MAE + Naive
  C: heuristic_imr + full features + single MAE + Naive
  D: heuristic_imr + full + ensemble + Naive
  E: heuristic_imr + full + ensemble + Empirical NV
  F: no recovery + full + ensemble + Empirical NV

Runs on the stratified 2000-SP subsample for tractability.
Outputs: output/benchmark/ablation_results.csv
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
from src.recovery import heuristic_imr
from src.inventory import DEFAULT_INV_CFG, compute_policies, residual_quantile_from_cv
from src.metrics import wape, wpe, mae
from pipeline_leakfree import load_data, train_cv, retrain_final, set_all_seeds


MINIMAL_FCOLS = [
    's_l1', 's_l2', 's_l3', 's_l7', 's_l14',
    's_m7', 's_sd7', 'dow', 'wknd', 'month',
    'discount', 'holiday_flag', 'activity_flag',
    'avg_temperature', 'avg_humidity', 'precpt',
]


def apply_recovery(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == 'none':
        df = df.copy()
        df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
        df['dem_rec'] = df['sale_amount'].astype('float32')
        return df
    return heuristic_imr(df, blend=0.5)


def select_features(train_feat: pd.DataFrame, mode: str, fb: FeatureBuilder) -> list:
    if mode == 'minimal':
        return [c for c in MINIMAL_FCOLS if c in train_feat.columns]
    return fb.feature_cols(train_feat)


def train_single_mae(train_feat: pd.DataFrame, fcols: list):
    TARGET = 'sale_amount'
    dates = sorted(train_feat['dt'].unique())
    warmup = dates[27]
    t_mask = train_feat['dt'] > warmup
    val_cutoff = dates[-8]
    v_mask = t_mask & (train_feat['dt'] >= val_cutoff)
    to_mask = t_mask & (train_feat['dt'] < val_cutoff)
    params = {
        'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 255,
        'learning_rate': 0.03, 'feature_fraction': 0.75, 'bagging_fraction': 0.75,
        'bagging_freq': 5, 'min_child_samples': 30, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtr = lgb.Dataset(train_feat.loc[to_mask, fcols].values,
                      label=train_feat.loc[to_mask, TARGET].values)
    dvl = lgb.Dataset(train_feat.loc[v_mask, fcols].values,
                      label=train_feat.loc[v_mask, TARGET].values, reference=dtr)
    mdl = lgb.train(params, dtr, num_boost_round=1500,
                    valid_sets=[dvl],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return mdl


def evaluate_config(name: str, cfg: dict, train_raw: pd.DataFrame,
                    ev_raw: pd.DataFrame) -> dict:
    t0 = time.time()
    tr = apply_recovery(train_raw.copy(), cfg['recovery'])
    fb = FeatureBuilder(n_clusters=min(25, max(2, tr['sp'].nunique() // 10)),
                        seed=42).fit(tr)
    tf = fb.transform(tr)

    fcols = select_features(tf, cfg['features'], fb)

    if cfg['ensemble']:
        cv_m, cv_res, best_iters, _ = train_cv(tf, fcols, cv_folds=2, val_size=7)
        finals = retrain_final(tf, fcols, best_iters)
        preds, ev_sorted = predict_eval_recursive(fb, finals, fcols, tr, ev_raw)
        ens = np.mean([preds['mae'], preds['huber'], preds['q50']], axis=0)
    else:
        # Single MAE: also build CV residuals for Empirical NV if needed
        cv_m, cv_res, _dummy, _ = train_cv(tf, fcols, cv_folds=2, val_size=7)
        mae_mdl = train_single_mae(tf, fcols)
        # For single-model, also need q10/q50/q90 for inventory. Share mae_mdl 3x:
        q50 = mae_mdl
        # Build a tiny q10/q90 pair for inventory compute (reuse MAE if not available)
        params_q = dict(objective='quantile', metric='quantile', num_leaves=255,
                        learning_rate=0.03, feature_fraction=0.75, bagging_fraction=0.75,
                        bagging_freq=5, min_child_samples=30, verbose=-1,
                        n_jobs=-1, seed=42)
        dates = sorted(tf['dt'].unique()); warmup = dates[27]
        mask = tf['dt'] > warmup
        dtr = lgb.Dataset(tf.loc[mask, fcols].values, label=tf.loc[mask, 'sale_amount'].values)
        params_q['alpha'] = 0.1
        q10_mdl = lgb.train(params_q, dtr, num_boost_round=300)
        params_q['alpha'] = 0.9
        q90_mdl = lgb.train(params_q, dtr, num_boost_round=300)
        preds, ev_sorted = predict_eval_recursive(
            fb,
            {'mae': mae_mdl, 'huber': mae_mdl, 'q50': q50, 'q10': q10_mdl, 'q90': q90_mdl},
            fcols, tr, ev_raw,
        )
        ens = preds['mae']

    y = ev_sorted['sale_amount'].values
    sp = ev_sorted['sp'].values
    wape_v = wape(y, ens)
    wpe_v = wpe(y, ens)

    cr = DEFAULT_INV_CFG['p'] / (DEFAULT_INV_CFG['p'] + DEFAULT_INV_CFG['w'])
    if cfg['policy'] == 'empirical_nv':
        sp_rq = residual_quantile_from_cv(cv_res, cr)
        preds_for_inv = {'ensemble': ens, 'q10': preds['q10'],
                         'q50': preds['q50'], 'q90': preds['q90']}
        rdf, _ = compute_policies(preds_for_inv, sp, tr, sp_rq,
                                   DEFAULT_INV_CFG, actuals_for_eval=y)
        row = rdf[rdf['Policy'] == 'Empirical Newsvendor'].iloc[0]
        profit = float(row['Avg Profit'])
        fill = float(row['Fill Rate'])
    else:
        # Naive Q = forecast
        from src.inventory import eval_policy
        Q = np.clip(ens, 0, None)
        row = eval_policy(Q, y, DEFAULT_INV_CFG, 'Naive')
        profit = float(row['Avg Profit'])
        fill = float(row['Fill Rate'])

    return {
        'config': name,
        'components': f"{cfg['recovery']}/{cfg['features']}/"
                      f"{'ensemble' if cfg['ensemble'] else 'single'}/{cfg['policy']}",
        'wape': wape_v,
        'wpe': wpe_v,
        'mae': mae(y, ens),
        'profit': profit,
        'fill_rate': fill,
        'time_s': time.time() - t0,
    }


ABLATION_CONFIGS = {
    'A': {'recovery': 'none',          'features': 'minimal', 'ensemble': False, 'policy': 'naive'},
    'B': {'recovery': 'heuristic_imr', 'features': 'minimal', 'ensemble': False, 'policy': 'naive'},
    'C': {'recovery': 'heuristic_imr', 'features': 'full',    'ensemble': False, 'policy': 'naive'},
    'D': {'recovery': 'heuristic_imr', 'features': 'full',    'ensemble': True,  'policy': 'naive'},
    'E': {'recovery': 'heuristic_imr', 'features': 'full',    'ensemble': True,  'policy': 'empirical_nv'},
    'F': {'recovery': 'none',          'features': 'full',    'ensemble': True,  'policy': 'empirical_nv'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=2000)
    ap.add_argument('--out', default='output/benchmark')
    ap.add_argument('--configs', default='A,B,C,D,E,F')
    args = ap.parse_args()

    set_all_seeds(42)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_sp}-SP subsample...")
    train, ev = load_data(n_sp=args.n_sp)
    print(f"  Train: {train.shape}, Eval: {ev.shape}")

    to_run = args.configs.split(',')
    rows = []
    for name in to_run:
        if name not in ABLATION_CONFIGS:
            print(f"  [skip] {name} not defined")
            continue
        cfg = ABLATION_CONFIGS[name]
        print(f"\n=== Config {name}: {cfg} ===")
        r = evaluate_config(name, cfg, train, ev)
        rows.append(r)
        print(f"  wape={r['wape']:.2f}  wpe={r['wpe']:+.2f}  profit={r['profit']:.4f}  "
              f"fill={r['fill_rate']:.3f}  t={r['time_s']:.1f}s")

    df = pd.DataFrame(rows)
    out_path = out_dir / 'ablation_results.csv'
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(df[['config', 'components', 'wape', 'wpe', 'profit', 'fill_rate']].to_string(index=False))


if __name__ == '__main__':
    main()
