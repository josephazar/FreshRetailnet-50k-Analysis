#!/usr/bin/env python3
"""
Leak-free 5-model benchmark (Naive, SSA, LightGBM, XGBoost, CatBoost).

The benchmark uses src.feature_builder.FeatureBuilder with train-only
aggregate lookups. All learned models receive the same 15-feature matrix.

Outputs: output/benchmark/benchmark_forecast_comparison.csv, benchmark_per_sp_predictions.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from src.feature_builder import FeatureBuilder, predict_eval_recursive
from src.metrics import wape, wpe, mae, rmse, corr, smape
from pipeline_leakfree import load_data, set_all_seeds


# Common 15-feature matrix used by every learned benchmark model.
MINIMAL_FCOLS = [
    's_l1', 's_l2', 's_l7', 's_l14',
    's_m7', 's_sd7',
    'dow', 'wknd', 'month',
    'discount', 'holiday_flag', 'activity_flag',
    'avg_temperature', 'avg_humidity', 'precpt',
]


def train_model(train_feat, fcols, model_name: str):
    """Train one of the 3 GBDT baselines with identical features, seeds, early stop."""
    TARGET = 'sale_amount'
    dates = sorted(train_feat['dt'].unique())
    warmup = dates[27]
    val_cutoff = dates[-8]
    t_mask = train_feat['dt'] > warmup
    v_mask = t_mask & (train_feat['dt'] >= val_cutoff)
    to_mask = t_mask & (train_feat['dt'] < val_cutoff)
    Xtr = train_feat.loc[to_mask, fcols].values
    ytr = train_feat.loc[to_mask, TARGET].values
    Xvl = train_feat.loc[v_mask, fcols].values
    yvl = train_feat.loc[v_mask, TARGET].values

    if model_name == 'lgbm':
        params = {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 127,
                  'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                  'bagging_freq': 5, 'min_child_samples': 30, 'verbose': -1,
                  'n_jobs': -1, 'seed': 42}
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_vl = lgb.Dataset(Xvl, label=yvl, reference=d_tr)
        mdl = lgb.train(params, d_tr, num_boost_round=1500, valid_sets=[d_vl],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        return mdl
    elif model_name == 'xgb':
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dvl = xgb.DMatrix(Xvl, label=yvl)
        params = {'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                  'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'min_child_weight': 30,
                  'tree_method': 'hist', 'seed': 42, 'verbosity': 0}
        mdl = xgb.train(params, dtr, num_boost_round=1500,
                        evals=[(dvl, 'val')], early_stopping_rounds=50,
                        verbose_eval=False)
        return mdl
    elif model_name == 'catboost':
        mdl = CatBoostRegressor(
            loss_function='MAE', eval_metric='MAE', iterations=1500,
            learning_rate=0.05, depth=8, l2_leaf_reg=3,
            random_seed=42, verbose=0,
        )
        mdl.fit(Xtr, ytr, eval_set=(Xvl, yvl), early_stopping_rounds=50, verbose=0)
        return mdl
    else:
        raise ValueError(model_name)


def predict_model(mdl, model_name: str, Xeval):
    if model_name == 'lgbm':
        return np.clip(mdl.predict(Xeval), 0, None)
    elif model_name == 'xgb':
        return np.clip(mdl.predict(xgb.DMatrix(Xeval)), 0, None)
    elif model_name == 'catboost':
        return np.clip(mdl.predict(Xeval), 0, None)


def _predict_recursive_single(fb, mdl, model_name: str, fcols, train_df, eval_df):
    """Recursive eval prediction for a single (non-ensemble) model."""
    hist = train_df[train_df['dt'] >= train_df['dt'].max() - pd.Timedelta(days=27)].copy()
    hist['cens'] = (hist['stock_hour6_22_cnt'] > 0).astype('int8')
    if 'dem_rec' not in hist.columns:
        hist['dem_rec'] = hist['sale_amount'].astype('float32')

    ev = eval_df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    ev['cens'] = (ev['stock_hour6_22_cnt'] > 0).astype('int8')
    ev['so_frac'] = (ev['stock_hour6_22_cnt'] / 16).astype('float32')
    if 'dem_rec' not in ev.columns:
        ev['dem_rec'] = 0.0
    ev_work = ev.copy()
    ev_work['sale_amount'] = 0.0
    ev_work['dem_rec'] = 0.0
    eval_dates = sorted(ev_work['dt'].unique())
    preds_out = np.zeros(len(ev_work), dtype='float32')

    for day in eval_dates:
        so_far = ev_work['dt'] <= day
        combined = pd.concat([hist, ev_work[so_far]], ignore_index=True)
        feat = fb.transform(combined)
        today = feat['dt'] == day
        X = feat.loc[today, fcols].fillna(0).values
        p = predict_model(mdl, model_name, X)
        # Write back
        rows = ev_work[ev_work['dt'] == day].index
        sp_today = feat.loc[today, 'sp'].values
        mapper = pd.Series(p, index=sp_today)
        ev_work.loc[rows, 'sale_amount'] = ev_work.loc[rows, 'sp'].map(mapper).values
        ev_work.loc[rows, 'dem_rec'] = ev_work.loc[rows, 'sale_amount']
        # Store in preds_out
        sp_in_ev = ev.loc[ev['dt'] == day, 'sp'].values
        pm = pd.Series(p, index=sp_today)
        preds_out[(ev['dt'] == day).values] = ev.loc[ev['dt'] == day, 'sp'].map(pm).values

    return preds_out, ev


def naive_seasonal(train_df, eval_df):
    """Naive: sale = sale on same DOW from last week (lag 7)."""
    ev = eval_df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    last_train = train_df[train_df['dt'] >= train_df['dt'].max() - pd.Timedelta(days=13)]
    last_train = last_train[last_train['dt'] >= last_train['dt'].max() - pd.Timedelta(days=6)]
    last_by_sp_dow = last_train.assign(
        dow=last_train['dt'].dt.dayofweek
    ).groupby(['sp', 'dow'])['sale_amount'].mean().to_dict()
    ev_dow = ev['dt'].dt.dayofweek.values
    keys = list(zip(ev['sp'].values, ev_dow))
    p = np.array([last_by_sp_dow.get(k, 0.0) for k in keys], dtype='float32')
    return np.clip(p, 0, None), ev


def ssa_baseline(train_df, eval_df):
    """Similar Scenario Average: mean of last 14 same-DOW training observations per SP."""
    ev = eval_df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    tr = train_df.copy()
    tr['dow'] = tr['dt'].dt.dayofweek
    last14 = tr[tr['dt'] >= tr['dt'].max() - pd.Timedelta(days=27)]
    mean_by = last14.groupby(['sp', 'dow'])['sale_amount'].mean().to_dict()
    ev_dow = ev['dt'].dt.dayofweek.values
    keys = list(zip(ev['sp'].values, ev_dow))
    p = np.array([mean_by.get(k, 0.0) for k in keys], dtype='float32')
    return np.clip(p, 0, None), ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=3000)
    ap.add_argument('--out', default='output/benchmark')
    args = ap.parse_args()

    set_all_seeds(42)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_sp}-SP subsample...")
    train, ev = load_data(n_sp=args.n_sp)
    train['cens'] = (train['stock_hour6_22_cnt'] > 0).astype('int8')
    train['dem_rec'] = train['sale_amount'].astype('float32')
    ev['cens'] = (ev['stock_hour6_22_cnt'] > 0).astype('int8')
    print(f"  Train: {train.shape}, Eval: {ev.shape}")

    # Fit FeatureBuilder on training only (this is THE leak-free piece)
    fb = FeatureBuilder(n_clusters=min(25, max(2, train['sp'].nunique() // 10)),
                        seed=42).fit(train)
    train_feat = fb.transform(train)
    fc_min = [c for c in MINIMAL_FCOLS if c in train_feat.columns]
    print(f"  Minimal features: {len(fc_min)}")

    ev_sorted = ev.sort_values(['sp', 'dt']).reset_index(drop=True)
    y_eval = ev_sorted['sale_amount'].values
    sp_eval = ev_sorted['sp'].values
    dt_eval = ev_sorted['dt'].values

    results = []

    # Naive
    t = time.time()
    p_nv, _ = naive_seasonal(train, ev_sorted)
    results.append({'Model': 'Naive Seasonal', 'time_s': time.time() - t,
                    'MAE': mae(y_eval, p_nv), 'RMSE': rmse(y_eval, p_nv),
                    'SMAPE': smape(y_eval, p_nv), 'Corr': corr(y_eval, p_nv),
                    'WAPE': wape(y_eval, p_nv), 'WPE': wpe(y_eval, p_nv),
                    'preds': p_nv})

    # SSA
    t = time.time()
    p_ssa, _ = ssa_baseline(train, ev_sorted)
    results.append({'Model': 'SSA', 'time_s': time.time() - t,
                    'MAE': mae(y_eval, p_ssa), 'RMSE': rmse(y_eval, p_ssa),
                    'SMAPE': smape(y_eval, p_ssa), 'Corr': corr(y_eval, p_ssa),
                    'WAPE': wape(y_eval, p_ssa), 'WPE': wpe(y_eval, p_ssa),
                    'preds': p_ssa})

    # LightGBM / XGBoost / CatBoost (ALL with identical minimal features)
    for name in ['lgbm', 'xgb', 'catboost']:
        t = time.time()
        print(f"  Training {name}...")
        mdl = train_model(train_feat, fc_min, name)
        p, _ = _predict_recursive_single(fb, mdl, name, fc_min, train, ev_sorted)
        pretty = {'lgbm': 'LightGBM-Minimal', 'xgb': 'XGBoost-Minimal',
                   'catboost': 'CatBoost-Minimal'}[name]
        results.append({'Model': pretty, 'time_s': time.time() - t,
                        'MAE': mae(y_eval, p), 'RMSE': rmse(y_eval, p),
                        'SMAPE': smape(y_eval, p), 'Corr': corr(y_eval, p),
                        'WAPE': wape(y_eval, p), 'WPE': wpe(y_eval, p),
                        'preds': p})

    # Emit comparison CSV
    rows_for_csv = [{k: v for k, v in r.items() if k != 'preds'} for r in results]
    df = pd.DataFrame(rows_for_csv)
    df.to_csv(out_dir / 'benchmark_forecast_comparison.csv', index=False)
    print("\n" + df[['Model', 'MAE', 'WAPE', 'WPE', 'Corr', 'time_s']].to_string(index=False))

    # Per-SP predictions for significance tests
    per_sp = pd.DataFrame({'sp': sp_eval, 'dt': dt_eval, 'actual': y_eval})
    for r in results:
        col = 'pred_' + r['Model'].lower().replace(' ', '_').replace('-', '_')
        per_sp[col] = r['preds']
    per_sp.to_csv(out_dir / 'benchmark_per_sp_predictions.csv', index=False)
    print(f"\nWrote per-SP predictions to {out_dir / 'benchmark_per_sp_predictions.csv'}")


if __name__ == '__main__':
    main()
