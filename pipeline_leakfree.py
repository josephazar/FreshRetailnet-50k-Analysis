#!/usr/bin/env python3
"""
FreshRetailNet-50K: leakage-free demand forecasting and inventory pipeline.

The pipeline provides:
  - Train-only aggregate features via src.feature_builder.FeatureBuilder
  - Recursive eval prediction (no cross-eval leak)
  - Out-of-sample inventory policy selection on validation fold
  - Honest recovery method naming: heuristic_imr OR tobit_mle
  - Full-train retrain of final models after CV

Outputs land in `output/pipeline/` by default.

Usage:
    python pipeline_leakfree.py --n_sp 0 --recovery tobit_mle
    python pipeline_leakfree.py --n_sp 2000 --recovery heuristic_imr
    python pipeline_leakfree.py --n_sp 0 --recovery heuristic_imr --calibration cv_scale
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('PYTHONHASHSEED', '42')

import argparse
import gc
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from src.feature_builder import FeatureBuilder, predict_eval_recursive
from src.recovery import heuristic_imr, tobit_mle
from src.calibration import (
    apply_calibrator_to_prediction_dict,
    calibrated_cv_residuals,
    learn_cv_scale_calibrator,
)
from src.inventory import (
    DEFAULT_INV_CFG,
    compute_policies,
    profit_optimal_cr,
    residual_quantile_from_cv,
    select_policy_on_validation,
)
from src.metrics import wape, wpe, mae, rmse, smape, corr


DATA_DIR = SCRIPT_DIR.parent / 'data' / 'freshretailnet' / 'raw' / 'data'
TRAIN_PATH = DATA_DIR / 'train.parquet'
EVAL_PATH = DATA_DIR / 'eval.parquet'

# Single source of truth for the LightGBM configuration; the manuscript
# (research/paper.tex, Model architecture) cites these exact values and
# tests/test_config_constants.py pins them.
LGBM_BASE_PARAMS = {
    'boosting_type': 'gbdt', 'num_leaves': 255, 'learning_rate': 0.03,
    'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 5,
    'min_child_samples': 30, 'lambda_l1': 0.1, 'lambda_l2': 1.0,
    'verbose': -1, 'n_jobs': -1, 'seed': 42,
}
LGBM_OBJECTIVES = {
    'mae': {'objective': 'regression_l1', 'metric': 'mae'},
    'huber': {'objective': 'huber', 'alpha': 1.0, 'metric': 'mae'},
    'q10': {'objective': 'quantile', 'alpha': 0.1, 'metric': 'quantile'},
    'q50': {'objective': 'quantile', 'alpha': 0.5, 'metric': 'quantile'},
    'q90': {'objective': 'quantile', 'alpha': 0.9, 'metric': 'quantile'},
}


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save_run_metadata(out_dir: Path, script_name: str, config: dict, runtime_s: float):
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=SCRIPT_DIR, text=True).strip()
    except Exception:
        sha = 'unknown'
    meta = {
        'script': script_name,
        'git_sha': sha,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'lightgbm': lgb.__version__,
        'config': config,
        'inventory_cfg': DEFAULT_INV_CFG,
        'runtime_s': runtime_s,
        'train_md5': _md5(TRAIN_PATH) if TRAIN_PATH.exists() else 'missing',
        'eval_md5': _md5(EVAL_PATH) if EVAL_PATH.exists() else 'missing',
    }
    (out_dir / 'run_metadata.json').write_text(
        json.dumps(meta, indent=2)
    )


def validate_unique_sp_dt(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Duplicate (sp, dt) rows would give predict_eval_recursive's per-day
    writeback (pd.Series indexed by sp) a non-unique index and crash mid-run;
    reject them at load time instead."""
    dup = int(df.duplicated(['sp', 'dt']).sum())
    if dup:
        print(f"  WARNING: {name} contains {dup} duplicate (sp, dt) rows; "
              f"keeping first occurrence of each")
        df = df.drop_duplicates(['sp', 'dt'], keep='first').reset_index(drop=True)
    return df


def load_data(n_sp: int = 0):
    cols = ['city_id', 'store_id', 'management_group_id', 'first_category_id',
            'second_category_id', 'third_category_id', 'product_id', 'dt',
            'sale_amount', 'stock_hour6_22_cnt', 'discount', 'holiday_flag',
            'activity_flag', 'precpt', 'avg_temperature', 'avg_humidity',
            'avg_wind_level']
    train = pd.read_parquet(TRAIN_PATH, columns=cols)
    train['sp'] = train['store_id'] * 10000 + train['product_id']
    train['dt'] = pd.to_datetime(train['dt'])
    ev = pd.read_parquet(EVAL_PATH, columns=cols)
    ev['sp'] = ev['store_id'] * 10000 + ev['product_id']
    ev['dt'] = pd.to_datetime(ev['dt'])
    if n_sp > 0 and n_sp < train['sp'].nunique():
        rng = np.random.default_rng(42)
        so = train.groupby('sp')['stock_hour6_22_cnt'].apply(lambda x: (x > 0).mean())
        so = so.reset_index()
        so.columns = ['sp', 'sor']
        so['bin'] = pd.qcut(so['sor'], q=5, labels=False, duplicates='drop')
        sampled = so.groupby('bin').apply(
            lambda x: x.sample(min(len(x), n_sp // 5), random_state=42)
        ).reset_index(drop=True)['sp'].values
        if len(sampled) < n_sp:
            extra = rng.choice(
                list(set(so['sp']) - set(sampled)),
                n_sp - len(sampled),
                replace=False,
            )
            sampled = np.concatenate([sampled, extra])
        sampled = set(sampled[:n_sp])
        train = train[train['sp'].isin(sampled)].copy()
        ev = ev[ev['sp'].isin(sampled)].copy()
    for c in train.select_dtypes('int64').columns:
        if c != 'sp':
            train[c] = train[c].astype('int32')
            ev[c] = ev[c].astype('int32')
    for c in train.select_dtypes('float64').columns:
        train[c] = train[c].astype('float32')
        ev[c] = ev[c].astype('float32')
    train = validate_unique_sp_dt(train, 'train')
    ev = validate_unique_sp_dt(ev, 'eval')
    return train, ev


def apply_recovery(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == 'none':
        df = df.copy()
        df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
        df['dem_rec'] = df['sale_amount'].astype('float32')
        return df
    elif method == 'heuristic_imr':
        return heuristic_imr(df, blend=0.5)
    elif method == 'tobit_mle':
        return tobit_mle(df)
    else:
        raise ValueError(f"unknown recovery method: {method}")


def train_cv(train_feat: pd.DataFrame, fcols: list, cv_folds: int = 4, val_size: int = 7,
             objectives: dict | None = None, target_col: str = 'sale_amount'):
    """Expanding-window temporal CV.

    objectives: subset of LGBM_OBJECTIVES to train (default: all five).
    target_col: training target column ('sale_amount' for the benchmark;
        'dem_rec' for censored-target experiments). Validation actuals are
        always observed sales semantics of target_col itself.

    Returns:
        cv_metrics: list of dicts per fold
        cv_residuals: list of (sp_keys, residuals) per fold for Empirical NV
        best_iters: dict of per-model per-fold best num_boost_round
        cv_val_frames: list of dicts per fold with keys
            {sp, dt, y, preds: {<objective names>, ensemble}}
            used by the out-of-sample inventory policy selector.
    """
    TARGET = target_col
    dates = sorted(train_feat['dt'].unique())
    warmup_date = dates[27]
    available = [d for d in dates if d > warmup_date]
    n_avail = len(available)

    fold_specs = []
    for i in range(cv_folds):
        val_end_idx = n_avail - (cv_folds - 1 - i) * val_size
        val_start_idx = val_end_idx - val_size
        fold_specs.append({
            'train_end': available[val_start_idx - 1],
            'val_start': available[val_start_idx],
            'val_end': available[min(val_end_idx - 1, n_avail - 1)],
        })

    cv_metrics = []
    cv_residuals = []  # list[(sp_keys, residuals)]
    cv_val_frames = []  # list[dict] with per-fold val preds + actuals
    objs = objectives if objectives is not None else LGBM_OBJECTIVES
    best_iters = {name: [] for name in objs}

    base = LGBM_BASE_PARAMS

    for fi, fs in enumerate(fold_specs):
        tr_mask = (train_feat['dt'] > warmup_date) & (train_feat['dt'] <= fs['train_end'])
        vl_mask = (train_feat['dt'] >= fs['val_start']) & (train_feat['dt'] <= fs['val_end'])
        Xtr = train_feat.loc[tr_mask, fcols].values
        ytr = train_feat.loc[tr_mask, TARGET].values
        Xvl = train_feat.loc[vl_mask, fcols].values
        yvl = train_feat.loc[vl_mask, TARGET].values
        sp_vl = train_feat.loc[vl_mask, 'sp'].values
        dt_vl = train_feat.loc[vl_mask, 'dt'].values
        wtr = np.ones(len(ytr), dtype='float32')
        cens_mask = train_feat.loc[tr_mask, 'cens'].values == 1
        wtr[cens_mask] = 0.5

        print(f"  Fold {fi+1}/{cv_folds}: train up to {fs['train_end'].date()}, "
              f"val {fs['val_start'].date()}..{fs['val_end'].date()}; "
              f"Ntr={len(ytr):,} Nvl={len(yvl):,}")
        fold_models = {}
        val_preds = {}
        for name, obj in objs.items():
            params = {**base, **obj}
            d_tr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
            d_vl = lgb.Dataset(Xvl, label=yvl, reference=d_tr)
            mdl = lgb.train(
                params, d_tr,
                num_boost_round=1500,
                valid_sets=[d_vl],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            fold_models[name] = mdl
            best_iters[name].append(mdl.best_iteration)
            val_preds[name] = np.clip(mdl.predict(Xvl), 0, None).astype('float32')
        # Ensemble over whichever point-forecast members were trained
        ens_members = [n for n in ('mae', 'huber', 'q50') if n in val_preds]
        if ens_members:
            val_preds['ensemble'] = np.mean(
                [val_preds[n] for n in ens_members], axis=0
            ).astype('float32')
        resid_key = 'mae' if 'mae' in val_preds else next(iter(val_preds))
        residuals = yvl - val_preds[resid_key]
        cv_residuals.append((sp_vl, residuals))
        cv_val_frames.append({
            'fold': fi + 1,
            'sp': sp_vl,
            'dt': dt_vl,
            'y': yvl,
            'preds': val_preds,
        })
        fold_mae = mean_absolute_error(yvl, val_preds[resid_key])
        cv_metrics.append({'fold': fi + 1, 'mae': float(fold_mae),
                           'n_train': int(len(ytr)), 'n_val': int(len(yvl))})

    return cv_metrics, cv_residuals, best_iters, cv_val_frames


def retrain_final(train_feat: pd.DataFrame, fcols: list, best_iters: dict,
                  warmup_days: int = 28, importance_out: Path | None = None,
                  objectives: dict | None = None, target_col: str = 'sale_amount'):
    """Retrain on the full training window at mean-of-best-folds rounds.

    importance_out: optional output directory; when set, dumps per-model
    gain importances to feature_importance.csv and saves boosters under
    models/ (used by scripts/export_feature_importance.py).
    objectives / target_col: see train_cv.
    """
    TARGET = target_col
    objs = objectives if objectives is not None else LGBM_OBJECTIVES
    dates = sorted(train_feat['dt'].unique())
    warmup_date = dates[warmup_days - 1]
    mask = train_feat['dt'] > warmup_date
    Xtr = train_feat.loc[mask, fcols].values
    ytr = train_feat.loc[mask, TARGET].values
    wtr = np.ones(len(ytr), dtype='float32')
    cens_mask = train_feat.loc[mask, 'cens'].values == 1
    wtr[cens_mask] = 0.5
    final = {}
    importance_rows = []
    for name, obj in objs.items():
        params = {**LGBM_BASE_PARAMS, **obj}
        nround = max(50, int(np.mean(best_iters[name])))
        d = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        mdl = lgb.train(params, d, num_boost_round=nround)
        final[name] = mdl
        print(f"  Final {name}: num_boost_round={nround}")
        if importance_out is not None:
            gains = mdl.feature_importance(importance_type='gain')
            importance_rows.extend(
                {'model': name, 'feature': f, 'gain': float(g)}
                for f, g in zip(fcols, gains)
            )
    if importance_out is not None:
        importance_out = Path(importance_out)
        importance_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(importance_rows).to_csv(
            importance_out / 'feature_importance.csv', index=False
        )
        mdl_dir = importance_out / 'models'
        mdl_dir.mkdir(exist_ok=True)
        for name, mdl in final.items():
            mdl.save_model(str(mdl_dir / f'{name}.txt'))
        print(f"  Wrote feature_importance.csv + models/ to {importance_out}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=0,
                    help='0=all 50K, else stratified subsample size')
    ap.add_argument('--recovery', choices=['none', 'heuristic_imr', 'tobit_mle'],
                    default='heuristic_imr')
    ap.add_argument('--out', default='output/pipeline')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--calibration', choices=['none', 'cv_scale'], default='none',
                    help='optional leakage-free post-forecast calibration from CV folds')
    ap.add_argument('--calibration_sp_shrink', type=float, default=28.0,
                    help='shrinkage strength for per-SP CV scale calibration')
    ap.add_argument('--calibration_horizon_shrink', type=float, default=5000.0,
                    help='shrinkage strength for horizon CV scale calibration')
    args = ap.parse_args()

    set_all_seeds(args.seed)
    t0 = time.time()

    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LEAK-FREE PIPELINE "
          f"(recovery={args.recovery}, n_sp={args.n_sp}, "
          f"calibration={args.calibration})")
    print("=" * 80)

    # ---- Load ----
    print("\n[1/7] Data load")
    train, ev = load_data(n_sp=args.n_sp)
    print(f"  Train: {train.shape}, Eval: {ev.shape}, SPs: {train['sp'].nunique()}")

    # ---- Recovery (training only; eval uses observed) ----
    print(f"\n[2/7] Recovery ({args.recovery})")
    train = apply_recovery(train, args.recovery)
    print(f"  train mean_sale={train['sale_amount'].mean():.4f} "
          f"mean_recovered={train['dem_rec'].mean():.4f}")

    # ---- FeatureBuilder fit on training only ----
    print("\n[3/7] FeatureBuilder.fit (training only)")
    fb = FeatureBuilder(n_clusters=min(50, max(2, train['sp'].nunique() // 10)),
                         seed=args.seed).fit(train)
    train_feat = fb.transform(train)
    fcols = fb.feature_cols(train_feat)
    print(f"  Features: {len(fcols)}")

    # ---- 4-fold temporal CV ----
    print("\n[4/7] Temporal CV (4-fold)")
    cv_metrics, cv_residuals, best_iters, cv_val_frames = train_cv(train_feat, fcols)
    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(out_dir / 'cv_results.csv', index=False)
    print(f"  CV mean MAE: {cv_df['mae'].mean():.4f} +/- {cv_df['mae'].std():.4f}")

    calibrator = None
    selection_calibrator = None
    cv_residuals_for_test = cv_residuals
    cv_residuals_for_selection = cv_residuals
    if args.calibration == 'cv_scale':
        print("\n[4b/7] CV scale calibration")
        calibrator = learn_cv_scale_calibrator(
            cv_val_frames,
            pred_key='ensemble',
            sp_shrink=args.calibration_sp_shrink,
            horizon_shrink=args.calibration_horizon_shrink,
        )
        selection_frames = cv_val_frames[:-1] if len(cv_val_frames) > 1 else cv_val_frames
        selection_calibrator = learn_cv_scale_calibrator(
            selection_frames,
            pred_key='ensemble',
            sp_shrink=args.calibration_sp_shrink,
            horizon_shrink=args.calibration_horizon_shrink,
        )
        cv_residuals_for_test = calibrated_cv_residuals(
            cv_val_frames, calibrator, pred_key='mae'
        )
        cv_residuals_for_selection = calibrated_cv_residuals(
            selection_frames, selection_calibrator, pred_key='mae'
        )
        (out_dir / 'cv_scale_calibration.json').write_text(
            json.dumps(calibrator, indent=2)
        )
        cm = calibrator['cv_metrics']
        print(f"  CV ensemble WAPE raw={cm['raw_wape']:.2f}% -> "
              f"calibrated={cm['calibrated_wape']:.2f}% "
              f"(WPE {cm['raw_wpe']:+.2f}% -> {cm['calibrated_wpe']:+.2f}%)")
        print("  Wrote cv_scale_calibration.json")

    # ---- Full-train retrain at validation-derived iteration counts ----
    print("\n[5/7] Full-train retrain")
    final_models = retrain_final(train_feat, fcols, best_iters)

    # ---- Recursive eval prediction (leak-free) ----
    print("\n[6/7] Recursive eval prediction")
    preds_dict, eval_sorted = predict_eval_recursive(
        fb, final_models, fcols, train, ev
    )
    raw_preds_dict = {name: values.copy() for name, values in preds_dict.items()}
    raw_preds_dict['ensemble'] = np.mean(
        [raw_preds_dict['mae'], raw_preds_dict['huber'], raw_preds_dict['q50']],
        axis=0,
    ).astype('float32')
    y_eval = eval_sorted['sale_amount'].values
    sp_eval = eval_sorted['sp'].values
    dt_eval = eval_sorted['dt'].values
    if calibrator is not None:
        preds_dict = apply_calibrator_to_prediction_dict(
            raw_preds_dict, sp_eval, dt_eval, calibrator
        )
        ensemble = preds_dict['ensemble']
    else:
        preds_dict = raw_preds_dict
        ensemble = preds_dict['ensemble']

    # Evaluate on eval
    metrics = {
        'ensemble': {
            'MAE': mae(y_eval, ensemble),
            'RMSE': rmse(y_eval, ensemble),
            'SMAPE': smape(y_eval, ensemble),
            'Corr': corr(y_eval, ensemble),
            'WAPE': wape(y_eval, ensemble),
            'WPE': wpe(y_eval, ensemble),
        }
    }
    for name in ['mae', 'huber', 'q50']:
        p = preds_dict[name]
        metrics[name] = {
            'MAE': mae(y_eval, p), 'RMSE': rmse(y_eval, p),
            'SMAPE': smape(y_eval, p), 'Corr': corr(y_eval, p),
            'WAPE': wape(y_eval, p), 'WPE': wpe(y_eval, p),
        }
    if calibrator is not None:
        for name in ['mae', 'huber', 'q50', 'ensemble']:
            p = raw_preds_dict[name]
            metrics[f'{name}_raw'] = {
                'MAE': mae(y_eval, p), 'RMSE': rmse(y_eval, p),
                'SMAPE': smape(y_eval, p), 'Corr': corr(y_eval, p),
                'WAPE': wape(y_eval, p), 'WPE': wpe(y_eval, p),
            }
    print(f"  Ensemble: MAE={metrics['ensemble']['MAE']:.4f}, "
          f"WAPE={metrics['ensemble']['WAPE']:.2f}%, "
          f"WPE={metrics['ensemble']['WPE']:+.2f}%")
    (out_dir / 'forecast_metrics.json').write_text(json.dumps(metrics, indent=2))

    # Save predictions for downstream statistical tests
    pred_df = pd.DataFrame({
        'sp': sp_eval, 'dt': eval_sorted['dt'].values, 'y_true': y_eval,
        'pred_mae': preds_dict['mae'], 'pred_huber': preds_dict['huber'],
        'pred_q50': preds_dict['q50'], 'pred_q10': preds_dict['q10'],
        'pred_q90': preds_dict['q90'], 'pred_ensemble': ensemble,
    })
    if calibrator is not None:
        for name in ['mae', 'huber', 'q10', 'q50', 'q90', 'ensemble']:
            pred_df[f'pred_{name}_raw'] = raw_preds_dict[name]
    pred_df.to_csv(out_dir / 'predictions.csv', index=False)

    # ---- OOS inventory: get residual quantile from CV folds ----
    print("\n[7/7] Inventory optimization (OOS)")
    cr_nv = profit_optimal_cr(DEFAULT_INV_CFG)
    sp_res_q = residual_quantile_from_cv(cv_residuals_for_test, cr=cr_nv)
    selection_sp_res_q = residual_quantile_from_cv(cv_residuals_for_selection, cr=cr_nv)
    print(f"  residual quantile learned from {len(sp_res_q)} SPs "
          f"from {len(cv_residuals_for_test)} folds")

    # Policy PRE-SELECTION on the last validation fold (not test set)
    # Use the last CV fold's predictions + actuals; this is the proper OOS signal
    # for choosing which of the candidate policies to report on test.
    last = cv_val_frames[-1]
    val_inv_preds_raw = {
        'ensemble': last['preds']['ensemble'],
        'q10': last['preds']['q10'],
        'q50': last['preds']['q50'],
        'q90': last['preds']['q90'],
    }
    if selection_calibrator is not None:
        val_inv_preds = apply_calibrator_to_prediction_dict(
            val_inv_preds_raw, last['sp'], last['dt'], selection_calibrator
        )
    else:
        val_inv_preds = val_inv_preds_raw
    print(f"  Pre-selecting policy on CV fold {last['fold']} "
          f"(N_val={len(last['y'])})...")
    preselected = select_policy_on_validation(
        val_preds=val_inv_preds,
        val_sp_keys=last['sp'],
        val_actuals=last['y'],
        train_df=train,
        sp_residual_quantile=selection_sp_res_q,
        cfg=DEFAULT_INV_CFG,
        criterion='Avg Profit',
        fill_rate_floor=0.90,
    )
    print(f"  Pre-selected policy (FR>=0.90, max profit): {preselected}")

    # Now compute ALL policies on TEST for transparency / reporting,
    # but report the preselected one as the headline.
    inv_preds = {
        'ensemble': ensemble,
        'q10': preds_dict['q10'],
        'q50': preds_dict['q50'],
        'q90': preds_dict['q90'],
    }
    rdf, _ = compute_policies(
        preds=inv_preds, sp_keys=sp_eval, train_df=train,
        sp_residual_quantile=sp_res_q, cfg=DEFAULT_INV_CFG,
        actuals_for_eval=y_eval,
    )
    # Mark preselected row for clarity
    rdf['preselected'] = (rdf['Policy'] == preselected).astype(int)
    rdf = rdf.sort_values('Avg Profit', ascending=False)
    rdf.to_csv(out_dir / 'inventory_results.csv', index=False)
    print(f"\n  Top-3 policies by TEST profit (transparency):")
    print(rdf[['Policy', 'Avg Profit', 'Fill Rate', 'Waste%', 'preselected']]
          .head(3).to_string(index=False))

    # Extract preselected row's TEST metrics
    sel_row = rdf[rdf['Policy'] == preselected]
    sel_metrics = sel_row.iloc[0].to_dict() if len(sel_row) else {}

    # Validation-fold metrics for the preselected policy (the basis for selection)
    val_rdf, _ = compute_policies(
        preds=val_inv_preds, sp_keys=last['sp'], train_df=train,
        sp_residual_quantile=selection_sp_res_q, cfg=DEFAULT_INV_CFG,
        actuals_for_eval=last['y'],
    )
    val_sel_row = val_rdf[val_rdf['Policy'] == preselected]
    val_metrics = val_sel_row.iloc[0].to_dict() if len(val_sel_row) else {}

    selection = {
        'preselected_policy': preselected,
        'selection_criterion': 'max Avg Profit s.t. Fill Rate >= 0.90',
        'selection_basis': (f"CV fold {last['fold']} validation data "
                            f"(N_val={len(last['y'])}, dates prior to test window)"),
        'residual_distribution_source': 'cv_fold_validation (OOS)',
        'forecast_calibration': (
            {'method': 'none'} if calibrator is None else {
                'method': calibrator['method'],
                'artifact': 'cv_scale_calibration.json',
                'prediction_key_fit': calibrator['pred_key'],
                'cv_raw_wape': calibrator['cv_metrics']['raw_wape'],
                'cv_calibrated_wape': calibrator['cv_metrics']['calibrated_wape'],
                'selection_calibrator_fit': (
                    'folds before held-out selector fold'
                    if len(cv_val_frames) > 1 else 'held-out selector fold'
                ),
            }
        ),
        'validation_fold_metrics': {k: val_metrics.get(k) for k in
                                     ['Avg Profit', 'Fill Rate', 'Waste%', 'SO Rate']},
        'test_metrics_reported': {k: sel_metrics.get(k) for k in
                                   ['Avg Profit', 'Fill Rate', 'Waste%', 'SO Rate']},
        'note': ("Policy was pre-selected on the held-out CV validation fold; "
                 "its test-set metrics are reported WITHOUT further selection. "
                 "Other policies are shown in inventory_results.csv for "
                 "transparency only and must not be cherry-picked post-hoc."),
    }
    (out_dir / 'inventory_selection.json').write_text(json.dumps(selection, indent=2, default=str))

    # ---- Metadata ----
    save_run_metadata(out_dir, 'pipeline_leakfree', vars(args), time.time() - t0)
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs in {out_dir}/")


if __name__ == '__main__':
    main()
