#!/usr/bin/env python3
"""Accuracy ablation: can targeted changes beat the paper's recursive ensemble
under the SAME leak-free protocol?

Arms (all share one data load, recovery, FeatureBuilder fit and feature frame):

  A0_recursive_baseline  paper configuration: recursive ensemble(mae,huber,q50)
  A1_direct_horizon      direct multi-horizon (horizon-as-feature), no recursion
  A2_tweedie_member      A0 + a Tweedie member in the ensemble (count-aware)
  A3_recentered          A0 + CV-scale recentering (grid chosen on fold-4 val)
  A4a_demrec_target      recursive ensemble trained on recovered demand target
  A4b_tobit_objective    recursive single model, Tobit-style censored objective
  A5_combo               direct(mae,huber,q50,tweedie) + recentering

FIXED SELECTION RULE (no test-set selection): the kept arm is the one
with the lowest WAPE on the LAST CV VALIDATION FOLD (fold 4); for recentering
arms the calibrator used at selection time is fit on folds 1-3 only. Test-set
metrics are reported for ALL arms for transparency, mirroring the paper's
inventory-policy table.

Vocabulary (for readers new to this repo):
  recursive   = the 1-step model is applied 7x over the eval week, feeding its
                own predictions back into lag features (pipeline_leakfree.py)
  direct      = one model trained on (history, horizon) pairs predicts each
                day from the forecast origin; no feedback (src/direct_forecast.py)
  recentering = post-hoc multiplicative scale calibration learned on CV
                validation folds only (src/calibration.py)
  WAPE / WPE  = weighted absolute / signed percentage error (src/metrics.py);
                WPE < 0 means the model under-predicts in aggregate

Run from the repo root:
    python scripts/accuracy_ablation.py --n_sp 1500
(1,500 SPs is the recorded study configuration; about 20 min on a laptop CPU.
Writes to `output/accuracy_ablation/` by default.)

Outputs:
  accuracy_ablation_results.csv     one row per arm (val + test metrics)
  accuracy_ablation_metadata.json   config, selection rule, winner, runtimes
  predictions_<arm>.csv             per-row test predictions for audit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_leakfree import (  # noqa: E402
    LGBM_BASE_PARAMS,
    LGBM_OBJECTIVES,
    apply_recovery,
    load_data,
    retrain_final,
    set_all_seeds,
    train_cv,
)
from src.calibration import (  # noqa: E402
    apply_scale_calibrator,
    learn_cv_scale_calibrator,
)
from src.censored_objectives import (  # noqa: E402
    make_tobit_l2_objective,
    mae_feval,
    per_sp_sigma,
)
from src.direct_forecast import (  # noqa: E402
    WARMUP_DAYS,
    predict_eval_direct,
    retrain_direct_final,
    train_direct_cv,
)
from src.feature_builder import FeatureBuilder, predict_eval_recursive  # noqa: E402
from src.metrics import wape, wpe, mae  # noqa: E402


POINT_OBJS = {k: LGBM_OBJECTIVES[k] for k in ('mae', 'huber', 'q50')}
# variance_power 1.3: between Poisson (1.0) and Gamma (2.0); the standard
# choice for zero-inflated retail count data.
TWEEDIE_OBJS = {'tweedie': {'objective': 'tweedie',
                            'tweedie_variance_power': 1.3, 'metric': 'mae'}}
# sp_shrink=1e9 acts as "infinite shrinkage": the per-SP factors collapse to
# 1.0, effectively disabling the per-SP correction (global+horizon only).
RECENTER_GRID = [
    {'sp_shrink': 28.0, 'horizon_shrink': 5000.0},   # paper's calibration default
    {'sp_shrink': 28.0, 'horizon_shrink': 100.0},    # stronger horizon correction
    {'sp_shrink': 1e9, 'horizon_shrink': 5000.0},    # global+horizon only
    {'sp_shrink': 1e9, 'horizon_shrink': 100.0},
]
SELECTION_RULE = ('lowest WAPE on CV fold-4 validation; recentering calibrators '
                  'fit on folds 1-3 at selection time, refit on folds 1-4 for test')


def _metrics(y: np.ndarray, p: np.ndarray, dt: np.ndarray) -> dict:
    dt = pd.to_datetime(pd.Series(dt))
    days = sorted(dt.unique())
    d1 = (dt == days[0]).values
    dl = (dt == days[-1]).values
    return {
        'wape': wape(y, p), 'wpe': wpe(y, p), 'mae': mae(y, p),
        'wape_d1': wape(y[d1], p[d1]), 'wape_dlast': wape(y[dl], p[dl]),
        # Fraction of rows with pred >= actual (demand covered) -- NOT the
        # under-prediction rate. 0.50 = perfectly centred median forecast.
        'p_under': float(np.mean(y <= p)),
    }


def _ens(preds: dict, members: tuple) -> np.ndarray:
    return np.mean([preds[m] for m in members], axis=0).astype('float32')


def _fold4_sales(frame: dict, train: pd.DataFrame) -> np.ndarray:
    """Observed sales aligned to a CV frame's rows (frames built on dem_rec
    targets store y=dem_rec; arm selection always scores against sales)."""
    key = pd.DataFrame({'sp': frame['sp'], 'dt': frame['dt']})
    merged = key.merge(train[['sp', 'dt', 'sale_amount']], on=['sp', 'dt'],
                       how='left', validate='one_to_one')
    return merged['sale_amount'].values.astype('float64')


def _recenter(frames: list, fold4: dict, fold4_y: np.ndarray,
              test_pred: np.ndarray, test_sp, test_dt) -> tuple:
    """Grid-select recentering on fold-4 val, refit on all folds, apply to test.

    Returns (val_pred_fold4, test_pred_calibrated, chosen_params).
    """
    best = None
    for g in RECENTER_GRID:
        cal13 = learn_cv_scale_calibrator(frames[:3], pred_key='ensemble', **g)
        v = apply_scale_calibrator(fold4['preds']['ensemble'], fold4['sp'],
                                   fold4['dt'], cal13)
        w = wape(fold4_y, v)
        if best is None or w < best[0]:
            best = (w, g, v)
    _, g, vpred = best
    cal_full = learn_cv_scale_calibrator(frames, pred_key='ensemble', **g)
    tpred = apply_scale_calibrator(test_pred, test_sp, test_dt, cal_full)
    return vpred, tpred, g


def train_tobit_cv(train_feat, fcols, sigma_map, base_params,
                   cv_folds: int = 4, val_size: int = 7):
    """Expanding-window CV for the single Tobit-objective model (arm A4b).

    Fold boundaries replicate pipeline_leakfree.train_cv exactly so that the
    fold-4 validation window is comparable across arms.

    NOTE: there is deliberately NO 0.5 down-weighting of censored rows here.
    The Tobit loss models censoring explicitly (survival-term gradient pushes
    predictions above the observed lower bound), so down-weighting censored
    rows on top of it would double-count the correction.

    Returns (best_iters list, fold4 dict with keys y/pred/dt).
    """
    dates = sorted(train_feat['dt'].unique())
    warmup_date = dates[WARMUP_DAYS - 1]
    available = [d for d in dates if d > warmup_date]
    n_avail = len(available)
    best_iters, fold4 = [], None
    for i in range(cv_folds):
        val_end_idx = n_avail - (cv_folds - 1 - i) * val_size
        val_start_idx = val_end_idx - val_size
        tr_mask = (train_feat['dt'] > warmup_date) & \
                  (train_feat['dt'] <= available[val_start_idx - 1])
        vl_mask = (train_feat['dt'] >= available[val_start_idx]) & \
                  (train_feat['dt'] <= available[min(val_end_idx - 1, n_avail - 1)])
        Xtr = train_feat.loc[tr_mask, fcols].values
        ytr = train_feat.loc[tr_mask, 'sale_amount'].values
        cens_tr = train_feat.loc[tr_mask, 'cens'].values == 1
        # SPs absent from sigma_map (all-censored history) fall back to the
        # same floor per_sp_sigma uses.
        sig_tr = train_feat.loc[tr_mask, 'sp'].map(sigma_map).fillna(0.1).values
        Xvl = train_feat.loc[vl_mask, fcols].values
        yvl = train_feat.loc[vl_mask, 'sale_amount'].values
        fobj = make_tobit_l2_objective(cens_tr, sig_tr)
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_vl = lgb.Dataset(Xvl, label=yvl, reference=d_tr)
        # LightGBM >= 4 (pinned in requirements.txt): a custom objective is
        # passed as params['objective']; the old fobj= kwarg was removed in 4.0.
        mdl = lgb.train({**base_params, 'objective': fobj}, d_tr,
                        num_boost_round=1500, valid_sets=[d_vl], feval=mae_feval,
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        best_iters.append(mdl.best_iteration)
        if i == cv_folds - 1:
            fold4 = {
                'pred': np.clip(mdl.predict(Xvl), 0, None),
                'y': yvl.astype('float64'),
                'dt': train_feat.loc[vl_mask, 'dt'].values,
            }
    return best_iters, fold4


def retrain_tobit_final(train_feat, fcols, sigma_map, base_params, best_iters):
    """Full-train Tobit retrain at mean-of-best-folds rounds (see train_tobit_cv)."""
    dates = sorted(train_feat['dt'].unique())
    warmup_date = dates[WARMUP_DAYS - 1]
    mask = train_feat['dt'] > warmup_date
    Xtr = train_feat.loc[mask, fcols].values
    ytr = train_feat.loc[mask, 'sale_amount'].values
    cens_tr = train_feat.loc[mask, 'cens'].values == 1
    sig_tr = train_feat.loc[mask, 'sp'].map(sigma_map).fillna(0.1).values
    fobj = make_tobit_l2_objective(cens_tr, sig_tr)
    nround = max(50, int(np.mean(best_iters)))
    mdl = lgb.train({**base_params, 'objective': fobj},
                    lgb.Dataset(Xtr, label=ytr), num_boost_round=nround)
    print(f"  [tobit] final num_boost_round={nround}")
    return mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=1500,
                    help='number of store-product (sp) series to sample '
                         '(stratified by stockout rate); 0 = all 50,000')
    ap.add_argument('--out', default='output/accuracy_ablation',
                    help='output directory name, created under the repo root')
    ap.add_argument('--seed', type=int, default=42,
                    help='global seed for numpy/LightGBM reproducibility')
    args = ap.parse_args()

    set_all_seeds(args.seed)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    runtimes, results = {}, []

    print(f"[1/8] Load + recovery (n_sp={args.n_sp})")
    train, ev = load_data(n_sp=args.n_sp)
    train = apply_recovery(train, 'heuristic_imr')
    fb = FeatureBuilder(n_clusters=min(50, max(2, train['sp'].nunique() // 10)),
                        seed=args.seed).fit(train)
    train_feat = fb.transform(train)
    fcols = fb.feature_cols(train_feat)
    print(f"  SPs={train['sp'].nunique():,} features={len(fcols)}")

    # NOTE: record() closes over test_sp/test_dt, which are defined after the
    # first recursive eval pass (step [4/8]); it must not be called earlier.
    def record(arm, val_y, val_p, val_dt, y, p, dt, note=''):
        row = {'arm': arm}
        row.update({f'val_{k}': v for k, v in _metrics(val_y, val_p, val_dt).items()
                    if k in ('wape', 'wpe')})
        row.update({f'test_{k}': v for k, v in _metrics(y, p, dt).items()})
        row['note'] = note
        results.append(row)
        pd.DataFrame({'sp': test_sp, 'dt': test_dt, 'y_true': y, 'pred': p}) \
            .to_csv(out_dir / f'predictions_{arm}.csv', index=False)
        print(f"  {arm}: val WAPE={row['val_wape']:.2f}  "
              f"test WAPE={row['test_wape']:.2f} WPE={row['test_wpe']:+.2f}")

    # ---- A0 + A2 share one CV (A2 adds a tweedie CV) and one recursive pass
    print("\n[2/8] A0 baseline CV (mae,huber,q50)")
    t0 = time.time()
    _, _, bi0, frames0 = train_cv(train_feat, fcols, objectives=POINT_OBJS)
    final0 = retrain_final(train_feat, fcols, bi0, objectives=POINT_OBJS)
    runtimes['A0_cv_final'] = time.time() - t0

    print("\n[3/8] A2 tweedie member CV")
    t0 = time.time()
    _, _, bi_tw, frames_tw = train_cv(train_feat, fcols, objectives=TWEEDIE_OBJS)
    final_tw = retrain_final(train_feat, fcols, bi_tw, objectives=TWEEDIE_OBJS)
    runtimes['A2_cv_final'] = time.time() - t0

    print("\n[4/8] Recursive eval pass (A0+A2 models together)")
    t0 = time.time()
    rec_models = {**final0, **final_tw}
    preds_rec, eval_sorted = predict_eval_recursive(fb, rec_models, fcols, train, ev)
    runtimes['recursive_pass_A0A2'] = time.time() - t0
    test_y = eval_sorted['sale_amount'].values.astype('float64')
    test_sp = eval_sorted['sp'].values
    test_dt = eval_sorted['dt'].values

    f4 = frames0[-1]
    f4_y = f4['y'].astype('float64')   # target sale_amount -> sales already
    val_dt = f4['dt']

    a0_val = f4['preds']['ensemble']
    a0_test = _ens(preds_rec, ('mae', 'huber', 'q50'))
    record('A0_recursive_baseline', f4_y, a0_val, val_dt, test_y, a0_test, test_dt,
           note='paper configuration')

    # A2: 4-member ensemble (fold rows align: identical fold specs and ordering)
    tw4 = frames_tw[-1]
    a2_val = np.mean([f4['preds']['mae'], f4['preds']['huber'],
                      f4['preds']['q50'], tw4['preds']['tweedie']], axis=0)
    a2_test = _ens(preds_rec, ('mae', 'huber', 'q50', 'tweedie'))
    record('A2_tweedie_member', f4_y, a2_val, val_dt, test_y, a2_test, test_dt,
           note='tweedie_variance_power=1.3; write-back feedback unchanged (3-member)')

    # ---- A3 recentering on the baseline
    print("\n[5/8] A3 recentering grid")
    a3_val, a3_test, g3 = _recenter(frames0, f4, f4_y, a0_test, test_sp, test_dt)
    record('A3_recentered', f4_y, a3_val, val_dt, test_y, a3_test, test_dt,
           note=f"grid winner {g3}")

    # ---- A1 direct multi-horizon
    print("\n[6/8] A1 direct horizon-as-feature")
    t0 = time.time()
    _, bi_d, frames_d = train_direct_cv(train_feat, fcols, POINT_OBJS,
                                        LGBM_BASE_PARAMS)
    final_d = retrain_direct_final(train_feat, fcols, POINT_OBJS,
                                   LGBM_BASE_PARAMS, bi_d)
    preds_dir, _ = predict_eval_direct(fb, final_d, fcols, train, ev)
    runtimes['A1_direct'] = time.time() - t0
    d4 = frames_d[-1]
    d4_y = d4['y'].astype('float64')
    a1_val = d4['preds']['ensemble']
    a1_test = _ens(preds_dir, ('mae', 'huber', 'q50'))
    record('A1_direct_horizon', d4_y, a1_val, d4['dt'], test_y, a1_test, test_dt,
           note='no recursion; origin-anchored history + target-day covariates')

    # ---- A4a recovered-demand target
    print("\n[7/8] A4a dem_rec target + A4b tobit objective")
    t0 = time.time()
    _, _, bi_r, frames_r = train_cv(train_feat, fcols, objectives=POINT_OBJS,
                                    target_col='dem_rec')
    final_r = retrain_final(train_feat, fcols, bi_r, objectives=POINT_OBJS,
                            target_col='dem_rec')
    preds_r, _ = predict_eval_recursive(fb, final_r, fcols, train, ev)
    runtimes['A4a'] = time.time() - t0
    r4 = frames_r[-1]
    r4_sales = _fold4_sales(r4, train)
    a4a_val = r4['preds']['ensemble']
    a4a_test = _ens(preds_r, ('mae', 'huber', 'q50'))
    record('A4a_demrec_target', r4_sales, a4a_val, r4['dt'], test_y, a4a_test,
           test_dt, note='target=recovered demand; scored against observed sales')

    # ---- A4b tobit objective (single model, custom loss)
    t0 = time.time()
    sigma_map = per_sp_sigma(train)
    bi_tobit, tobit_fold4 = train_tobit_cv(train_feat, fcols, sigma_map,
                                           LGBM_BASE_PARAMS)
    mdl_tobit = retrain_tobit_final(train_feat, fcols, sigma_map,
                                    LGBM_BASE_PARAMS, bi_tobit)
    preds_tobit, _ = predict_eval_recursive(fb, {'tobit': mdl_tobit}, fcols, train, ev)
    runtimes['A4b'] = time.time() - t0
    record('A4b_tobit_objective', tobit_fold4['y'], tobit_fold4['pred'],
           tobit_fold4['dt'], test_y, preds_tobit['tobit'], test_dt,
           note='latent-demand head; expected to overshoot on censored eval days')

    # ---- A5 combo: direct 4-member + recentering
    print("\n[8/8] A5 combo (direct + tweedie + recentering)")
    t0 = time.time()
    _, bi_dtw, frames_dtw = train_direct_cv(train_feat, fcols, TWEEDIE_OBJS,
                                            LGBM_BASE_PARAMS)
    final_dtw = retrain_direct_final(train_feat, fcols, TWEEDIE_OBJS,
                                     LGBM_BASE_PARAMS, bi_dtw)
    preds_dtw, _ = predict_eval_direct(fb, final_dtw, fcols, train, ev)
    combo_frames = []
    for fr3, frt in zip(frames_d, frames_dtw):
        p = dict(fr3['preds'])
        p['tweedie'] = frt['preds']['tweedie']
        p['ensemble'] = np.mean([p['mae'], p['huber'], p['q50'], p['tweedie']],
                                axis=0).astype('float32')
        combo_frames.append({**fr3, 'preds': p})
    c4 = combo_frames[-1]
    combo_test_raw = np.mean([preds_dir['mae'], preds_dir['huber'],
                              preds_dir['q50'], preds_dtw['tweedie']],
                             axis=0).astype('float32')
    a5_val, a5_test, g5 = _recenter(combo_frames, c4, c4['y'].astype('float64'),
                                    combo_test_raw, test_sp, test_dt)
    runtimes['A5_combo'] = time.time() - t0
    record('A5_combo', c4['y'].astype('float64'), a5_val, c4['dt'],
           test_y, a5_test, test_dt,
           note=f"direct 4-member + recentering {g5}")

    # ---- Fixed validation selection + outputs
    rdf = pd.DataFrame(results)
    winner = rdf.loc[rdf['val_wape'].idxmin(), 'arm']
    rdf['preselected'] = (rdf['arm'] == winner).astype(int)
    rdf.to_csv(out_dir / 'accuracy_ablation_results.csv', index=False)

    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=SCRIPT_DIR, text=True).strip()
    except Exception:
        sha = 'unknown'
    meta = {
        'script': 'accuracy_ablation',
        'git_sha': sha,
        'config': vars(args),
        'selection_rule': SELECTION_RULE,
        'preselected_arm': winner,
        'recenter_grid': RECENTER_GRID,
        'runtimes_s': {k: round(v, 1) for k, v in runtimes.items()},
        'total_runtime_s': round(time.time() - t_start, 1),
        'lightgbm': lgb.__version__,
    }
    (out_dir / 'accuracy_ablation_metadata.json').write_text(json.dumps(meta, indent=2))
    print(f"\nPreselected on fold-4 validation: {winner}")
    print(rdf[['arm', 'val_wape', 'test_wape', 'test_wpe', 'preselected']]
          .to_string(index=False))
    print(f"\nDone in {meta['total_runtime_s']:.0f}s -> {out_dir}/")


if __name__ == '__main__':
    main()
