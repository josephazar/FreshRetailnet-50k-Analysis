"""Headline run -- full-data pipeline for the gate-selected configuration.

The fixed common multi-step validation study in
scripts/multistep_validation.py retained the incumbent

    A3: recursive forecasting, three-member ensemble (MAE + Huber + Q50),
        with CV scale recentering (sp_shrink=28, horizon_shrink=5000,
        the pipeline default)

as the headline configuration. The challenger A5 (direct multi-horizon +
Tweedie member + recentering) had a genuinely lower pooled validation WAPE
(gates G1, G2, G4 passed; paired store-clustered 95% CI for the WAPE
difference excludes zero) but FAILED bias gate G3: its pooled validation
|WPE| exceeded the incumbent's by more than the pre-declared 1.0 pp
tolerance. Under the fixed rule, any gate failure retains the incumbent;
the held-out test never participated in the decision.

Only the gate-selected A3 family is eligible for this headline runner; the
seven-arm comparison remains in scripts/multistep_validation.py.

This script runs that ONE configuration end-to-end on the full dataset and
produces every headline artifact from the same coherent run:
  - forecast metrics (calibrated and raw) on the held-out evaluation window;
  - per-row predictions (with stockout hours, so the uncensored-subset
    diagnostic E4 can be computed without reloading raw data);
  - fold-causal, nested, true multi-step CV using the same outer protocol as
    the promotion study; this supplies final boosting rounds, calibration,
    residuals, and the LAST-fold inventory-policy selection without exposing
    any selector-fold actual to recovery or feature fitting;
  - test evaluation of all 20 policies plus per-row order quantities for
    every policy (policy_orders.parquet) so the profit-inference study E1
    can run store-clustered bootstraps without re-running the pipeline;
  - per-model gain feature importances;
  - run metadata (git SHA, environment, data hashes, phase runtimes).

Scientific choices fixed before the headline result is observed:
  - recovery: heuristic IMR (blend 0.5), as in the reported configuration;
  - calibration: cv_scale with the pipeline-default shrinkage
    (sp_shrink=28, horizon_shrink=5000); NO grid search anywhere;
  - Empirical-Newsvendor residuals are CROSS-FITTED residuals of the
    deployed point forecast: the residual for CV
    fold f uses a calibrator fit on folds < f only, so no fold's residual
    is scaled by a calibrator that saw that fold. Fold 1 seeds the
    calibrators and contributes no residuals. Test-time EmpNV pools folds
    2-4; fold-4 selection pools folds 2-3;
  - policy selection: max Avg Profit subject to Fill Rate >= 0.90 on the
    last validation fold (floors 0.85/0.95 reported as sensitivity); the
    historical statistics behind the candidate policies (means, variances,
    Gamma/KDE fits, sigma scales) use only data strictly BEFORE the
    selection week, mirroring the test-time
    relationship where all of train precedes the evaluation window;
  - profit is evaluated against OBSERVED SALES under censoring; it is a
    proxy whose direction of bias relative to latent-demand profit is not
    identified, and it is never described as a bound.

Run (full data; expect several hours):
    ./venv/bin/python scripts/headline_run.py --n_sp 0 --out output/headline
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

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_leakfree import (  # noqa: E402
    LGBM_OBJECTIVES,
    apply_recovery,
    load_data,
    retrain_final,
    save_run_metadata,
    set_all_seeds,
)
from src.calibration import (  # noqa: E402
    apply_calibrator_to_prediction_dict,
    apply_scale_calibrator,
    learn_cv_scale_calibrator,
)
from src.feature_builder import FeatureBuilder, predict_eval_recursive  # noqa: E402
from src.inventory import (  # noqa: E402
    DEFAULT_INV_CFG,
    compute_policies,
    profit_optimal_cr,
    residual_quantile_from_cv,
)
from src.metrics import corr, mae, rmse, smape, wape, wpe  # noqa: E402
from scripts.multistep_validation import (  # noqa: E402
    INNER_VAL_DAYS,
    _fold_specs,
    _nested_fit,
    _weights,
)

HEADLINE_ARM = 'A3_recursive_recentered'
RECENTER_CFG = {'sp_shrink': 28.0, 'horizon_shrink': 5000.0}
FILL_RATE_FLOORS = [0.85, 0.90, 0.95]
HEADLINE_FLOOR = 0.90

# Per-family configuration. Ensemble members define the deployed point
# forecast; q10/q90 are trained only as quantile heads for the inventory
# policies (never ensemble members).
FAMILIES = {
    'A3_recursive_recentered': {
        'design': 'recursive',
        'members': ('mae', 'huber', 'q50'),
        'objectives': {k: LGBM_OBJECTIVES[k]
                       for k in ('mae', 'huber', 'q50', 'q10', 'q90')},
    },
}


def _member_ensemble(preds: dict, members: tuple) -> np.ndarray:
    return np.mean([preds[m] for m in members], axis=0).astype('float32')


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {'MAE': round(mae(y, p), 4), 'RMSE': round(rmse(y, p), 4),
            'SMAPE': round(smape(y, p), 4), 'Corr': round(corr(y, p), 4),
            'WAPE': round(wape(y, p), 4), 'WPE': round(wpe(y, p), 4)}


def _fold_causal_recursive_cv(raw_train: pd.DataFrame, objectives: dict,
                              members: tuple, seed: int):
    """Run the retained recursive family under the common outer folds.

    Recovery and aggregate features are refit through each fold origin.
    Boosting rounds are selected on an inner temporal split, models are refit
    on the complete outer-training window, and the outer week is forecast
    recursively from the fold origin.
    """
    dates = sorted(raw_train['dt'].unique())
    warmup_date, specs = _fold_specs(dates)
    n_clusters = min(50, max(2, raw_train['sp'].nunique() // 10))
    best_iters = {name: [] for name in objectives}
    frames, metrics = [], []
    fcols_ref = None

    for spec in specs:
        fold, train_end, val_days = (
            spec['fold'], spec['train_end'], spec['val_days'])
        print(f"  Fold {fold}/4: train through "
              f"{pd.Timestamp(train_end).date()}, true multi-step validation")

        recovered = apply_recovery(
            raw_train[raw_train['dt'] <= train_end].copy(), 'heuristic_imr')
        fb_fold = FeatureBuilder(n_clusters=n_clusters, seed=seed).fit(recovered)
        feat = fb_fold.transform(recovered)
        fcols = fb_fold.feature_cols(feat)
        if fcols_ref is None:
            fcols_ref = fcols
        assert fcols == fcols_ref, 'feature set differs across folds'

        upcast = [c for c in fcols
                  if feat[c].dtype in (np.float64, np.int32)]
        if upcast:
            feat[upcast] = feat[upcast].astype('float32')

        days_fold = [d for d in dates if warmup_date < d <= train_end]
        inner_val_days = days_fold[-INNER_VAL_DAYS:]
        inner_end = days_fold[-INNER_VAL_DAYS - 1]
        inner_train = ((feat['dt'] > warmup_date)
                       & (feat['dt'] <= inner_end))
        inner_val = feat['dt'].isin(inner_val_days)
        outer_train = ((feat['dt'] > warmup_date)
                       & (feat['dt'] <= train_end))

        models, rounds = _nested_fit(
            objectives,
            feat.loc[inner_train, fcols].values,
            feat.loc[inner_train, 'sale_amount'].values,
            _weights(feat, inner_train),
            feat.loc[inner_val, fcols].values,
            feat.loc[inner_val, 'sale_amount'].values,
            feat.loc[outer_train, fcols].values,
            feat.loc[outer_train, 'sale_amount'].values,
            _weights(feat, outer_train),
            return_rounds=True,
        )
        for name, nround in rounds.items():
            best_iters[name].append(nround)

        val_raw = raw_train[raw_train['dt'].isin(val_days)][
            [c for c in raw_train.columns if c not in ('cens', 'dem_rec')]
        ].copy()
        preds, val_sorted = predict_eval_recursive(
            fb_fold, models, fcols, recovered, val_raw)
        preds['ensemble'] = _member_ensemble(preds, members)
        y = val_sorted['sale_amount'].values.astype('float64')
        frame = {
            'fold': fold,
            'sp': val_sorted['sp'].values,
            'dt': val_sorted['dt'].values,
            'y': y,
            'preds': preds,
        }
        frames.append(frame)
        metrics.append({
            'fold': fold,
            'mae_members': round(mae(y, preds['ensemble']), 4),
            'wape_members': round(wape(y, preds['ensemble']), 4),
            'wpe_members': round(wpe(y, preds['ensemble']), 4),
            'n_train': int(outer_train.sum()),
            'n_val': int(len(y)),
            **{f'best_iter_{name}': rounds[name] for name in objectives},
        })

        del feat, recovered, fb_fold, models, preds, val_sorted

    return metrics, best_iters, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=0,
                    help='0 = full dataset (headline); >0 = subsample smoke')
    ap.add_argument('--out', default='output/headline')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--arm', default=HEADLINE_ARM, choices=sorted(FAMILIES),
                    help='forecasting family; default = the gate-selected '
                         'headline configuration')
    args = ap.parse_args()
    fam = FAMILIES[args.arm]
    members = fam['members']

    set_all_seeds(args.seed)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    phase_t = {}

    # ---- [1/7] data ------------------------------------------------------
    print(f"[1/7] Load (n_sp={args.n_sp or 'ALL'})")
    t = time.time()
    raw_train, ev = load_data(n_sp=args.n_sp)
    phase_t['load'] = time.time() - t
    print(f"  train={len(raw_train):,} rows, eval={len(ev):,} rows, "
          f"SPs={raw_train['sp'].nunique():,}")

    # ---- [2/7] fold-causal CV --------------------------------------------
    if fam['design'] != 'recursive':
        raise ValueError('The gate-retained headline must use recursive CV')
    print(f"[2/7] Fold-causal nested recursive CV "
          f"({len(fam['objectives'])} objectives, 4 folds)")
    t = time.time()
    cv_metrics, best_iters, cv_val_frames = _fold_causal_recursive_cv(
        raw_train, fam['objectives'], members, args.seed)
    phase_t['cv'] = time.time() - t
    pd.DataFrame(cv_metrics).to_csv(out_dir / 'cv_results.csv', index=False)

    # ---- [3/7] recentering calibrators (fixed config, no search) ----------
    print("[3/7] CV scale recentering (fixed config)")
    t = time.time()
    calibrator = learn_cv_scale_calibrator(
        cv_val_frames, pred_key='ensemble', **RECENTER_CFG)
    selection_frames = cv_val_frames[:-1] if len(cv_val_frames) > 1 \
        else cv_val_frames
    selection_calibrator = learn_cv_scale_calibrator(
        selection_frames, pred_key='ensemble', **RECENTER_CFG)

    # Empirical-Newsvendor safety stock = CROSS-FITTED residual quantiles of
    # the deployed (calibrated ensemble) forecast: the
    # residual for fold f uses a calibrator fit on folds < f only, so no
    # fold's residual is scaled by a calibrator that saw that fold. Fold 1
    # seeds the calibrators and contributes no residuals.
    def _crossfit_residuals(frames):
        out = []
        for i in range(1, len(frames)):
            cal_i = learn_cv_scale_calibrator(
                frames[:i], pred_key='ensemble', **RECENTER_CFG)
            fr = frames[i]
            p = apply_scale_calibrator(fr['preds']['ensemble'], fr['sp'],
                                       fr['dt'], cal_i)
            out.append((fr['sp'], fr['y'] - p))
        return out

    cv_residuals_for_test = _crossfit_residuals(cv_val_frames)       # f2-f4
    cv_residuals_for_selection = _crossfit_residuals(selection_frames)  # f2-f3
    (out_dir / 'cv_scale_calibration.json').write_text(
        json.dumps(calibrator, indent=2))
    phase_t['calibration'] = time.time() - t
    print(f"  global_scale={calibrator['global_scale']:.4f} "
          f"(raw {calibrator['raw_global_scale']:.4f}); "
          f"cv raw WAPE {calibrator['cv_metrics']['raw_wape']:.2f} -> "
          f"calibrated {calibrator['cv_metrics']['calibrated_wape']:.2f}")

    # ---- [4/7] full-train features ---------------------------------------
    print("[4/7] Full-train recovery + features")
    t = time.time()
    train = apply_recovery(raw_train.copy(), 'heuristic_imr')
    fb = FeatureBuilder(
        n_clusters=min(50, max(2, train['sp'].nunique() // 10)),
        seed=args.seed,
    ).fit(train)
    train_feat = fb.transform(train)
    fcols = fb.feature_cols(train_feat)
    upcast = [c for c in fcols
              if train_feat[c].dtype in (np.float64, np.int32)]
    if upcast:
        train_feat[upcast] = train_feat[upcast].astype('float32')
    phase_t['full_train_features'] = time.time() - t
    print(f"  features={len(fcols)} (downcast {len(upcast)} columns)")

    # ---- [5/7] final retrain + held-out prediction ------------------------
    print("[5/7] Final retrain + held-out evaluation window")
    t = time.time()
    final_models = retrain_final(train_feat, fcols, best_iters,
                                 objectives=fam['objectives'])
    preds_raw, eval_sorted = predict_eval_recursive(
        fb, final_models, fcols, train, ev)
    design_cols = fcols
    preds_raw['ensemble'] = _member_ensemble(preds_raw, members)
    sp_eval = eval_sorted['sp'].values
    dt_eval = eval_sorted['dt'].values
    y_eval = eval_sorted['sale_amount'].values.astype('float64')
    preds_cal = apply_calibrator_to_prediction_dict(
        preds_raw, sp_eval, dt_eval, calibrator)
    phase_t['retrain_predict'] = time.time() - t

    # feature importances (gain) for the appendix / parsimony discussion
    imp_rows = []
    for name, mdl in final_models.items():
        for feat_name, g in zip(design_cols, mdl.feature_importance('gain')):
            imp_rows.append({'model': name, 'feature': feat_name,
                             'gain': float(g)})
    pd.DataFrame(imp_rows).to_csv(out_dir / 'feature_importance.csv',
                                  index=False)

    # forecast metrics: calibrated headline + raw, full window and the
    # uncensored diagnostic subset (E4; non-random subset, diagnostic only)
    uncens = eval_sorted['stock_hour6_22_cnt'].values == 0
    fmetrics = {}
    for name in ('ensemble',) + members:
        fmetrics[name] = _metrics(y_eval, preds_cal[name])
        fmetrics[f'{name}_raw'] = _metrics(y_eval, preds_raw[name])
    fmetrics['ensemble_uncensored_subset'] = _metrics(
        y_eval[uncens], preds_cal['ensemble'][uncens])
    fmetrics['ensemble_censored_subset'] = _metrics(
        y_eval[~uncens], preds_cal['ensemble'][~uncens])
    fmetrics['_uncensored_subset_note'] = (
        'rows with zero stockout hours; non-random diagnostic subset, not a '
        'validation of latent-demand accuracy')
    fmetrics['_n_rows'] = {'total': int(len(y_eval)),
                           'uncensored': int(uncens.sum())}
    (out_dir / 'forecast_metrics.json').write_text(
        json.dumps(fmetrics, indent=2))
    print(f"  headline (calibrated ensemble): "
          f"WAPE {fmetrics['ensemble']['WAPE']:.2f} "
          f"WPE {fmetrics['ensemble']['WPE']:+.2f} | raw: "
          f"WAPE {fmetrics['ensemble_raw']['WAPE']:.2f} "
          f"WPE {fmetrics['ensemble_raw']['WPE']:+.2f}")

    pred_df = pd.DataFrame({'sp': sp_eval, 'dt': dt_eval, 'y_true': y_eval,
                            'stockout_hours':
                                eval_sorted['stock_hour6_22_cnt'].values})
    for name in dict.fromkeys(('ensemble',) + members + ('q10', 'q90')):
        pred_df[f'pred_{name}'] = preds_cal[name]
        pred_df[f'pred_{name}_raw'] = preds_raw[name]
    pred_df.to_csv(out_dir / 'predictions.csv', index=False)

    # ---- [6/7] inventory: selection on last validation fold ---------------
    print("[6/7] Inventory policy selection (last validation fold)")
    t = time.time()
    cr_nv = profit_optimal_cr(DEFAULT_INV_CFG)
    sp_res_q = residual_quantile_from_cv(cv_residuals_for_test, cr=cr_nv)
    selection_sp_res_q = residual_quantile_from_cv(
        cv_residuals_for_selection, cr=cr_nv)

    last = cv_val_frames[-1]
    val_inv_preds_raw = {k: last['preds'][k]
                         for k in ('ensemble', 'q10', 'q50', 'q90')}
    val_inv_preds = apply_calibrator_to_prediction_dict(
        val_inv_preds_raw, last['sp'], last['dt'], selection_calibrator)

    # Amendment A5: the historical statistics behind the candidate policies
    # (per-SP means/variances, Gamma/KDE fits, sigma scales) must end
    # strictly BEFORE the selection week, mirroring the test-time
    # relationship where all of train precedes the evaluation window.
    selection_train = train[train['dt'] < pd.Timestamp(last['dt'].min())]

    # One policy evaluation on the selection fold; the fill-rate floors are
    # applied to its rows (identical to select_policy_on_validation's
    # filter + first-occurrence idxmax, without recomputing policies).
    val_rdf, _ = compute_policies(
        preds=val_inv_preds, sp_keys=last['sp'], train_df=selection_train,
        sp_residual_quantile=selection_sp_res_q, cfg=DEFAULT_INV_CFG,
        actuals_for_eval=last['y'])
    floor_selection = {}
    for floor in FILL_RATE_FLOORS:
        cand = val_rdf[val_rdf['Fill Rate'] >= floor]
        if cand.empty:
            print(f"  [WARN] no policy meets FR>={floor}; using all")
            cand = val_rdf
        floor_selection[f'{floor:.2f}'] = \
            cand.loc[cand['Avg Profit'].idxmax(), 'Policy']
    preselected = floor_selection[f'{HEADLINE_FLOOR:.2f}']
    print(f"  preselected @floor {HEADLINE_FLOOR}: {preselected}")
    for fl, pol in floor_selection.items():
        print(f"    floor {fl}: {pol}")

    val_rdf['preselected'] = (val_rdf['Policy'] == preselected).astype(int)
    val_rdf.to_csv(out_dir / 'inventory_validation_results.csv', index=False)

    # ---- [7/7] inventory: held-out test evaluation ------------------------
    print("[7/7] Inventory test evaluation (all 20 policies)")
    inv_preds = {k: preds_cal[k] for k in ('ensemble', 'q10', 'q50', 'q90')}
    rdf, policy_Q = compute_policies(
        preds=inv_preds, sp_keys=sp_eval, train_df=train,
        sp_residual_quantile=sp_res_q, cfg=DEFAULT_INV_CFG,
        actuals_for_eval=y_eval)
    rdf['preselected'] = (rdf['Policy'] == preselected).astype(int)
    rdf = rdf.sort_values('Avg Profit', ascending=False)
    rdf.to_csv(out_dir / 'inventory_results.csv', index=False)

    # Per-row order quantities for EVERY policy: the profit-inference study
    # (E1) recomputes per-row profit from these and clusters by store
    # (store_id = sp // 10000) without re-running the pipeline.
    orders = pd.DataFrame({'sp': sp_eval, 'dt': dt_eval, 'y_true': y_eval})
    for pol_name, q_arr in policy_Q.items():
        orders[pol_name] = q_arr.astype('float32')
    orders.to_parquet(out_dir / 'policy_orders.parquet', index=False)
    phase_t['inventory'] = time.time() - t

    sel_row = rdf[rdf['Policy'] == preselected].iloc[0]
    val_row = val_rdf[val_rdf['Policy'] == preselected].iloc[0]
    selection = {
        'preselected_policy': preselected,
        'selection_criterion':
            f'max Avg Profit s.t. Fill Rate >= {HEADLINE_FLOOR}',
        'newsvendor_critical_ratio': {
            'value': round(cr_nv, 6),
            'formula': 'Cu/(Cu+Co), Cu=unit_revenue-unit_cost+p, '
                       'Co=unit_cost+h+w (profit-optimal for the stated '
                       'objective)'},
        'selection_basis': 'last CV validation fold; selector calibrator, '
                           'EmpNV residuals, and policy history statistics '
                           'from data strictly before the fold',
        'fill_rate_floor_sensitivity': floor_selection,
        'residual_distribution_source':
            'cv_fold_validation residuals of the calibrated ensemble, '
            'cross-fitted (calibrator for fold f fit on folds < f); '
            'out-of-sample w.r.t. model training; recovery and FeatureBuilder '
            'are refit through each fold origin',
        'forecast_calibration': {
            'method': 'cv_scale', **RECENTER_CFG,
            'artifact': 'cv_scale_calibration.json',
            'prediction_key_fit': 'ensemble'},
        'validation_fold_metrics': {
            k: float(val_row[k]) for k in
            ('Avg Profit', 'Fill Rate', 'Waste%', 'SO Rate')},
        'test_metrics_reported': {
            k: float(sel_row[k]) for k in
            ('Avg Profit', 'Fill Rate', 'Waste%', 'SO Rate')},
        'note': 'profit is an observed-sales proxy under censored '
                'evaluation; direction of bias vs latent-demand profit is '
                'not identified',
    }
    (out_dir / 'inventory_selection.json').write_text(
        json.dumps(selection, indent=2))

    # ---- metadata ---------------------------------------------------------
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                         capture_output=True, text=True).stdout.strip()
    headline_meta = {
        'script': 'headline_run',
        'git_sha': sha,
        'arm': args.arm,
        'is_gate_selected_headline': args.arm == HEADLINE_ARM,
        'gate_provenance': {
            'gate_study': 'output/multistep_validation',
            'decision': 'validation-only promotion gate G1-G4, decided '
                        'before any full-data test evaluation; challenger '
                        'A5 failed bias gate G3, incumbent A3 retained'},
        'ensemble_members': list(members),
        'recenter_cfg': RECENTER_CFG,
        'phase_runtimes_s': {k: round(v, 1) for k, v in phase_t.items()},
        'runtime_s_total': round(time.time() - t0, 1),
    }
    (out_dir / 'headline_run_metadata.json').write_text(
        json.dumps(headline_meta, indent=2))
    save_run_metadata(out_dir, 'headline_run', vars(args), time.time() - t0)
    print(f"Done in {headline_meta['runtime_s_total']:.0f}s -> {out_dir}/")


if __name__ == '__main__':
    main()
