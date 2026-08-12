"""Common multi-step validation of the forecasting arms.

Every arm is validated under ONE regime: true 7-day multi-step forecasts of
each fold's validation window from information available at the fold origin
only. Recursive and direct arms are evaluated under the same multi-step
regime, and the full-data headline configuration is decided through a
promotion gate evaluated on validation quantities only. The held-out test set
never participates.

The protocol enforces:
  A1. FOLD-CAUSAL RECOVERY AND FEATURES. Demand recovery (heuristic IMR) and
      FeatureBuilder are refit per fold on data up to the fold's train_end
      only, so no validation-window actual can enter recovery, aggregate
      lookups, calendar profiles, or cluster statistics used to predict that
      window. Within a fold, every arm shares the same builder, so arms are
      still compared under identical information sets.
  A2. NESTED EARLY STOPPING. Boosting iterations are chosen on an INNER
      temporal split (the last 7 days of the fold's training window, using
      each family's native training design); the model is then refit on the
      full fold-training window at that fixed iteration count. The outer
      validation fold is touched exactly once, by the final multi-step
      evaluation, for every family alike.
  A3. Validation predictions flow through the SAME code paths as the final
      test evaluation (predict_eval_recursive / predict_eval_direct), so the
      validation regime is structurally identical to deployment.
  A4. A paired, store-clustered bootstrap CI for the candidate-minus-
      incumbent pooled WAPE difference is reported as a descriptive measure.

Run:
    ./venv/bin/python scripts/multistep_validation.py --n_sp 1500
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
    set_all_seeds,
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
    _origin_design,
    build_direct_table,
    direct_feature_cols,
    predict_eval_direct,
)
from src.feature_builder import FeatureBuilder, predict_eval_recursive  # noqa: E402
from src.metrics import wape, wpe  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed protocol constants
# ---------------------------------------------------------------------------
PROTOCOL_NAME = 'fold-causal multi-step validation with nested early stopping'
SELECTION_RULE = (
    'lowest pooled multi-step validation WAPE over folds 2-4; every arm '
    'predicts each fold window 7 days ahead from the fold origin; recovery '
    'and FeatureBuilder refit per fold on data up to the fold train_end '
    '(fold-causal); boosting iterations chosen on an inner temporal split '
    'of the fold-training window (nested; outer fold untouched by '
    'training); recentering calibrators fit only on earlier folds (default '
    'config, sp_shrink=28, horizon_shrink=5000); fold 1 excluded from '
    'pooling (calibrator seed); eligible arms: A0, A1, A2, A3, A5; '
    'diagnostic-only: A4a, A4b (latent-demand heads)')
PROMOTION_GATE = (
    'incumbent = A3 (recursive + CV recentering); candidate promotes iff '
    'G1 pooled val WAPE < incumbent, '
    'G2 wins >= 2 of 3 pooled folds, G3 |pooled val WPE| <= incumbent + '
    '1.0 pp, G4 runtime <= 3x incumbent; test never participates')
ELIGIBLE = ['A0_recursive', 'A1_direct', 'A2_recursive_tweedie',
            'A3_recursive_recentered', 'A5_direct_tweedie_recentered']
DIAGNOSTIC = ['A4a_demrec_target', 'A4b_tobit_objective']
INCUMBENT = 'A3_recursive_recentered'
POOLED_FOLDS = [2, 3, 4]          # 1-based fold ids used for selection
RECENTER_CFG = {'sp_shrink': 28.0, 'horizon_shrink': 5000.0}
INNER_VAL_DAYS = 7                # nested early-stopping split length
BOOTSTRAP_B = 2000                # store-clustered CI resamples
BOOTSTRAP_SEED = 42

POINT_OBJS = {k: LGBM_OBJECTIVES[k] for k in ('mae', 'huber', 'q50')}
TWEEDIE_OBJ = {'tweedie': {'objective': 'tweedie',
                           'tweedie_variance_power': 1.3, 'metric': 'mae'}}
REC_OBJS = {**POINT_OBJS, **TWEEDIE_OBJ}


def _fold_specs(dates, cv_folds=4, val_size=7):
    warmup_date = dates[WARMUP_DAYS - 1]
    available = [d for d in dates if d > warmup_date]
    n = len(available)
    specs = []
    for i in range(cv_folds):
        end = n - (cv_folds - 1 - i) * val_size
        start = end - val_size
        specs.append({'fold': i + 1,
                      'train_end': available[start - 1],
                      'val_days': available[start:end]})
    return warmup_date, specs


def _nested_fit(objectives, X_in, y_in, w_in, X_iv, y_iv,
                X_out, y_out, w_out, fobj=None, fobj_outer=None,
                return_rounds=False):
    """Nested early stopping: choose iterations on the inner split, refit on
    the full outer-training design at that fixed count (no valid set)."""
    models = {}
    rounds = {}
    for name, obj in objectives.items():
        params = {**LGBM_BASE_PARAMS, **obj}
        d_in = lgb.Dataset(X_in, label=y_in, weight=w_in)
        d_iv = lgb.Dataset(X_iv, label=y_iv, reference=d_in)
        kw = {'feval': mae_feval} if fobj is not None else {}
        inner_params = dict(params)
        if fobj is not None:
            inner_params['objective'] = fobj
        inner = lgb.train(inner_params, d_in, num_boost_round=1500,
                          valid_sets=[d_iv],
                          callbacks=[lgb.early_stopping(50),
                                     lgb.log_evaluation(0)], **kw)
        nround = max(50, inner.best_iteration)
        rounds[name] = nround
        outer_params = dict(params)
        if fobj_outer is not None:
            outer_params['objective'] = fobj_outer
        models[name] = lgb.train(
            outer_params, lgb.Dataset(X_out, label=y_out, weight=w_out),
            num_boost_round=nround)
        del d_in, d_iv, inner
    return (models, rounds) if return_rounds else models


def _weights(feat, mask, weight_censored=True):
    w = np.ones(int(mask.sum()), dtype='float32')
    if weight_censored:
        w[feat.loc[mask, 'cens'].values == 1] = 0.5
    return w


def _paired_store_ci(fr_a, fr_b, folds, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Store-clustered paired bootstrap CI (95%) for pooled WAPE(a) -
    WAPE(b), in percentage points. Negative = arm a better."""
    sp = np.concatenate([fr_a[f]['sp'] for f in folds])
    for f in folds:  # both arms must be evaluated on identical rows
        assert np.array_equal(fr_a[f]['sp'], fr_b[f]['sp'])
        assert np.array_equal(fr_a[f]['y'], fr_b[f]['y'])
    y = np.concatenate([fr_a[f]['y'] for f in folds])
    pa = np.concatenate([fr_a[f]['pred'] for f in folds])
    pb = np.concatenate([fr_b[f]['pred'] for f in folds])
    g = pd.DataFrame({'store': (sp // 10000).astype('int64'),
                      'ea': np.abs(y - pa), 'eb': np.abs(y - pb),
                      'y': y}).groupby('store').sum()
    A, Bs, Y = g['ea'].values, g['eb'].values, g['y'].values
    point = (A.sum() / Y.sum() - Bs.sum() / Y.sum()) * 100
    rng = np.random.default_rng(seed)
    n = len(A)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        ys = Y[idx].sum()
        diffs[i] = (A[idx].sum() / ys - Bs[idx].sum() / ys) * 100
    return {'point_pp': round(float(point), 3),
            'ci95_lo_pp': round(float(np.percentile(diffs, 2.5)), 3),
            'ci95_hi_pp': round(float(np.percentile(diffs, 97.5)), 3),
            'n_stores': int(n), 'B': B, 'cluster': 'store'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=1500,
                    help='stratified subsample size used by the accuracy ablation')
    ap.add_argument('--out', default='output/multistep_validation',
                    help='output directory under the repo root')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[1/5] Load (n_sp={args.n_sp}); recovery/features are per-fold")
    train, _ev = load_data(n_sp=args.n_sp)
    dates = sorted(train['dt'].unique())
    warmup_date, specs = _fold_specs(dates)
    n_clusters = min(50, max(2, train['sp'].nunique() // 10))
    print(f"  SPs={train['sp'].nunique():,} folds={[s['fold'] for s in specs]}")

    frames = {a: {} for a in ['A0_recursive', 'A1_direct',
                              'A2_recursive_tweedie', 'A5_direct_base',
                              'A4a_demrec_target', 'A4b_tobit_objective']}
    runtimes = {k: 0.0 for k in ['recursive', 'direct', 'a4a', 'a4b']}
    fcols_ref = None

    for s in specs:
        f, T, val_days = s['fold'], s['train_end'], s['val_days']
        print(f"\n[2/5] Fold {f}: origin {pd.Timestamp(T).date()}, "
              f"val {pd.Timestamp(val_days[0]).date()}..")

        # -- fold-causal recovery + features --
        rec_f = apply_recovery(train[train['dt'] <= T].copy(), 'heuristic_imr')
        fb = FeatureBuilder(n_clusters=n_clusters, seed=args.seed).fit(rec_f)
        feat = fb.transform(rec_f)
        fcols = fb.feature_cols(feat)
        if fcols_ref is None:
            fcols_ref = fcols
        assert fcols == fcols_ref, 'feature set differs across folds'
        val_raw = train[train['dt'].isin(val_days)][
            [c for c in train.columns if c not in ('cens', 'dem_rec')]].copy()

        # -- nested split inside the fold-training window --
        days_f = [d for d in dates if warmup_date < d <= T]
        inner_val_days = days_f[-INNER_VAL_DAYS:]
        inner_T = days_f[-INNER_VAL_DAYS - 1]
        outer_mask = (feat['dt'] > warmup_date) & (feat['dt'] <= T)
        in_tr_mask = (feat['dt'] > warmup_date) & (feat['dt'] <= inner_T)
        in_vl_mask = feat['dt'].isin(inner_val_days)

        # ---- recursive family (A0/A2 from one model set + one pass) ----
        t0 = time.time()
        rec_models = _nested_fit(
            REC_OBJS,
            feat.loc[in_tr_mask, fcols].values,
            feat.loc[in_tr_mask, 'sale_amount'].values,
            _weights(feat, in_tr_mask),
            feat.loc[in_vl_mask, fcols].values,
            feat.loc[in_vl_mask, 'sale_amount'].values,
            feat.loc[outer_mask, fcols].values,
            feat.loc[outer_mask, 'sale_amount'].values,
            _weights(feat, outer_mask))
        preds, val_sorted = predict_eval_recursive(fb, rec_models, fcols,
                                                   rec_f, val_raw)
        runtimes['recursive'] += time.time() - t0
        y = val_sorted['sale_amount'].values.astype('float64')
        sp_v, dt_v = val_sorted['sp'].values, val_sorted['dt'].values
        frames['A0_recursive'][f] = {
            'sp': sp_v, 'dt': dt_v, 'y': y,
            'pred': np.mean([preds['mae'], preds['huber'], preds['q50']],
                            axis=0)}
        frames['A2_recursive_tweedie'][f] = {
            'sp': sp_v, 'dt': dt_v, 'y': y,
            'pred': np.mean([preds['mae'], preds['huber'], preds['q50'],
                             preds['tweedie']], axis=0)}

        # ---- direct family (A1/A5 base from one model set) ----
        t0 = time.time()
        table = build_direct_table(feat, fcols, horizons=7)
        dcols = direct_feature_cols(fcols)
        din_tr = (table['dt'] > warmup_date) & (table['dt'] <= inner_T)
        dout = (table['dt'] > warmup_date) & (table['dt'] <= T)
        X_iv, meta_iv = _origin_design(feat, fcols, inner_val_days[0],
                                       list(inner_val_days))
        y_iv = meta_iv.merge(feat[['sp', 'dt', 'sale_amount']],
                             on=['sp', 'dt'], how='left')['sale_amount'].values
        d_models = _nested_fit(
            REC_OBJS,
            table.loc[din_tr, dcols].fillna(0).values,
            table.loc[din_tr, 'sale_amount'].values,
            _weights(table, din_tr),
            X_iv, y_iv,
            table.loc[dout, dcols].fillna(0).values,
            table.loc[dout, 'sale_amount'].values,
            _weights(table, dout))
        del table
        d_preds, d_sorted = predict_eval_direct(fb, d_models, fcols,
                                                rec_f, val_raw)
        runtimes['direct'] += time.time() - t0
        assert np.array_equal(d_sorted['sp'].values, sp_v)
        assert np.array_equal(d_sorted['dt'].values, dt_v)
        frames['A1_direct'][f] = {
            'sp': sp_v, 'dt': dt_v, 'y': y,
            'pred': np.mean([d_preds['mae'], d_preds['huber'],
                             d_preds['q50']], axis=0)}
        frames['A5_direct_base'][f] = {
            'sp': sp_v, 'dt': dt_v, 'y': y,
            'pred': np.mean([d_preds['mae'], d_preds['huber'],
                             d_preds['q50'], d_preds['tweedie']], axis=0)}

        # ---- A4a diagnostic: recovered-demand target, recursive ----
        t0 = time.time()
        a4a_models = _nested_fit(
            POINT_OBJS,
            feat.loc[in_tr_mask, fcols].values,
            feat.loc[in_tr_mask, 'dem_rec'].values,
            _weights(feat, in_tr_mask, weight_censored=False),
            feat.loc[in_vl_mask, fcols].values,
            feat.loc[in_vl_mask, 'dem_rec'].values,
            feat.loc[outer_mask, fcols].values,
            feat.loc[outer_mask, 'dem_rec'].values,
            _weights(feat, outer_mask, weight_censored=False))
        p4a, vs4a = predict_eval_recursive(fb, a4a_models, fcols,
                                           rec_f, val_raw)
        runtimes['a4a'] += time.time() - t0
        frames['A4a_demrec_target'][f] = {
            'sp': vs4a['sp'].values, 'dt': vs4a['dt'].values,
            'y': vs4a['sale_amount'].values.astype('float64'),
            'pred': np.mean([p4a['mae'], p4a['huber'], p4a['q50']], axis=0)}

        # ---- A4b diagnostic: Tobit objective, recursive ----
        t0 = time.time()
        sig_map = per_sp_sigma(train[train['dt'] <= T])
        cens_in = feat.loc[in_tr_mask, 'cens'].values == 1
        sg_in = feat.loc[in_tr_mask, 'sp'].map(sig_map).fillna(0.1).values
        cens_out = feat.loc[outer_mask, 'cens'].values == 1
        sg_out = feat.loc[outer_mask, 'sp'].map(sig_map).fillna(0.1).values
        tob_models = _nested_fit(
            {'tobit': {'metric': 'None'}},
            feat.loc[in_tr_mask, fcols].values,
            feat.loc[in_tr_mask, 'sale_amount'].values,
            None,
            feat.loc[in_vl_mask, fcols].values,
            feat.loc[in_vl_mask, 'sale_amount'].values,
            feat.loc[outer_mask, fcols].values,
            feat.loc[outer_mask, 'sale_amount'].values,
            None,
            fobj=make_tobit_l2_objective(cens_in, sg_in),
            fobj_outer=make_tobit_l2_objective(cens_out, sg_out))
        p4b, vs4b = predict_eval_recursive(fb, tob_models, fcols,
                                           rec_f, val_raw)
        runtimes['a4b'] += time.time() - t0
        frames['A4b_tobit_objective'][f] = {
            'sp': vs4b['sp'].values, 'dt': vs4b['dt'].values,
            'y': vs4b['sale_amount'].values.astype('float64'),
            'pred': p4b['tobit']}

    # ---- Recentering arms: fold-f calibrator fit on folds < f only ----
    print("\n[3/5] Recentering (calibrators from earlier folds only)")
    frames['A3_recursive_recentered'] = {}
    frames['A5_direct_tweedie_recentered'] = {}
    for base, out in (('A0_recursive', 'A3_recursive_recentered'),
                      ('A5_direct_base', 'A5_direct_tweedie_recentered')):
        for f in POOLED_FOLDS:
            earlier = [{'fold': g, 'sp': frames[base][g]['sp'],
                        'dt': frames[base][g]['dt'], 'y': frames[base][g]['y'],
                        'preds': {'ensemble': frames[base][g]['pred']}}
                       for g in range(1, f)]
            cal = learn_cv_scale_calibrator(earlier, pred_key='ensemble',
                                            **RECENTER_CFG)
            fr = frames[base][f]
            frames[out][f] = {**fr, 'pred': apply_scale_calibrator(
                fr['pred'], fr['sp'], fr['dt'], cal)}

    # ---- Pool folds 2-4, evaluate rule + gate ----
    print("\n[4/5] Pooled selection (folds 2-4)")
    arm_runtime = {'A0_recursive': runtimes['recursive'],
                   'A2_recursive_tweedie': runtimes['recursive'],
                   'A3_recursive_recentered': runtimes['recursive'],
                   'A1_direct': runtimes['direct'],
                   'A5_direct_tweedie_recentered': runtimes['direct'],
                   'A4a_demrec_target': runtimes['a4a'],
                   'A4b_tobit_objective': runtimes['a4b']}
    rows = []
    for arm in ELIGIBLE + DIAGNOSTIC:
        pool_y = np.concatenate([frames[arm][f]['y'] for f in POOLED_FOLDS])
        pool_p = np.concatenate([frames[arm][f]['pred'] for f in POOLED_FOLDS])
        per_fold = {f: wape(frames[arm][f]['y'], frames[arm][f]['pred'])
                    for f in POOLED_FOLDS}
        rows.append({'arm': arm,
                     'eligible': arm in ELIGIBLE,
                     'pooled_val_wape': wape(pool_y, pool_p),
                     'pooled_val_wpe': wpe(pool_y, pool_p),
                     **{f'wape_fold{f}': v for f, v in per_fold.items()},
                     'runtime_s': round(arm_runtime[arm], 1)})
    rdf = pd.DataFrame(rows)

    elig = rdf[rdf['eligible']].set_index('arm')
    candidate = elig['pooled_val_wape'].idxmin()
    inc = elig.loc[INCUMBENT]
    cand = elig.loc[candidate]
    g1 = bool(cand['pooled_val_wape'] < inc['pooled_val_wape'])
    fold_wins = sum(cand[f'wape_fold{f}'] < inc[f'wape_fold{f}']
                    for f in POOLED_FOLDS)
    g2 = bool(fold_wins >= 2)
    g3 = bool(abs(cand['pooled_val_wpe']) <= abs(inc['pooled_val_wpe']) + 1.0)
    g4 = bool(cand['runtime_s'] <= 3.0 * inc['runtime_s'])
    promoted = candidate if (candidate != INCUMBENT and g1 and g2 and g3 and g4) \
        else INCUMBENT
    rdf['candidate'] = (rdf['arm'] == candidate).astype(int)
    rdf['headline'] = (rdf['arm'] == promoted).astype(int)
    rdf.to_csv(out_dir / 'multistep_validation_results.csv', index=False)

    # Descriptive paired CI; the gate itself uses point estimates.
    ci = _paired_store_ci(frames[candidate], frames[INCUMBENT], POOLED_FOLDS) \
        if candidate != INCUMBENT else None

    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=SCRIPT_DIR, text=True).strip()
    except Exception:
        sha = 'unknown'
    meta = {
        'script': 'multistep_validation',
        'protocol': PROTOCOL_NAME,
        'git_sha': sha,
        'config': vars(args),
        'selection_rule': SELECTION_RULE,
        'promotion_gate': PROMOTION_GATE,
        'candidate': candidate,
        'gate': {'G1_pooled_wape': g1, 'G2_fold_wins': f'{fold_wins}/3',
                 'G2_pass': g2, 'G3_bias': g3, 'G4_runtime': g4},
        'candidate_minus_incumbent_wape_ci': ci,
        'headline_configuration': promoted,
        'incumbent': INCUMBENT,
        'runtime_s_total': round(time.time() - t_start, 1),
    }
    (out_dir / 'multistep_validation_metadata.json').write_text(
        json.dumps(meta, indent=2))

    print("\n[5/5] Results")
    print(rdf[['arm', 'eligible', 'pooled_val_wape', 'pooled_val_wpe',
               'wape_fold2', 'wape_fold3', 'wape_fold4',
               'candidate', 'headline']].to_string(index=False))
    print(f"\ncandidate={candidate}  gates: G1={g1} G2={fold_wins}/3 "
          f"G3={g3} G4={g4}  -> headline configuration: {promoted}")
    if ci:
        print(f"paired store-clustered WAPE diff (candidate - incumbent): "
              f"{ci['point_pp']:+.3f} pp  95% CI "
              f"[{ci['ci95_lo_pp']:+.3f}, {ci['ci95_hi_pp']:+.3f}]  "
              f"({ci['n_stores']} stores, B={ci['B']})")
    print(f"Done in {meta['runtime_s_total']:.0f}s -> {out_dir}/")


if __name__ == '__main__':
    main()
