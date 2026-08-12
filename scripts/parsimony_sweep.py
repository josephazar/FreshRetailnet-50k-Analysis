"""Feature-parsimony sweep with training-fold-only ranking.

Protocol:
  - Feature ranking from TRAINING-FOLD gain importances only: mean gain
    share of the point-objective models (MAE/Huber/Q50) across CV folds
    1-3; fold 4 and the held-out test never contribute to the ranking.
  - Configurations: top-k for k in {5, 10, 20}, the full 117-feature set,
    and the pipeline's minimal configuration (MINIMAL_FCOLS from
    scripts/ablation.py; its actual length is reported).
  - Metrics per configuration: pooled multi-step validation WAPE/WPE over
    folds 2-4 under the common fold-causal regime (recovery + FeatureBuilder
    refit per fold, nested early stopping, recursive 3-member ensemble --
    the headline family), fit+predict runtime, and design-matrix memory.
  - Test WAPE/WPE are also reported for transparency: full-train refit per
    configuration, single evaluation pass. Transparency only -- nothing is
    selected on test.

Recentering is deliberately absent here: the sweep isolates the effect of
feature count on the raw ensemble; the scale calibrator is orthogonal
post-processing shared by any configuration.

Run (1,500-SP; expect ~1-2 h):
    ./venv/bin/python scripts/parsimony_sweep.py --n_sp 1500

Outputs (output/parsimony/):
    parsimony_ranking.csv   feature, mean gain share, rank (folds 1-3 only)
    parsimony_results.csv   one row per configuration
    parsimony_metadata.json provenance
"""
from __future__ import annotations

import argparse
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
from scripts.ablation import MINIMAL_FCOLS  # noqa: E402
from scripts.multistep_validation import (  # noqa: E402
    INNER_VAL_DAYS,
    PROTOCOL_NAME,
    _fold_specs,
    _weights,
)

POINT_OBJS = {k: LGBM_OBJECTIVES[k] for k in ('mae', 'huber', 'q50')}
K_GRID = [5, 10, 20]
RANKING_FOLDS = [1, 2, 3]
POOLED_FOLDS = [2, 3, 4]


def _nested_point_models(feat, fcols, in_tr, in_vl, outer):
    """Nested early stopping (amendment A2) for the point-objective family."""
    models = {}
    for name, obj in POINT_OBJS.items():
        params = {**LGBM_BASE_PARAMS, **obj}
        d_in = lgb.Dataset(feat.loc[in_tr, fcols].values,
                           label=feat.loc[in_tr, 'sale_amount'].values,
                           weight=_weights(feat, in_tr))
        d_iv = lgb.Dataset(feat.loc[in_vl, fcols].values,
                           label=feat.loc[in_vl, 'sale_amount'].values,
                           reference=d_in)
        inner = lgb.train(params, d_in, num_boost_round=1500,
                          valid_sets=[d_iv],
                          callbacks=[lgb.early_stopping(50),
                                     lgb.log_evaluation(0)])
        nround = max(50, inner.best_iteration)
        models[name] = lgb.train(
            params,
            lgb.Dataset(feat.loc[outer, fcols].values,
                        label=feat.loc[outer, 'sale_amount'].values,
                        weight=_weights(feat, outer)),
            num_boost_round=nround)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=1500)
    ap.add_argument('--out', default='output/parsimony')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[1/4] Load (n_sp={args.n_sp}); recovery/features are per-fold")
    train, ev = load_data(n_sp=args.n_sp)
    dates = sorted(train['dt'].unique())
    warmup_date, specs = _fold_specs(dates)
    n_clusters = min(50, max(2, train['sp'].nunique() // 10))

    # per-fold contexts (1,500-SP frames are small enough to cache)
    ctx = {}
    for s in specs:
        f, T, val_days = s['fold'], s['train_end'], s['val_days']
        rec_f = apply_recovery(train[train['dt'] <= T].copy(),
                               'heuristic_imr')
        fb = FeatureBuilder(n_clusters=n_clusters, seed=args.seed).fit(rec_f)
        feat = fb.transform(rec_f)
        days_f = [d for d in dates if warmup_date < d <= T]
        ctx[f] = {
            'rec': rec_f, 'fb': fb, 'feat': feat, 'T': T,
            'val_raw': train[train['dt'].isin(val_days)][
                [c for c in train.columns
                 if c not in ('cens', 'dem_rec')]].copy(),
            'outer': (feat['dt'] > warmup_date) & (feat['dt'] <= T),
            'in_tr': (feat['dt'] > warmup_date)
            & (feat['dt'] <= days_f[-INNER_VAL_DAYS - 1]),
            'in_vl': feat['dt'].isin(days_f[-INNER_VAL_DAYS:]),
        }
    fcols_full = ctx[1]['fb'].feature_cols(ctx[1]['feat'])

    print(f"[2/4] Ranking from folds {RANKING_FOLDS} gain importances "
          f"(fold 4 and test never used)")
    gain_share = np.zeros(len(fcols_full))
    n_models = 0
    for f in RANKING_FOLDS:
        c = ctx[f]
        models = _nested_point_models(c['feat'], fcols_full,
                                      c['in_tr'], c['in_vl'], c['outer'])
        for mdl in models.values():
            g = mdl.feature_importance('gain').astype('float64')
            tot = g.sum()
            if tot > 0:
                gain_share += g / tot
                n_models += 1
    gain_share /= max(n_models, 1)
    ranking = pd.DataFrame({'feature': fcols_full,
                            'mean_gain_share': gain_share})
    ranking = ranking.sort_values('mean_gain_share', ascending=False)
    ranking['rank'] = range(1, len(ranking) + 1)
    ranking.to_csv(out_dir / 'parsimony_ranking.csv', index=False)

    minimal = [c for c in MINIMAL_FCOLS if c in fcols_full]
    configs = [(f'top{k}', list(ranking['feature'].head(k))) for k in K_GRID]
    configs += [(f'minimal{len(minimal)}', minimal),
                (f'full{len(fcols_full)}', fcols_full)]

    print(f"[3/4] Sweep: {len(configs)} configurations x folds "
          f"{POOLED_FOLDS} + one test pass each")
    # deployment-path context for the transparency test pass (shared by all
    # configurations)
    rec_full = apply_recovery(train.copy(), 'heuristic_imr')
    fb_full = FeatureBuilder(n_clusters=n_clusters,
                             seed=args.seed).fit(rec_full)
    feat_full = fb_full.transform(rec_full)
    days_full = [d for d in dates if d > warmup_date]
    full_ctx = {
        'outer': feat_full['dt'] > warmup_date,
        'in_tr': (feat_full['dt'] > warmup_date)
        & (feat_full['dt'] <= days_full[-INNER_VAL_DAYS - 1]),
        'in_vl': feat_full['dt'].isin(days_full[-INNER_VAL_DAYS:]),
    }
    rows = []
    for cname, cols in configs:
        t0 = time.time()
        ys, ps, per_fold = [], [], {}
        xbytes = 0
        for f in POOLED_FOLDS:
            c = ctx[f]
            models = _nested_point_models(c['feat'], cols,
                                          c['in_tr'], c['in_vl'], c['outer'])
            xbytes = max(xbytes, int(c['feat'].loc[c['outer'], cols]
                                     .values.nbytes))
            preds, val_sorted = predict_eval_recursive(
                c['fb'], models, cols, c['rec'], c['val_raw'])
            y = val_sorted['sale_amount'].values.astype('float64')
            p = np.mean([preds['mae'], preds['huber'], preds['q50']], axis=0)
            ys.append(y)
            ps.append(p)
            per_fold[f] = wape(y, p)
        y_all, p_all = np.concatenate(ys), np.concatenate(ps)

        # transparency-only test pass: full-train deployment path
        fmodels = _nested_point_models(feat_full, cols, full_ctx['in_tr'],
                                       full_ctx['in_vl'], full_ctx['outer'])
        tpreds, tsorted = predict_eval_recursive(fb_full, fmodels, cols,
                                                 rec_full, ev)
        ty = tsorted['sale_amount'].values.astype('float64')
        tp = np.mean([tpreds['mae'], tpreds['huber'], tpreds['q50']], axis=0)

        row = {'config': cname, 'k': len(cols),
               'pooled_val_wape': wape(y_all, p_all),
               'pooled_val_wpe': wpe(y_all, p_all),
               **{f'wape_fold{f}': per_fold[f] for f in POOLED_FOLDS},
               'test_wape': wape(ty, tp), 'test_wpe': wpe(ty, tp),
               'design_matrix_mb': round(xbytes / 1e6, 1),
               'runtime_s': round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  {cname}: pooled val WAPE {row['pooled_val_wape']:.2f} "
              f"WPE {row['pooled_val_wpe']:+.2f} | test WAPE "
              f"{row['test_wape']:.2f} ({row['runtime_s']:.0f}s)")

    print("[4/4] Results")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'parsimony_results.csv', index=False)
    print(df.to_string(index=False))

    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                         capture_output=True, text=True).stdout.strip()
    meta = {'script': 'parsimony_sweep', 'git_sha': sha, 'args': vars(args),
            'protocol': PROTOCOL_NAME,
            'ranking_basis': 'mean gain share, MAE/Huber/Q50 outer models, '
                             f'folds {RANKING_FOLDS} only',
            'family': 'recursive 3-member ensemble (headline family), '
                      'no recentering (orthogonal post-processing)',
            'test_note': 'test columns are transparency only; nothing is '
                         'selected on test',
            'runtime_s_total': round(time.time() - t_start, 1)}
    (out_dir / 'parsimony_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"Done in {meta['runtime_s_total']:.0f}s -> {out_dir}/")


if __name__ == '__main__':
    main()
