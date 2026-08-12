"""Leakage stress tests. Phase 1 blocker -- every test MUST pass."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.feature_builder import FeatureBuilder, predict_eval_recursive, predict_eval_single_origin


@pytest.fixture(scope='module')
def synth():
    np.random.seed(0)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    rows = []
    for s in range(1, 4):
        for p in range(1, 4):
            for dt in dates:
                rows.append({
                    'sp': s * 10000 + p,
                    'store_id': s, 'product_id': p, 'city_id': 1,
                    'management_group_id': 1,
                    'first_category_id': 10,
                    'second_category_id': 20 + (p % 2),
                    'third_category_id': 30 + (p % 3),
                    'dt': dt,
                    'sale_amount': float(np.random.poisson(4 + s + p)),
                    'stock_hour6_22_cnt': 16,
                    'discount': 0.1, 'holiday_flag': 0, 'activity_flag': 0,
                    'precpt': 0.5, 'avg_temperature': 15.0,
                    'avg_humidity': 60.0, 'avg_wind_level': 2.0,
                })
    df = pd.DataFrame(rows)
    df['cens'] = 0
    df['dem_rec'] = df['sale_amount'].astype('float32')
    train = df[df['dt'] < '2024-03-25'].copy()
    ev = df[df['dt'] >= '2024-03-25'].copy()
    return train, ev


def test_aggregate_invariance_to_eval_actuals(synth):
    """
    The critical test: aggregate features at eval rows must be IDENTICAL
    whether eval's sale_amount is the true value or garbage.
    """
    train, ev = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)

    ev_true = ev.copy()
    ev_mut = ev.copy()
    ev_mut['sale_amount'] = 9999.0
    ev_mut['dem_rec'] = 9999.0

    _, feat_true, _ = predict_eval_single_origin(fb, {}, [], train, ev_true)
    _, feat_mut, _ = predict_eval_single_origin(fb, {}, [], train, ev_mut)

    # Aggregate features must be identical (looked up from .fit())
    agg_cols = ['sp_m', 'pd_m', 'st_m', 'ct_m',
                'cat1_m', 'cat2_m', 'cat3_m', 'cat3_s',
                'dow_prof', 'dow_prof_r',
                'store_daily_vol', 'store_so_rate', 'cat2_daily_m',
                'cluster_m', 'cluster_s', 'temp_dev', 'hum_dev']
    for c in agg_cols:
        np.testing.assert_array_almost_equal(
            feat_true[c].values, feat_mut[c].values, decimal=5,
            err_msg=f"Feature '{c}' leaks eval actuals")


def test_recursive_is_stable_under_eval_target_shuffle(synth):
    """
    With recursive inference, shuffling eval sale_amount should NOT change
    the predictions (because predict_eval_recursive overwrites them anyway).
    """
    import lightgbm as lgb
    train, ev = synth

    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fc = fb.feature_cols(train_feat)

    # Train a tiny model
    params = {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 31,
              'learning_rate': 0.1, 'verbose': -1, 'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(train_feat[fc].fillna(0).values, label=train_feat['sale_amount'].values)
    mdl = lgb.train(params, d, num_boost_round=50)
    models = {'mae': mdl, 'huber': mdl, 'q50': mdl}

    ev_a = ev.copy()
    ev_b = ev.copy()
    ev_b['sale_amount'] = np.random.permutation(ev_b['sale_amount'].values)
    ev_b['dem_rec'] = ev_b['sale_amount'].astype('float32')

    preds_a, _ = predict_eval_recursive(fb, models, fc, train, ev_a)
    preds_b, _ = predict_eval_recursive(fb, models, fc, train, ev_b)

    np.testing.assert_array_almost_equal(
        preds_a['mae'], preds_b['mae'], decimal=4,
        err_msg="Recursive prediction leaks eval actuals")


def test_recursive_invariant_to_constant_garbage_actuals(synth):
    """Stronger than the shuffle test: replace eval actuals with a CONSTANT
    (a permutation preserves the marginal distribution; a constant does not).
    Predictions through the real recursive path must be identical."""
    import lightgbm as lgb
    train, ev = synth

    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fc = fb.feature_cols(train_feat)
    params = {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 31,
              'learning_rate': 0.1, 'verbose': -1, 'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(train_feat[fc].fillna(0).values, label=train_feat['sale_amount'].values)
    mdl = lgb.train(params, d, num_boost_round=50)
    models = {'mae': mdl, 'huber': mdl, 'q50': mdl}

    ev_a = ev.copy()
    ev_b = ev.copy()
    ev_b['sale_amount'] = 9999.0
    ev_b['dem_rec'] = 9999.0

    preds_a, _ = predict_eval_recursive(fb, models, fc, train, ev_a)
    preds_b, _ = predict_eval_recursive(fb, models, fc, train, ev_b)

    np.testing.assert_array_almost_equal(
        preds_a['mae'], preds_b['mae'], decimal=4,
        err_msg="Recursive predictions depend on eval-period actuals")


def test_leaky_pattern_fails_invariance(synth):
    """Demonstrate that concatenate-and-recompute fails the same
    invariance check the clean path passes.

    The prohibited pattern computes
    aggregate features over pd.concat([train, eval]) -- e.g. per-SP means via
    groupby().transform('mean') on the combined frame. This minimal
    reproduction of that pattern shows eval-row feature values CHANGE when
    eval actuals change, i.e. the construction leaks; the FeatureBuilder path
    (test_aggregate_invariance_to_eval_actuals) does not. Cited by the paper's
    leakage-verification paragraph.
    """
    train, ev = synth

    def leaky_sp_mean_feature(train_df, eval_df):
        combined = pd.concat([train_df, eval_df], ignore_index=True)
        combined['sp_m'] = combined.groupby('sp')['sale_amount'].transform('mean')
        return combined.loc[combined['dt'].isin(eval_df['dt'].unique()), 'sp_m'].values

    feat_true = leaky_sp_mean_feature(train, ev)
    ev_mut = ev.copy()
    ev_mut['sale_amount'] = 9999.0
    feat_mut = leaky_sp_mean_feature(train, ev_mut)

    assert not np.allclose(feat_true, feat_mut), (
        "expected concat+transform to leak eval actuals into "
        "eval-row features; if this now passes, the reproduction no longer "
        "matches the documented prohibited behavior"
    )


def test_horizon_independence(synth):
    """
    Predictions for eval day 7 must NOT depend on eval day 1-6 actuals.
    """
    import lightgbm as lgb
    train, ev = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fc = fb.feature_cols(train_feat)
    params = {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 31,
              'learning_rate': 0.1, 'verbose': -1, 'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(train_feat[fc].fillna(0).values, label=train_feat['sale_amount'].values)
    mdl = lgb.train(params, d, num_boost_round=50)
    models = {'mae': mdl, 'huber': mdl, 'q50': mdl}

    eval_dates = sorted(ev['dt'].unique())
    last_day = eval_dates[-1]

    ev_a = ev.copy()
    ev_b = ev.copy()
    # Mangle all eval days except the last
    mask = ev_b['dt'] != last_day
    ev_b.loc[mask, 'sale_amount'] = 99999.0
    ev_b.loc[mask, 'dem_rec'] = 99999.0

    preds_a, feat_a = predict_eval_recursive(fb, models, fc, train, ev_a)
    preds_b, feat_b = predict_eval_recursive(fb, models, fc, train, ev_b)

    # Last-day rows must have identical predictions
    mask_last_a = (feat_a['dt'] == last_day).values
    mask_last_b = (feat_b['dt'] == last_day).values
    np.testing.assert_array_almost_equal(
        preds_a['mae'][mask_last_a], preds_b['mae'][mask_last_b], decimal=4,
        err_msg="Last-day predictions depend on earlier-day actuals")
