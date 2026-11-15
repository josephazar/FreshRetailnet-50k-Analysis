"""Pin the numerical equivalence of the two recursive-prediction paths.

benchmark_comparison._predict_recursive_single duplicates the day-by-day
recursion of src.feature_builder.predict_eval_recursive (delegation is blocked
by the xgboost DMatrix wrapper in predict_model). The paper claims the
benchmark runs under a protocol identical to the main pipeline; this test
locks the two implementations together so an edit to either cannot silently
desynchronize them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.feature_builder import FeatureBuilder, predict_eval_recursive


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


def test_single_model_recursion_matches_pipeline_path(synth):
    lgb = pytest.importorskip('lightgbm')
    bench = pytest.importorskip('benchmark_comparison')
    train, ev = synth

    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fc = fb.feature_cols(train_feat)
    params = {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 31,
              'learning_rate': 0.1, 'verbose': -1, 'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(train_feat[fc].fillna(0).values,
                    label=train_feat['sale_amount'].values)
    mdl = lgb.train(params, d, num_boost_round=50)

    p_single, ev_single = bench._predict_recursive_single(
        fb, mdl, 'lgbm', fc, train, ev
    )
    # With a single model named 'mae', the pipeline path writes back exactly
    # that model's predictions, so the two recursions see identical state.
    preds, ev_sorted = predict_eval_recursive(fb, {'mae': mdl}, fc, train, ev)

    pd.testing.assert_frame_equal(
        ev_single[['sp', 'dt']].reset_index(drop=True),
        ev_sorted[['sp', 'dt']].reset_index(drop=True),
    )
    np.testing.assert_allclose(
        p_single, preds['mae'], rtol=1e-5, atol=1e-6,
        err_msg=("benchmark _predict_recursive_single diverged from "
                 "src.feature_builder.predict_eval_recursive"),
    )


def test_single_model_recursion_matches_pipeline_path_with_stockouts():
    """Same lock-step check on a panel whose stock status varies, so the two
    recursions must also agree on how evaluation-week stock status is held
    (the constant-status fixture above cannot tell)."""
    lgb = pytest.importorskip('lightgbm')
    bench = pytest.importorskip('benchmark_comparison')
    from tests.stock_panel import flip_stock_status, make_stock_panel
    train, ev = make_stock_panel()

    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fc = fb.feature_cols(train_feat)
    params = {'objective': 'regression_l1', 'metric': 'mae', 'num_leaves': 31,
              'learning_rate': 0.1, 'verbose': -1, 'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(train_feat[fc].fillna(0).values,
                    label=train_feat['sale_amount'].values)
    mdl = lgb.train(params, d, num_boost_round=50)

    p_single, _ = bench._predict_recursive_single(fb, mdl, 'lgbm', fc, train, ev)
    preds, _ = predict_eval_recursive(fb, {'mae': mdl}, fc, train, ev)
    np.testing.assert_allclose(p_single, preds['mae'], rtol=1e-5, atol=1e-6)
    p_flip, _ = bench._predict_recursive_single(
        fb, mdl, 'lgbm', fc, train, flip_stock_status(ev))
    np.testing.assert_array_equal(
        p_single, p_flip,
        err_msg="benchmark recursion reads evaluation-week stock status")
