"""Tests for the direct multi-horizon (horizon-as-feature) forecasting path."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.direct_forecast import (
    HORIZON_COL,
    build_direct_table,
    classify_feature_cols,
    direct_feature_cols,
    predict_eval_direct,
    retrain_direct_final,
    train_direct_cv,
)
from src.feature_builder import FeatureBuilder


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


def test_feature_classification_is_complete(synth):
    train, _ = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)
    feat = fb.transform(train)
    fcols = fb.feature_cols(feat)
    hist, tday = classify_feature_cols(fcols)
    assert len(hist) + len(tday) == len(fcols)
    assert not set(hist) & set(tday)
    # the real pipeline's 117 features split 68/49; the synthetic panel
    # produces the same families, so both groups must be non-trivial
    assert len(hist) > 30 and len(tday) > 20


def test_direct_table_h1_matches_baseline_design(synth):
    train, _ = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)
    feat = fb.transform(train)
    fcols = fb.feature_cols(feat)
    table = build_direct_table(feat, fcols, horizons=3)
    h1 = table[table[HORIZON_COL] == 1]
    assert len(h1) == len(feat)
    feat_sorted = feat.sort_values(['sp', 'dt']).reset_index(drop=True)
    np.testing.assert_array_almost_equal(
        h1[fcols].values, feat_sorted[fcols].values, decimal=5,
        err_msg='h=1 direct design must equal the one-step baseline design')


def test_direct_table_history_is_shifted(synth):
    train, _ = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)
    feat = fb.transform(train).sort_values(['sp', 'dt']).reset_index(drop=True)
    fcols = fb.feature_cols(feat)
    table = build_direct_table(feat, fcols, horizons=3)
    sp = feat['sp'].iloc[0]
    day = feat[feat['sp'] == sp]['dt'].iloc[40]
    base_row = feat[(feat['sp'] == sp) & (feat['dt'] == day)]
    h3_row = table[(table['sp'] == sp) & (table['dt'] == day)
                   & (table[HORIZON_COL] == 3)]
    # history of the h=3 sample for target `day` equals the h=1 history
    # two days earlier (information through day-3)
    earlier = feat[(feat['sp'] == sp)
                   & (feat['dt'] == day - pd.Timedelta(days=2))]
    assert float(h3_row['s_l1'].iloc[0]) == float(earlier['s_l1'].iloc[0])
    # target-day columns are NOT shifted
    assert float(h3_row['dow_sin'].iloc[0]) == float(base_row['dow_sin'].iloc[0])


def _tiny_models(train_feat, fcols, fb):
    import lightgbm as lgb
    objs = {'mae': {'objective': 'regression_l1', 'metric': 'mae'}}
    base = {'num_leaves': 15, 'learning_rate': 0.1, 'verbose': -1,
            'n_jobs': 1, 'seed': 42}
    _, bi, frames = train_direct_cv(train_feat, fcols, objs, base,
                                    num_boost_round=30, early_stopping=10)
    final = retrain_direct_final(train_feat, fcols, objs, base, bi)
    return final, frames


def test_direct_predictions_invariant_to_eval_actuals(synth):
    """The strongest property of the direct path: evaluation actuals cannot
    influence predictions at all -- not even through write-back."""
    train, ev = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fcols = fb.feature_cols(train_feat)
    final, _ = _tiny_models(train_feat, fcols, fb)

    ev_a = ev.copy()
    ev_b = ev.copy()
    ev_b['sale_amount'] = 9999.0
    ev_b['dem_rec'] = 9999.0

    pa, sa = predict_eval_direct(fb, final, fcols, train, ev_a)
    pb, sb = predict_eval_direct(fb, final, fcols, train, ev_b)
    np.testing.assert_array_equal(
        pa['mae'], pb['mae'],
        err_msg='direct predictions depend on eval actuals')
    assert len(pa['mae']) == len(sa)


def test_direct_cv_frames_shape(synth):
    train, _ = synth
    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fcols = fb.feature_cols(train_feat)
    _, frames = _tiny_models(train_feat, fcols, fb)
    assert len(frames) == 4
    for fr in frames:
        n_sp = train['sp'].nunique()
        assert len(fr['y']) == n_sp * 7
        assert 'ensemble' in fr['preds']
        assert np.isfinite(fr['preds']['ensemble']).all()
