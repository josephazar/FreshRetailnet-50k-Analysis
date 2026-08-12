"""Golden tests for FeatureBuilder on a minimal synthetic panel."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.feature_builder import FeatureBuilder


@pytest.fixture
def small_panel():
    np.random.seed(0)
    dates = pd.date_range('2024-01-01', periods=45, freq='D')
    rows = []
    for store in [1, 2]:
        for prod in [1, 2]:
            for dt in dates:
                rows.append({
                    'sp': store * 10000 + prod,
                    'store_id': store, 'product_id': prod, 'city_id': 1,
                    'management_group_id': 1,
                    'first_category_id': 10,
                    'second_category_id': 20 + prod,
                    'third_category_id': 30 + prod,
                    'dt': dt, 'sale_amount': float(5 + store + prod + (dt.dayofweek == 5)),
                    'stock_hour6_22_cnt': 0, 'discount': 0.1, 'holiday_flag': 0,
                    'activity_flag': 0, 'precpt': 0.5, 'avg_temperature': 15.0,
                    'avg_humidity': 60.0, 'avg_wind_level': 2.0,
                })
    df = pd.DataFrame(rows)
    df['cens'] = 0
    df['dem_rec'] = df['sale_amount'].astype('float32')
    return df


def test_fit_then_transform_produces_features(small_panel):
    fb = FeatureBuilder(n_clusters=2).fit(small_panel)
    out = fb.transform(small_panel)
    assert 'sp_m' in out.columns
    assert 'cluster_m' in out.columns
    assert 'dow_prof' in out.columns
    assert out.shape[0] == small_panel.shape[0]
    fc = fb.feature_cols(out)
    assert len(fc) > 80, f"expected 80+ features, got {len(fc)}"


def test_group_stats_match_training(small_panel):
    fb = FeatureBuilder(n_clusters=2).fit(small_panel)
    expected_sp_m = small_panel.groupby('sp')['sale_amount'].mean().to_dict()
    assert fb.group_stats_['sp']['m'] == expected_sp_m


def test_transform_raises_without_fit():
    fb = FeatureBuilder(n_clusters=2)
    with pytest.raises(RuntimeError):
        fb.transform(pd.DataFrame())


def test_cluster_assignment_deterministic(small_panel):
    fb1 = FeatureBuilder(n_clusters=2, seed=42).fit(small_panel)
    fb2 = FeatureBuilder(n_clusters=2, seed=42).fit(small_panel)
    assert fb1.sp_cluster_map_ == fb2.sp_cluster_map_


def test_duplicate_sp_dt_rows_handled(small_panel):
    """Duplicate (sp, dt) rows would crash predict_eval_recursive's per-day
    writeback (non-unique Series index); the load-time guard must dedupe."""
    import lightgbm as lgb
    from pipeline_leakfree import validate_unique_sp_dt
    from src.feature_builder import predict_eval_recursive

    train = small_panel[small_panel['dt'] < '2024-02-08'].copy()
    ev = small_panel[small_panel['dt'] >= '2024-02-08'].copy()
    # Inject a duplicated eval row
    ev_dup = pd.concat([ev, ev.iloc[[0]]], ignore_index=True)
    assert ev_dup.duplicated(['sp', 'dt']).sum() == 1

    ev_clean = validate_unique_sp_dt(ev_dup, 'eval')
    assert ev_clean.duplicated(['sp', 'dt']).sum() == 0
    assert len(ev_clean) == len(ev)

    # The deduped frame must run through the real recursive path end to end
    fb = FeatureBuilder(n_clusters=2).fit(train)
    train_feat = fb.transform(train)
    fc = fb.feature_cols(train_feat)
    params = {'objective': 'regression_l1', 'num_leaves': 15,
              'learning_rate': 0.1, 'verbose': -1, 'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(train_feat[fc].fillna(0).values,
                    label=train_feat['sale_amount'].values)
    mdl = lgb.train(params, d, num_boost_round=10)
    preds, ev_sorted = predict_eval_recursive(fb, {'mae': mdl}, fc, train, ev_clean)
    assert np.isfinite(preds['mae']).all()
    assert len(preds['mae']) == len(ev_sorted)
