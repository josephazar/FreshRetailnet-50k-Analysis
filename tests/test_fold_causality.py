"""Regression test: the headline CV is fold-causal end to end.

The fold-causal guarantee is that nothing a validation week contains --
sales actuals included -- can reach the recovery step, the feature
builder, the boosting-round choice, or the model that predicts that week.
The strongest executable statement of that guarantee: perturbing the sales
actuals inside the LAST fold's validation window must leave every fold's
predictions bit-identical (only the targets change), because every fold
trains and fits strictly before that window.

A negative control makes the test non-vacuous: perturbing an EARLIER
fold's window must change a LATER fold's predictions, because later folds
legitimately train on that data.  A reverted fit-once-on-the-full-file
implementation fails the first assertion immediately.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline_leakfree import LGBM_OBJECTIVES
from scripts.headline_run import _fold_causal_recursive_cv
from scripts.multistep_validation import _fold_specs

OBJS = {'mae': LGBM_OBJECTIVES['mae']}


@pytest.fixture(scope='module')
def raw_panel():
    """Raw frame as load_data would return it (no cens/dem_rec columns).

    stock_hour6_22_cnt follows the dataset card's semantics: the number of
    OUT-of-stock hours between 6:00 and 22:00 (0 = fully available).
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    rows = []
    for s in range(1, 4):
        for p in range(1, 4):
            for dt in dates:
                stockout_hours = int(rng.integers(0, 6)) \
                    if rng.random() < 0.3 else 0
                base = float(rng.poisson(4 + s + p))
                rows.append({
                    'sp': s * 10000 + p,
                    'store_id': s, 'product_id': p, 'city_id': 1,
                    'management_group_id': 1,
                    'first_category_id': 10,
                    'second_category_id': 20 + (p % 2),
                    'third_category_id': 30 + (p % 3),
                    'dt': dt,
                    'sale_amount': base * (1 - stockout_hours / 32),
                    'stock_hour6_22_cnt': stockout_hours,
                    'discount': 0.1, 'holiday_flag': 0, 'activity_flag': 0,
                    'precpt': 0.5, 'avg_temperature': 15.0,
                    'avg_humidity': 60.0, 'avg_wind_level': 2.0,
                })
    return pd.DataFrame(rows)


def _run(raw):
    _metrics, _iters, frames = _fold_causal_recursive_cv(
        raw, OBJS, ('mae',), seed=42)
    return frames


def _perturb(raw, val_days):
    out = raw.copy()
    mask = out['dt'].isin(val_days)
    out.loc[mask, 'sale_amount'] = out.loc[mask, 'sale_amount'] * 3.0 + 7.0
    return out


def test_all_fold_predictions_invariant_to_last_window_actuals(raw_panel):
    dates = sorted(raw_panel['dt'].unique())
    _warmup, specs = _fold_specs(dates)
    base = _run(raw_panel)
    pert = _run(_perturb(raw_panel, specs[-1]['val_days']))
    for fb, fp in zip(base, pert):
        np.testing.assert_allclose(
            fb['preds']['mae'], fp['preds']['mae'],
            err_msg=f"fold {fb['fold']} predictions changed when the last "
                    f"validation window's actuals were perturbed")
    # the targets themselves must differ on the last fold (sanity)
    assert not np.allclose(base[-1]['y'], pert[-1]['y'])


def test_later_folds_do_learn_from_earlier_windows(raw_panel):
    dates = sorted(raw_panel['dt'].unique())
    _warmup, specs = _fold_specs(dates)
    base = _run(raw_panel)
    pert = _run(_perturb(raw_panel, specs[0]['val_days']))
    # fold 1 predicts its own (perturbed) window from earlier data only
    np.testing.assert_allclose(base[0]['preds']['mae'],
                               pert[0]['preds']['mae'])
    # a later fold trains on that window, so its model must differ
    assert not np.allclose(base[-1]['preds']['mae'],
                           pert[-1]['preds']['mae'])
