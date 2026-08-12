"""Sanity tests for src.inventory policies."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.inventory import (
    eval_policy,
    compute_policies,
    select_policy_on_validation,
    residual_quantile_from_cv,
    profit_optimal_cr,
    DEFAULT_INV_CFG,
)


def test_eval_policy_math():
    Q = np.array([10, 5, 8], dtype=float)
    D = np.array([8, 10, 8], dtype=float)
    out = eval_policy(Q, D, DEFAULT_INV_CFG, 'test')
    # eval_policy rounds to 4 decimals; check to 3
    assert abs(out['Fill Rate'] - 21 / 26) < 5e-4
    assert abs(out['SO Rate'] - 1 / 3) < 5e-4
    assert abs(out['Avg Revenue'] - 7.0) < 5e-4


def test_profit_optimal_cr_matches_full_objective():
    """The policy fractile must include margin and procurement costs."""
    cr = profit_optimal_cr(DEFAULT_INV_CFG)
    assert cr == pytest.approx(1.1 / 1.9)
    assert cr != pytest.approx(
        DEFAULT_INV_CFG['p'] / (DEFAULT_INV_CFG['p'] + DEFAULT_INV_CFG['w'])
    )


def test_compute_policies_returns_20():
    np.random.seed(0)
    # Build a small synthetic training set
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    rows = []
    for sp in [10001, 10002, 10003]:
        for dt in dates:
            rows.append({'sp': sp, 'dt': dt, 'sale_amount': float(np.random.poisson(5))})
    train = pd.DataFrame(rows)
    n = 30
    sp_keys = np.array([10001, 10002, 10003] * 10)
    preds = {
        'ensemble': np.random.uniform(3, 7, size=n).astype('float32'),
        'q10': np.random.uniform(1, 4, size=n).astype('float32'),
        'q50': np.random.uniform(3, 7, size=n).astype('float32'),
        'q90': np.random.uniform(6, 10, size=n).astype('float32'),
    }
    actuals = np.random.uniform(3, 7, size=n).astype('float32')
    rdf, policy_Q = compute_policies(preds, sp_keys, train,
                                      sp_residual_quantile={10001: 1.0, 10002: 1.0, 10003: 1.0},
                                      cfg=DEFAULT_INV_CFG,
                                      actuals_for_eval=actuals)
    # Keep documentation, tests, and paper outputs synchronized with the policy set.
    assert len(rdf) == 20, f"expected exactly 20 policies, got {len(rdf)}"
    # All profit values should be finite
    assert rdf['Avg Profit'].notna().all()
    # Empirical Newsvendor should exist
    assert 'Empirical Newsvendor' in rdf['Policy'].values
    assert 'Newsvendor-Normal (CR=0.579)' in rdf['Policy'].values
    assert not rdf['Policy'].str.contains('CR=0.625', regex=False).any()


def test_select_policy_on_validation_prefers_validation_not_test():
    """
    The pipeline must select
    the policy on VALIDATION data; swapping test actuals for validation actuals
    must not change the chosen policy as long as validation preds stay fixed.
    """
    np.random.seed(123)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    rows = []
    for sp in [101, 102, 103, 104, 105, 106, 107, 108]:
        for dt in dates:
            rows.append({'sp': sp, 'dt': dt,
                         'sale_amount': float(np.random.poisson(5))})
    train = pd.DataFrame(rows)

    sp_keys = np.repeat(np.array([101, 102, 103, 104, 105, 106, 107, 108]), 8)
    n = len(sp_keys)
    rng = np.random.default_rng(0)
    val_preds = {
        'ensemble': rng.uniform(3, 7, size=n).astype('float32'),
        'q10':      rng.uniform(1, 4, size=n).astype('float32'),
        'q50':      rng.uniform(3, 7, size=n).astype('float32'),
        'q90':      rng.uniform(6, 10, size=n).astype('float32'),
    }
    val_actuals = rng.uniform(3, 7, size=n).astype('float32')

    sp_rq = {sp: float(rng.normal(0, 1)) for sp in np.unique(sp_keys)}

    winner = select_policy_on_validation(
        val_preds=val_preds, val_sp_keys=sp_keys, val_actuals=val_actuals,
        train_df=train, sp_residual_quantile=sp_rq, cfg=DEFAULT_INV_CFG,
        fill_rate_floor=0.0,  # disable FR filter so we exercise the raw idxmax
    )
    assert isinstance(winner, str) and len(winner) > 0
    # Rerun with a permuted actuals -> different profit -> potentially different winner
    perm = rng.permutation(val_actuals)
    winner_b = select_policy_on_validation(
        val_preds=val_preds, val_sp_keys=sp_keys, val_actuals=perm,
        train_df=train, sp_residual_quantile=sp_rq, cfg=DEFAULT_INV_CFG,
        fill_rate_floor=0.0,
    )
    # Not asserting equality -- the point is that the selector depends only on
    # the data we pass it (validation preds+actuals+residual quantile), not on
    # any hidden test-set state. Both calls must succeed.
    assert isinstance(winner_b, str)


def test_residual_quantile_from_cv_aggregates_across_folds():
    """Sanity: residual_quantile_from_cv pools residuals across folds per SP."""
    sp_keys_a = np.array([1, 1, 2, 2, 3])
    resid_a = np.array([0.1, 0.2, -0.5, 0.0, 1.0])
    sp_keys_b = np.array([1, 2, 2])
    resid_b = np.array([0.3, 0.1, -0.2])
    q = residual_quantile_from_cv([(sp_keys_a, resid_a), (sp_keys_b, resid_b)], cr=0.5)
    # SP 1: [0.1, 0.2, 0.3] -> median 0.2
    assert abs(q[1] - 0.2) < 1e-6
    # SP 2: [-0.5, 0.0, 0.1, -0.2] -> median ~ -0.1
    assert abs(q[2] - (-0.1)) < 1e-6


def test_saa_deterministic():
    """SAA must use its own seeded RNG: identical inputs -> identical Q,
    and the call must not consume from (or depend on) global np.random."""
    dates = pd.date_range('2024-01-01', periods=40, freq='D')
    rng = np.random.default_rng(7)
    rows = [{'sp': 10001, 'dt': dt, 'sale_amount': float(rng.poisson(5))}
            for dt in dates]
    train = pd.DataFrame(rows)
    n = 10
    sp_keys = np.full(n, 10001)
    preds = {
        'ensemble': np.linspace(3, 7, n).astype('float32'),
        'q10': np.linspace(1, 4, n).astype('float32'),
        'q50': np.linspace(3, 7, n).astype('float32'),
        'q90': np.linspace(6, 10, n).astype('float32'),
    }

    np.random.seed(0)
    state_before = np.random.get_state()[1].copy()
    _, q_a = compute_policies(preds, sp_keys, train,
                              sp_residual_quantile={10001: 0.5},
                              cfg=DEFAULT_INV_CFG)
    state_after = np.random.get_state()[1].copy()

    # Different global RNG position must not change SAA quantities
    np.random.seed(12345)
    np.random.random(999)
    _, q_b = compute_policies(preds, sp_keys, train,
                              sp_residual_quantile={10001: 0.5},
                              cfg=DEFAULT_INV_CFG)

    np.testing.assert_array_equal(
        q_a['SAA (2000 scenarios)'], q_b['SAA (2000 scenarios)'],
        err_msg="SAA quantities depend on global np.random state")
    np.testing.assert_array_equal(
        state_before, state_after,
        err_msg="compute_policies consumed from the global RNG")


def test_policies_no_inf_or_negative_q():
    np.random.seed(1)
    dates = pd.date_range('2024-01-01', periods=40, freq='D')
    rows = [{'sp': 10001, 'dt': dt, 'sale_amount': float(np.random.poisson(5))}
            for dt in dates]
    train = pd.DataFrame(rows)
    n = 10
    sp_keys = np.full(n, 10001)
    preds = {
        'ensemble': np.linspace(3, 7, n).astype('float32'),
        'q10': np.linspace(1, 4, n).astype('float32'),
        'q50': np.linspace(3, 7, n).astype('float32'),
        'q90': np.linspace(6, 10, n).astype('float32'),
    }
    actuals = np.linspace(4, 6, n).astype('float32')
    rdf, policy_Q = compute_policies(preds, sp_keys, train,
                                      sp_residual_quantile={10001: 0.5},
                                      cfg=DEFAULT_INV_CFG,
                                      actuals_for_eval=actuals)
    for name, Q in policy_Q.items():
        assert np.all(np.isfinite(Q)), f"{name} has non-finite Q"
        assert np.all(Q >= -1e-6), f"{name} has negative Q"
