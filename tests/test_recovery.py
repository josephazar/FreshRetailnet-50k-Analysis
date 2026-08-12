"""Sanity tests for src.recovery (heuristic_imr and tobit_mle)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.recovery import heuristic_imr, tobit_mle, _fit_tobit_mle_group


@pytest.fixture
def synth_censored():
    """Build a tiny panel where we know the latent demand.

    Stock availability: stock_hour6_22_cnt = 0 means uncensored, 16 = full stockout.
    Observed sale = min(latent_demand, effective_capacity).
    """
    np.random.seed(0)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    rows = []
    for s in [1, 2]:
        for p in [1, 2]:
            for dt in dates:
                latent = float(np.random.poisson(5 + s + p))
                # 30% of days have stockouts uniform in 0..16
                stockout_hrs = 0 if np.random.rand() > 0.3 else np.random.randint(1, 16)
                if stockout_hrs == 0:
                    sale = latent
                else:
                    # fraction of day stocked
                    stocked_frac = (16 - stockout_hrs) / 16
                    sale = min(latent, latent * stocked_frac + np.random.rand())
                rows.append({
                    'sp': s * 10000 + p, 'store_id': s, 'product_id': p,
                    'city_id': 1, 'management_group_id': 1,
                    'first_category_id': 10, 'second_category_id': 20,
                    'third_category_id': 30, 'dt': dt,
                    'sale_amount': float(sale), 'stock_hour6_22_cnt': int(stockout_hrs),
                    'latent': latent,
                })
    return pd.DataFrame(rows)


def test_heuristic_imr_does_not_decrease_demand(synth_censored):
    df = synth_censored[['sp', 'store_id', 'product_id', 'city_id',
                         'management_group_id', 'first_category_id',
                         'second_category_id', 'third_category_id', 'dt',
                         'sale_amount', 'stock_hour6_22_cnt']].copy()
    out = heuristic_imr(df, blend=0.5)
    # Recovery must be >= observed sales everywhere
    assert (out['dem_rec'] >= out['sale_amount'] - 1e-5).all(), \
        "heuristic_imr decreased demand below observed sales"
    # Average recovery should be >= average sale_amount
    assert out['dem_rec'].mean() >= out['sale_amount'].mean() - 1e-5


def test_tobit_mle_does_not_decrease_demand(synth_censored):
    df = synth_censored[['sp', 'store_id', 'product_id', 'city_id',
                         'management_group_id', 'first_category_id',
                         'second_category_id', 'third_category_id', 'dt',
                         'sale_amount', 'stock_hour6_22_cnt']].copy()
    out = tobit_mle(df)
    assert (out['dem_rec'] >= out['sale_amount'] - 1e-5).all()
    # At least some SP params should have been fit
    sp_params = out.attrs.get('tobit_sp_params', {})
    fit_count = sum(v is not None for v in sp_params.values())
    assert fit_count >= 1, "tobit_mle did not fit any SP-level MLEs"


def test_tobit_nonconvergence_falls_back(synth_censored, monkeypatch):
    """When the optimizer reports failure, tobit_mle must NOT use the stale
    iterate: SP fits route through the category fallback, category fits fall
    back to moment estimates, and the diagnostics record the counts."""
    import src.recovery as rec

    class _FailedResult:
        success = False
        x = np.array([1e6, 1e6])  # garbage iterate that must never be used

    monkeypatch.setattr(rec, 'minimize', lambda *a, **k: _FailedResult())

    df = synth_censored[['sp', 'store_id', 'product_id', 'city_id',
                         'management_group_id', 'first_category_id',
                         'second_category_id', 'third_category_id', 'dt',
                         'sale_amount', 'stock_hour6_22_cnt']].copy()
    out = rec.tobit_mle(df)

    diag = out.attrs['tobit_fit_diagnostics']
    assert diag['n_sp_nonconverged'] + diag['n_cat_nonconverged'] > 0, \
        "diagnostics did not record the forced non-convergence"
    # Recovery must still be finite and >= observed sales (moment fallback used)
    assert np.isfinite(out['dem_rec']).all()
    assert (out['dem_rec'] >= out['sale_amount'] - 1e-5).all()
    # The garbage iterate (1e6) must not have leaked into recovered demand
    assert out['dem_rec'].max() < 1e5, \
        "non-converged optimizer iterate leaked into recovered demand"


def test_tobit_mle_recovers_known_mu_on_synthetic():
    """When we know the truth, Tobit MLE should recover mu within tolerance."""
    rng = np.random.default_rng(0)
    true_mu, true_sigma = 5.0, 2.0
    n = 500
    d = rng.normal(true_mu, true_sigma, size=n)
    # Censor 30% of observations with threshold S=4
    cens = (d > 4).astype(int)
    y = np.where(d > 4, 4.0, d)  # observed value
    mu_hat, sigma_hat = _fit_tobit_mle_group(y, cens)
    # Tobit should recover mu closer to 5.0 than the uncensored mean (which
    # would be ~3.5 because all high-demand days are capped at 4).
    uncens_mean = float(np.mean(y[cens == 0]))
    assert abs(mu_hat - true_mu) < abs(uncens_mean - true_mu) + 0.5, \
        f"Tobit mu_hat={mu_hat:.2f} not better than naive uncensored mean {uncens_mean:.2f}"
