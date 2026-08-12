"""Tests for the Tobit-style censored LightGBM objective."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy import stats

from src.censored_objectives import make_tobit_l2_objective, per_sp_sigma


class _FakeDataset:
    def __init__(self, label):
        self._label = np.asarray(label, dtype=float)

    def get_label(self):
        return self._label


def _negloglik(mu, y, cens, sigma):
    """Reference per-row Tobit negative log-likelihood (sigma^2-rescaled)."""
    nll = 0.0
    for m, yi, c, s in zip(mu, y, cens, sigma):
        z = (yi - m) / s
        if c:
            nll += -s * s * stats.norm.logsf(z)
        else:
            nll += 0.5 * (yi - m) ** 2
    return nll


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    n = 40
    y = rng.gamma(3.0, 2.0, size=n)
    cens = rng.random(n) < 0.4
    sigma = np.full(n, 1.7)
    mu = y + rng.normal(0, 1.0, size=n)

    fobj = make_tobit_l2_objective(cens, sigma)
    grad, hess = fobj(mu, _FakeDataset(y))

    eps = 1e-5
    for i in range(0, n, 7):
        mu_p, mu_m = mu.copy(), mu.copy()
        mu_p[i] += eps
        mu_m[i] -= eps
        fd = (_negloglik(mu_p, y, cens, sigma)
              - _negloglik(mu_m, y, cens, sigma)) / (2 * eps)
        assert abs(grad[i] - fd) < 1e-3, \
            f"row {i} (cens={cens[i]}): grad {grad[i]:.6f} vs FD {fd:.6f}"
    assert (hess > 0).all(), "Tobit hessian must be positive (convex in mu)"


def test_censored_gradient_pushes_predictions_up():
    """At mu == y (prediction equals the censoring bound), the censored
    gradient must be negative (loss decreases as mu increases): observed
    sales are a lower bound, so the model is pushed above them."""
    y = np.array([5.0, 5.0])
    cens = np.array([True, False])
    sigma = np.array([2.0, 2.0])
    fobj = make_tobit_l2_objective(cens, sigma)
    grad, _ = fobj(np.array([5.0, 5.0]), _FakeDataset(y))
    assert grad[0] < 0, "censored row at the bound must be pushed upward"
    assert grad[1] == 0, "uncensored row at the target has zero gradient"


def test_tobit_objective_recovers_latent_mean_in_training():
    """End-to-end: train LightGBM with the Tobit objective on synthetic
    censored data; it must estimate latent demand less negatively-biased
    than the same model trained on censored targets with plain L2."""
    lgb = pytest.importorskip('lightgbm')
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(0, 1, size=(n, 3))
    latent = 5.0 + 1.5 * x[:, 0] + rng.normal(0, 1.0, size=n)
    cap = np.where(rng.random(n) < 0.45, latent - np.abs(rng.normal(1.0, 0.5, n)),
                   np.inf)
    y_obs = np.minimum(latent, cap)
    cens = np.isfinite(cap) & (y_obs < latent)
    sigma = np.full(n, 1.0)

    params = {'num_leaves': 15, 'learning_rate': 0.1, 'verbose': -1,
              'n_jobs': 1, 'seed': 42}
    d = lgb.Dataset(x, label=y_obs)
    mdl_tobit = lgb.train(
        {**params, 'objective': make_tobit_l2_objective(cens, sigma)},
        d, num_boost_round=150)
    mdl_l2 = lgb.train({**params, 'objective': 'regression'},
                       lgb.Dataset(x, label=y_obs), num_boost_round=150)

    bias_tobit = float(np.mean(mdl_tobit.predict(x) - latent))
    bias_l2 = float(np.mean(mdl_l2.predict(x) - latent))
    assert bias_l2 < -0.1, "sanity: plain L2 on censored targets is biased down"
    assert bias_tobit > bias_l2 + 0.05, \
        f"Tobit head must reduce downward bias (tobit {bias_tobit:.3f} vs l2 {bias_l2:.3f})"


def test_per_sp_sigma_floor():
    import pandas as pd
    df = pd.DataFrame({'sp': [1, 1, 2], 'sale_amount': [3.0, 3.0, 4.0],
                       'cens': [0, 0, 0]})
    s = per_sp_sigma(df)
    assert s[1] >= 0.10 and s[2] >= 0.10
