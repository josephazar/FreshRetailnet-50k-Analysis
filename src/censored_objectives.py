"""Censoring-aware training objectives for LightGBM.

Tobit-style loss: on stockout days, recorded sales y are a LOWER BOUND on
latent demand. Treating them as exact targets biases the model downward.
The Tobit likelihood replaces the squared error on censored rows with the
Gaussian survival term, whose gradient pushes predictions ABOVE the observed
lower bound instead of toward it.

Per-row negative log-likelihood (sigma fixed per row, estimated upstream):
    uncensored:  0.5 * ((y - mu)/sigma)^2
    censored:    -log(1 - Phi((y - mu)/sigma))

Gradients/Hessians w.r.t. mu, rescaled by sigma^2 so the uncensored part
matches plain L2 (keeps LightGBM's default learning-rate regime sensible):
    uncensored:  grad = mu - y                  hess = 1
    censored:    grad = -sigma * lam(z)         hess = lam(z) * (lam(z) - z)
where z = (y - mu)/sigma and lam(z) = phi(z) / (1 - Phi(z)) is the inverse
Mills ratio (computed via exp(logpdf - logsf) for numerical stability).
The censored Hessian is positive for all z, so the loss is convex in mu.

The model trained with this objective estimates LATENT DEMAND. Evaluated
against censored sales (the benchmark's ground truth) it is expected to
overshoot on stocked-out days -- that asymmetry is the point, and the
accuracy-ablation study reports it honestly.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def make_tobit_l2_objective(cens_mask: np.ndarray, sigma: np.ndarray):
    """Build a LightGBM fobj for Tobit-style censored regression.

    cens_mask : bool array aligned with the training rows (True = censored).
    sigma     : per-row positive scale (e.g. per-SP demand std, clipped).
    """
    cens_mask = np.asarray(cens_mask, dtype=bool)
    # Defensive second floor (0.05, below per_sp_sigma's 0.10 default) so a
    # caller-supplied sigma can never make grad/hess blow up.
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 0.05, None)

    def fobj(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y = dataset.get_label().astype(np.float64)
        mu = preds.astype(np.float64)
        grad = mu - y          # uncensored: plain (rescaled) L2
        hess = np.ones_like(grad)
        if cens_mask.any():
            z = (y[cens_mask] - mu[cens_mask]) / sigma[cens_mask]
            # Numerical guards, not tuning knobs: above z=8 the inverse Mills
            # ratio grows ~linearly and would explode the gradient; below
            # z=-30, logsf(z) ~ 0 and lam vanishes anyway.
            z = np.clip(z, -30.0, 8.0)
            # lam = phi(z) / (1 - Phi(z)), stable in the right tail
            lam = np.exp(stats.norm.logpdf(z) - stats.norm.logsf(z))
            grad[cens_mask] = -sigma[cens_mask] * lam
            hess[cens_mask] = np.clip(lam * (lam - z), 1e-6, None)
        return grad, hess

    return fobj


def mae_feval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
    """Early-stopping metric for custom-objective training (MAE vs labels)."""
    y = dataset.get_label()
    return 'mae', float(np.mean(np.abs(preds - y))), False


def per_sp_sigma(train_df, sp_col: str = 'sp', target_col: str = 'sale_amount',
                 floor: float = 0.10) -> dict:
    """Per-SP scale estimates from uncensored training rows (moment-based).

    Returns a {sp: sigma} dict (per store-product, NOT per row). To build the
    per-ROW sigma array the Tobit objective needs, map it onto the training
    rows: ``df['sp'].map(sigma_map).fillna(floor)`` -- SPs absent from the map
    (e.g. an all-censored history) get the same floor used here.
    """
    unc = train_df[train_df['cens'] == 0] if 'cens' in train_df.columns else train_df
    s = unc.groupby(sp_col)[target_col].std()
    return s.fillna(floor).clip(lower=floor).to_dict()
