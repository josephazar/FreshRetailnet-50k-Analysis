"""
Censored demand recovery methods.

Two explicitly labeled methods:
  - heuristic_imr: an IMR-inspired heuristic with a blending weight; this is
    not maximum-likelihood Tobit estimation.
  - tobit_mle: proper maximum-likelihood Tobit estimation per SP (or per
    cluster for SPs with few uncensored obs).

Time-weighted partial-stockout recovery and rolling-average full-stockout
recovery are shared stages used by both methods; only the final correction
differs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


OP = 16  # operating hours (6am-10pm)

# Hourly demand profile (16 operating hours: 6am-10pm)
HOURLY_PROFILE = np.array([
    0.08, 0.09, 0.10, 0.08,
    0.06, 0.05, 0.04, 0.04,
    0.05, 0.06, 0.07, 0.08,
    0.09, 0.08, 0.06, 0.06,
], dtype=np.float32)
HOURLY_PROFILE = HOURLY_PROFILE / HOURLY_PROFILE.sum()


def _time_weighted_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: scale partial-stockout sales by the fraction of demand
    profile that remained available. No leakage concern (row-local)."""
    so_hrs = df['stock_hour6_22_cnt'].values.astype(int)
    partial = (df['stock_hour6_22_cnt'] > 0) & (df['stock_hour6_22_cnt'] < OP)
    avail_weights = np.array(
        [HOURLY_PROFILE[:OP - n].sum() if n < OP else 0.01 for n in range(OP + 1)]
    )
    partial_idx = df.index[partial]
    partial_so = so_hrs[partial]
    recovered = (
        df.loc[partial_idx, 'sale_amount'].values
        / avail_weights[partial_so].clip(min=0.01)
    ).astype('float32')
    df.loc[partial_idx, 'dem_rec'] = recovered
    return df


def _rolling_avg_full_stockout(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Stage 2: full-stockout days get the rolling average of non-stockout
    days in the past window. Per-SP, causal by construction."""
    full = df['stock_hour6_22_cnt'] >= OP
    no_so = (df['cens'] == 0).astype(float)
    df['_s'] = df['sale_amount'] * no_so
    rs = df.groupby('sp')['_s'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    # rolling count of non-stockout days per SP
    rc = df.groupby('sp').apply(
        lambda g: no_so.loc[g.index].rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True).sort_index()
    avg_ns = rs / rc.clip(lower=1)
    df.loc[full, 'dem_rec'] = avg_ns[full]
    df.drop('_s', axis=1, inplace=True)
    return df


def heuristic_imr(
    df: pd.DataFrame,
    blend: float = 0.5,
    cap_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Apply the fast IMR-inspired recovery heuristic.

    Estimates per-SP mu/sigma from NON-CENSORED observations only
    (this is biased downward -- a true Tobit does joint MLE), applies
    IMR correction with a blending weight.

    Returns df with 'dem_rec' column populated. Only modifies censored rows
    beyond what Stage 1/2 already set.
    """
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['dem_rec'] = df['sale_amount'].astype(np.float64)

    df = _time_weighted_partial(df)
    df = _rolling_avg_full_stockout(df)

    # Stats from non-censored rows only (biased but fast)
    sp_stats = df[df['cens'] == 0].groupby('sp')['sale_amount'].agg(['mean', 'std', 'count'])
    sp_stats.columns = ['sp_mu', 'sp_sigma', 'sp_n']
    sp_stats['sp_sigma'] = sp_stats['sp_sigma'].fillna(0).clip(lower=0.05)
    cat_stats = df[df['cens'] == 0].groupby('second_category_id')['sale_amount'].agg(['mean', 'std'])
    cat_stats.columns = ['cat_mu', 'cat_sigma']
    cat_stats['cat_sigma'] = cat_stats['cat_sigma'].fillna(0).clip(lower=0.05)

    df = df.merge(sp_stats[['sp_mu', 'sp_sigma', 'sp_n']], left_on='sp',
                  right_index=True, how='left')
    df = df.merge(cat_stats[['cat_mu', 'cat_sigma']], left_on='second_category_id',
                  right_index=True, how='left')
    use_cat = (df['sp_n'].fillna(0) < 5)
    df.loc[use_cat, 'sp_mu'] = df.loc[use_cat, 'cat_mu']
    df.loc[use_cat, 'sp_sigma'] = df.loc[use_cat, 'cat_sigma']
    df['sp_mu'] = df['sp_mu'].fillna(df['sale_amount'].mean())
    df['sp_sigma'] = df['sp_sigma'].fillna(0.5)

    cens_mask = df['cens'] == 1
    S = df.loc[cens_mask, 'sale_amount'].values.astype(np.float64)
    mu_vals = df.loc[cens_mask, 'sp_mu'].values.astype(np.float64)
    sigma_vals = df.loc[cens_mask, 'sp_sigma'].values.astype(np.float64)
    z = (S - mu_vals) / sigma_vals.clip(min=0.05)
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    imr = phi_z / (1 - Phi_z).clip(min=1e-6)
    correction = np.clip(sigma_vals * imr, 0, cap_multiplier * sigma_vals)
    corrected = np.maximum(
        df.loc[cens_mask, 'dem_rec'].values,
        df.loc[cens_mask, 'dem_rec'].values + correction * blend,
    )
    df.loc[cens_mask, 'dem_rec'] = corrected
    df.drop(['sp_mu', 'sp_sigma', 'sp_n', 'cat_mu', 'cat_sigma'],
            axis=1, inplace=True, errors='ignore')
    df['dem_rec'] = df['dem_rec'].clip(lower=0).astype('float32')
    return df


# --------------------------------------------------------------------------
# Tobit MLE
# --------------------------------------------------------------------------
def _tobit_negll(theta: np.ndarray, y_obs: np.ndarray, cens: np.ndarray) -> float:
    """
    Intercept-only Tobit negative log-likelihood.
        For uncensored (cens=0): y = d ~ N(mu, sigma^2) -> log phi((y-mu)/sigma) - log sigma
        For censored (cens=1):   y = S <= d, so contribution is log P(d > S)
                                 = log[1 - Phi((S - mu)/sigma)]
    theta = [mu, log_sigma].
    """
    mu, log_s = theta
    sigma = np.exp(log_s)
    if sigma < 1e-4:
        return 1e10
    z = (y_obs - mu) / sigma
    uncens_mask = cens == 0
    ll = 0.0
    if uncens_mask.any():
        ll += float(np.sum(stats.norm.logpdf(z[uncens_mask]) - np.log(sigma)))
    cens_mask = ~uncens_mask
    if cens_mask.any():
        z_c = z[cens_mask]
        # log(1 - Phi(z))  == logsf
        ll += float(np.sum(stats.norm.logsf(z_c)))
    return -ll


def _moment_estimates(y_obs: np.ndarray, cens: np.ndarray) -> tuple[float, float]:
    """Moment-based (mu, sigma) from uncensored rows; biased but always defined."""
    mu0 = float(np.nanmean(y_obs[cens == 0])) if (cens == 0).any() else float(np.nanmean(y_obs))
    s0 = max(float(np.nanstd(y_obs[cens == 0])) if (cens == 0).any() else float(np.nanstd(y_obs)), 0.1)
    return mu0, s0


def _fit_tobit_mle_group(y_obs: np.ndarray, cens: np.ndarray) -> tuple[float, float] | None:
    """Fit intercept-only Tobit for one group.

    Returns (mu, sigma), or None when the optimizer fails to converge --
    callers must fall back explicitly rather than silently using whatever
    iterate the optimizer stopped at.
    """
    if len(y_obs) < 5:
        return _moment_estimates(y_obs, cens)
    mu0, s0 = _moment_estimates(y_obs, cens)
    try:
        res = minimize(
            _tobit_negll, x0=np.array([mu0, np.log(s0)]),
            args=(y_obs, cens), method='Nelder-Mead',
            options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200},
        )
        if not res.success:
            return None
        mu_hat, log_s_hat = res.x
        return float(mu_hat), float(np.exp(log_s_hat))
    except Exception:
        return None


def tobit_mle(
    df: pd.DataFrame,
    min_group_n: int = 15,
    min_cens_n: int = 2,
) -> pd.DataFrame:
    """Per-SP Tobit MLE with per-cat2 fallback for small groups.

    Recovers demand for censored rows as:
        d_hat = mu + sigma * lambda(z)  (conditional expectation on truncated tail)
    where z = (S - mu) / sigma and lambda is the inverse Mills ratio.
    For uncensored rows, d_hat = y (observed).
    """
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['dem_rec'] = df['sale_amount'].astype(np.float64)

    # Stages 1 and 2 (same as heuristic)
    df = _time_weighted_partial(df)
    df = _rolling_avg_full_stockout(df)

    # Fit per-SP Tobit
    sp_params = {}
    n_sp_nonconverged = 0
    sp_group = df.groupby('sp')
    for sp_id, g in sp_group:
        y = g['sale_amount'].values.astype(np.float64)
        c = g['cens'].values.astype(np.int8)
        # Fall back to category MLE if too few rows or too few censored
        if len(y) < min_group_n or c.sum() < min_cens_n:
            sp_params[sp_id] = None
        else:
            fit = _fit_tobit_mle_group(y, c)
            if fit is None:
                # Non-converged MLE: route through the category fallback
                # instead of trusting the optimizer's last iterate.
                n_sp_nonconverged += 1
            sp_params[sp_id] = fit

    # Per-cat2 fallback fit
    cat_params = {}
    n_cat_nonconverged = 0
    for cat_id, g in df.groupby('second_category_id'):
        y = g['sale_amount'].values.astype(np.float64)
        c = g['cens'].values.astype(np.int8)
        fit = _fit_tobit_mle_group(y, c)
        if fit is None:
            n_cat_nonconverged += 1
            fit = _moment_estimates(y, c)
        cat_params[cat_id] = fit
    if n_sp_nonconverged or n_cat_nonconverged:
        print(f"  tobit_mle: {n_sp_nonconverged} SP fits and "
              f"{n_cat_nonconverged} category fits did not converge; "
              f"used category/moment fallbacks")

    # Apply corrections to censored rows
    cens_mask = df['cens'] == 1
    cens_rows = df[cens_mask]
    mu_arr = np.empty(len(cens_rows), dtype=np.float64)
    sigma_arr = np.empty(len(cens_rows), dtype=np.float64)
    for i, (sp_id, cat_id) in enumerate(zip(cens_rows['sp'].values,
                                            cens_rows['second_category_id'].values)):
        p = sp_params.get(sp_id)
        if p is None:
            p = cat_params.get(cat_id, (float(df['sale_amount'].mean()), 0.5))
        mu_arr[i], sigma_arr[i] = p
    sigma_arr = np.clip(sigma_arr, 0.05, None)

    S = cens_rows['sale_amount'].values.astype(np.float64)
    z = (S - mu_arr) / sigma_arr
    imr = stats.norm.pdf(z) / (1 - stats.norm.cdf(z)).clip(min=1e-6)
    # E[D | D > S] = mu + sigma * lambda((S-mu)/sigma)
    cond_exp = mu_arr + sigma_arr * imr
    # Take max of already-recovered dem_rec and conditional expectation
    current = cens_rows['dem_rec'].values
    df.loc[cens_mask, 'dem_rec'] = np.maximum(current, cond_exp)
    df['dem_rec'] = df['dem_rec'].clip(lower=0).astype('float32')

    # Store fitted params + convergence diagnostics for downstream inspection
    df.attrs['tobit_sp_params'] = sp_params
    df.attrs['tobit_cat_params'] = cat_params
    df.attrs['tobit_fit_diagnostics'] = {
        'n_sp_nonconverged': n_sp_nonconverged,
        'n_cat_nonconverged': n_cat_nonconverged,
    }
    return df
