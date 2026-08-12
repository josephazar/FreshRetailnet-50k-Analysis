"""Inventory policy evaluation and out-of-sample policy selection.

Empirical Newsvendor residuals are learned from validation predictions, and
the policy is selected on a held-out validation fold before one test-window
evaluation. Held-out test outcomes never affect policy construction or choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma as gamma_dist, gaussian_kde


DEFAULT_INV_CFG = {
    'h': 0.10, 'p': 0.50, 'w': 0.30,
    'alpha': 0.95, 'shelf_life': 3, 'lead_time': 1,
    'unit_revenue': 1.0, 'unit_cost': 0.40,
    'saa_seed': 42,
}


def profit_optimal_cr(cfg: dict) -> float:
    """Profit-optimal Newsvendor fractile for the FULL stated objective.

    Under the profit function of eval_policy (revenue on sold units,
    procurement on ordered units, holding + waste on leftovers, shortage
    penalty on unmet demand, zero salvage), the marginal costs are
        C_u = unit_revenue - unit_cost + p   (per unit of underage)
        C_o = unit_cost + h + w              (per unit of overage)
    and the optimal order satisfies F(Q*) = C_u / (C_u + C_o).  With the
    default configuration this is 1.1 / 1.9 ~= 0.579. The penalty-only service
    fractile p / (p + w) = 0.625 ignores the revenue margin and procurement
    cost and is not the optimizer of the stated profit function.
    """
    cu = cfg['unit_revenue'] - cfg['unit_cost'] + cfg['p']
    co = cfg['unit_cost'] + cfg['h'] + cfg['w']
    return cu / (cu + co)


def eval_policy(Q: np.ndarray, D: np.ndarray, cfg: dict, name: str = "") -> dict:
    Q = np.asarray(Q, dtype=float)
    D = np.asarray(D, dtype=float)
    sold = np.minimum(Q, D)
    over = np.maximum(Q - D, 0)
    under = np.maximum(D - Q, 0)
    revenue = sold * cfg['unit_revenue']
    procurement = Q * cfg['unit_cost']
    holding = cfg['h'] * over
    stockout = cfg['p'] * under
    waste = cfg['w'] * over
    total_cost = procurement + holding + stockout + waste
    profit = revenue - total_cost
    return {
        'Policy': name,
        'SL(Type1)': round(float(np.mean(Q >= D)), 4),
        'Fill Rate': round(float(np.sum(sold) / max(np.sum(D), 1)), 4),
        'Avg Order': round(float(np.mean(Q)), 4),
        'Avg Demand': round(float(np.mean(D)), 4),
        'Avg Revenue': round(float(np.mean(revenue)), 4),
        'Avg TotalCost': round(float(np.mean(total_cost)), 4),
        'Avg Profit': round(float(np.mean(profit)), 4),
        'SO Rate': round(float(np.mean(D > Q)), 4),
        'Avg Waste': round(float(np.mean(over)), 4),
        'Waste%': round(float(np.sum(over) / max(np.sum(Q), 1) * 100), 2),
    }


def compute_policies(
    preds: dict,
    sp_keys: np.ndarray,
    train_df: pd.DataFrame,
    sp_residual_quantile: dict | None,
    cfg: dict,
    actuals_for_eval: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute all 20 policies.

    Parameters
    ----------
    preds : dict with keys 'ensemble', 'q10', 'q50', 'q90' (and optionally 'mae'/'huber').
    sp_keys : per-row SP identifiers.
    train_df : training frame (for historical mu/sigma baselines).
    sp_residual_quantile : per-SP safety stock at critical ratio,
        learned from VALIDATION fold residuals (NOT test). If None,
        Empirical NV is computed with a pooled global quantile from train.
    cfg : cost configuration.
    actuals_for_eval : per-row actual demand (censored sales) used only for
        eval_policy metrics. Stays available for transparency but the
        policy construction does NOT use it.

    Returns
    -------
    (rdf, policy_quantities): DataFrame with one row per policy (metrics
    computed only if actuals_for_eval is provided); dict of policy -> Q arr.
    """
    mu = preds['ensemble']
    q10, q50, q90 = preds['q10'], preds['q50'], preds['q90']

    sp_std_map = train_df.groupby('sp')['sale_amount'].std().fillna(0.1).to_dict()
    sigma_base = np.array([max(sp_std_map.get(k, 0.1), 0.05) for k in sp_keys])
    # Residual std from training-period residuals would be ideal; using sp_std
    # alone here -- validation residuals contribute via sp_residual_quantile
    sigma = np.sqrt(0.5 * sigma_base ** 2 + 0.5 * np.var(mu - np.mean(mu)))

    cr_nv = profit_optimal_cr(cfg)
    # cr_per is an explicitly heuristic shelf-life-tempered fractile (one of
    # the 20 candidates), not a claimed optimum of the profit objective.
    wf = max(0, 1 - cfg['shelf_life'] * 0.3)
    co_eff = cfg['h'] + cfg['w'] * wf
    cr_per = cfg['p'] / (cfg['p'] + co_eff)

    sp_hist_mean = train_df.groupby('sp')['sale_amount'].mean().to_dict()
    sp_hist_std = train_df.groupby('sp')['sale_amount'].std().fillna(0).to_dict()

    policy_Q = {}
    policy_Q['Static MinMax (hist μ+σ)'] = np.array(
        [sp_hist_mean.get(k, 0) + sp_hist_std.get(k, 0) for k in sp_keys]
    ).clip(min=0)
    policy_Q['Static Mean-Only (no buffer)'] = np.array(
        [sp_hist_mean.get(k, 0) for k in sp_keys]
    ).clip(min=0)
    z_nv = stats.norm.ppf(cr_nv)
    policy_Q[f'Newsvendor-Normal (CR={cr_nv:.3f})'] = np.clip(mu + z_nv * sigma, 0, None)
    t = (cr_nv - 0.5) / (0.9 - 0.5)
    policy_Q['Quantile Newsvendor'] = np.clip(q50 + t * (q90 - q50), 0, None)
    z_alpha = stats.norm.ppf(cfg['alpha'])
    policy_Q[f"Service Level (α={cfg['alpha']})"] = np.clip(mu + z_alpha * sigma, 0, None)
    # Local RNG: SAA sampling must not depend on global np.random state
    # (which shifts whenever upstream code consumes random numbers).
    saa_rng = np.random.default_rng(cfg.get('saa_seed', 42))
    samp = saa_rng.normal(mu[:, None], sigma[:, None], (len(mu), 2000))
    samp = np.clip(samp, 0, None)
    policy_Q['SAA (2000 scenarios)'] = np.quantile(samp, cr_nv, axis=1)
    z_per = stats.norm.ppf(cr_per)
    policy_Q['Perishable Cost Opt'] = np.clip(mu + z_per * sigma, 0, None)
    cv = sigma / np.maximum(mu, 0.01)
    z_dyn = np.where(cv < 0.5, 0.5, np.where(cv < 1.0, 1.0, 1.5))
    policy_Q['Dynamic SS (CV-adaptive)'] = np.clip(mu + z_dyn * sigma, 0, None)
    policy_Q['Q50 Direct'] = np.clip(q50, 0, None)
    policy_Q['Q90 Direct'] = np.clip(q90, 0, None)
    policy_Q['Naive (Q=forecast)'] = np.clip(mu, 0, None)
    for ss in [0.25, 0.5, 0.75, 1.0, 1.5]:
        policy_Q[f'SS ({ss}σ)'] = np.clip(mu + ss * sigma, 0, None)

    # ----- Empirical Newsvendor: uses OOS residual quantile from validation -----
    if sp_residual_quantile is not None:
        # Use per-SP residual quantile learned on validation fold
        global_q = float(np.quantile(list(sp_residual_quantile.values()), 0.5)
                         if sp_residual_quantile else 0.0)
        emp_safety = np.array(
            [sp_residual_quantile.get(k, global_q) for k in sp_keys]
        )
    else:
        # Fallback: use training-period error estimate per SP via naive sp-std
        emp_safety = sigma * z_nv  # degenerate, acts like Newsvendor-Normal
    policy_Q['Empirical Newsvendor'] = np.clip(mu + emp_safety, 0, None)

    # Gamma NV: fits Gamma on training-period historical demand only (OK — no leak)
    unique_sps = np.unique(sp_keys)
    sp_demand_train = train_df.groupby('sp')['sale_amount'].apply(np.array).to_dict()
    Q_gamma = np.zeros(len(mu))
    for sp_id in unique_sps:
        mask = sp_keys == sp_id
        hist_demand = sp_demand_train.get(sp_id, np.array([0.5]))
        positive = hist_demand[hist_demand > 0.01]
        if len(positive) >= 5:
            try:
                shape, _, scale = gamma_dist.fit(positive, floc=0)
                p0 = np.mean(hist_demand <= 0.01)
                adj_cr = (cr_nv - p0) / max(1 - p0, 0.01)
                adj_cr = np.clip(adj_cr, 0.01, 0.99)
                Q_gamma[mask] = gamma_dist.ppf(adj_cr, shape, scale=scale)
            except Exception:
                Q_gamma[mask] = mu[mask] + 0.5 * sigma[mask]
        else:
            Q_gamma[mask] = mu[mask] + 0.5 * sigma[mask]
    policy_Q['Zero-Inflated Gamma NV'] = np.clip(Q_gamma, 0, None)

    # KDE NV: fits on training historical demand -> OK
    Q_kde = np.zeros(len(mu))
    for sp_id in unique_sps:
        mask = sp_keys == sp_id
        hist_demand = sp_demand_train.get(sp_id, np.array([0.5]))
        if len(hist_demand) >= 10:
            try:
                kde = gaussian_kde(hist_demand, bw_method='silverman')
                x_grid = np.linspace(0, max(hist_demand) * 2 + 1, 300)
                cdf = np.cumsum(kde(x_grid)) * (x_grid[1] - x_grid[0])
                cdf = cdf / cdf[-1]
                idx = np.searchsorted(cdf, cr_nv)
                Q_kde[mask] = x_grid[min(idx, len(x_grid) - 1)]
            except Exception:
                Q_kde[mask] = mu[mask] + 0.5 * sigma[mask]
        else:
            Q_kde[mask] = mu[mask] + 0.5 * sigma[mask]
    policy_Q['KDE Newsvendor'] = np.clip(Q_kde, 0, None)

    # Quantile-Direct Interpolated
    if cr_nv <= 0.1:
        Q_interp = q10
    elif cr_nv <= 0.5:
        tt = (cr_nv - 0.1) / (0.5 - 0.1)
        Q_interp = q10 + tt * (q50 - q10)
    elif cr_nv <= 0.9:
        tt = (cr_nv - 0.5) / (0.9 - 0.5)
        Q_interp = q50 + tt * (q90 - q50)
    else:
        Q_interp = q90
    policy_Q['Quantile-Direct Interp'] = np.clip(Q_interp, 0, None)

    # Evaluate if actuals provided
    if actuals_for_eval is not None:
        rows = [eval_policy(Q, actuals_for_eval, cfg, name) for name, Q in policy_Q.items()]
        rdf = pd.DataFrame(rows)
    else:
        rdf = pd.DataFrame({'Policy': list(policy_Q)})
    return rdf, policy_Q


def residual_quantile_from_cv(
    cv_residuals: list[tuple[np.ndarray, np.ndarray]],
    cr: float,
) -> dict:
    """Aggregate validation-fold per-SP residuals into quantile lookup.

    Parameters
    ----------
    cv_residuals : list of (sp_keys, residuals) pairs, one per fold.
    cr : critical ratio (quantile to extract).

    Returns
    -------
    dict sp_id -> safety-stock quantile value.
    """
    rows = []
    for sp_keys, resid in cv_residuals:
        rows.append(pd.DataFrame({'sp': sp_keys, 'resid': resid}))
    df_r = pd.concat(rows, ignore_index=True)
    q = df_r.groupby('sp')['resid'].quantile(cr).to_dict()
    return q


def select_policy_on_validation(
    val_preds: dict,
    val_sp_keys: np.ndarray,
    val_actuals: np.ndarray,
    train_df: pd.DataFrame,
    sp_residual_quantile: dict,
    cfg: dict,
    criterion: str = 'Avg Profit',
    fill_rate_floor: float = 0.90,
) -> str:
    """Pre-select the winning policy on validation data (no test leakage).

    Respects a fill-rate floor to avoid picking a trivially-high-profit,
    low-service policy; if no policy meets the floor, returns the
    max-profit policy with a warning printed.
    """
    rdf, _ = compute_policies(
        preds=val_preds,
        sp_keys=val_sp_keys,
        train_df=train_df,
        sp_residual_quantile=sp_residual_quantile,
        cfg=cfg,
        actuals_for_eval=val_actuals,
    )
    filtered = rdf[rdf['Fill Rate'] >= fill_rate_floor]
    if len(filtered) == 0:
        print(f"  [WARN] no policy meets FR>={fill_rate_floor} on validation; "
              f"returning max-{criterion}")
        filtered = rdf
    winner = filtered.loc[filtered[criterion].idxmax(), 'Policy']
    return winner
