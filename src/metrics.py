"""
Metric helpers + per-SP paired bootstrap + Holm-Bonferroni with preregistered family.

Fixes statistical_significance.py issues:
  - line 104-161: bootstrap was row-level; now cluster-bootstrapped per SP.
  - line 393-404: Holm family counted AFTER NaN filtering; now preregistered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wape(y: np.ndarray, p: np.ndarray) -> float:
    sy = float(np.sum(y))
    return float(np.sum(np.abs(y - p)) / sy * 100) if sy > 0 else 0.0


def wpe(y: np.ndarray, p: np.ndarray) -> float:
    sy = float(np.sum(y))
    return float(np.sum(p - y) / sy * 100) if sy > 0 else 0.0


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y: np.ndarray, p: np.ndarray) -> float:
    # Note: we use the convention |y|+|p| denominator (not /2). Same as paper.
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-8)) * 100)


def corr(y: np.ndarray, p: np.ndarray) -> float:
    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def per_sp_metrics(y: np.ndarray, p: np.ndarray, sp_keys: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({'sp': sp_keys, 'y': y, 'p': p})
    out = df.groupby('sp').apply(lambda g: pd.Series({
        'mae': mae(g['y'].values, g['p'].values),
        'wape': wape(g['y'].values, g['p'].values),
        'wpe': wpe(g['y'].values, g['p'].values),
        'n': len(g),
    })).reset_index()
    return out


def cluster_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    sp_keys: np.ndarray,
    metric_fn,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Per-SP clustered bootstrap. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    sp_ids = np.unique(sp_keys)
    # build sp -> row index lookup
    lookup = {sp: np.where(sp_keys == sp)[0] for sp in sp_ids}
    point = metric_fn(y, p)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot_sps = rng.choice(sp_ids, size=len(sp_ids), replace=True)
        idxs = np.concatenate([lookup[s] for s in boot_sps])
        vals[b] = metric_fn(y[idxs], p[idxs])
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_wilcoxon(
    ref_vals: np.ndarray,
    alt_vals: np.ndarray,
    alternative: str = 'less',
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (p_value, rank_biserial)."""
    mask = (~np.isnan(ref_vals)) & (~np.isnan(alt_vals))
    r = ref_vals[mask]
    a = alt_vals[mask]
    if len(r) < 10:
        return float('nan'), 0.0
    try:
        stat, p = stats.wilcoxon(r, a, alternative=alternative,
                                 zero_method='wilcox', correction=False)
    except Exception:
        return float('nan'), 0.0
    # Rank-biserial effect size
    diff = r - a
    n_pos = float(np.sum(diff > 0))
    n_neg = float(np.sum(diff < 0))
    n_total = n_pos + n_neg
    rb = (n_pos - n_neg) / n_total if n_total > 0 else 0.0
    return float(p), float(rb)


def holm_bonferroni(
    p_values: list[float],
    family_size: int | None = None,
) -> list[float]:
    """Holm-Bonferroni correction.

    family_size: Fixed family size. If None, uses len(p_values). If
    given and larger than len(p_values), inflates correction accordingly
    (i.e., treats missing tests as if they failed).

    Returns adjusted p-values (same order as input).
    """
    n = len(p_values)
    m = family_size if family_size is not None else n
    if m < n:
        raise ValueError("family_size cannot be smaller than len(p_values)")
    # Sort indices by p
    idx = sorted(range(n), key=lambda i: (float('inf') if np.isnan(p_values[i])
                                           else p_values[i]))
    adjusted = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(idx):
        p = p_values[i]
        if np.isnan(p):
            adjusted[i] = float('nan')
            continue
        adj = min(1.0, p * (m - rank))
        adj = max(adj, prev)
        adjusted[i] = adj
        prev = adj
    return adjusted
