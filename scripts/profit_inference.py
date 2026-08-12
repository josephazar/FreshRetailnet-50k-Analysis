"""Profit inference for the headline run (store-clustered).

Fixed analysis specification:
  - Cluster bootstrap with clusters = STORES (products within a store share
    demand shocks), 10,000 resamples, seed 42, percentile 95% CIs for each
    policy's mean per-row profit on the held-out evaluation window.
  - PAIRED differences (same store resamples for both policies) for three
    pre-declared comparisons: selected vs the static baseline, selected vs
    naive forecast ordering, selected vs the Empirical Newsvendor.
  - Two-sided paired Wilcoxon signed-rank tests on per-store mean profit
    differences, Holm-corrected within the family of 3 comparisons.
  - Absolute differences and effect sizes (Cohen's d_z on store-level
    differences) reported alongside ratios.

CIs are computed for ALL policies (cheap once per-store sums exist); the
manuscript's display subset is a presentation choice, not a selection.

Profit here is evaluated against OBSERVED SALES under censoring: it is a
proxy whose direction of bias relative to latent-demand profit is not
identified. Nothing in this script or its outputs may be described as a
bound.

Run (after scripts/headline_run.py has produced --run_dir):
    ./venv/bin/python scripts/profit_inference.py --run_dir output/headline

Outputs (inside --run_dir):
    profit_inference.csv           per policy: point estimate + 95% CI
    profit_paired_differences.csv  the 3 fixed comparisons with CI, Holm-
                                   corrected Wilcoxon p, effect size
    profit_inference_metadata.json provenance
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from src.inventory import DEFAULT_INV_CFG  # noqa: E402

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42
COMPARISONS = [
    ('static baseline', 'Static MinMax (hist μ+σ)'),
    ('naive forecast ordering', 'Naive (Q=forecast)'),
    ('empirical newsvendor', 'Empirical Newsvendor'),
]


def _row_profit(Q: np.ndarray, D: np.ndarray, cfg: dict) -> np.ndarray:
    """Per-row profit, identical to src.inventory.eval_policy's arithmetic
    (kept unrounded here; eval_policy rounds its aggregates)."""
    sold = np.minimum(Q, D)
    over = np.maximum(Q - D, 0.0)
    under = np.maximum(D - Q, 0.0)
    revenue = sold * cfg['unit_revenue']
    cost = (Q * cfg['unit_cost'] + cfg['h'] * over + cfg['p'] * under
            + cfg['w'] * over)
    return revenue - cost


def _holm(pvals: list) -> list:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='output/headline')
    ap.add_argument('--B', type=int, default=BOOTSTRAP_B)
    ap.add_argument('--seed', type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()
    run_dir = SCRIPT_DIR / args.run_dir
    t0 = time.time()

    orders = pd.read_parquet(run_dir / 'policy_orders.parquet')
    selection = json.loads((run_dir / 'inventory_selection.json').read_text())
    selected = selection['preselected_policy']
    run_meta = json.loads(
        (run_dir / 'run_metadata.json').read_text())
    cfg = run_meta.get('inventory_cfg', DEFAULT_INV_CFG)

    policies = [c for c in orders.columns if c not in ('sp', 'dt', 'y_true')]
    assert selected in policies, f'selected policy {selected!r} not in orders'
    D = orders['y_true'].values.astype('float64')
    store = (orders['sp'].values // 10000).astype('int64')
    print(f"[1/3] {len(orders):,} rows, {len(policies)} policies, "
          f"{len(np.unique(store)):,} stores; selected = {selected}")

    # per-store per-policy profit sums (+ shared row counts)
    stores, inv = np.unique(store, return_inverse=True)
    n_stores = len(stores)
    P = np.zeros((n_stores, len(policies)))
    for j, pol in enumerate(policies):
        prof = _row_profit(orders[pol].values.astype('float64'), D, cfg)
        P[:, j] = np.bincount(inv, weights=prof, minlength=n_stores)
    N = np.bincount(inv, minlength=n_stores).astype('float64')
    point = P.sum(axis=0) / N.sum()

    print(f"[2/3] Store-clustered bootstrap (B={args.B:,}, seed {args.seed})")
    rng = np.random.default_rng(args.seed)
    sel_j = policies.index(selected)
    cmp_j = {label: policies.index(name) for label, name in COMPARISONS}
    boot_means = np.empty((args.B, len(policies)))
    for b in range(args.B):
        idx = rng.integers(0, n_stores, n_stores)
        boot_means[b] = P[idx].sum(axis=0) / N[idx].sum()
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)

    pol_df = pd.DataFrame({
        'Policy': policies,
        'avg_profit': point,
        'ci95_lo': ci_lo,
        'ci95_hi': ci_hi,
        'selected': [int(p == selected) for p in policies],
    }).sort_values('avg_profit', ascending=False)
    pol_df.to_csv(run_dir / 'profit_inference.csv', index=False)

    # paired comparisons: same bootstrap draws -> paired difference CIs;
    # Wilcoxon + effect size on per-store mean profits
    store_means = P / N[:, None]
    raw_p, rows = [], []
    for label, name in COMPARISONS:
        j = cmp_j[label]
        diff_point = point[sel_j] - point[j]
        diff_draws = boot_means[:, sel_j] - boot_means[:, j]
        d_store = store_means[:, sel_j] - store_means[:, j]
        try:
            w_stat, w_p = wilcoxon(d_store, alternative='two-sided')
        except ValueError:            # all-zero differences
            w_stat, w_p = np.nan, 1.0
        sd = d_store.std(ddof=1)
        rows.append({
            'comparison': f'selected vs {label}',
            'selected_policy': selected,
            'other_policy': name,
            'diff_point': diff_point,
            'diff_ci95_lo': float(np.percentile(diff_draws, 2.5)),
            'diff_ci95_hi': float(np.percentile(diff_draws, 97.5)),
            'ratio': point[sel_j] / point[j] if point[j] != 0 else np.nan,
            'wilcoxon_stat': float(w_stat),
            'wilcoxon_p': float(w_p),
            'cohens_dz_store': float(d_store.mean() / sd) if sd > 0
            else np.nan,
            'n_stores': n_stores,
        })
        raw_p.append(float(w_p))
    holm = _holm(raw_p)
    for r, hp in zip(rows, holm):
        r['wilcoxon_p_holm'] = hp
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(run_dir / 'profit_paired_differences.csv', index=False)

    print("[3/3] Results")
    with pd.option_context('display.width', 200):
        print(pol_df.head(8).to_string(index=False))
        print()
        print(cmp_df[['comparison', 'diff_point', 'diff_ci95_lo',
                      'diff_ci95_hi', 'wilcoxon_p_holm',
                      'cohens_dz_store']].to_string(index=False))

    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                         capture_output=True, text=True).stdout.strip()
    meta = {
        'script': 'profit_inference', 'git_sha': sha, 'args': vars(args),
        'cluster': 'store (sp // 10000)', 'n_stores': n_stores,
        'B': args.B, 'seed': args.seed,
        'comparisons_family': [f'selected vs {label}'
                               for label, _ in COMPARISONS],
        'correction': 'Holm, family of 3, two-sided',
        'fill_rate_floor_sensitivity':
            selection.get('fill_rate_floor_sensitivity'),
        'inventory_cfg': cfg,
        'note': ('profit is an observed-sales proxy under censored '
                 'evaluation; direction of bias vs latent-demand profit is '
                 'not identified; not a bound'),
        'runtime_s': round(time.time() - t0, 1),
    }
    (run_dir / 'profit_inference_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"Done in {meta['runtime_s']:.0f}s -> {run_dir}/")


if __name__ == '__main__':
    main()
