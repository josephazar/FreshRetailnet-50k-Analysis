#!/usr/bin/env python3
"""
Statistical Significance Tests for FreshRetailNet-50K Benchmark
================================================================
Computes:
  1. Bootstrap confidence intervals (95%) for MAE, WAPE, WPE, and Profit
  2. Paired Wilcoxon signed-rank tests between LGBM-Minimal (best) vs all others
  3. Per-SP aggregated metrics for robust paired comparisons

Reads: output/benchmark_per_sp_predictions.csv
Writes: output/statistical_significance.json
        output/statistical_significance.csv
        output/bootstrap_ci_comparison.png

Usage:
    python statistical_significance.py
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
ALPHA = 0.05

# ============================================================================
# Load Data
# ============================================================================
def load_data():
    """Load per-SP predictions from benchmark comparison."""
    path = os.path.join(OUTPUT_DIR, 'benchmark_per_sp_predictions.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run benchmark_comparison.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df['sp'].nunique():,} SPs")
    return df


# ============================================================================
# Identify model columns
# ============================================================================
def get_model_names(df):
    """Extract model names from column prefixes."""
    ae_cols = [c for c in df.columns if c.startswith('ae_')]
    models = [c.replace('ae_', '') for c in ae_cols]
    return models


# ============================================================================
# Per-SP Aggregation
# ============================================================================
def aggregate_per_sp(df, models):
    """Compute per-SP metrics for paired tests."""
    sp_metrics = {}
    for sp, g in df.groupby('sp'):
        y = g['y_true'].values
        total_actual = y.sum()
        sp_data = {'y_total': total_actual, 'n_days': len(g)}
        for m in models:
            preds = g[f'pred_{m}'].values
            ae = g[f'ae_{m}'].values
            profit = g[f'profit_{m}'].values

            sp_data[f'mae_{m}'] = ae.mean()
            sp_data[f'profit_{m}'] = profit.mean()
            # WAPE per SP: sum(|error|) / sum(actual)
            if total_actual > 0:
                sp_data[f'wape_{m}'] = ae.sum() / total_actual * 100
                sp_data[f'wpe_{m}'] = (preds.sum() - y.sum()) / total_actual * 100
            else:
                sp_data[f'wape_{m}'] = np.nan
                sp_data[f'wpe_{m}'] = np.nan

        sp_metrics[sp] = sp_data

    sp_df = pd.DataFrame(sp_metrics).T
    sp_df.index.name = 'sp'
    return sp_df


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================
def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, statistic=np.mean):
    """Compute bootstrap confidence interval for a statistic."""
    values = values[~np.isnan(values)]
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[np.random.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    point = statistic(values)
    return point, lower, upper


def bootstrap_wape(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap CI for WAPE = sum(|error|)/sum(actual)*100."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ae = np.abs(y_true - y_pred)
    n = len(y_true)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        s_actual = y_true[idx].sum()
        if s_actual > 0:
            boot_stats[i] = ae[idx].sum() / s_actual * 100
        else:
            boot_stats[i] = np.nan

    boot_stats = boot_stats[~np.isnan(boot_stats)]
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    point = ae.sum() / max(y_true.sum(), 1e-8) * 100
    return point, lower, upper


def bootstrap_wpe(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap CI for WPE = sum(pred-actual)/sum(actual)*100."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    n = len(y_true)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        s_actual = y_true[idx].sum()
        if s_actual > 0:
            boot_stats[i] = err[idx].sum() / s_actual * 100
        else:
            boot_stats[i] = np.nan

    boot_stats = boot_stats[~np.isnan(boot_stats)]
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    point = err.sum() / max(y_true.sum(), 1e-8) * 100
    return point, lower, upper


# ============================================================================
# Paired Wilcoxon Signed-Rank Tests
# ============================================================================
def paired_wilcoxon(sp_df, models, reference='LGBM_Minimal', metric='mae'):
    """Paired Wilcoxon signed-rank test: reference vs each other model."""
    results = []
    ref_vals = sp_df[f'{metric}_{reference}'].dropna().values

    for m in models:
        if m == reference:
            continue
        alt_vals = sp_df[f'{metric}_{m}'].dropna().values

        # Ensure same SPs
        mask = ~np.isnan(ref_vals) & ~np.isnan(alt_vals)
        r = ref_vals[mask]
        a = alt_vals[mask]

        if len(r) < 10:
            continue

        diffs = a - r  # positive = reference is better (lower metric)

        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(r, a, alternative='two-sided')
        except ValueError:
            stat, p_value = np.nan, np.nan

        # Effect size: matched-pairs rank-biserial correlation
        n_pairs = len(diffs[diffs != 0])
        if n_pairs > 0:
            ranks = stats.rankdata(np.abs(diffs[diffs != 0]))
            r_plus = ranks[diffs[diffs != 0] > 0].sum()
            r_minus = ranks[diffs[diffs != 0] < 0].sum()
            effect_size = (r_plus - r_minus) / (r_plus + r_minus)
        else:
            effect_size = 0.0

        results.append({
            'Comparison': f'{reference} vs {m}',
            'Metric': metric.upper(),
            'N_pairs': int(mask.sum()),
            f'Mean_{reference}': round(r.mean(), 5),
            f'Mean_{m}': round(a.mean(), 5),
            'Mean_diff': round((a - r).mean(), 5),
            'Median_diff': round(np.median(a - r), 5),
            'W_statistic': round(stat, 1) if not np.isnan(stat) else None,
            'p_value': round(p_value, 6) if not np.isnan(p_value) else None,
            'Significant': p_value < ALPHA if not np.isnan(p_value) else None,
            'Effect_size_r': round(effect_size, 4),
            'Interpretation': interpret_effect(effect_size),
        })

    return results


def interpret_effect(r):
    """Interpret rank-biserial correlation effect size."""
    r = abs(r)
    if r < 0.1:
        return 'negligible'
    elif r < 0.3:
        return 'small'
    elif r < 0.5:
        return 'medium'
    else:
        return 'large'


# ============================================================================
# Visualization
# ============================================================================
def plot_bootstrap_cis(ci_results, models):
    """Create forest plot of bootstrap CIs."""
    metrics = ['MAE', 'WAPE', 'WPE', 'Profit']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'Naive_Seasonal': '#95a5a6',
        'SSA': '#e67e22',
        'DLinear': '#3498db',
        'LGBM_Minimal': '#2ecc71',
        'XGBoost_Minimal': '#9b59b6',
        'CatBoost_Minimal': '#e74c3c',
    }

    display_names = {
        'Naive_Seasonal': 'Naive Seasonal',
        'SSA': 'SSA',
        'DLinear': 'DLinear',
        'LGBM_Minimal': 'LightGBM-Min.',
        'XGBoost_Minimal': 'XGBoost-Min.',
        'CatBoost_Minimal': 'CatBoost-Min.',
    }

    for ax, metric in zip(axes.flat, metrics):
        y_positions = np.arange(len(models))
        for i, m in enumerate(models):
            key = f'{metric}_{m}'
            if key in ci_results:
                point, lower, upper = ci_results[key]
                color = colors.get(m, '#333')
                ax.errorbar(
                    point, i, xerr=[[point - lower], [upper - point]],
                    fmt='o', color=color, capsize=4, capthick=1.5,
                    markersize=8, linewidth=1.5,
                    label=display_names.get(m, m)
                )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([display_names.get(m, m) for m in models], fontsize=9)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_title(f'{metric} — 95% Bootstrap CI ({N_BOOTSTRAP:,} resamples)', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'bootstrap_ci_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE TESTS")
    print("  FreshRetailNet-50K Benchmark Comparison")
    print("=" * 70)

    # Load data
    df = load_data()
    models = get_model_names(df)
    print(f"  Models: {models}")

    # ------------------------------------------------------------------
    # 1. Bootstrap CIs on global metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. Bootstrap Confidence Intervals (95%)")
    print("=" * 70)

    y_true = df['y_true'].values
    ci_results = {}
    ci_table = []

    for m in models:
        preds = df[f'pred_{m}'].values
        ae = df[f'ae_{m}'].values
        profit = df[f'profit_{m}'].values

        # MAE
        mae_pt, mae_lo, mae_hi = bootstrap_ci(ae, statistic=np.mean)
        ci_results[f'MAE_{m}'] = (mae_pt, mae_lo, mae_hi)

        # WAPE
        wape_pt, wape_lo, wape_hi = bootstrap_wape(y_true, preds)
        ci_results[f'WAPE_{m}'] = (wape_pt, wape_lo, wape_hi)

        # WPE
        wpe_pt, wpe_lo, wpe_hi = bootstrap_wpe(y_true, preds)
        ci_results[f'WPE_{m}'] = (wpe_pt, wpe_lo, wpe_hi)

        # Profit
        prof_pt, prof_lo, prof_hi = bootstrap_ci(profit, statistic=np.mean)
        ci_results[f'Profit_{m}'] = (prof_pt, prof_lo, prof_hi)

        row = {
            'Model': m.replace('_', ' '),
            'MAE': f'{mae_pt:.4f}',
            'MAE_95CI': f'[{mae_lo:.4f}, {mae_hi:.4f}]',
            'WAPE(%)': f'{wape_pt:.2f}',
            'WAPE_95CI': f'[{wape_lo:.2f}, {wape_hi:.2f}]',
            'WPE(%)': f'{wpe_pt:.2f}',
            'WPE_95CI': f'[{wpe_lo:.2f}, {wpe_hi:.2f}]',
            'Profit': f'{prof_pt:.4f}',
            'Profit_95CI': f'[{prof_lo:.4f}, {prof_hi:.4f}]',
        }
        ci_table.append(row)
        print(f"  {m:20s}  MAE={mae_pt:.4f} [{mae_lo:.4f}, {mae_hi:.4f}]  "
              f"WAPE={wape_pt:.2f}% [{wape_lo:.2f}, {wape_hi:.2f}]  "
              f"Profit={prof_pt:.4f} [{prof_lo:.4f}, {prof_hi:.4f}]")

    # ------------------------------------------------------------------
    # 2. Per-SP aggregation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Per-SP Aggregation for Paired Tests")
    print("=" * 70)
    sp_df = aggregate_per_sp(df, models)
    print(f"  Aggregated {len(sp_df)} SPs")

    # ------------------------------------------------------------------
    # 3. Paired Wilcoxon tests (LGBM-Minimal vs all)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Paired Wilcoxon Signed-Rank Tests")
    print("   Reference: LGBM_Minimal (best MAE)")
    print("=" * 70)

    wilcoxon_results = []

    # Test on MAE
    mae_tests = paired_wilcoxon(sp_df, models, reference='LGBM_Minimal', metric='mae')
    wilcoxon_results.extend(mae_tests)

    # Test on profit
    profit_tests = paired_wilcoxon(sp_df, models, reference='LGBM_Minimal', metric='profit')
    wilcoxon_results.extend(profit_tests)

    # Test on WAPE
    wape_tests = paired_wilcoxon(sp_df, models, reference='LGBM_Minimal', metric='wape')
    wilcoxon_results.extend(wape_tests)

    for r in wilcoxon_results:
        sig = "***" if r['Significant'] else "n.s."
        print(f"  {r['Comparison']:40s}  {r['Metric']:8s}  "
              f"p={r['p_value']:.6f} {sig}  "
              f"effect={r['Effect_size_r']:+.4f} ({r['Interpretation']})")

    # ------------------------------------------------------------------
    # 4. Holm-Bonferroni correction for multiple comparisons
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Holm-Bonferroni Correction for Multiple Comparisons")
    print("=" * 70)

    p_values = [(i, r['p_value']) for i, r in enumerate(wilcoxon_results)
                if r['p_value'] is not None]
    p_values.sort(key=lambda x: x[1])

    m_tests = len(p_values)
    for rank, (idx, p) in enumerate(p_values):
        adjusted_alpha = ALPHA / (m_tests - rank)
        wilcoxon_results[idx]['Holm_significant'] = p < adjusted_alpha
        wilcoxon_results[idx]['Holm_threshold'] = round(adjusted_alpha, 6)

    n_sig = sum(1 for r in wilcoxon_results if r.get('Holm_significant', False))
    print(f"  {m_tests} tests total, {n_sig} significant after Holm-Bonferroni correction")

    # ------------------------------------------------------------------
    # 5. CI Overlap Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Bootstrap CI Overlap Analysis (MAE)")
    print("=" * 70)

    ref = 'LGBM_Minimal'
    ref_lo = ci_results[f'MAE_{ref}'][1]
    ref_hi = ci_results[f'MAE_{ref}'][2]
    overlap_results = []

    for m in models:
        if m == ref:
            continue
        m_lo = ci_results[f'MAE_{m}'][1]
        m_hi = ci_results[f'MAE_{m}'][2]
        overlaps = m_lo <= ref_hi and ref_lo <= m_hi
        overlap_pct = 0.0
        if overlaps:
            overlap_start = max(ref_lo, m_lo)
            overlap_end = min(ref_hi, m_hi)
            ref_width = ref_hi - ref_lo
            if ref_width > 0:
                overlap_pct = (overlap_end - overlap_start) / ref_width * 100

        status = f"overlap {overlap_pct:.0f}%" if overlaps else "NO overlap"
        overlap_results.append({
            'Comparison': f'{ref} vs {m}',
            'Ref_CI': f'[{ref_lo:.4f}, {ref_hi:.4f}]',
            'Alt_CI': f'[{m_lo:.4f}, {m_hi:.4f}]',
            'Overlaps': overlaps,
            'Overlap_pct': round(overlap_pct, 1),
        })
        print(f"  {ref} vs {m:20s}  {status}")

    # ------------------------------------------------------------------
    # 6. Visualization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. Creating Visualizations")
    print("=" * 70)
    plot_bootstrap_cis(ci_results, models)

    # ------------------------------------------------------------------
    # 7. Save all results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. Saving Results")
    print("=" * 70)

    # CSV summary
    ci_df = pd.DataFrame(ci_table)
    ci_csv_path = os.path.join(OUTPUT_DIR, 'statistical_significance.csv')
    ci_df.to_csv(ci_csv_path, index=False)
    print(f"  Saved: {ci_csv_path}")

    # Wilcoxon CSV
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_csv_path = os.path.join(OUTPUT_DIR, 'wilcoxon_tests.csv')
    wilcoxon_df.to_csv(wilcoxon_csv_path, index=False)
    print(f"  Saved: {wilcoxon_csv_path}")

    # JSON with all results
    json_out = {
        'config': {
            'n_bootstrap': N_BOOTSTRAP,
            'ci_level': CI_LEVEL,
            'alpha': ALPHA,
            'reference_model': 'LGBM_Minimal',
        },
        'bootstrap_cis': {
            m.replace('_', ' '): {
                'MAE': {'point': ci_results[f'MAE_{m}'][0],
                        'lower': ci_results[f'MAE_{m}'][1],
                        'upper': ci_results[f'MAE_{m}'][2]},
                'WAPE': {'point': ci_results[f'WAPE_{m}'][0],
                         'lower': ci_results[f'WAPE_{m}'][1],
                         'upper': ci_results[f'WAPE_{m}'][2]},
                'WPE': {'point': ci_results[f'WPE_{m}'][0],
                        'lower': ci_results[f'WPE_{m}'][1],
                        'upper': ci_results[f'WPE_{m}'][2]},
                'Profit': {'point': ci_results[f'Profit_{m}'][0],
                           'lower': ci_results[f'Profit_{m}'][1],
                           'upper': ci_results[f'Profit_{m}'][2]},
            }
            for m in models
        },
        'wilcoxon_tests': wilcoxon_results,
        'ci_overlap': overlap_results,
    }

    json_path = os.path.join(OUTPUT_DIR, 'statistical_significance.json')
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\nBootstrap 95% CIs:")
    for row in ci_table:
        print(f"  {row['Model']:20s}  MAE={row['MAE']} {row['MAE_95CI']}  "
              f"WAPE={row['WAPE(%)']}% {row['WAPE_95CI']}")

    print(f"\nWilcoxon tests ({n_sig}/{m_tests} significant after Holm-Bonferroni):")
    for r in wilcoxon_results:
        sig = "***" if r.get('Holm_significant', False) else "n.s."
        print(f"  {r['Comparison']:40s}  {r['Metric']:8s}  p={r['p_value']:.6f} {sig}")


if __name__ == '__main__':
    main()
