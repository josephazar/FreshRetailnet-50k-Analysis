#!/usr/bin/env python3
"""
Generate standalone publication figures for the paper.

Produces focused figures in place of multi-panel composites (`pareto_frontier.png`,
`abc_xyz_matrix.png`) with individual figures so each plot can be referenced,
captioned and sized independently in the manuscript.

Inputs:
    output/headline/inventory_results.csv (Pareto + top-policies)
    output/benchmark/abc_xyz_segmentation.csv      (optional segmentation cache)
    output/headline/predictions.csv      (forecast diagnostics F1-F4)
    output/headline/feature_importance.csv (LightGBM gain importances, F5)
    ../data/freshretailnet/raw/data/{train,eval}.parquet (stockout shading, F1)

Outputs (written to research/images/):
    fig_pareto_cost_fillrate.png            profit-service operating points
    fig_top_policies_profit.png             bar chart of top policies
    fig_abcxyz_matrix.png                   9-cell ABC-XYZ heatmap
    fig_abcxyz_demand.png                   demand distribution by ABC class
    fig_abcxyz_cv.png                       CV distribution by XYZ class
    fig_forecast_examples.png               forecast vs actual, 3 example SPs
    fig_wape_ecdf_by_class.png              per-SP WAPE ECDF by XYZ class
    fig_wape_by_horizon.png                 WAPE degradation over the 7-day horizon
    fig_interval_coverage.png               Q10-Q90 empirical coverage
    fig_feature_importance.png              top-20 features by gain share
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


ROOT = Path(__file__).parent.parent.resolve()
IMG_OUT = ROOT / 'research' / 'images'
IMG_OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette (print-safe; no red/green ambiguity)
OKABE_ITO = {
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999',
}
FAMILY_COLORS = {
    'non_normal': OKABE_ITO['vermillion'],
    'normal_safety': OKABE_ITO['blue'],
    'static': OKABE_ITO['grey'],
    'other': OKABE_ITO['green'],
}


def _policy_family(name: str) -> str:
    if any(tag in name for tag in ['Empirical', 'Gamma', 'KDE', 'Interp']):
        return 'non_normal'
    if any(tag in name for tag in ['Static']):
        return 'static'
    if any(tag in name for tag in ['SS (', 'Service Level', 'Perishable', 'Dynamic']):
        return 'normal_safety'
    return 'other'


def figure_pareto_cost_fillrate(inv: pd.DataFrame, preselected: str | None):
    """Profit vs. fill-rate operating points across all 20 policies."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    colors = FAMILY_COLORS
    for fam, c in colors.items():
        sub = inv[inv['Policy'].map(_policy_family) == fam]
        if len(sub) == 0:
            continue
        ax.scatter(sub['Fill Rate'], sub['Avg Profit'], s=60, c=c,
                   edgecolors='black', linewidths=0.5,
                   label=fam.replace('_', ' ').title(), zorder=4)

    ax.axvline(0.90, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    # Place the FR floor label at the bottom so it cannot collide with
    # the preselected-policy annotation near the circle at FR=0.90.
    ymin, ymax = ax.get_ylim()
    ax.text(0.895, ymin + 0.03 * (ymax - ymin), 'FR floor = 0.90',
            fontsize=8, color='gray', ha='right', va='bottom', rotation=90)

    if preselected and preselected in inv['Policy'].values:
        row = inv[inv['Policy'] == preselected].iloc[0]
        ax.scatter([row['Fill Rate']], [row['Avg Profit']],
                   s=260, facecolors='none', edgecolors='black',
                   linewidths=2.0, zorder=5,
                   label=f'Preselected: {preselected}')
        ax.annotate(preselected, (row['Fill Rate'], row['Avg Profit']),
                    textcoords='offset points', xytext=(12, 12),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    ax.set_xlabel('Fill rate')
    ax.set_ylabel('Average profit (per SP-day)')
    ax.set_title('Profit--service operating frontier across 20 policies')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower left', framealpha=0.95)

    out = IMG_OUT / 'fig_pareto_cost_fillrate.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def figure_top_policies_profit(inv: pd.DataFrame, preselected: str | None, top_n: int = 12):
    """Horizontal bar chart: top-N policies by average profit."""
    top = inv.nlargest(top_n, 'Avg Profit').iloc[::-1]  # ascending for horizontal
    colors = [FAMILY_COLORS[_policy_family(p)] for p in top['Policy']]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.barh(range(len(top)), top['Avg Profit'].values, color=colors,
                   edgecolor='black', linewidth=0.4)

    if preselected:
        for i, p in enumerate(top['Policy'].values):
            if p == preselected:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2.0)
                bars[i].set_hatch('//')
                break

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['Policy'].values, fontsize=9)
    ax.set_xlabel('Average profit (per SP-day)')
    ax.set_title(f'Top {top_n} policies by profit on the test window')
    for i, v in enumerate(top['Avg Profit'].values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8)
    ax.grid(True, axis='x', alpha=0.25)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=FAMILY_COLORS['non_normal'], edgecolor='black',
              label='Non-parametric'),
        Patch(facecolor=FAMILY_COLORS['normal_safety'], edgecolor='black',
              label='Normal / safety stock'),
        Patch(facecolor=FAMILY_COLORS['static'], edgecolor='black',
              label='Static (no ML)'),
        Patch(facecolor=FAMILY_COLORS['other'], edgecolor='black',
              label='Other'),
    ]
    if preselected:
        legend_handles.append(
            Patch(facecolor='white', edgecolor='black', hatch='//',
                  label='Preselected (validation fold)')
        )
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.95)

    out = IMG_OUT / 'fig_top_policies_profit.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def build_abc_xyz_segmentation(train_df: pd.DataFrame) -> pd.DataFrame:
    """Derive ABC-XYZ classes from training sales only.

    ABC ranks store-products by cumulative demand volume (equivalent to
    revenue here because the evaluation uses one common unit revenue). XYZ
    groups the coefficient of variation into smooth, variable, and erratic
    demand. The returned frame is sorted by descending total demand.
    """
    required = {'sale_amount'}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f'Missing segmentation columns: {sorted(missing)}')

    frame = train_df.copy()
    if 'sp' not in frame.columns:
        id_cols = {'store_id', 'product_id'}
        missing_ids = id_cols.difference(frame.columns)
        if missing_ids:
            raise ValueError(
                'Segmentation needs either sp or store_id and product_id'
            )
        frame['sp'] = frame['store_id'] * 10000 + frame['product_id']

    stats = frame.groupby('sp')['sale_amount'].agg(
        total_demand='sum', mean_demand='mean', std_demand='std', n_days='count'
    ).reset_index()
    stats['std_demand'] = stats['std_demand'].fillna(0.0)
    stats['cv'] = stats['std_demand'] / stats['mean_demand'].clip(lower=0.001)
    stats = stats.sort_values(
        ['total_demand', 'sp'], ascending=[False, True], kind='mergesort'
    ).reset_index(drop=True)

    total = float(stats['total_demand'].sum())
    if total <= 0:
        raise ValueError('ABC-XYZ segmentation requires positive total demand')
    stats['cum_share'] = stats['total_demand'].cumsum() / total
    stats['ABC'] = np.where(
        stats['cum_share'] <= 0.80, 'A',
        np.where(stats['cum_share'] <= 0.95, 'B', 'C'),
    )
    stats['XYZ'] = np.where(
        stats['cv'] <= 0.5, 'X',
        np.where(stats['cv'] <= 1.0, 'Y', 'Z'),
    )
    stats['Segment'] = stats['ABC'] + stats['XYZ']
    return stats


def figure_abcxyz_matrix(seg: pd.DataFrame):
    """9-cell ABC-XYZ heatmap with counts and percentages."""
    abc_labels = ['A', 'B', 'C']
    xyz_labels = ['X', 'Y', 'Z']
    matrix = np.zeros((3, 3), dtype=int)
    for i, a in enumerate(abc_labels):
        for j, x in enumerate(xyz_labels):
            matrix[i, j] = int(((seg['ABC'] == a) & (seg['XYZ'] == x)).sum())
    total = int(matrix.sum())
    assert total == len(seg), (
        f"ABC-XYZ matrix sums to {total} but segmentation has {len(seg)} SPs; "
        f"some rows have classes outside A-C/X-Z"
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['X\n(Smooth,\nCV<0.5)',
                        'Y\n(Variable,\n0.5-1.0)',
                        'Z\n(Erratic,\nCV>1.0)'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['A\n(top 80%\nrevenue)',
                        'B\n(next 15%)',
                        'C\n(bottom 5%)'])
    ax.set_title('ABC-XYZ segmentation: store-product counts per cell')

    for i in range(3):
        for j in range(3):
            v = matrix[i, j]
            pct = v / total * 100 if total > 0 else 0
            color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{v:,}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label('Store-product count')

    out = IMG_OUT / 'fig_abcxyz_matrix.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def figure_abcxyz_demand(seg: pd.DataFrame):
    """Demand distribution stacked by ABC class."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {'A': OKABE_ITO['vermillion'], 'B': OKABE_ITO['orange'],
              'C': OKABE_ITO['blue']}
    xmax = float(seg['mean_demand'].quantile(0.95))
    for abc, c in colors.items():
        subset = seg.loc[seg['ABC'] == abc, 'mean_demand']
        ax.hist(subset, bins=50, range=(0, xmax), alpha=0.6,
                label=f'Class {abc} (n={len(subset):,})',
                color=c, edgecolor='none')
    ax.set_xlabel('Mean daily demand (units)')
    ax.set_ylabel('Store-product count')
    ax.set_title('Daily-demand distribution by ABC class')
    ax.set_xlim(0, xmax)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.25)

    out = IMG_OUT / 'fig_abcxyz_demand.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def figure_abcxyz_cv(seg: pd.DataFrame):
    """Coefficient-of-variation distribution stacked by XYZ class."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {'X': OKABE_ITO['green'], 'Y': OKABE_ITO['orange'],
              'Z': OKABE_ITO['vermillion']}
    clipped = seg['cv'].clip(upper=3.0)
    bins = np.linspace(0, 3.0, 60)
    for xyz, c in colors.items():
        subset = clipped[seg['XYZ'] == xyz]
        ax.hist(subset, bins=bins, alpha=0.6,
                label=f'Class {xyz} (n={len(subset):,})',
                color=c, edgecolor='none')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ymax = ax.get_ylim()[1]
    ax.text(0.5, ymax * 0.97, 'CV=0.5', fontsize=8, color='gray',
            ha='center', va='top')
    ax.text(1.0, ymax * 0.97, 'CV=1.0', fontsize=8, color='gray',
            ha='center', va='top')
    ax.set_xlabel('Coefficient of variation (CV)')
    ax.set_ylabel('Store-product count')
    ax.set_title('Demand-variability (CV) distribution by XYZ class')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.25)

    out = IMG_OUT / 'fig_abcxyz_cv.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


# ---------------------------------------------------------------------------
# Forecast diagnostics (F1-F4) from the corrected headline run.
# ---------------------------------------------------------------------------
def load_predictions_with_segments(preds_path: Path, seg_path: Path) -> pd.DataFrame:
    """predictions.csv + ABC/XYZ segment labels + 1-based horizon day."""
    pr = pd.read_csv(preds_path, parse_dates=['dt'])
    seg = pd.read_csv(seg_path)[['sp', 'ABC', 'XYZ', 'Segment', 'total_demand']]
    pr = pr.merge(seg, on='sp', how='left')
    pr['horizon_day'] = (pr['dt'] - pr['dt'].min()).dt.days + 1
    return pr


def _per_sp_wape(pr: pd.DataFrame, pred_col: str = 'pred_ensemble') -> pd.DataFrame:
    g = pr.groupby('sp')
    num = g.apply(lambda x: float(np.abs(x['y_true'] - x[pred_col]).sum()))
    den = g['y_true'].sum()
    out = pd.DataFrame({'sp': num.index, 'abs_err': num.values, 'y_sum': den.values})
    n_zero = int((out['y_sum'] <= 0).sum())
    if n_zero:
        print(f'  [info] dropping {n_zero} SPs with zero eval demand from per-SP WAPE')
    out = out[out['y_sum'] > 0].copy()
    out['wape'] = 100.0 * out['abs_err'] / out['y_sum']
    return out.merge(pr[['sp', 'XYZ', 'ABC', 'Segment']].drop_duplicates('sp'),
                     on='sp', how='left')


def figure_forecast_examples(pr: pd.DataFrame, train_parquet: Path,
                             eval_parquet: Path, history_days: int = 21):
    """Forecast vs actual for three representative SPs (AX / BY / CZ).

    Deterministic, cherry-pick-proof selection: within each target segment,
    pick the SP whose ensemble WAPE is closest to the segment median
    (ties -> larger total eval demand). For CZ, restrict to SPs with at
    least one stockout-affected eval day so the censoring shading is shown.
    """
    spw = _per_sp_wape(pr)

    # Stockout hours per (sp, dt) for shading: eval + last train days
    ev_meta = pd.read_parquet(
        eval_parquet, columns=['store_id', 'product_id', 'dt', 'stock_hour6_22_cnt'])
    ev_meta['sp'] = ev_meta['store_id'] * 10000 + ev_meta['product_id']
    ev_meta['dt'] = pd.to_datetime(ev_meta['dt'])

    sp_so = ev_meta[ev_meta['stock_hour6_22_cnt'] > 0].groupby('sp').size()

    chosen = []
    for seg_name in ['AX', 'BY', 'CZ']:
        cand = spw[spw['Segment'] == seg_name].copy()
        if seg_name == 'CZ':
            with_so = cand[cand['sp'].isin(sp_so.index)]
            if len(with_so):
                cand = with_so
        if not len(cand):
            print(f'  [warn] no candidate SP for segment {seg_name}')
            continue
        med = cand['wape'].median()
        cand['dist'] = (cand['wape'] - med).abs()
        cand = cand.sort_values(['dist', 'y_sum'], ascending=[True, False])
        chosen.append((seg_name, int(cand.iloc[0]['sp']), float(cand.iloc[0]['wape'])))

    tr_hist = pd.read_parquet(
        train_parquet,
        columns=['store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt'])
    tr_hist['sp'] = tr_hist['store_id'] * 10000 + tr_hist['product_id']
    tr_hist['dt'] = pd.to_datetime(tr_hist['dt'])
    sps = [sp for _, sp, _ in chosen]
    tr_hist = tr_hist[tr_hist['sp'].isin(sps)]
    cutoff = tr_hist['dt'].max() - pd.Timedelta(days=history_days - 1)
    tr_hist = tr_hist[tr_hist['dt'] >= cutoff]
    ev_meta = ev_meta[ev_meta['sp'].isin(sps)]

    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.0, 7.5), sharex=True)
    if len(chosen) == 1:
        axes = [axes]
    test_start = pr['dt'].min()
    for ax, (seg_name, sp, w) in zip(axes, chosen):
        h = tr_hist[tr_hist['sp'] == sp].sort_values('dt')
        p = pr[pr['sp'] == sp].sort_values('dt')
        e = ev_meta[ev_meta['sp'] == sp]

        ax.plot(h['dt'], h['sale_amount'], color=OKABE_ITO['black'],
                lw=1.0, marker='o', ms=2.5, label='Observed sales')
        ax.plot(p['dt'], p['y_true'], color=OKABE_ITO['black'],
                lw=1.0, marker='o', ms=2.5)
        ax.plot(p['dt'], p['pred_ensemble'], color=OKABE_ITO['blue'],
                lw=1.6, marker='s', ms=3, label='Ensemble forecast')
        ax.fill_between(p['dt'], p['pred_q10'], p['pred_q90'],
                        color=OKABE_ITO['skyblue'], alpha=0.30,
                        label='Q10--Q90 interval')
        so_days = set(pd.to_datetime(
            pd.concat([e.loc[e['stock_hour6_22_cnt'] > 0, 'dt'],
                       h.loc[h['stock_hour6_22_cnt'] > 0, 'dt']])
        ).dt.normalize())
        for d in sorted(so_days):
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                       color=OKABE_ITO['grey'], alpha=0.25, lw=0)
        ax.axvline(test_start - pd.Timedelta(hours=12), color='gray',
                   linestyle='--', lw=1.0)
        ax.set_title(f'{seg_name}  |  SP {sp}  |  per-SP WAPE {w:.1f}\\%'
                     .replace('\\%', '%'), fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel('Units / day')
    axes[0].text(test_start, axes[0].get_ylim()[1] * 0.95, ' test start',
                 fontsize=8, color='gray', va='top')
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor=OKABE_ITO['grey'], alpha=0.25,
                         label='Stockout hours observed'))
    labels.append('Stockout hours observed')
    axes[0].legend(handles, labels, loc='upper left', framealpha=0.95, ncol=2)
    axes[-1].set_xlabel('Date')
    fig.autofmt_xdate()

    out = IMG_OUT / 'fig_forecast_examples.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out} (SPs: {[(s, sp) for s, sp, _ in chosen]})')


def figure_wape_ecdf_by_class(pr: pd.DataFrame):
    """ECDF of per-SP WAPE, one curve per XYZ variability class."""
    spw = _per_sp_wape(pr)
    pooled = 100.0 * float(np.abs(pr['y_true'] - pr['pred_ensemble']).sum()
                           / pr['y_true'].sum())

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {'X': OKABE_ITO['green'], 'Y': OKABE_ITO['orange'],
              'Z': OKABE_ITO['vermillion']}
    for xyz, c in colors.items():
        vals = np.sort(spw.loc[spw['XYZ'] == xyz, 'wape'].clip(upper=200).values)
        if not len(vals):
            continue
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        med = float(np.median(vals))
        ax.step(vals, ecdf, where='post', color=c, lw=1.8,
                label=f'{xyz} (n={len(vals):,}, median {med:.1f}%)')
        ax.plot([med], [0.5], marker='o', ms=5, color=c, zorder=5)
    ax.axvline(pooled, color='gray', linestyle=':', lw=1.2)
    ax.text(pooled + 2, 0.06, f'pooled WAPE {pooled:.1f}%', fontsize=8,
            color='gray')
    ax.set_xlabel('Per-SP WAPE (%) -- clipped at 200%')
    ax.set_ylabel('Fraction of store-products')
    ax.set_title('Per-SP forecast error by demand-variability class')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.95)

    out = IMG_OUT / 'fig_wape_ecdf_by_class.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def figure_wape_by_horizon(pr: pd.DataFrame):
    """Pooled WAPE per forecast horizon day (1..7), per model."""
    model_cols = {
        'Ensemble': ('pred_ensemble', OKABE_ITO['blue'], 2.2),
        'MAE': ('pred_mae', OKABE_ITO['grey'], 1.0),
        'Huber': ('pred_huber', OKABE_ITO['orange'], 1.0),
        'Q50': ('pred_q50', OKABE_ITO['green'], 1.0),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    days = sorted(pr['horizon_day'].unique())
    for label, (col, c, lw) in model_cols.items():
        wapes = []
        for d in days:
            sub = pr[pr['horizon_day'] == d]
            wapes.append(100.0 * float(np.abs(sub['y_true'] - sub[col]).sum()
                                       / sub['y_true'].sum()))
        ax.plot(days, wapes, marker='o', ms=4, color=c, lw=lw, label=label)
        if label == 'Ensemble':
            for d, w in zip(days, wapes):
                ax.annotate(f'{w:.1f}', (d, w), textcoords='offset points',
                            xytext=(0, 7), fontsize=8, ha='center')
    ax.set_xlabel('Forecast horizon day')
    ax.set_ylabel('WAPE (%)')
    ax.set_title('Forecast error growth over the 7-day recursive horizon')
    ax.set_xticks(days)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', framealpha=0.95)

    out = IMG_OUT / 'fig_wape_by_horizon.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def figure_interval_coverage(pr: pd.DataFrame):
    """Empirical coverage of the [Q10, Q90] interval vs the 80% nominal."""
    in_int = ((pr['y_true'] >= pr['pred_q10']) &
              (pr['y_true'] <= pr['pred_q90']))
    under_q50 = pr['y_true'] <= pr['pred_q50']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))

    # Panel (a): coverage overall + by XYZ class
    groups = ['All', 'X', 'Y', 'Z']
    cov80, cov50 = [], []
    for gname in groups:
        m = slice(None) if gname == 'All' else (pr['XYZ'] == gname).values
        cov80.append(float(in_int.values[m].mean()))
        cov50.append(float(under_q50.values[m].mean()))
    x = np.arange(len(groups))
    ax1.bar(x - 0.17, cov80, width=0.34, color=OKABE_ITO['blue'],
            edgecolor='black', lw=0.4, label='P(Q10 $\\leq$ y $\\leq$ Q90)')
    ax1.bar(x + 0.17, cov50, width=0.34, color=OKABE_ITO['orange'],
            edgecolor='black', lw=0.4, label='P(y $\\leq$ Q50)')
    ax1.axhline(0.80, color=OKABE_ITO['blue'], linestyle='--', lw=1.0)
    ax1.axhline(0.50, color=OKABE_ITO['orange'], linestyle='--', lw=1.0)
    ax1.text(len(groups) - 0.45, 0.805, 'nominal 0.80', fontsize=8,
             color=OKABE_ITO['blue'], va='bottom', ha='right')
    ax1.text(len(groups) - 0.45, 0.505, 'nominal 0.50', fontsize=8,
             color=OKABE_ITO['orange'], va='bottom', ha='right')
    for xi, v in zip(x, cov80):
        ax1.text(xi - 0.17, v + 0.012, f'{v:.2f}', fontsize=8, ha='center')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_ylabel('Empirical coverage')
    ax1.set_title('(a) Coverage by variability class')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=8)
    ax1.grid(True, axis='y', alpha=0.25)

    # Panel (b): coverage by horizon day
    days = sorted(pr['horizon_day'].unique())
    cov_by_day = [float(in_int[pr['horizon_day'] == d].mean()) for d in days]
    ax2.plot(days, cov_by_day, marker='o', ms=4, color=OKABE_ITO['blue'], lw=1.8)
    ax2.axhline(0.80, color='gray', linestyle='--', lw=1.0)
    ax2.text(days[-1], 0.805, 'nominal 0.80', fontsize=8, color='gray',
             va='bottom', ha='right')
    ax2.set_xlabel('Forecast horizon day')
    ax2.set_ylabel('Empirical coverage of [Q10, Q90]')
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xticks(days)
    ax2.set_title('(b) Coverage by horizon day')
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out = IMG_OUT / 'fig_interval_coverage.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def figure_feature_importance(fi_path: Path, top_n: int = 20):
    """Top-N features by mean gain share across the point-forecast models."""
    fi = pd.read_csv(fi_path)
    point_models = [m for m in ['mae', 'huber', 'q50'] if m in fi['model'].unique()]
    fi = fi[fi['model'].isin(point_models)].copy()
    # Normalize gain within each model, then average shares across models
    fi['gain_share'] = fi.groupby('model')['gain'].transform(lambda g: g / g.sum())
    mean_share = (fi.groupby('feature')['gain_share'].mean()
                    .sort_values(ascending=False))
    top = mean_share.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.barh(range(len(top)), top.values * 100, color=OKABE_ITO['blue'],
            edgecolor='black', lw=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel('Mean gain share across MAE/Huber/Q50 models (%)')
    ax.set_title(f'Top {top_n} features by LightGBM gain')
    ax.grid(True, axis='x', alpha=0.25)

    out = IMG_OUT / 'fig_feature_importance.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inv', default='output/headline/inventory_results.csv')
    ap.add_argument('--seg', default='output/benchmark/abc_xyz_segmentation.csv')
    ap.add_argument('--sel', default='output/headline/inventory_selection.json')
    ap.add_argument('--preds', default='output/headline/predictions.csv')
    ap.add_argument('--fi', default='output/headline/feature_importance.csv')
    ap.add_argument('--train-parquet',
                    default=str(ROOT.parent / 'data' / 'freshretailnet' / 'raw'
                                / 'data' / 'train.parquet'))
    ap.add_argument('--eval-parquet',
                    default=str(ROOT.parent / 'data' / 'freshretailnet' / 'raw'
                                / 'data' / 'eval.parquet'))
    args = ap.parse_args()

    inv_path = ROOT / args.inv
    seg_path = ROOT / args.seg
    sel_path = ROOT / args.sel
    preds_path = ROOT / args.preds
    fi_path = ROOT / args.fi
    train_parquet = Path(args.train_parquet)
    eval_parquet = Path(args.eval_parquet)

    if not inv_path.exists():
        print(f'[error] {inv_path} missing; run pipeline_leakfree.py first')
        return 1
    inv = pd.read_csv(inv_path)

    preselected = None
    if sel_path.exists():
        preselected = json.loads(sel_path.read_text()).get('preselected_policy')
        print(f'  preselected policy: {preselected}')

    print('Building Pareto / top-policies figures...')
    figure_pareto_cost_fillrate(inv, preselected)
    figure_top_policies_profit(inv, preselected)

    if not seg_path.exists() and train_parquet.exists():
        print(f'  {seg_path} missing; deriving ABC-XYZ classes from training data')
        train_for_seg = pd.read_parquet(
            train_parquet,
            columns=['store_id', 'product_id', 'sale_amount'],
        )
        seg = build_abc_xyz_segmentation(train_for_seg)
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        seg.to_csv(seg_path, index=False)
        print(f'  wrote {seg_path}')

    if seg_path.exists():
        seg = pd.read_csv(seg_path)
        print('Building ABC-XYZ figures...')
        figure_abcxyz_matrix(seg)
        figure_abcxyz_demand(seg)
        figure_abcxyz_cv(seg)
    else:
        print(f'[warn] {seg_path} missing; ABC-XYZ figures skipped')

    if preds_path.exists() and seg_path.exists():
        print('Building forecast diagnostics (F1-F4)...')
        pr = load_predictions_with_segments(preds_path, seg_path)
        figure_wape_ecdf_by_class(pr)
        figure_wape_by_horizon(pr)
        figure_interval_coverage(pr)
        if train_parquet.exists() and eval_parquet.exists():
            figure_forecast_examples(pr, train_parquet, eval_parquet)
        else:
            print(f'[warn] parquet files missing; forecast-examples figure skipped')
    else:
        print(f'[warn] {preds_path} missing; forecast diagnostics skipped')

    if fi_path.exists():
        print('Building feature-importance figure...')
        figure_feature_importance(fi_path)
    else:
        print(f'[warn] {fi_path} missing; feature-importance figure skipped '
              f'(run scripts/export_feature_importance.py)')

    print(f'\nDone. Figures in {IMG_OUT}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
