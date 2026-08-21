#!/usr/bin/env python3
"""Robustness of the stockout-stratum error ordering in the headline run.

The headline error stratified by stockout exposure is NOT monotonic in that
exposure: the lightly censored stratum (1-4 stockout hours) scores a *lower*
WAPE than the zero-stockout stratum, while the heavily censored stratum (5+
hours) scores much higher.  A reading in which the evaluation protocol alone
drives the zero-stockout error would predict a monotone ordering, so the
reversal is the load-bearing fact and it has to survive conditioning.

This script tests whether the reversal is an artifact of composition:

  (a) overall     per-stratum WAPE, WPE, mean observed sales, row counts.
  (b) decile      the same within deciles of demand level, where deciles are
                  formed on each store-product's MEAN y_true over the test
                  window, so decile membership is a series-level property and
                  a series never splits across deciles.
  (b2) decile on  the same deciles rebuilt on each store-product's mean y_true
       uncensored  over its ZERO-STOCKOUT rows only, so no truncated value
       basis       enters the basis. Which days are uncensored is itself a
                   non-random selection, so this complements rather than
                   supersedes (b), whose basis is computed over censored rows.
  (c) date        the same within each test date, controlling for day effects.
  (d) paired      a within-store-product comparison restricted to series that
                  contain BOTH 0h and light rows, which removes between-series
                  composition entirely.

WAPE = 100 * sum|pred - y| / sum(y); it is undefined where sum(y) = 0 and is
reported as empty in that case.  Because WAPE normalizes by the observed-sales
level and that level differs sharply across strata, mean_y is carried on every
stratum row so the denominator is auditable next to every ratio.

Run (after scripts/headline_run.py has produced the predictions):
    ./venv/bin/python scripts/stratum_robustness.py

Output (inside --out):
    stockout_stratum_robustness.csv   tidy long format, one metric per row
    stockout_stratum_robustness_metadata.json
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

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.metrics import paired_wilcoxon  # noqa: E402

MODEL = 'ensemble'
STRATA = ('0h', 'light', 'heavy')
N_DECILES = 10


def assign_stratum(hours: np.ndarray) -> np.ndarray:
    """0h = no stockout, light = 1-4 stockout hours, heavy = 5+."""
    return np.where(hours == 0, '0h', np.where(hours <= 4, 'light', 'heavy'))


def stratum_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-group x per-stratum sums plus WAPE, WPE and mean observed sales."""
    g = df.groupby(keys + ['stratum'], observed=True).agg(
        n_rows=('y_true', 'size'),
        sum_y=('y_true', 'sum'),
        sum_abs_err=('abs_err', 'sum'),
        sum_err=('err', 'sum'),
    ).reset_index()
    ok = g['sum_y'] > 0
    g['wape_pct'] = np.where(ok, 100 * g['sum_abs_err'] / g['sum_y'], np.nan)
    g['wpe_pct'] = np.where(ok, 100 * g['sum_err'] / g['sum_y'], np.nan)
    g['mean_y'] = g['sum_y'] / g['n_rows']
    return g


def emit(rows: list[dict], analysis: str, group_key, detail: str,
         stratum: str, metric: str, value) -> None:
    rows.append({'analysis': analysis, 'group_key': str(group_key),
                 'group_detail': detail, 'stratum': stratum,
                 'metric': metric, 'value': float(value)})


def emit_strata(rows: list[dict], tab: pd.DataFrame, analysis: str,
                group_key, detail: str) -> dict[str, float]:
    """Write the four per-stratum metrics and return each stratum's WAPE."""
    wapes: dict[str, float] = {}
    for stratum in STRATA:
        sel = tab[tab['stratum'] == stratum]
        if sel.empty:
            continue
        r = sel.iloc[0]
        emit(rows, analysis, group_key, detail, stratum, 'n_rows', r['n_rows'])
        emit(rows, analysis, group_key, detail, stratum, 'mean_y', r['mean_y'])
        emit(rows, analysis, group_key, detail, stratum, 'wape_pct', r['wape_pct'])
        emit(rows, analysis, group_key, detail, stratum, 'wpe_pct', r['wpe_pct'])
        wapes[stratum] = float(r['wape_pct'])
    base = wapes.get('0h', np.nan)
    for stratum in ('light', 'heavy'):
        if stratum not in wapes or not np.isfinite(base):
            continue
        diff = wapes[stratum] - base
        emit(rows, analysis, group_key, detail, f'{stratum}_vs_0h',
             'wape_minus_0h_pp', diff)
        emit(rows, analysis, group_key, detail, f'{stratum}_vs_0h',
             'lower_wape_than_0h_flag', float(diff < 0))
    return wapes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='output_v5_headline/predictions.csv',
                    help='headline per-row predictions CSV')
    ap.add_argument('--out', default='results/current',
                    help='directory the tidy CSV and metadata are written to')
    args = ap.parse_args()
    t0 = time.time()

    preds_path = Path(args.preds)
    if not preds_path.is_absolute():
        preds_path = ROOT / preds_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path, usecols=['sp', 'dt', 'y_true',
                                          'stockout_hours', f'pred_{MODEL}'])
    df = df.rename(columns={f'pred_{MODEL}': 'pred'})
    df['stratum'] = assign_stratum(df['stockout_hours'].to_numpy())
    df['err'] = df['pred'] - df['y_true']
    df['abs_err'] = df['err'].abs()
    print(f"Loaded {len(df):,} rows / {df['sp'].nunique():,} store-products / "
          f"{df['dt'].nunique()} test dates from {preds_path}")

    rows: list[dict] = []

    # ---- (a) overall -----------------------------------------------------
    detail = f'all {len(df):,} evaluation rows of the headline {MODEL} forecast'
    overall = stratum_table(df.assign(_all=0), ['_all'])
    emit(rows, 'overall', 'all', detail, 'all', 'n_rows', len(df))
    emit(rows, 'overall', 'all', detail, 'all', 'n_sp', df['sp'].nunique())
    emit(rows, 'overall', 'all', detail, 'all', 'mean_y', df['y_true'].mean())
    w_overall = emit_strata(rows, overall, 'overall', 'all', detail)

    # ---- (b) demand deciles ---------------------------------------------
    # Deciles are a SERIES-level property: formed on each store-product's mean
    # y_true over the whole test window, so a series sits in exactly one decile
    # and cannot migrate between deciles with its own stockout status.
    sp_mean = df.groupby('sp')['y_true'].mean()
    dec = pd.qcut(sp_mean, N_DECILES, labels=False, duplicates='drop') + 1
    if dec.nunique() != N_DECILES:
        raise RuntimeError(f'demand deciles collapsed to {dec.nunique()} bins')
    df['decile'] = df['sp'].map(dec).astype(int)
    dec_tab = stratum_table(df, ['decile'])
    edges = sp_mean.groupby(dec).agg(['min', 'max', 'size'])
    dec_flags = []
    for d in range(1, N_DECILES + 1):
        e = edges.loc[d]
        detail = (f'decile {d}/10 of series-level mean y_true, '
                  f'range [{e["min"]:.4f}, {e["max"]:.4f}]')
        emit(rows, 'decile', d, detail, 'all', 'n_sp', e['size'])
        emit(rows, 'decile', d, detail, 'all', 'series_mean_y_min', e['min'])
        emit(rows, 'decile', d, detail, 'all', 'series_mean_y_max', e['max'])
        w = emit_strata(rows, dec_tab[dec_tab['decile'] == d], 'decile', d, detail)
        dec_flags.append(w['light'] < w['0h'])
    n_dec_light_lower = int(np.sum(dec_flags))
    emit(rows, 'decile', 'all', f'summary over all {N_DECILES} demand deciles',
         'light_vs_0h', 'n_groups', N_DECILES)
    emit(rows, 'decile', 'all', f'summary over all {N_DECILES} demand deciles',
         'light_vs_0h', 'n_groups_light_lower', n_dec_light_lower)

    # ---- (b2) deciles on an uncensored demand basis -----------------------
    # The basis in (b) is each series' mean observed sales over the whole test
    # window, which stockouts themselves depress, so conditioning on it does not
    # cleanly control for demand level. Here the deciles are rebuilt on each
    # series' mean over its ZERO-STOCKOUT rows only, so no truncated value
    # enters the basis; which days are uncensored is itself non-random (a
    # stockout can remove a high-demand day from the mean), so this basis
    # complements the one in (b). Series with no zero-stockout row cannot be placed
    # on this basis and are excluded; the excluded count is emitted so the
    # restriction is visible in the artifact rather than only in the code.
    sp_mean_unc = df[df['stratum'] == '0h'].groupby('sp')['y_true'].mean()
    n_sp_total = int(df['sp'].nunique())
    n_excluded = n_sp_total - int(len(sp_mean_unc))
    sub = df[df['sp'].isin(sp_mean_unc.index)].copy()
    dec_u = pd.qcut(sp_mean_unc, N_DECILES, labels=False, duplicates='drop') + 1
    if dec_u.nunique() != N_DECILES:
        raise RuntimeError(f'uncensored deciles collapsed to {dec_u.nunique()} bins')
    sub['decile'] = sub['sp'].map(dec_u).astype(int)
    dec_u_tab = stratum_table(sub, ['decile'])
    edges_u = sp_mean_unc.groupby(dec_u).agg(['min', 'max', 'size'])
    dec_u_flags = []
    for d in range(1, N_DECILES + 1):
        e = edges_u.loc[d]
        detail = (f'decile {d}/10 of series-level mean y_true computed over '
                  f'zero-stockout rows only, '
                  f'range [{e["min"]:.4f}, {e["max"]:.4f}]')
        emit(rows, 'decile_uncensored_basis', d, detail, 'all', 'n_sp', e['size'])
        emit(rows, 'decile_uncensored_basis', d, detail, 'all',
             'series_mean_y_min', e['min'])
        emit(rows, 'decile_uncensored_basis', d, detail, 'all',
             'series_mean_y_max', e['max'])
        w = emit_strata(rows, dec_u_tab[dec_u_tab['decile'] == d],
                        'decile_uncensored_basis', d, detail)
        wl, w0 = w.get('light'), w.get('0h')
        dec_u_flags.append(bool(wl is not None and w0 is not None and wl < w0))
    n_dec_u_light_lower = int(np.sum(dec_u_flags))
    summ_u = (f'summary over all {N_DECILES} deciles on the zero-stockout-only '
              f'demand basis; {n_excluded} of {n_sp_total} series have no '
              f'zero-stockout row and are excluded')
    emit(rows, 'decile_uncensored_basis', 'all', summ_u,
         'light_vs_0h', 'n_groups', N_DECILES)
    emit(rows, 'decile_uncensored_basis', 'all', summ_u,
         'light_vs_0h', 'n_groups_light_lower', n_dec_u_light_lower)
    emit(rows, 'decile_uncensored_basis', 'all', summ_u,
         'all', 'n_sp_excluded_no_uncensored_row', n_excluded)

    # ---- (c) test dates --------------------------------------------------
    dates = sorted(df['dt'].unique())
    date_tab = stratum_table(df, ['dt'])
    date_flags = []
    for i, d in enumerate(dates, start=1):
        detail = f'test date {d} (day {i} of {len(dates)})'
        emit(rows, 'date', d, detail, 'all', 'n_rows',
             int((df['dt'] == d).sum()))
        w = emit_strata(rows, date_tab[date_tab['dt'] == d], 'date', d, detail)
        date_flags.append(w['light'] < w['0h'])
    n_date_light_lower = int(np.sum(date_flags))
    emit(rows, 'date', 'all', f'summary over all {len(dates)} test dates',
         'light_vs_0h', 'n_groups', len(dates))
    emit(rows, 'date', 'all', f'summary over all {len(dates)} test dates',
         'light_vs_0h', 'n_groups_light_lower', n_date_light_lower)

    # ---- (d) within-store-product paired comparison ----------------------
    pair_src = df[df['stratum'].isin(['0h', 'light'])]
    per_sp = stratum_table(pair_src, ['sp'])
    wide = per_sp.pivot(index='sp', columns='stratum',
                        values=['n_rows', 'sum_y', 'wape_pct', 'mean_y'])
    present = wide[('n_rows', '0h')].notna() & wide[('n_rows', 'light')].notna()
    defined = present & (wide[('sum_y', '0h')] > 0) & (wide[('sum_y', 'light')] > 0)
    n_present, n_defined = int(present.sum()), int(defined.sum())

    pair = wide[defined]
    w0 = pair[('wape_pct', '0h')].to_numpy(dtype=float)
    wl = pair[('wape_pct', 'light')].to_numpy(dtype=float)
    diff = wl - w0
    n_light_lower = int((diff < 0).sum())
    n_0h_lower = int((diff > 0).sum())
    n_tied = int((diff == 0).sum())
    share_light_lower = n_light_lower / n_defined
    median_diff_pp = float(np.median(diff))
    wilcoxon_p, _ = paired_wilcoxon(wl, w0, alternative='less')

    paired_sps = pair.index.to_numpy()
    paired_rows = pair_src[pair_src['sp'].isin(paired_sps)]
    paired_tab = stratum_table(paired_rows.assign(_all=0), ['_all'])
    detail = ('store-products holding at least one 0h row and one light row '
              'with positive observed sales in both, so a per-series WAPE is '
              'defined in each stratum')
    emit(rows, 'paired', 'all', detail, 'all',
         'n_sp_with_rows_in_both_strata', n_present)
    emit(rows, 'paired', 'all', detail, 'all',
         'n_sp_paired_wape_defined', n_defined)
    for stratum in ('0h', 'light'):
        sel = paired_tab[paired_tab['stratum'] == stratum].iloc[0]
        emit(rows, 'paired', 'all', detail, stratum, 'n_rows', sel['n_rows'])
        emit(rows, 'paired', 'all', detail, stratum, 'mean_y', sel['mean_y'])
        emit(rows, 'paired', 'all', detail, stratum, 'pooled_wape_pct',
             sel['wape_pct'])
        vals = wl if stratum == 'light' else w0
        emit(rows, 'paired', 'all', detail, stratum, 'median_sp_wape_pct',
             float(np.median(vals)))
        emit(rows, 'paired', 'all', detail, stratum, 'mean_sp_wape_pct',
             float(np.mean(vals)))
    emit(rows, 'paired', 'all', detail, 'light_vs_0h', 'n_sp', n_defined)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'n_sp_light_lower', n_light_lower)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'n_sp_0h_lower', n_0h_lower)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h', 'n_sp_tied', n_tied)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'share_sp_light_lower', share_light_lower)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'median_sp_wape_diff_pp', median_diff_pp)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'median_sp_wape_diff_frac', median_diff_pp / 100)
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'mean_sp_wape_diff_pp', float(np.mean(diff)))
    for q in (10, 25, 75, 90):
        emit(rows, 'paired', 'all', detail, 'light_vs_0h',
             f'p{q}_sp_wape_diff_pp', float(np.percentile(diff, q)))
    emit(rows, 'paired', 'all', detail, 'light_vs_0h',
         'wilcoxon_signed_rank_p_light_less', wilcoxon_p)

    out = pd.DataFrame(rows, columns=['analysis', 'group_key', 'group_detail',
                                      'stratum', 'metric', 'value'])
    csv_path = out_dir / 'stockout_stratum_robustness.csv'
    out.to_csv(csv_path, index=False, float_format='%.6g')

    # ---- readable summary ------------------------------------------------
    print('\n(a) overall')
    print(overall[['stratum', 'n_rows', 'mean_y', 'wape_pct', 'wpe_pct']]
          .set_index('stratum').loc[list(STRATA)].round(4).to_string())
    print('\n(b) WAPE by demand decile (series-level mean y_true)')
    piv = dec_tab.pivot(index='decile', columns='stratum', values='wape_pct')
    piv = piv[list(STRATA)]
    piv['light-0h'] = piv['light'] - piv['0h']
    piv['light_lower'] = piv['light-0h'] < 0
    print(piv.round(4).to_string())
    print(f"    light < 0h in {n_dec_light_lower} of {N_DECILES} deciles")
    print('\n(b2) WAPE by demand decile (zero-stockout-only demand basis)')
    pivu = dec_u_tab.pivot(index='decile', columns='stratum', values='wape_pct')
    pivu = pivu[list(STRATA)]
    pivu['light-0h'] = pivu['light'] - pivu['0h']
    pivu['light_lower'] = pivu['light-0h'] < 0
    print(pivu.round(4).to_string())
    print(f"    light < 0h in {n_dec_u_light_lower} of {N_DECILES} deciles "
          f"({n_excluded} series excluded: no zero-stockout row)")
    print('\n(c) WAPE by test date')
    pivd = date_tab.pivot(index='dt', columns='stratum', values='wape_pct')
    pivd = pivd[list(STRATA)]
    pivd['light-0h'] = pivd['light'] - pivd['0h']
    pivd['light_lower'] = pivd['light-0h'] < 0
    print(pivd.round(4).to_string())
    print(f"    light < 0h on {n_date_light_lower} of {len(dates)} test dates")
    print('\n(d) within-store-product paired 0h vs light')
    print(f"    store-products with rows in both strata : {n_present:,}")
    print(f"    with a defined WAPE in both strata      : {n_defined:,}")
    print(f"    light lower                             : {n_light_lower:,} "
          f"({100 * share_light_lower:.1f}%)")
    print(f"    median paired WAPE difference           : {median_diff_pp:+.4f} pp "
          f"({median_diff_pp / 100:+.4f} as a fraction)")
    print(f"    Wilcoxon signed-rank p (light < 0h)     : {wilcoxon_p:.3g}")

    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    meta = {
        'script': 'stratum_robustness', 'git_sha': sha, 'args': vars(args),
        'model': MODEL,
        'prediction_column': f'pred_{MODEL}',
        'prediction_column_note': (
            'the calibrated ensemble column; predictions.csv also carries '
            'pred_ensemble_raw, on which the light-below-0h reversal also holds'),
        'strata': {'0h': 'stockout_hours == 0', 'light': '1 <= h <= 4',
                   'heavy': 'h >= 5'},
        'wape_definition': '100 * sum|pred - y| / sum(y)',
        'headline_finding': (
            'error is not monotonic in stockout exposure: light < 0h < heavy '
            f'overall ({w_overall["light"]:.4f} < {w_overall["0h"]:.4f} < '
            f'{w_overall["heavy"]:.4f} WAPE), the light-below-0h reversal holds '
            f'in {n_dec_light_lower}/{N_DECILES} demand deciles '
            f'({n_dec_u_light_lower}/{N_DECILES} on a zero-stockout-only '
            f'demand basis) and on '
            f'{n_date_light_lower}/{len(dates)} test dates, and within the '
            f'{n_defined} store-products measured in both strata light is lower '
            f'for {100 * share_light_lower:.1f}% (median paired difference '
            f'{median_diff_pp:+.4f} WAPE points)'),
        'note': ('strata are NON-RANDOM subsets: stockout exposure correlates '
                 'with demand level, promotions, and store practices. These '
                 'figures characterize where the pooled error comes from; they '
                 'do not validate latent-demand accuracy and are not bounds in '
                 'either direction. The reversal refutes a monotone '
                 'protocol-only reading of the 0h stratum, it does not '
                 'establish a causal account of its own.'),
        'runtime_s': round(time.time() - t0, 1),
    }
    (out_dir / 'stockout_stratum_robustness_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"\nDone in {meta['runtime_s']:.0f}s -> {csv_path}")


if __name__ == '__main__':
    main()
