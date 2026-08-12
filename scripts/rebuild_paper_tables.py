#!/usr/bin/env python3
"""
Regenerate every numeric table in research/paper.tex from output CSVs.

The paper source uses `\\input{generated_tables/<name>.tex}` for each numeric
table, so paper.tex content stays in sync with the latest pipeline run.
Tests (tests/test_paper_csv_sync.py) regenerate the tables into a temp dir
and compare byte-for-byte to the committed ones, failing CI on any drift.

Every table's source file is pinned in TABLE_SOURCES: a missing source is a
hard error (no silent directory fallback), and each emitted table carries a
`% source: <path> md5=<12 hex>` provenance line so the producing artifact is
auditable from the .tex alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
RESULTS = SCRIPT_DIR / 'results' / 'current'
DEFAULT_GEN = SCRIPT_DIR / 'research' / 'generated_tables'

# Single source of truth: compact, versioned result artifacts committed with
# the companion repository. Full local run directories can be regenerated but
# are deliberately excluded because their predictions and order matrices are
# large.
TABLE_SOURCES = {
    'inventory_results.csv': RESULTS,
    'inventory_selection.json': RESULTS,
    'forecast_metrics.json': RESULTS,
    'profit_inference.csv': RESULTS,
    'profit_paired_differences.csv': RESULTS,
    'inventory_cost_sensitivity.csv': RESULTS,
    'ablation_results.csv': RESULTS,
    'tobit_sensitivity.csv': RESULTS,
    'benchmark_forecast_comparison.csv': RESULTS,
    'wilcoxon_holm.csv': RESULTS,
    'multistep_validation_results.csv': RESULTS,
    'parsimony_results.csv': RESULTS,
}

ACCURACY_ARM_LABELS = {
    'A0_recursive_baseline': 'Recursive ensemble (paper baseline)',
    'A1_direct_horizon': 'Direct multi-horizon',
    'A2_tweedie_member': 'Recursive + Tweedie member',
    'A3_recentered': 'Recursive + CV recentering',
    'A4a_demrec_target': 'Recovered-demand target',
    'A4b_tobit_objective': 'Tobit objective (latent-demand head)',
    'A5_combo': 'Direct + Tweedie + recentering',
}


def _resolve(name: str) -> Path:
    """Resolve a table source via the manifest. Missing file = hard error;
    we never silently fall back to another run directory."""
    base = TABLE_SOURCES.get(name)
    if base is None:
        raise KeyError(f"{name} is not registered in TABLE_SOURCES")
    p = base / name
    if not p.exists():
        raise FileNotFoundError(
            f"Table source {p} is missing. Regenerate it with the pipeline "
            f"before rebuilding paper tables; refusing to fall back to "
            f"another run directory."
        )
    return p


def _prov(*paths: Path) -> str:
    """Provenance comment naming each source file and a short content hash."""
    parts = []
    for p in paths:
        digest = hashlib.md5(p.read_bytes()).hexdigest()[:12]
        parts.append(f"{p.relative_to(SCRIPT_DIR)} md5={digest}")
    return "% source: " + "; ".join(parts)


def _latex_escape_policy(name: str) -> str:
    """Convert a policy name to a LaTeX-safe representation.

    Handles Unicode sigma (U+03C3) and common special characters that
    pdflatex's default font cannot render directly.
    """
    if name is None:
        return ''
    s = str(name)
    # Unicode Greek sigma -> LaTeX math sigma
    s = s.replace('\u03c3', '$\\sigma$')
    s = s.replace('\u03b1', '$\\alpha$')
    # Underscores and percent signs must be escaped in text mode
    s = s.replace('_', '\\_')
    s = s.replace('%', '\\%')
    return s


def emit_table_policies() -> str:
    """Top test-window policies plus the validation-preselected row and CIs."""
    csv = _resolve('inventory_results.csv')
    ci_csv = _resolve('profit_inference.csv')
    df = pd.read_csv(csv)
    ci = pd.read_csv(ci_csv).set_index('Policy')
    preselected_name = _read_preselected_policy()
    pre_row = df[df['Policy'] == preselected_name]
    if len(pre_row) != 1:
        raise ValueError(f"Expected one row for preselected policy {preselected_name!r}")
    selected_profit = float(pre_row.iloc[0]['Avg Profit'])
    # Show every test-window policy that out-earns the validation-selected
    # policy, plus the selected row. This makes threshold crossings visible
    # without inviting a post-hoc winner swap.
    top = (df[df['Avg Profit'] >= selected_profit]
           .sort_values('Avg Profit', ascending=False))
    lines = [
        "% Auto-generated from the corrected headline inventory and inference artifacts",
        _prov(csv, ci_csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Every inventory policy that out-earns the validation-selected policy "
        "on the held-out test window, plus the selected policy itself. "
        "The row marked $\\star$ is the validation-preselected policy whose test metrics we "
        "report as the headline; all other rows are transparency only. Profit intervals are "
        "store-clustered bootstrap 95\\% CIs (898 stores, 10{,}000 resamples). Every policy "
        "above the starred row fails the 0.90 fill-rate floor on the selection fold. "
        "Newsvendor-Normal, SAA, and SS($0.25\\sigma$) cross the threshold on test, "
        "illustrating near-threshold sampling variation without changing eligibility.}",
        "\\label{tab:policies}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        " & \\textbf{Policy} & \\textbf{SL} & \\textbf{Fill Rate} & "
        "\\textbf{Profit}$\\,\\uparrow$ & \\textbf{Profit 95\\% CI} & "
        "\\textbf{SO Rate} & \\textbf{Waste\\%} \\\\",
        "\\midrule",
    ]
    for _, r in top.iterrows():
        pol = _latex_escape_policy(r['Policy'])
        mark = '$\\star$' if r['Policy'] == preselected_name else ''
        c = ci.loc[r['Policy']]
        ci_text = f"[{c['ci95_lo']:.3f}, {c['ci95_hi']:.3f}]"
        row = (f"{mark} & {pol} & {r['SL(Type1)']:.3f} & {r['Fill Rate']:.3f} & "
               f"{r['Avg Profit']:.4f} & {ci_text} & {r['SO Rate']:.3f} & "
               f"{r['Waste%']:.1f}\\% \\\\")
        if r['Policy'] == preselected_name:
            row = (f"{mark} & \\textbf{{{pol}}} & "
                   f"\\textbf{{{r['SL(Type1)']:.3f}}} & \\textbf{{{r['Fill Rate']:.3f}}} & "
                   f"\\textbf{{{r['Avg Profit']:.4f}}} & \\textbf{{{ci_text}}} & "
                   f"\\textbf{{{r['SO Rate']:.3f}}} & "
                   f"\\textbf{{{r['Waste%']:.1f}}}\\% \\\\")
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def _read_preselected_policy() -> str:
    """Read the validation-fold preselected policy name from inventory_selection.json."""
    p = _resolve('inventory_selection.json')
    data = json.loads(p.read_text())
    name = data.get('preselected_policy')
    if not name:
        raise ValueError(f"{p} has no 'preselected_policy' key")
    return name


def emit_table_before_after() -> str:
    """tab:before_after -- Static MinMax vs ML Forecast vs ML+Optimization.

    The ML+Optimization row uses whichever policy was PRE-SELECTED on the
    validation fold (read from inventory_selection.json), not a hardcoded
    policy name. If no selection file is present, falls back to the
    Empirical Newsvendor row for backward compatibility.
    """
    csv = _resolve('inventory_results.csv')
    ci_csv = _resolve('profit_inference.csv')
    df = pd.read_csv(csv)
    ci = pd.read_csv(ci_csv).set_index('Policy')

    def find_row(name: str) -> pd.Series | None:
        # Exact match first; fall back to substring
        exact = df[df['Policy'] == name]
        if len(exact):
            return exact.iloc[0]
        fuzzy = df[df['Policy'].str.contains(name, case=False, regex=False)]
        return fuzzy.iloc[0] if len(fuzzy) else None

    static = find_row('Static MinMax')
    naive = find_row('Naive')
    preselected_name = _read_preselected_policy()
    chosen = find_row(preselected_name)

    if any(r is None for r in (static, naive, chosen)):
        raise ValueError(
            "inventory_results.csv is missing one of the required rows "
            f"(Static MinMax / Naive / {preselected_name})"
        )

    # Escape underscores / percent / unicode in policy name for LaTeX
    chosen_latex = _latex_escape_policy(preselected_name)
    lines = [
        "% Auto-generated from the corrected headline inventory and inference artifacts",
        _prov(csv, ci_csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{End-to-end business impact from one corrected coherent full-scale run: "
        "static rules vs ML vs ML + optimization. Profit intervals are store-clustered "
        "bootstrap 95\\% CIs (898 stores, 10{,}000 resamples). "
        f"The validation-preselected policy for this run is {chosen_latex} "
        "(see Section~\\ref{sec:optimization}).}",
        "\\label{tab:before_after}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        " & \\textbf{Approach} & \\textbf{SL} & \\textbf{Fill Rate} & "
        "\\textbf{Profit}$\\,\\uparrow$ & \\textbf{Profit 95\\% CI} & "
        "\\textbf{SO Rate} & \\textbf{Waste\\%} \\\\",
        "\\midrule",
        f"(a) & Static MinMax & {static['SL(Type1)']:.3f} & {static['Fill Rate']:.3f} & "
        f"{static['Avg Profit']:.4f} & [{ci.loc[static['Policy'], 'ci95_lo']:.3f}, "
        f"{ci.loc[static['Policy'], 'ci95_hi']:.3f}] & {static['SO Rate']:.3f} & "
        f"{static['Waste%']:.1f}\\% \\\\",
        f"(b) & ML Forecast Only & {naive['SL(Type1)']:.3f} & {naive['Fill Rate']:.3f} & "
        f"{naive['Avg Profit']:.4f} & [{ci.loc[naive['Policy'], 'ci95_lo']:.3f}, "
        f"{ci.loc[naive['Policy'], 'ci95_hi']:.3f}] & {naive['SO Rate']:.3f} & "
        f"{naive['Waste%']:.1f}\\% \\\\",
        f"(c) & ML + preselected {chosen_latex} & "
        f"{chosen['SL(Type1)']:.3f} & \\textbf{{{chosen['Fill Rate']:.3f}}} & "
        f"\\textbf{{{chosen['Avg Profit']:.4f}}} & "
        f"[{ci.loc[chosen['Policy'], 'ci95_lo']:.3f}, "
        f"{ci.loc[chosen['Policy'], 'ci95_hi']:.3f}] & "
        f"{chosen['SO Rate']:.3f} & {chosen['Waste%']:.1f}\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def emit_table_full_results() -> str:
    """tab:full_results -- forecast metrics for full run."""
    mj = _resolve('forecast_metrics.json')
    data = json.loads(mj.read_text())
    rows = []
    for name in ['mae', 'huber', 'q50', 'ensemble']:
        key = f'{name}_raw'
        if key not in data:
            continue
        r = data[key]
        rows.append({
            'Model': name.capitalize(),
            'MAE': r.get('MAE', float('nan')),
            'RMSE': r.get('RMSE', float('nan')),
            'Corr': r.get('Corr', float('nan')),
            'WAPE': r.get('WAPE', float('nan')),
            'WPE': r.get('WPE', float('nan')),
        })
    lines = [
        "% Auto-generated from the corrected coherent headline forecast artifact",
        _prov(mj),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Raw ensemble-member evaluation on the 7-day held-out window, before "
        "the validation-defined recentering in Table~\\ref{tab:calibration_context}.}",
        "\\label{tab:full_results}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{Corr} & "
        "\\textbf{WAPE(\\%)}$\\,\\downarrow$ & \\textbf{WPE(\\%)} \\\\",
        "\\midrule",
    ]
    best = {
        'MAE': min(r['MAE'] for r in rows),
        'RMSE': min(r['RMSE'] for r in rows),
        'Corr': max(r['Corr'] for r in rows),
        'WAPE': min(r['WAPE'] for r in rows),
    }
    for r in rows:
        def fmt(metric, precision):
            value = r[metric]
            return (f"\\textbf{{{value:.{precision}f}}}"
                    if value == best.get(metric)
                    else f"{value:.{precision}f}")
        lines.append(
            f"{r['Model']} & {fmt('MAE', 4)} & {fmt('RMSE', 4)} & "
            f"{fmt('Corr', 4)} & {fmt('WAPE', 2)} & {r['WPE']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_calibration_context() -> str:
    """Raw and recentered ensembles from the same corrected headline run."""
    metrics = _resolve('forecast_metrics.json')
    data = json.loads(metrics.read_text())
    rows = [
        {
            'run': 'Leak-free ensemble (raw)',
            'source': 'Uncalibrated ensemble of the coherent headline run',
            'wape': data['ensemble_raw']['WAPE'],
            'wpe': data['ensemble_raw']['WPE'],
        },
        {
            'run': 'CV-scale calibrated ensemble',
            'source': 'Headline configuration retained by validation gate',
            'wape': data['ensemble']['WAPE'],
            'wpe': data['ensemble']['WPE'],
        },
    ]

    lines = [
        "% Auto-generated from the corrected coherent headline forecast artifact",
        _prov(metrics),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Effect of the CPU-only recentering calibration in one coherent "
        "full-scale run. The calibrated configuration was retained by the "
        "validation gate in Table~\\ref{tab:accuracy_ablation}; published "
        "deep-learning values from other protocols are discussed, not tabulated, "
        "in Section~\\ref{sec:discussion}.}",
        "\\label{tab:calibration_context}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}p{3.2cm}p{6.2cm}cc@{}}",
        "\\toprule",
        "\\textbf{Run} & \\textbf{Source / status} & "
        "\\textbf{WAPE(\\%)}$\\,\\downarrow$ & \\textbf{WPE(\\%)} \\\\",
        "\\midrule",
    ]
    for r in rows:
        source = _latex_escape_policy(r['source'])
        lines.append(
            f"{r['run']} & {source} & {float(r['wape']):.2f} & "
            f"{float(r['wpe']):+.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_ablation() -> str:
    """tab:ablation: decompose the components along four dimensions.

    Instead of a single slash-joined configuration string (which is not
    publication-appropriate), the ablation table presents four separate
    columns -- Recovery, Features, Model, and Policy -- each with a
    readable natural-language value. The Config letter (A--F) indexes
    the rows. $\\Delta$Profit is the change vs. the previous row.
    """
    csv = _resolve('ablation_results.csv')
    df = pd.read_csv(csv)

    def _split(component_str: str) -> tuple[str, str, str, str]:
        parts = [p.strip() for p in str(component_str).split('/')]
        while len(parts) < 4:
            parts.append('-')
        # Compact academic abbreviations -- full glossary is given in the caption.
        rec = {
            'none': 'None',
            'heuristic_imr': 'IMR',
            'tobit_mle': 'Tobit MLE',
        }.get(parts[0], parts[0])
        feat = {
            'minimal': 'basic (14)',
            'full': 'full (117)',
        }.get(parts[1], parts[1])
        mdl = {
            'single': 'MAE',
            'ensemble': 'Ensemble',
        }.get(parts[2], parts[2])
        pol = {
            'naive': 'Na{\\"\\i}ve',
            'empirical_nv': 'Emp.\\ NV',
        }.get(parts[3], parts[3])
        return rec, feat, mdl, pol

    lines = [
        "% Auto-generated from ablation_results.csv",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Component-wise ablation of the leak-free pipeline. Columns: "
        "\\emph{Recovery} (None; IMR = inverse-Mills-ratio heuristic; "
        "Tobit MLE = maximum-likelihood Tobit); \\emph{Features} (basic = 14 "
        "lag/calendar features; full = 117 features); \\emph{Model} "
        "(MAE = single $L_1$ LightGBM; Ensemble = MAE + Huber + quantile "
        "ensemble); \\emph{Policy} (Na{\\\"\\i}ve: $Q=\\hat y$; "
        "Emp.\\ NV: Empirical Newsvendor). Each row changes a single stage "
        "relative to the row above; $\\Delta$Profit is the incremental "
        "change in average profit.}",
        "\\label{tab:ablation}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}c l l l l r r r r@{}}",
        "\\toprule",
        " & \\textbf{Recovery} & \\textbf{Features} & \\textbf{Model} & "
        "\\textbf{Policy} & \\textbf{WAPE(\\%)} & \\textbf{WPE(\\%)} & "
        "\\textbf{Profit} & \\textbf{$\\Delta$Profit} \\\\",
        "\\midrule",
    ]
    prev_profit = None
    for _, r in df.iterrows():
        rec, feat, mdl, pol = _split(r.get('components', ''))
        delta = (r['profit'] - prev_profit) if prev_profit is not None else 0.0
        lines.append(
            f"({r['config']}) & {rec} & {feat} & {mdl} & {pol} & "
            f"{r['wape']:.2f} & {r['wpe']:+.2f} & {r['profit']:.4f} & "
            f"{delta:+.4f} \\\\"
        )
        prev_profit = r['profit']
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_tobit_sensitivity() -> str:
    csv = _resolve('tobit_sensitivity.csv')
    df = pd.read_csv(csv)
    lines = [
        "% Auto-generated by scripts/rebuild_paper_tables.py from tobit_sensitivity.csv",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Recovery comparison under the full leak-free protocol on a "
        "stratified 1{,}500-SP subsample: heuristic-IMR blending weight "
        "$w \\in \\{0, 0.25, 0.5, 0.75, 1.0\\}$ vs the per-store-product Tobit "
        "MLE baseline. Uplift is the mean recovered-demand increase on censored "
        "rows; WAPE/WPE are downstream forecast metrics; Profit is average "
        "per SP-day under the OOS Empirical Newsvendor policy for that recovery arm.}",
        "\\label{tab:tobit_sensitivity}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "\\textbf{Config} & \\textbf{Blend $w$} & \\textbf{Uplift(\\%)} & "
        "\\textbf{WAPE(\\%)} & \\textbf{WPE(\\%)} & \\textbf{Profit} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        # Build a human-readable configuration label
        if pd.isna(r['blend']):
            row_label = "Tobit MLE"
        else:
            row_label = f"IMR heuristic, $w = {r['blend']:.2f}$"
        # Blend column repeats the value (MLE = ---)
        blend_val = '---' if pd.isna(r['blend']) else f"{r['blend']:.2f}"
        lines.append(
            f"{row_label} & {blend_val} & {r['uplift_pct']:.1f} & "
            f"{r['wape']:.2f} & {r['wpe']:+.2f} & {r['profit']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_cost_sensitivity() -> str:
    csv = _resolve('inventory_cost_sensitivity.csv')
    df = pd.read_csv(csv)
    lines = [
        "% Auto-generated by scripts/rebuild_paper_tables.py from inventory_cost_sensitivity.csv",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Preselected-policy profit across cost configurations ($p_s$, $w$).}",
        "\\label{tab:cost_sensitivity}",
        "\\footnotesize",
        "\\begin{tabular}{@{}ccccc@{}}",
        "\\toprule",
        "\\textbf{$p_s$} & \\textbf{w} & \\textbf{Profit} & "
        "\\textbf{Fill Rate} & \\textbf{Waste\\%} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['p']:.2f} & {r['w']:.2f} & {r['profit']:.4f} & "
            f"{r['fill_rate']:.3f} & {r['waste_pct']:.1f}\\% \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_benchmark_forecast() -> str:
    """tab:benchmark_forecast — leak-free benchmark comparison."""
    csv = _resolve('benchmark_forecast_comparison.csv')
    df = pd.read_csv(csv)
    lines = [
        "% Auto-generated by scripts/rebuild_paper_tables.py from benchmark_forecast_comparison.csv",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Forecast accuracy comparison under leak-free conditions "
        "(stratified subsample; identical features across GBDT models).}",
        "\\label{tab:benchmark_forecast}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{MAE}$\\,\\downarrow$ & \\textbf{RMSE}$\\,\\downarrow$ & "
        "\\textbf{SMAPE}$\\,\\downarrow$ & \\textbf{WAPE}$\\,\\downarrow$ & "
        "\\textbf{WPE} & \\textbf{Corr}$\\,\\uparrow$ \\\\",
        "\\midrule",
    ]
    # Mark best per column (lower-is-better for MAE/RMSE/SMAPE/WAPE, |WPE| close to 0)
    best_mae = df['MAE'].min()
    best_wape = df['WAPE'].min()
    best_corr = df['Corr'].max()
    for _, r in df.iterrows():
        fm = lambda v, d: f"\\textbf{{{v:.{d}f}}}" if False else f"{v:.{d}f}"
        mae_v = f"\\textbf{{{r['MAE']:.4f}}}" if r['MAE'] == best_mae else f"{r['MAE']:.4f}"
        wape_v = f"\\textbf{{{r['WAPE']:.2f}}}" if r['WAPE'] == best_wape else f"{r['WAPE']:.2f}"
        corr_v = f"\\textbf{{{r['Corr']:.4f}}}" if r['Corr'] == best_corr else f"{r['Corr']:.4f}"
        lines.append(
            f"{r['Model']} & {mae_v} & {r['RMSE']:.4f} & "
            f"{r['SMAPE']:.1f}\\% & {wape_v} & {r['WPE']:+.2f} & {corr_v} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_accuracy_ablation() -> str:
    """Common-protocol validation ablation and fixed promotion-gate result."""
    csv = _resolve('multistep_validation_results.csv')
    df = pd.read_csv(csv)
    labels = {
        'A0_recursive': 'Recursive ensemble (baseline)',
        'A1_direct': 'Direct multi-horizon',
        'A2_recursive_tweedie': 'Recursive + Tweedie member',
        'A3_recursive_recentered': 'Recursive + CV recentering',
        'A5_direct_tweedie_recentered': 'Direct + Tweedie + recentering',
        'A4a_demrec_target': 'Recovered-demand target (diagnostic)',
        'A4b_tobit_objective': 'Tobit objective (diagnostic)',
    }
    order = list(labels)
    df['_o'] = df['arm'].map({a: i for i, a in enumerate(order)})
    df = df.sort_values('_o')
    lines = [
        "% Auto-generated from the common multi-step validation artifact",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Common-protocol accuracy ablation (stratified 1{,}500-SP "
        "subsample). Every arm produces a true 7-day multi-step forecast from "
        "each fold origin, with fold-causal recovery/features and nested early "
        "stopping. Selection pools folds 2--4; fold 1 seeds recentering. The "
        "dagger marks the lowest-WAPE gate candidate; the star marks the "
        "retained headline after the candidate failed bias gate G3. The held-out "
        "test window never participates.}",
        "\\label{tab:accuracy_ablation}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        " & \\textbf{Forecasting variant} & \\textbf{Pooled val WAPE}$\\,\\downarrow$ & "
        "\\textbf{Pooled val WPE} & \\textbf{Fold 2} & \\textbf{Fold 3} & "
        "\\textbf{Fold 4} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        label = labels.get(r['arm'], r['arm'])
        mark = '$\\star$' if int(r['headline']) else (
            '$\\dagger$' if int(r['candidate']) else '')
        cells = (f"{r['pooled_val_wape']:.2f} & {r['pooled_val_wpe']:+.2f} & "
                 f"{r['wape_fold2']:.2f} & {r['wape_fold3']:.2f} & "
                 f"{r['wape_fold4']:.2f}")
        if int(r['headline']):
            label = f"\\textbf{{{label}}}"
            cells = ' & '.join(
                f"\\textbf{{{cell.strip()}}}" for cell in cells.split('&'))
        lines.append(f"{mark} & {label} & {cells} \\\\")
        if r['arm'] == 'A5_direct_tweedie_recentered':
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_parsimony() -> str:
    """Feature-parsimony sweep from the current validation artifact."""
    csv = _resolve('parsimony_results.csv')
    df = pd.read_csv(csv)
    labels = {
        'top5': 'Top-5 (gain-ranked)',
        'top10': 'Top-10 (gain-ranked)',
        'top20': 'Top-20 (gain-ranked)',
        'minimal16': 'Minimal (hand-picked)',
        'full117': 'Full feature set',
    }
    lines = [
        "% Auto-generated from the feature-parsimony artifact",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\caption{Feature-parsimony sweep (stratified 1{,}500-SP subsample, "
        "recursive three-member ensemble, common fold-causal validation). "
        "Feature ranking uses folds 1--3 only; fold 4 and the held-out test "
        "never contribute. Test columns are transparency only.}",
        "\\label{tab:parsimony}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        "\\textbf{Feature set} & $k$ & \\textbf{Pooled val WAPE}$\\,\\downarrow$ & "
        "\\textbf{Pooled val WPE} & \\textbf{Test WAPE} & \\textbf{Test WPE} & "
        "\\textbf{Memory (MB)} & \\textbf{Runtime (s)} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{labels.get(r['config'], r['config'])} & {int(r['k'])} & "
            f"{r['pooled_val_wape']:.2f} & {r['pooled_val_wpe']:+.2f} & "
            f"{r['test_wape']:.2f} & {r['test_wpe']:+.2f} & "
            f"{r['design_matrix_mb']:.1f} & {r['runtime_s']:.0f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def emit_table_wilcoxon() -> str:
    csv = _resolve('wilcoxon_holm.csv')
    df = pd.read_csv(csv)
    # Some upstream runs may include models not present in the benchmark CSV.
    # Keep only implemented comparisons in the manuscript tables.
    notes = df['note'] if 'note' in df.columns else ''
    df = df[~pd.Series(notes).fillna('').str.lower().eq('model not in predictions csv')].copy()
    if len(df) == 0:
        raise ValueError("wilcoxon_holm.csv contains no implemented comparisons")
    n_comp = len(df)
    lines = [
        "% Auto-generated by scripts/rebuild_paper_tables.py from wilcoxon_holm.csv",
        _prov(csv),
        "\\begin{table}[!tbp]",
        "\\centering",
        f"\\caption{{Paired Wilcoxon signed-rank tests: LightGBM-Minimal vs each alternative "
        f"(Holm--Bonferroni over {n_comp} paired hypotheses).}}",
        "\\label{tab:wilcoxon}",
        "\\footnotesize",
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "\\textbf{Comparison} & \\textbf{Metric} & \\textbf{$p$-raw} & "
        "\\textbf{$p$-Holm} & \\textbf{Effect ($r$)} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        p_raw = 'NaN' if pd.isna(r['p_raw']) else f"{r['p_raw']:.2e}"
        p_holm = 'NaN' if pd.isna(r['p_holm']) else f"{r['p_holm']:.2e}"
        lines.append(
            f"vs {r['alt_model']} & {r['metric']} & {p_raw} & {p_holm} & {r['r_biserial']:+.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(DEFAULT_GEN),
                    help='directory to write table_*.tex into '
                         '(tests use a temp dir for byte-comparison)')
    args = ap.parse_args()
    gen = Path(args.out_dir)
    gen.mkdir(parents=True, exist_ok=True)

    tables = {
        'table_policies.tex': emit_table_policies(),
        'table_before_after.tex': emit_table_before_after(),
        'table_full_results.tex': emit_table_full_results(),
        'table_calibration_context.tex': emit_table_calibration_context(),
        'table_ablation.tex': emit_table_ablation(),
        'table_tobit_sensitivity.tex': emit_table_tobit_sensitivity(),
        'table_cost_sensitivity.tex': emit_table_cost_sensitivity(),
        'table_benchmark_forecast.tex': emit_table_benchmark_forecast(),
        'table_wilcoxon.tex': emit_table_wilcoxon(),
        'table_accuracy_ablation.tex': emit_table_accuracy_ablation(),
        'table_parsimony.tex': emit_table_parsimony(),
    }
    for fname, content in tables.items():
        (gen / fname).write_text(content)
        print(f"  wrote {fname} ({len(content)} bytes)")
    print(f"\nDone. Tables in {gen}/")


if __name__ == '__main__':
    main()
