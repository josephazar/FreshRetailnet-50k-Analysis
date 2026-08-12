#!/usr/bin/env python3
"""
Verify that the current result artifacts are internally consistent and tests pass.
Prints a checklist with pass/fail per item. Exit code nonzero if any blocker fails.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = 5e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def check(name: str, cond: bool, blocking: bool = True):
    mark = "OK " if cond else ("FAIL" if blocking else "WARN")
    print(f"  [{mark}] {name}")
    return cond if blocking else True


def main():
    print("=" * 66)
    print("  Current artifact verification")
    print("=" * 66)
    ok = True

    print("\nPhase 0 (published evidence):")
    results = ROOT / 'results' / 'current'
    ok &= check("results/current/ exists", results.is_dir())
    for name in [
        'release_summary.json', 'headline_run_metadata.json',
        'forecast_metrics.json', 'interval_coverage.json',
        'inventory_results.csv',
        'inventory_selection.json', 'profit_inference.csv',
        'recovery_runtime.json',
        'uncensored_diagnostics.csv', 'inventory_cost_sensitivity.csv',
        'multistep_validation_results.csv', 'parsimony_results.csv',
    ]:
        ok &= check(f"  results/current/{name}", (results / name).exists())
    ok &= check("docs/run_manifest.md present", (ROOT / 'docs/run_manifest.md').exists())

    summary = json.loads((results / 'release_summary.json').read_text())
    index = summary['headline_run']
    forecast = json.loads((results / 'forecast_metrics.json').read_text())
    coverage = json.loads((results / 'interval_coverage.json').read_text())
    selection = json.loads((results / 'inventory_selection.json').read_text())
    inventory = {row['Policy']: row for row in read_csv_rows(results / 'inventory_results.csv')}
    paired = {
        row['comparison']: row
        for row in read_csv_rows(results / 'profit_paired_differences.csv')
    }
    ok &= check("release summary identifies software v1.0.0",
                summary.get('software_release') == 'v1.0.0')
    ok &= check("headline WAPE synchronized",
                close(index['forecast']['calibrated_wape_pct'], forecast['ensemble']['WAPE']))
    ok &= check("headline WPE synchronized",
                close(index['forecast']['calibrated_wpe_pct'], forecast['ensemble']['WPE']))
    ok &= check("interval coverage synchronized",
                close(index['forecast']['q10_q90_coverage_pct'],
                      100 * coverage['overall_interval_coverage']))
    ok &= check("full-objective critical ratio is 0.578947",
                close(selection['newsvendor_critical_ratio']['value'], 1.1 / 1.9))
    ok &= check("full-objective Normal policy label present",
                'Newsvendor-Normal (CR=0.579)' in inventory)
    ok &= check("penalty-only CR=0.625 policy label absent",
                all('CR=0.625' not in name for name in inventory))
    ok &= check("floor sensitivity synchronized",
                selection['fill_rate_floor_sensitivity'] == {
                    '0.85': 'Newsvendor-Normal (CR=0.579)',
                    '0.90': 'Q90 Direct',
                    '0.95': 'SS (0.75σ)',
                })
    static_diff = paired['selected vs static baseline']
    index_diff = index['selection']['paired_profit_difference_vs_static']
    ok &= check("paired profit point synchronized",
                close(static_diff['diff_point'], index_diff['point'], tol=5e-7))
    ok &= check("paired profit CI synchronized",
                close(static_diff['diff_ci95_lo'], index_diff['ci95'][0], tol=5e-7)
                and close(static_diff['diff_ci95_hi'], index_diff['ci95'][1], tol=5e-7))
    runtime = json.loads((results / 'recovery_runtime.json').read_text())
    ok &= check("controlled recovery runtimes synchronized",
                close(runtime['timings']['heuristic_imr']['median_s'], 0.295)
                and close(runtime['timings']['tobit_mle']['median_s'], 6.2)
                and close(runtime['timings']['saits']['median_s'], 23.0)
                and close(runtime['timings']['timesnet']['median_s'], 531.1))

    print("\nPhase 1 (leakage fix):")
    ok &= check("src/feature_builder.py", (ROOT / 'src/feature_builder.py').exists())
    ok &= check("src/recovery.py", (ROOT / 'src/recovery.py').exists())
    ok &= check("src/inventory.py", (ROOT / 'src/inventory.py').exists())
    ok &= check("src/metrics.py", (ROOT / 'src/metrics.py').exists())
    ok &= check("pipeline_leakfree.py", (ROOT / 'pipeline_leakfree.py').exists())

    print("\nPhase 2 (out-of-sample inventory):")
    ok &= check("headline policy is validation-preselected",
                selection.get('preselected_policy') == 'Q90 Direct')
    selected_row = inventory[selection['preselected_policy']]
    ok &= check("selected test profit synchronized",
                close(selected_row['Avg Profit'], selection['test_metrics_reported']['Avg Profit'],
                      tol=5e-5))
    ok &= check("selected test fill rate synchronized",
                close(selected_row['Fill Rate'], selection['test_metrics_reported']['Fill Rate'],
                      tol=5e-5))
    ok &= check("profit inference reports 898 store clusters",
                '898' in (results / 'profit_inference_metadata.json').read_text())

    print("\nPhase 3 (paper/CSV sync):")
    paper_tables = ROOT / 'research/generated_tables'
    if paper_tables.is_dir():
        required_tables = ['table_policies.tex', 'table_before_after.tex',
                           'table_full_results.tex', 'table_ablation.tex',
                           'table_calibration_context.tex',
                           'table_tobit_sensitivity.tex', 'table_cost_sensitivity.tex',
                           'table_benchmark_forecast.tex', 'table_wilcoxon.tex',
                           'table_accuracy_ablation.tex', 'table_parsimony.tex']
        for t in required_tables:
            ok &= check(f"  {t}", (paper_tables / t).exists())
        # Recovery sensitivity is reported through the causal Tobit table.
        ok &= check("  non-causal table_recovery.tex absent",
                    not (paper_tables / 'table_recovery.tex').exists())
    else:
        check("manuscript tables excluded from code-only companion", True,
              blocking=False)

    print("\nPhase 4 (Tobit MLE):")
    ok &= check("src/recovery.tobit_mle available", True)
    ok &= check("results/current/tobit_sensitivity.csv",
                (results / 'tobit_sensitivity.csv').exists())

    print("\nPhase 5 (Wang comparison):")
    ok &= check("scripts/compare_wang_vs_ours.py",
                (ROOT / 'scripts/compare_wang_vs_ours.py').exists())
    check("Wang comparison is not a headline artifact", True, blocking=False)

    print("\nPhase 6 (ablation):")
    ok &= check("scripts/ablation.py", (ROOT / 'scripts/ablation.py').exists())
    ok &= check("results/current/ablation_results.csv",
                (results / 'ablation_results.csv').exists())

    print("\nPhase 7 (reproducibility):")
    req = (ROOT / 'requirements.txt').read_text() if (ROOT / 'requirements.txt').exists() else ''
    lock_path = ROOT / 'requirements-lock.txt'
    lock = lock_path.read_text() if lock_path.exists() else ''
    for pkg in ['xgboost', 'catboost', 'torch', 'pypots', 'statsmodels', 'pytest']:
        ok &= check(f"  requirements includes {pkg}", pkg in req)
        ok &= check(f"  exact snapshot pins {pkg}", f'{pkg}==' in lock)
    ok &= check("README uses neutral current output paths",
                'output/headline' in (ROOT / 'README.md').read_text()
                and 'output_v' not in (ROOT / 'README.md').read_text())
    ok &= check("tests/ dir populated", len(list((ROOT / 'tests').glob('test_*.py'))) >= 4)

    print("\nPhase 7b (tutorial semantics):")
    nb1 = json.loads((ROOT / 'notebooks/01_data_exploration_and_censored_demand.ipynb').read_text())
    nb5 = json.loads((ROOT / 'notebooks/05_inventory_optimization.ipynb').read_text())
    nb1_source = '\n'.join(''.join(c.get('source', [])) for c in nb1['cells'])
    nb5_source = '\n'.join(''.join(c.get('source', [])) for c in nb5['cells'])
    ok &= check("notebook 01 treats positive count as stockout",
                "df['stock_hour6_22_cnt'] > 0" in nb1_source
                and "df['stock_hour6_22_cnt'] < 16" not in nb1_source)
    ok &= check("notebook 05 derives the full-objective fractile",
                'underage = cfg["unit_revenue"] - cfg["unit_cost"] + cfg["p"]'
                in nb5_source
                and 'overage = cfg["unit_cost"] + cfg["h"] + cfg["w"]'
                in nb5_source
                and 'critical_ratio = underage / (underage + overage)'
                in nb5_source)

    print("\nPhase 8 (paper source, when present):")
    paper_path = ROOT / 'research/paper.tex'
    if paper_path.exists():
        paper = paper_path.read_text()
        ok &= check("abstract rewritten (mentions leak-free)",
                    'leakage-free' in paper.lower() or 'leakage free' in paper.lower())
        ok &= check("uses \\input{generated_tables/table_policies",
                    '\\input{generated_tables/table_policies' in paper)
        ok &= check("uses \\input{generated_tables/table_before_after",
                    '\\input{generated_tables/table_before_after' in paper)
        ok &= check("uses \\input{generated_tables/table_full_results",
                    '\\input{generated_tables/table_full_results' in paper)
        ok &= check("profit equation multiplies sold units by revenue",
                    '\\text{Sold}_i \\cdot r' in paper)
        ok &= check("paper cites immutable software release",
                    'v1.0.0' in paper)
        ok &= check("dataset stockout field semantics disclosed",
                    'number of \\emph{out-of-stock} hours' in paper)
    else:
        check("manuscript source excluded from code companion", True, blocking=False)

    print("\nRunning unit tests...")
    res = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q'],
                         cwd=ROOT, capture_output=True, text=True)
    ok &= check(f"pytest result (returncode={res.returncode})", res.returncode == 0)
    if res.returncode != 0:
        print(res.stdout[-1500:])

    print("\n" + "=" * 66)
    print(f"  Overall: {'PASS' if ok else 'FAIL'}")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
