# Censored demand forecasting and inventory optimization for fresh retail

An end-to-end, leakage-free pipeline for perishable retail: estimate demand
under stockouts, forecast it with gradient boosting, and convert forecasts into
order quantities with Newsvendor-family policies. The implementation is built
and evaluated on
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K):
50,000 store-product series with 90 days of daily sales records and hourly
stock-status vectors.

This repository contains the current implementation, tests, compact result
artifacts, and reproducibility metadata for software release `v1.1.0`.

## Results

Fold-causal protocol on all 50,000 store-products:

| Metric | Value |
|---|---|
| Forecast WAPE, raw ensemble | 33.54% |
| Forecast WAPE, validation-defined recentering | 32.79% |
| Forecast bias (WPE), raw / recentered | -8.23% / -2.62% |
| Q10-Q90 interval coverage | 70.3% observed / 80% nominal |
| Validation-selected inventory policy | Q90 Direct |
| Test profit / fill rate | 0.2413 / 0.956 |
| Profit difference vs. Static MinMax | +0.0762 per SP-day; 95% CI [0.0718, 0.0810] |

These figures predate the stock-status fix described under [Errata](#errata).
Re-scored with the fix, the forecast WAPE rises by about 0.2 points and the
inventory figures are essentially unchanged.

The inventory result is an observed-sales proxy under censored evaluation. Its
bias relative to latent-demand profit is not identified. The complete compact
evidence bundle is in `results/current/`, and `docs/run_manifest.md` documents
the authoritative execution.

## Repository layout

```text
pipeline_leakfree.py          Main recovery, feature, CV, forecast, and inventory pipeline
benchmark_comparison.py       Common-protocol forecasting benchmark

src/
  feature_builder.py          Train-only fit/transform features and recursive inference
  direct_forecast.py          Direct multi-horizon forecasting
  recovery.py                 IMR heuristic and Tobit MLE demand recovery
  censored_objectives.py      Tobit-style custom LightGBM objective
  inventory.py                Newsvendor policies and validation-only selection
  calibration.py              Cross-validation forecast recentering
  metrics.py                  WAPE/WPE, clustered bootstrap, and Holm correction

scripts/
  headline_run.py             Full fold-causal 50,000-series execution
  multistep_validation.py     Common multi-step validation and promotion gate
  accuracy_ablation.py        Seven forecasting variants
  parsimony_sweep.py          Validation-only feature-parsimony analysis
  hyperparam_grid.py          Validation-only robustness grid
  profit_inference.py         Store-clustered policy intervals and paired tests
  uncensored_diagnostics.py   Stockout-exposure diagnostic strata
  stratum_robustness.py       Stockout-stratum error ordering under conditioning
  cost_sensitivity.py         Inventory profit across cost grids
  remeasure_recovery_runtime.py  Controlled recovery runtime benchmark
  statistical_significance.py   Benchmark confidence intervals and paired tests
  rebuild_paper_tables.py     Regenerate manuscript tables from `results/current`
  make_paper_figures.py       Regenerate manuscript figures
  verify_outputs.py           Cross-check code, metrics, and compact artifacts

tests/                        Synthetic-data test suite; no dataset required
notebooks/                    Result-backed tutorials plus data and simulation labs
docs/run_manifest.md          Current full-scale run configuration and metrics
results/current/              Current compact results and provenance
```

## Installation

Python 3.12 is recommended for the exact environment snapshot.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-lock.txt
```

Use `requirements.txt` instead when compatible minimum versions are preferable
to the exact platform snapshot.

## Data

The dataset is not included. Download FreshRetailNet-50K Version 1.0 and place
the files at:

```text
../data/freshretailnet/raw/data/train.parquet
../data/freshretailnet/raw/data/eval.parquet
```

Expected MD5 checksums:

| File | MD5 |
|---|---|
| `train.parquet` | `501f47af53495f01592ea8caf1c83978` |
| `eval.parquet` | `14878135c5b048d4e0dac44ae0282879` |

Set `DATA_DIR` near the top of `pipeline_leakfree.py` when using another local
location.

## Running the pipeline

```bash
# Full fold-causal pipeline, all 50,000 store-products (about 2.6 h CPU)
python scripts/headline_run.py --n_sp 0 --out output/headline
python scripts/profit_inference.py --run_dir output/headline
python scripts/cost_sensitivity.py --run-dir output/headline

# Five-model benchmark on a stratified 1,500-series sample
python benchmark_comparison.py --n_sp 1500 --out output/benchmark
python scripts/statistical_significance.py \
  --preds output/benchmark/benchmark_per_sp_predictions.csv \
  --out output/benchmark

# Validation studies
python scripts/multistep_validation.py --n_sp 1500 --out output/multistep_validation
python scripts/hyperparam_grid.py --n_sp 1500 --out output/hyperparameter_grid
python scripts/parsimony_sweep.py --n_sp 1500 --out output/parsimony

# Verify committed evidence and regenerate paper assets
python scripts/verify_outputs.py
python scripts/rebuild_paper_tables.py
python scripts/make_paper_figures.py
```

All experiments use deterministic seeds and write configuration, input hashes,
runtime, and software provenance alongside their outputs.

The inventory objective implies underage cost `Cu = r - c + p_s = 1.1`,
overage cost `Co = c + h + w = 0.8`, and profit-optimal critical ratio
`Cu / (Cu + Co) = 0.578947`. Policy selection is constrained by the declared
validation-fold fill-rate floor.

## Tests

```bash
pytest -q
```

The tests cover feature and prediction invariance, recursive forecasting,
censored objectives, inventory policies, calibration, metric inference,
notebook structure, notebook semantics, and synchronization of the compact
result artifacts.

## Leakage controls

- Aggregate features are fitted on training data only and applied through
  immutable lookup tables.
- Multi-step recursive inference never uses an evaluation-day actual, sales or
  stock status, as an input for a later evaluation day. Up to `v1.1.0` stock
  status was not covered; see [Errata](#errata).
- Calibration and policy selection use validation folds only; the held-out test
  window is evaluated once.
- Negative-control tests reproduce the prohibited concatenate-and-recompute
  pattern and confirm that the invariance checks detect it.

## Errata

**Evaluation-week stock status in recursive inference (fixed after `v1.1.0`).**
In `v1.0.0` and `v1.1.0`, recursive inference zeroed evaluation-week sales but
kept the realised evaluation-week stock status. For evaluation days 2-7 the
stockout-history features `so1`, `sor7`, `sor14`, `soh7` and `soh14`
therefore read stockouts that are not known at the forecast origin. Every
recursive forecast shared this path: the headline run's validation folds and
test window, the validation studies, and the benchmark. The features are now
held at their first-forecast-day values, which read training days only, the
rule the direct forecaster already applied. The leakage tests now also mutate
evaluation-week stock status, on a synthetic panel where stock status carries
signal; the earlier fixtures held it constant, which is why they missed it.

The committed artifacts in `results/current/` predate the fix. Re-scoring the
test window with the same final models and the committed calibrator:

| Metric | Committed | With fix |
|---|---|---|
| Forecast WAPE, raw ensemble | 33.54% | 33.71% |
| Forecast WAPE, validation-defined recentering | 32.79% | 32.98% |
| Forecast bias (WPE), raw / recentered | -8.23% / -2.62% | -8.48% / -2.88% |
| Q10-Q90 interval coverage | 70.3% | 70.0% |
| Q90 Direct test profit / fill rate | 0.2413 / 0.956 | 0.2416 / 0.955 |

The recentered figures reuse a calibrator learned from validation forecasts
that had the same issue, so a rerun of `scripts/headline_run.py` gives the
exact values. In the multi-step validation study the promotion decision is
unchanged: with the fix, the challenger's bias excess over the incumbent grows
from 1.03 to 1.18 points against the 1.0-point tolerance, so the incumbent is
still retained. The benchmark, ablation and sensitivity tables were not
re-scored.

## Citation

If this code or its accompanying artifacts are useful in your research, please
cite the published paper:

> Azar, J. A Reproducible, Leakage-Free Pipeline for Censored Demand Forecasting
> and Inventory Optimization on FreshRetailNet-50K. *Inventions* **2026**, *11*(5),
> 89. [Article](https://www.mdpi.com/2411-5134/11/5/89) ·
> [DOI](https://doi.org/10.3390/inventions11050089)

```bibtex
@article{azar2026freshretailnet,
  author  = {Azar, Joseph},
  title   = {A Reproducible, Leakage-Free Pipeline for Censored Demand Forecasting and Inventory Optimization on FreshRetailNet-50K},
  journal = {Inventions},
  year    = {2026},
  volume  = {11},
  number  = {5},
  article-number = {89},
  url     = {https://www.mdpi.com/2411-5134/11/5/89},
  issn    = {2411-5134},
  doi     = {10.3390/inventions11050089}
}
```

Machine-readable citation metadata is available in `CITATION.cff`.

## License

The code is MIT licensed. FreshRetailNet-50K retains its own license at the
dataset source.
