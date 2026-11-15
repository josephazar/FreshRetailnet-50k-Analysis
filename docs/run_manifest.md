# Current Run Manifest

This file documents the authoritative full-scale result bundle in
`results/current/`. Outputs outside this bundle are not part of the public
code companion. Release `v1.1.0` added robustness diagnostics, and `v1.2.0`
fixes the evaluation-week stock-status issue described under Known issue. Both
preserve the headline outputs and their original `v1.0.0` execution
provenance.

| Field | Value |
|---|---|
| Code companion release | `v1.2.0` |
| Headline run provenance | `v1.0.0` |
| Execution date | 2026-08-12 |
| Store-product series | 50,000 |
| Test rows | 350,000 |
| Evaluation protocol | Fold-causal recovery and features; nested temporal validation; validation-only calibration and policy selection |
| Recovery | Heuristic IMR (`weight=0.5`) with Tobit sensitivity analysis |
| Forecast configuration | Recursive LightGBM ensemble with validation-defined recentering |
| Raw WAPE / WPE | 33.54% / -8.23% |
| Calibrated WAPE / WPE | 32.79% / -2.62% |
| Q10-Q90 interval coverage | 70.27% observed / 80% nominal |
| Selected policy | Q90 Direct, selected on the final validation fold under a 0.90 fill-rate floor |
| Test profit / fill rate | 0.2413 / 0.9564 |
| Profit difference vs. Static MinMax | +0.076210 per store-product day; 95% store-clustered CI [0.071787, 0.081004] |
| Critical ratio | `Cu/(Cu+Co) = 0.578947`, with `Cu=1.1` and `Co=0.8` |
| Runtime | 9,219.7 seconds (about 2.6 hours, CPU) |
| Dataset | FreshRetailNet-50K Version 1.0 |
| Train MD5 | `501f47af53495f01592ea8caf1c83978` |
| Evaluation MD5 | `14878135c5b048d4e0dac44ae0282879` |

Profit is evaluated against observed censored sales. It is an operational
proxy, not an estimate or bound for profit under latent demand.

## Known issue

The recursive inference used by this run read realised evaluation-week stock
status through five stockout-history features (README, Errata). Re-scored with
the same final models and the `v1.2.0` stock-status fix, raw / recentered WAPE are
33.71% / 32.98% and Q90 Direct profit / fill rate are 0.2416 / 0.955. Exact
fixed-protocol values require a rerun of `scripts/headline_run.py`.
