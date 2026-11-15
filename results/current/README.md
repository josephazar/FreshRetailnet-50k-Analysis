# Current Result Artifacts

This directory is the source of truth for the compact, human-readable results
reported by the software release and paper. Large prediction and per-row order
files are excluded; the documented scripts regenerate them. These artifacts
predate the evaluation-week stock-status fix described in the top-level README
(Errata), which also gives the re-scored headline figures.

`release_summary.json` is the top-level index. `run_metadata.json` records the
50,000-series execution environment and input hashes. The remaining CSV and
JSON files contain forecast metrics, validation studies, inventory-policy
selection, cost sensitivity, uncertainty estimates, and diagnostics.

`interval_coverage.json` records Q10-Q90 coverage overall and for each
forecast-horizon day. Its 70.3% observed coverage is below the 80% nominal
level and should be interpreted as an unresolved calibration limitation.

`uncensored_diagnostics.csv` reports error by stockout-exposure stratum and
carries `mean_y`, the stratum mean of observed sales, because WAPE normalizes
by that level and it differs sharply across strata.
`stockout_stratum_robustness.csv` records whether the resulting error ordering
is monotone in stockout exposure. It is not: the lightly censored stratum
(1-4 stockout hours) attains a lower WAPE than the zero-stockout stratum
overall, in every demand decile, on six of seven test dates, and within the
store-products observed in both strata. The strata remain non-random subsets,
so the reversal refutes a monotone protocol-only reading of the zero-stockout
error rather than supplying a causal account of it.

The headline configuration uses the full-objective critical ratio
`Cu/(Cu+Co) = 0.578947`. Q90 Direct is selected on the validation fold under
the declared 0.90 fill-rate floor. Its test-window profit is evaluated against
observed censored sales and is therefore a proxy, not an estimate or bound for
latent-demand profit.
