"""Uncensored-subset diagnostics for the headline run.

Report WAPE/WPE of the headline forecast on evaluation rows with zero stockout hours,
explicitly as a NON-RANDOM diagnostic subset (stockouts correlate with
demand level, promotions, and store practices). This is a transparency
diagnostic; it is not a validation of latent-demand
accuracy and NOT a bound in either direction -- under censoring the
direction of bias relative to true demand is not identified.

Strata by stockout intensity are included so the reader can see how the
error metrics move with censoring exposure rather than a single split.

`mean_y` is reported alongside the error metrics because WAPE normalizes by
the observed-sales level, which differs sharply across the strata; the strata
are therefore not on a common denominator and their WAPEs are not directly
comparable as model-quality statements.

Run (after scripts/headline_run.py has produced --run_dir):
    ./venv/bin/python scripts/uncensored_diagnostics.py --run_dir output/headline

Output (inside --run_dir):
    uncensored_diagnostics.csv     one row per stratum x model, with the
                                   stratum mean of observed sales (mean_y)
    uncensored_diagnostics_metadata.json
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

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from src.metrics import mae, wape, wpe  # noqa: E402

STRATA = [('uncensored (0h)', 0, 0), ('light (1-4h)', 1, 4),
          ('heavy (5h+)', 5, np.inf), ('all rows', 0, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='output/headline')
    args = ap.parse_args()
    run_dir = SCRIPT_DIR / args.run_dir
    t0 = time.time()

    df = pd.read_csv(run_dir / 'predictions.csv')
    hours = df['stockout_hours'].values
    y = df['y_true'].values.astype('float64')
    models = sorted({c[5:] for c in df.columns
                     if c.startswith('pred_') and not c.endswith('_raw')})

    rows = []
    for label, lo, hi in STRATA:
        m = (hours >= lo) & (hours <= hi)
        for model in models:
            p = df.loc[m, f'pred_{model}'].values.astype('float64')
            rows.append({'stratum': label, 'model': model,
                         'n_rows': int(m.sum()),
                         'share_rows': round(float(m.mean()), 4),
                         'wape': round(wape(y[m], p), 4),
                         'wpe': round(wpe(y[m], p), 4),
                         'mae': round(mae(y[m], p), 4),
                         'mean_y': round(float(y[m].mean()), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / 'uncensored_diagnostics.csv', index=False)
    print(out[out['model'] == 'ensemble'].to_string(index=False))

    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                         capture_output=True, text=True).stdout.strip()
    meta = {
        'script': 'uncensored_diagnostics', 'git_sha': sha,
        'args': vars(args),
        'note': ('non-random diagnostic subsets: stockout exposure '
                 'correlates with demand level and store practices; these '
                 'figures are transparency diagnostics, not latent-demand '
                 'validation and not bounds in either direction'),
        'runtime_s': round(time.time() - t0, 1),
    }
    (run_dir / 'uncensored_diagnostics_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"Done in {meta['runtime_s']:.0f}s -> {run_dir}/")


if __name__ == '__main__':
    main()
