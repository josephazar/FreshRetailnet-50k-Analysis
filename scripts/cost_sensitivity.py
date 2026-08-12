#!/usr/bin/env python3
"""Cost sensitivity for the validation-selected policy in a headline run.

The policy is never rebuilt here.  Instead, its saved per-row order quantities
from ``policy_orders.parquet`` are re-evaluated under each cost configuration.
This keeps the policy, observed-sales evaluation rows, and arithmetic exactly
aligned with the headline inventory table.

Run after ``scripts/headline_run.py``::

    ./venv/bin/python scripts/cost_sensitivity.py --run-dir output/headline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent.resolve()
GRID_P = (0.3, 0.5, 0.7, 1.0)
GRID_W = (0.1, 0.3, 0.5, 0.7)


def _metrics(order: np.ndarray, demand: np.ndarray, cfg: dict) -> dict:
    """Match ``src.inventory.eval_policy`` without rounded intermediates."""
    sold = np.minimum(order, demand)
    over = np.maximum(order - demand, 0.0)
    under = np.maximum(demand - order, 0.0)
    revenue = sold * cfg['unit_revenue']
    total_cost = (order * cfg['unit_cost'] + cfg['h'] * over
                  + cfg['p'] * under + cfg['w'] * over)
    return {
        'profit': float(np.mean(revenue - total_cost)),
        'fill_rate': float(sold.sum() / max(demand.sum(), 1.0)),
        'waste_pct': float(over.sum() / max(order.sum(), 1.0) * 100.0),
        'so_rate': float(np.mean(demand > order)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='output/headline')
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    orders_path = run_dir / 'policy_orders.parquet'
    selection_path = run_dir / 'inventory_selection.json'
    metadata_path = run_dir / 'run_metadata.json'
    missing = [p for p in (orders_path, selection_path, metadata_path)
               if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required headline artifact(s): ' +
                                ', '.join(str(p) for p in missing))

    started = time.time()
    orders = pd.read_parquet(orders_path)
    selected = json.loads(selection_path.read_text())['preselected_policy']
    cfg = json.loads(metadata_path.read_text())['inventory_cfg']
    if selected not in orders:
        raise KeyError(f"Selected policy {selected!r} is absent from {orders_path}")

    order = orders[selected].to_numpy(dtype='float64')
    demand = orders['y_true'].to_numpy(dtype='float64')
    rows = []
    for stockout_penalty in GRID_P:
        for waste_penalty in GRID_W:
            scenario_cfg = {**cfg, 'p': stockout_penalty, 'w': waste_penalty}
            rows.append({
                'p': stockout_penalty,
                'w': waste_penalty,
                'policy': selected,
                **_metrics(order, demand, scenario_cfg),
            })

    result = pd.DataFrame(rows)
    out_path = run_dir / 'inventory_cost_sensitivity.csv'
    result.to_csv(out_path, index=False)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    (run_dir / 'inventory_cost_sensitivity_metadata.json').write_text(
        json.dumps({
            'script': 'cost_sensitivity',
            'git_sha': sha,
            'run_dir': args.run_dir,
            'source': 'saved policy_orders.parquet; no policy re-estimation',
            'selected_policy': selected,
            'grid_p': list(GRID_P),
            'grid_w': list(GRID_W),
            'base_inventory_cfg': cfg,
            'runtime_s': round(time.time() - started, 3),
        }, indent=2) + '\n')
    print(f'Wrote {out_path}')
    print(result.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
