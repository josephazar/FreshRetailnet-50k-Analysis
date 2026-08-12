"""Controlled recovery-runtime benchmark (recovery call only).

The benchmark measures wall-clock time for
IMR, Tobit MLE, SAITS, and TimesNet on the SAME 2,000-SP stratified
subsample (seed 42), same machine, same environment, measurement boundary =
the recovery/imputation call only -- no downstream forecaster training.
The manuscript's runtime ratios are replaced by these remeasured numbers.

Boundary definitions (documented in the output):
  - heuristic IMR / Tobit MLE: the apply_recovery(...) call on the training
    frame (includes the per-SP MLE fits for Tobit -- that IS its recovery
    cost).
  - SAITS / TimesNet (pypots): model.fit(...) + impute on the same panel
    with censored entries masked as missing. Deep imputers must be trained
    on the panel before they can impute, so training time is part of the
    recovery cost by construction. Configuration (epochs, batch size,
    default architectures, CPU device) is recorded alongside the timings;
    ratios in the manuscript cite this configuration explicitly.

Process layout: the deep imputers run in CHILD processes that never import
LightGBM -- PyTorch and LightGBM each bundle their own libomp on macOS and
loading both in one process aborts with OMP error #179.

Run (expect tens of minutes; nothing else should share the CPU):
    ./venv/bin/python scripts/remeasure_recovery_runtime.py

Output: output/recovery_runtime/recovery_runtime.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()

N_SP = 2000
SEED = 42
IMR_REPS = 5
DEEP_EPOCHS = 10
DEEP_BATCH = 32


def _child(model_name: str, panel_path: str, out_path: str):
    """Time one deep imputer. Imports only numpy/torch/pypots."""
    import numpy as np
    import torch
    torch.manual_seed(SEED)
    panel = np.load(panel_path)
    n_steps, n_feat = panel.shape[1], panel.shape[2]
    t0 = time.time()
    if model_name == 'saits':
        from pypots.imputation import SAITS
        mdl = SAITS(n_steps=n_steps, n_features=n_feat, n_layers=2,
                    d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128,
                    epochs=DEEP_EPOCHS, batch_size=DEEP_BATCH, device='cpu')
    elif model_name == 'timesnet':
        from pypots.imputation import TimesNet
        mdl = TimesNet(n_steps=n_steps, n_features=n_feat, n_layers=2,
                       top_k=3, d_model=64, d_ffn=64, n_kernels=6,
                       epochs=DEEP_EPOCHS, batch_size=DEEP_BATCH,
                       device='cpu')
    else:
        raise ValueError(model_name)
    mdl.fit({'X': panel})
    mdl.impute({'X': panel})
    Path(out_path).write_text(json.dumps(
        {'median_s': round(time.time() - t0, 1), 'reps_s': 1}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='output/recovery_runtime')
    ap.add_argument('--child', choices=['saits', 'timesnet'])
    ap.add_argument('--panel')
    ap.add_argument('--child_out')
    args = ap.parse_args()

    if args.child:
        _child(args.child, args.panel, args.child_out)
        return

    # parent: classic methods + panel export (LightGBM may load here)
    import numpy as np
    sys.path.insert(0, str(SCRIPT_DIR))
    from pipeline_leakfree import apply_recovery, load_data, set_all_seeds

    out_dir = SCRIPT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)
    train, _ev = load_data(n_sp=N_SP)
    timings = {}

    print(f"[1/4] heuristic IMR ({IMR_REPS} reps)", flush=True)
    reps = []
    for _ in range(IMR_REPS):
        t0 = time.time()
        apply_recovery(train.copy(), 'heuristic_imr')
        reps.append(time.time() - t0)
    timings['heuristic_imr'] = {'median_s': round(float(np.median(reps)), 3),
                                'reps_s': [round(r, 3) for r in reps]}
    print(f"  median {timings['heuristic_imr']['median_s']}s", flush=True)

    print("[2/4] Tobit MLE (per-SP fits; single rep)", flush=True)
    t0 = time.time()
    apply_recovery(train.copy(), 'tobit_mle')
    timings['tobit_mle'] = {'median_s': round(time.time() - t0, 1),
                            'reps_s': 1}
    print(f"  {timings['tobit_mle']['median_s']}s", flush=True)

    df = train[['sp', 'dt', 'sale_amount', 'stock_hour6_22_cnt']].copy()
    df['val'] = df['sale_amount'].astype('float32')
    df.loc[df['stock_hour6_22_cnt'] > 0, 'val'] = np.nan
    panel = df.pivot(index='sp', columns='dt',
                     values='val').values[:, :, None].astype('float32')
    panel_path = out_dir / 'panel_2000sp.npy'
    np.save(panel_path, panel)

    for i, name in enumerate(('saits', 'timesnet')):
        print(f"[{3 + i}/4] {name} (child process, {DEEP_EPOCHS} epochs, "
              f"CPU) panel {panel.shape}", flush=True)
        child_out = out_dir / f'{name}_timing.json'
        r = subprocess.run(
            [sys.executable, __file__, '--child', name,
             '--panel', str(panel_path), '--child_out', str(child_out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {name} FAILED:\n{r.stderr[-2000:]}", flush=True)
            timings[name] = {'median_s': None, 'error': 'child failed'}
        else:
            timings[name] = json.loads(child_out.read_text())
            print(f"  {timings[name]['median_s']}s", flush=True)

    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                         capture_output=True, text=True).stdout.strip()
    base = timings['heuristic_imr']['median_s']
    result = {
        'script': 'remeasure_recovery_runtime', 'git_sha': sha,
        'n_sp': N_SP, 'seed': SEED,
        'boundary': 'recovery/imputation call only; deep imputers include '
                    'their required panel training (fit+impute)',
        'deep_config': {'epochs': DEEP_EPOCHS, 'batch_size': DEEP_BATCH,
                        'device': 'cpu', 'library': 'pypots'},
        'machine': {'platform': platform.platform(),
                    'processor': platform.machine()},
        'timings': timings,
        'ratios_vs_imr': {k: (round(v['median_s'] / base, 1)
                              if v.get('median_s') else None)
                          for k, v in timings.items()},
    }
    (out_dir / 'recovery_runtime.json').write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result['ratios_vs_imr'], indent=2), flush=True)
    print(f"-> {out_dir}/recovery_runtime.json", flush=True)


if __name__ == '__main__':
    main()
