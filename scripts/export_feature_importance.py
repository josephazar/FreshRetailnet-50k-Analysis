#!/usr/bin/env python3
"""One-off export of LightGBM gain importances for the paper.

Replays the reported configuration (full 50K SPs, heuristic_imr,
seed 42) through load -> recovery -> FeatureBuilder -> CV -> full-train
retrain, and dumps per-model gain importances + boosters into output/benchmark/.
It does NOT touch any existing output/benchmark metric/prediction artifacts and
skips the expensive recursive eval prediction entirely.

Usage:
    ./venv/bin/python scripts/export_feature_importance.py [--n_sp 0] [--out output/benchmark]
"""
import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_leakfree import (  # noqa: E402
    apply_recovery,
    load_data,
    retrain_final,
    set_all_seeds,
    train_cv,
)
from src.feature_builder import FeatureBuilder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sp', type=int, default=0,
                    help='0=all 50K, else stratified subsample size')
    ap.add_argument('--recovery', default='heuristic_imr',
                    choices=['none', 'heuristic_imr', 'tobit_mle'])
    ap.add_argument('--out', default='output/benchmark')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    t0 = time.time()
    out_dir = SCRIPT_DIR / args.out

    print(f"[1/4] Data load (n_sp={args.n_sp})")
    train, _ev = load_data(n_sp=args.n_sp)
    print(f"  Train: {train.shape}, SPs: {train['sp'].nunique()}")

    print(f"[2/4] Recovery ({args.recovery})")
    train = apply_recovery(train, args.recovery)

    print("[3/4] FeatureBuilder fit/transform + temporal CV (for best_iters)")
    fb = FeatureBuilder(n_clusters=min(50, max(2, train['sp'].nunique() // 10)),
                        seed=args.seed).fit(train)
    train_feat = fb.transform(train)
    fcols = fb.feature_cols(train_feat)
    print(f"  Features: {len(fcols)}")
    _, _, best_iters, _ = train_cv(train_feat, fcols)

    print("[4/4] Full-train retrain with importance export")
    retrain_final(train_feat, fcols, best_iters, importance_out=out_dir)
    print(f"Done in {time.time() - t0:.1f}s; "
          f"wrote {out_dir / 'feature_importance.csv'}")


if __name__ == '__main__':
    main()
