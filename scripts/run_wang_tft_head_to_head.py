#!/usr/bin/env python3
"""Run Wang et al. TFT+TimesNet baseline on the same split and compare apples-to-apples."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = ROOT / "freshretailnet_repos" / "frn-50k-baseline"
DATA_ROOT = ROOT.parent / "data" / "freshretailnet" / "raw" / "data"


def run_cmd(cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    cmd_display = " ".join(map(str, cmd))
    print(f"[run] {cmd_display}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def python_version(python_exec: Path) -> tuple[int, int]:
    result = subprocess.run(
        [str(python_exec), "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = (int(x) for x in result.stdout.strip().split())
    return major, minor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="output/benchmark/predictions.csv",
                    help="Leak-free predictions CSV from this repo.")
    ap.add_argument("--out", default="output/benchmark",
                    help="Directory that will store wang_vs_ours_head_to_head.csv.")
    ap.add_argument("--train_path", default=str(DATA_ROOT / "train.parquet"),
                    help="Local train parquet for both recovery and TFT training.")
    ap.add_argument("--eval_path", default=str(DATA_ROOT / "eval.parquet"),
                    help="Local eval parquet for TFT prediction/evaluation.")
    ap.add_argument("--baseline_root", default=str(BASELINE_ROOT),
                    help="Path to Dingdong-Inc/frn-50k-baseline.")
    ap.add_argument("--model", default="TimesNet",
                    help="Recovery model to run in frn-50k-baseline (default: TimesNet).")
    ap.add_argument("--python", default=str(ROOT / "venv" / "bin" / "python"),
                    help="Python executable used to run the baseline (defaults to ./venv/bin/python).")
    ap.add_argument("--skip_recovery", action="store_true",
                    help="Skip demand recovery and reuse existing demand.parquet.")
    ap.add_argument("--skip_training", action="store_true",
                    help="Skip TFT training and reuse existing checkpoints.")
    ap.add_argument("--skip_prediction", action="store_true",
                    help="Skip prediction and comparison-only flow (keeps existing Wang preds file).")
    ap.add_argument("--skip_compare", action="store_true",
                    help="Only run baseline scripts; skip running scripts/compare_wang_vs_ours.py.")
    ap.add_argument("--install_requirements", action="store_true",
                    help="Install frn-50k-baseline requirements before running baseline commands.")
    ap.add_argument("--requirements", default=str(ROOT / "requirements-frn-baseline.txt"),
                    help="Requirements file used when --install_requirements is set.")
    ap.add_argument("--use_gpu", action="store_true",
                    help="Enable GPU in baseline TFT scripts.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing Wang prediction output.")
    ap.add_argument("--clean_checkpoints", action="store_true", default=True,
                    help="Remove recovered checkpoints before TFT retrain.")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(args.baseline_root).resolve()
    python_exec = Path(args.python).resolve()

    if not python_exec.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exec}")
    if not baseline_root.exists():
        raise FileNotFoundError(f"Baseline repo not found: {baseline_root}")

    train_path = Path(args.train_path).resolve()
    eval_path = Path(args.eval_path).resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval parquet not found: {eval_path}")

    recovery_app = baseline_root / "latent_demand_recovery" / "exp" / "app.py"
    tft_train = baseline_root / "demand_forecasting" / "TFT" / "trainTFT.py"
    tft_predict = baseline_root / "demand_forecasting" / "TFT" / "predictTFT.py"
    demand_dir = baseline_root / "latent_demand_recovery" / "exp" / "demand"
    demand_parquet = demand_dir / "demand.parquet"

    wang_pred_out = out_dir / "wang_tft_timesnet_eval_preds.csv"
    compare_csv = out_dir / "wang_vs_ours_head_to_head.csv"
    req_txt = Path(args.requirements)
    major, minor = python_version(python_exec)
    if major == 3 and minor >= 11:
        print(
            "[warn] You are using Python >=3.11. "
            "The upstream Wang baseline requirements may require Python 3.8. "
            "Consider using --requirements with a pyproject-compatible stack if install fails."
        )

    if args.install_requirements:
        if not req_txt.exists():
            raise FileNotFoundError(f"Requirements file not found: {req_txt}")
        run_cmd([str(python_exec), "-m", "pip", "install", "-r", str(req_txt)],
                cwd=baseline_root)

    if not args.skip_recovery:
        if args.force and demand_parquet.exists():
            demand_parquet.unlink()
        run_cmd([str(python_exec), str(recovery_app), "--model", args.model, "--data_path", str(train_path)],
                cwd=baseline_root / "latent_demand_recovery" / "exp")

    needs_recovered_demand = (not args.skip_recovery) or (not args.skip_training) or (not args.skip_prediction)
    if needs_recovered_demand and not demand_parquet.exists():
        raise FileNotFoundError(f"Expected recovered demand not found: {demand_parquet}")

    if not args.skip_training and args.clean_checkpoints:
        ckpt_root = baseline_root / "demand_forecasting" / "TFT" / "lightning_logs" / "recovered"
        if ckpt_root.exists():
            shutil.rmtree(ckpt_root)

    if not args.skip_training:
        run_cmd([
            str(python_exec), str(tft_train),
            "--demand",
            "--demand_path", str(demand_parquet),
            "--train_path", str(train_path),
        ] + ([] if not args.use_gpu else ["--use_gpu"]),
            cwd=baseline_root / "demand_forecasting" / "TFT")

    if not args.skip_prediction:
        if args.force and wang_pred_out.exists():
            wang_pred_out.unlink()
        run_cmd([
            str(python_exec), str(tft_predict),
            "--demand",
            "--demand_path", str(demand_parquet),
            "--train_path", str(train_path),
            "--eval_path", str(eval_path),
            "--preds_out", str(wang_pred_out),
        ] + ([] if not args.use_gpu else ["--use_gpu"]),
            cwd=baseline_root / "demand_forecasting" / "TFT")

    if not args.skip_prediction and not wang_pred_out.exists():
        raise FileNotFoundError(f"Wang prediction output not created: {wang_pred_out}")

    if not args.skip_compare:
        run_cmd([
            str(python_exec), str(ROOT / "scripts" / "compare_wang_vs_ours.py"),
            "--ours", args.ours,
            "--wang_preds", str(wang_pred_out),
            "--out", str(out_dir),
        ], cwd=ROOT)

    print(f"\nDone. Wang predictions: {wang_pred_out}")
    if not args.skip_compare:
        print(f"Head-to-head file: {compare_csv}")


if __name__ == "__main__":
    main()
