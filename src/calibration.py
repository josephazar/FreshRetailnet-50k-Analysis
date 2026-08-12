"""
Leakage-free post-forecast calibration learned from CV validation folds.

The calibrator estimates multiplicative scale corrections from out-of-sample
CV predictions only. It can then be applied to the final held-out evaluation
window without using any eval actuals.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import wape, wpe


def _clip(x: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(min(max(x, lo), hi))


def _horizon_from_dt(dt_values: np.ndarray) -> np.ndarray:
    """Return 1-based horizon indices from sorted unique dates."""
    dt = pd.to_datetime(pd.Series(dt_values))
    dates = {d: i + 1 for i, d in enumerate(sorted(dt.unique()))}
    return dt.map(dates).to_numpy(dtype=np.int16)


def _cv_frames_to_df(cv_val_frames: list[dict[str, Any]], pred_key: str) -> pd.DataFrame:
    rows = []
    for frame in cv_val_frames:
        if pred_key not in frame['preds']:
            raise KeyError(f"prediction key '{pred_key}' missing from CV frame")
        sp = np.asarray(frame['sp'])
        y = np.asarray(frame['y'], dtype=float)
        pred = np.asarray(frame['preds'][pred_key], dtype=float)
        if 'dt' in frame:
            dt = np.asarray(frame['dt'])
            horizon = _horizon_from_dt(dt)
        else:
            dt = np.arange(len(y))
            horizon = np.ones(len(y), dtype=np.int16)
        rows.append(pd.DataFrame({
            'fold': int(frame.get('fold', 0)),
            'sp': sp,
            'dt': dt,
            'horizon': horizon,
            'y': y,
            'pred': pred,
        }))
    if not rows:
        raise ValueError("cv_val_frames must contain at least one frame")
    return pd.concat(rows, ignore_index=True)


def learn_cv_scale_calibrator(
    cv_val_frames: list[dict[str, Any]],
    pred_key: str = 'ensemble',
    sp_shrink: float = 28.0,
    horizon_shrink: float = 5000.0,
    global_bounds: tuple[float, float] = (0.80, 1.25),
    horizon_rel_bounds: tuple[float, float] = (0.85, 1.20),
    sp_bounds: tuple[float, float] = (0.75, 1.35),
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Learn a global + horizon + SP multiplicative calibrator from CV folds.

    The decomposition is:

        calibrated_pred = pred * global_scale * horizon_rel[h] * sp_scale[sp]

    Horizon effects are learned as relative deviations from the global scale.
    SP effects are learned after applying the global/horizon terms and shrunk
    toward 1.0, so unseen SPs naturally fall back to global + horizon scaling.
    """
    df = _cv_frames_to_df(cv_val_frames, pred_key)
    pred_sum = float(df['pred'].sum())
    y_sum = float(df['y'].sum())
    raw_global = y_sum / pred_sum if pred_sum > eps else 1.0
    global_scale = _clip(raw_global, global_bounds)

    horizon_rel: dict[str, float] = {}
    for h, g in df.groupby('horizon'):
        denom = float(g['pred'].sum())
        raw = float(g['y'].sum()) / denom if denom > eps else global_scale
        n = len(g)
        weight = n / (n + horizon_shrink)
        h_scale = weight * raw + (1.0 - weight) * global_scale
        rel = h_scale / global_scale if global_scale > eps else 1.0
        horizon_rel[str(int(h))] = _clip(rel, horizon_rel_bounds)

    h_arr = df['horizon'].map(lambda h: horizon_rel.get(str(int(h)), 1.0)).to_numpy(float)
    adjusted = df['pred'].to_numpy(float) * global_scale * h_arr
    df = df.copy()
    df['_adjusted'] = adjusted

    sp_scale: dict[str, float] = {}
    for sp, g in df.groupby('sp'):
        denom = float(g['_adjusted'].sum())
        if denom <= eps:
            continue
        raw = float(g['y'].sum()) / denom
        if not math.isfinite(raw):
            continue
        n = len(g)
        weight = n / (n + sp_shrink)
        scale = weight * raw + (1.0 - weight) * 1.0
        sp_scale[str(int(sp))] = _clip(scale, sp_bounds)

    raw_pred = df['pred'].to_numpy(float)
    cal_pred = apply_scale_calibrator(
        raw_pred,
        df['sp'].to_numpy(),
        df['dt'].to_numpy(),
        {
            'global_scale': global_scale,
            'horizon_rel': horizon_rel,
            'sp_scale': sp_scale,
        },
    )

    return {
        'method': 'cv_scale',
        'pred_key': pred_key,
        'global_scale': global_scale,
        'raw_global_scale': raw_global,
        'global_bounds': list(global_bounds),
        'horizon_rel_bounds': list(horizon_rel_bounds),
        'sp_bounds': list(sp_bounds),
        'sp_shrink': float(sp_shrink),
        'horizon_shrink': float(horizon_shrink),
        'horizon_rel': horizon_rel,
        'sp_scale': sp_scale,
        'cv_metrics': {
            'raw_wape': wape(df['y'].to_numpy(float), raw_pred),
            'raw_wpe': wpe(df['y'].to_numpy(float), raw_pred),
            'calibrated_wape': wape(df['y'].to_numpy(float), cal_pred),
            'calibrated_wpe': wpe(df['y'].to_numpy(float), cal_pred),
            'n_rows': int(len(df)),
            'n_sps': int(df['sp'].nunique()),
        },
    }


def apply_scale_calibrator(
    pred: np.ndarray,
    sp_keys: np.ndarray,
    dt_values: np.ndarray,
    calibrator: dict[str, Any],
) -> np.ndarray:
    """Apply a learned calibrator. Uses no actual target values."""
    pred = np.asarray(pred, dtype=float)
    sp_keys = np.asarray(sp_keys)
    horizon = _horizon_from_dt(np.asarray(dt_values))
    global_scale = float(calibrator.get('global_scale', 1.0))
    horizon_rel = calibrator.get('horizon_rel', {})
    sp_scale = calibrator.get('sp_scale', {})

    h_scale = np.array(
        [float(horizon_rel.get(str(int(h)), 1.0)) for h in horizon],
        dtype=float,
    )
    s_scale = np.array(
        [float(sp_scale.get(str(int(sp)), 1.0)) for sp in sp_keys],
        dtype=float,
    )
    return np.clip(pred * global_scale * h_scale * s_scale, 0, None).astype('float32')


def apply_calibrator_to_prediction_dict(
    preds: dict[str, np.ndarray],
    sp_keys: np.ndarray,
    dt_values: np.ndarray,
    calibrator: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Apply the same multiplicative calibrator to every prediction array."""
    return {
        name: apply_scale_calibrator(values, sp_keys, dt_values, calibrator)
        for name, values in preds.items()
    }


def calibrated_cv_residuals(
    cv_val_frames: list[dict[str, Any]],
    calibrator: dict[str, Any],
    pred_key: str = 'mae',
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return per-fold residuals after applying the calibrator to CV predictions."""
    out = []
    for frame in cv_val_frames:
        dt_values = frame.get('dt')
        if dt_values is None:
            dt_values = np.arange(len(frame['y']))
        p = apply_scale_calibrator(
            frame['preds'][pred_key],
            frame['sp'],
            dt_values,
            calibrator,
        )
        out.append((np.asarray(frame['sp']), np.asarray(frame['y'], dtype=float) - p))
    return out
