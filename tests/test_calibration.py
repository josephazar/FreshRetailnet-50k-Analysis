"""Tests for leakage-free CV forecast calibration."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    apply_scale_calibrator,
    calibrated_cv_residuals,
    learn_cv_scale_calibrator,
)
from src.metrics import wape


def _toy_cv_frames():
    dates_a = pd.date_range('2024-01-01', periods=3, freq='D')
    dates_b = pd.date_range('2024-01-08', periods=3, freq='D')
    frames = []
    for fold, dates, bump in [(1, dates_a, 0.0), (2, dates_b, 1.0)]:
        sp = np.array([101, 101, 101, 202, 202, 202])
        dt = np.tile(dates.values, 2)
        y = np.array([10, 12, 14, 20, 22, 24], dtype='float32') + bump
        pred = (y * 0.8).astype('float32')
        preds = {
            'mae': pred.copy(),
            'huber': pred.copy(),
            'q10': (pred * 0.8).astype('float32'),
            'q50': pred.copy(),
            'q90': (pred * 1.2).astype('float32'),
            'ensemble': pred.copy(),
        }
        frames.append({'fold': fold, 'sp': sp, 'dt': dt, 'y': y, 'preds': preds})
    return frames


def test_cv_scale_calibration_improves_simple_global_underprediction():
    frames = _toy_cv_frames()
    cal = learn_cv_scale_calibrator(frames, sp_shrink=1000.0, horizon_shrink=1000.0)

    last = frames[-1]
    raw = last['preds']['ensemble']
    adjusted = apply_scale_calibrator(raw, last['sp'], last['dt'], cal)

    assert cal['method'] == 'cv_scale'
    assert cal['cv_metrics']['calibrated_wape'] < cal['cv_metrics']['raw_wape']
    assert wape(last['y'], adjusted) < wape(last['y'], raw)


def test_unknown_sp_uses_global_and_horizon_fallback_without_targets():
    cal = learn_cv_scale_calibrator(_toy_cv_frames())
    pred = np.array([8.0, 9.0, 10.0], dtype='float32')
    sp = np.array([999, 999, 999])
    dt = pd.date_range('2024-02-01', periods=3, freq='D').values

    adjusted = apply_scale_calibrator(pred, sp, dt, cal)

    assert adjusted.shape == pred.shape
    assert np.all(np.isfinite(adjusted))
    assert float(np.mean(adjusted)) > float(np.mean(pred))


def test_calibrated_cv_residuals_return_post_calibration_errors():
    frames = _toy_cv_frames()
    cal = learn_cv_scale_calibrator(frames)

    raw_abs = []
    for frame in frames:
        raw_abs.extend(np.abs(frame['y'] - frame['preds']['mae']))
    residuals = calibrated_cv_residuals(frames, cal, pred_key='mae')
    cal_abs = np.concatenate([np.abs(resid) for _, resid in residuals])

    assert len(residuals) == len(frames)
    assert np.mean(cal_abs) < np.mean(raw_abs)
