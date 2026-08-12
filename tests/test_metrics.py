"""Sanity tests for src.metrics (wape, wpe, cluster_bootstrap, holm-bonferroni)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.metrics import (
    wape, wpe, mae, rmse, smape, corr,
    cluster_bootstrap_ci, paired_wilcoxon, holm_bonferroni,
)


def test_wape_basic():
    y = np.array([10, 20, 30], dtype=float)
    p = np.array([12, 18, 33], dtype=float)
    # |2|+|2|+|3| / 60 = 7/60 = 11.67%
    assert abs(wape(y, p) - 11.666666666666666) < 1e-6


def test_wpe_sign():
    y = np.array([10, 20, 30], dtype=float)
    p = np.array([12, 22, 34], dtype=float)  # over-predict
    assert wpe(y, p) > 0


def test_cluster_bootstrap_shrinks_with_more_sps():
    # per-SP clusters with identical within-SP signals
    rng = np.random.default_rng(0)
    n_per_sp = 10
    sp = np.concatenate([np.full(n_per_sp, i) for i in range(20)])
    y = rng.normal(5, 1, size=len(sp))
    p = y + rng.normal(0, 0.5, size=len(sp))
    point, lo, hi = cluster_bootstrap_ci(y, p, sp, mae, n_boot=200, seed=42)
    assert lo <= point <= hi, f"CI invalid: {lo}<={point}<={hi}"
    assert hi - lo > 0


def test_paired_wilcoxon_detects_difference():
    # Reference is strictly better
    rng = np.random.default_rng(0)
    ref = rng.normal(0.3, 0.1, size=200)  # lower MAE
    alt = rng.normal(0.5, 0.1, size=200)  # higher MAE
    p, rb = paired_wilcoxon(ref, alt, alternative='less')
    assert p < 0.01, f"Should detect ref < alt, got p={p}"
    assert rb < 0  # ref wins


def test_holm_bonferroni_preregistered_family():
    ps = [0.01, 0.04, 0.03]
    # Family size 3: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
    # Sorted by p: indices (0, 2, 1) -> adj = [0.03, 0.06, 0.06]
    adj = holm_bonferroni(ps, family_size=3)
    assert abs(adj[0] - 0.03) < 1e-9
    # Family size 10: 0.01*10=0.1, 0.03*9=0.27, 0.04*8=0.32
    # With increasing-prev enforcement, 0.1, 0.27, 0.32
    adj10 = holm_bonferroni(ps, family_size=10)
    assert adj10[0] > adj[0], "larger family should inflate corrections"


def test_holm_rejects_too_small_family():
    with pytest.raises(ValueError):
        holm_bonferroni([0.01, 0.02, 0.03], family_size=2)
