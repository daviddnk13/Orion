# ============================================================
# test_metrics.py — Unit tests for quant_gate.metrics
# Validates all 8 metric functions
# FIXED: arrays large enough for guard clauses
# ============================================================

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quant_gate.metrics import (
    pearson_corr,
    spearman_corr,
    mae,
    rmse,
    directional_accuracy,
    information_coefficient,
    quantile_hit_rate,
    regime_accuracy,
)


class TestPearsonCorr:

    def test_perfect_positive(self):
        x = np.arange(20, dtype=float)
        corr, pval = pearson_corr(x, x)
        assert abs(corr - 1.0) < 1e-10

    def test_perfect_negative(self):
        x = np.arange(20, dtype=float)
        corr, pval = pearson_corr(x, -x)
        assert abs(corr - (-1.0)) < 1e-10

    def test_uncorrelated(self):
        rng = np.random.RandomState(42)
        x = rng.randn(10000)
        y = rng.randn(10000)
        corr, pval = pearson_corr(x, y)
        assert abs(corr) < 0.05


class TestSpearmanCorr:

    def test_perfect_monotonic(self):
        x = np.arange(1, 21, dtype=float)
        corr, pval = spearman_corr(x, x ** 3)
        assert abs(corr - 1.0) < 1e-10

    def test_uncorrelated(self):
        rng = np.random.RandomState(42)
        x = rng.randn(10000)
        y = rng.randn(10000)
        corr, pval = spearman_corr(x, y)
        assert abs(corr) < 0.05


class TestMAE:

    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mae(x, x) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert abs(mae(y_true, y_pred) - 1.0) < 1e-10

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        assert abs(mae(y_true, y_pred) - mae(y_pred, y_true)) < 1e-10


class TestRMSE:

    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert rmse(x, x) == 0.0

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt(12.5)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_rmse_ge_mae(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(100)
        y_pred = rng.randn(100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestDirectionalAccuracy:

    def test_perfect_direction(self):
        # Need 10+ elements so np.diff gives 9+ (guard: valid.sum() >= 5)
        rng = np.random.RandomState(42)
        y_true = np.cumsum(rng.randn(20))
        y_pred = y_true * 0.8 + rng.randn(20) * 0.01  # same direction
        da = directional_accuracy(y_true, y_pred)
        assert da > 80.0, "Expected >80% for near-perfect direction, got {}".format(da)

    def test_random_near_50(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(10000)
        y_pred = rng.randn(10000)
        da = directional_accuracy(y_true, y_pred)
        assert 45.0 < da < 55.0


class TestInformationCoefficient:

    def test_ic_is_spearman(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(500)
        y_pred = y_true * 0.5 + rng.randn(500) * 0.5
        ic, pval = information_coefficient(y_true, y_pred)
        sp, _ = spearman_corr(y_true, y_pred)
        assert abs(ic - sp) < 1e-10


class TestQuantileHitRate:

    def test_perfect_ranking(self):
        y_true = np.arange(100, dtype=float)
        y_pred = np.arange(100, dtype=float)
        qhr = quantile_hit_rate(y_true, y_pred, n_quantiles=5)
        assert qhr > 95.0, "Expected ~100% for perfect ranking, got {}".format(qhr)

    def test_random_near_expected(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(10000)
        y_pred = rng.randn(10000)
        qhr = quantile_hit_rate(y_true, y_pred, n_quantiles=5)
        assert 15.0 < qhr < 25.0


class TestRegimeAccuracy:

    def test_perfect_regime(self):
        # Need 30+ elements (n_regimes=3, guard: n < n_regimes*10)
        y_true = np.linspace(0.01, 1.0, 50)
        ra = regime_accuracy(y_true, y_true)
        assert ra > 95.0, "Expected ~100% for perfect regime, got {}".format(ra)

    def test_random_regime(self):
        rng = np.random.RandomState(42)
        y_true = rng.rand(1000)
        y_pred = rng.rand(1000)
        ra = regime_accuracy(y_true, y_pred)
        assert 25.0 < ra < 45.0
