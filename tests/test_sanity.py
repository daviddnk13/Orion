# ============================================================
# test_sanity.py — Tests for sanity, error, and consistency checks
# Covers: valid PASS, leakage FAIL, zero-error FAIL, random FAIL
# ============================================================

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quant_gate.checks.sanity import run_sanity_checks
from quant_gate.checks.error import run_error_checks
from quant_gate.checks.consistency import run_consistency_checks


class TestSanityChecks:
    """Tests for sanity.py"""

    def test_valid_data_passes(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(500)
        y_pred = y_true * 0.3 + rng.randn(500) * 0.7
        result = run_sanity_checks(y_true, y_pred)
        assert result.passed, "Valid data should pass: {}".format(result.errors)

    def test_leakage_detected(self):
        y_true = np.random.randn(500)
        y_pred = y_true.copy()  # exact copy = leakage
        result = run_sanity_checks(y_true, y_pred)
        assert not result.passed, "Leakage should be detected"
        assert any("leakage" in e.lower() or "identical" in e.lower()
                    for e in result.errors)

    def test_constant_predictions_detected(self):
        y_true = np.random.randn(500)
        y_pred = np.full(500, 0.5)  # constant
        result = run_sanity_checks(y_true, y_pred)
        assert not result.passed, "Constant predictions should fail"

    def test_constant_target_detected(self):
        y_true = np.full(500, 1.0)  # constant target
        y_pred = np.random.randn(500)
        result = run_sanity_checks(y_true, y_pred)
        assert not result.passed, "Constant target should fail"

    def test_length_mismatch_detected(self):
        y_true = np.random.randn(500)
        y_pred = np.random.randn(400)
        result = run_sanity_checks(y_true, y_pred)
        assert not result.passed, "Length mismatch should fail"

    def test_nan_detected(self):
        y_true = np.random.randn(500)
        y_pred = np.random.randn(500)
        y_pred[10] = np.nan
        result = run_sanity_checks(y_true, y_pred)
        assert not result.passed, "NaN should be detected"

    def test_inf_detected(self):
        y_true = np.random.randn(500)
        y_pred = np.random.randn(500)
        y_pred[5] = np.inf
        result = run_sanity_checks(y_true, y_pred)
        assert not result.passed, "Inf should be detected"


class TestErrorChecks:
    """Tests for error.py"""

    def test_valid_errors_pass(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(500)
        y_pred = y_true * 0.3 + rng.randn(500) * 0.7
        result = run_error_checks(y_true, y_pred)
        assert result.passed, "Valid errors should pass: {}".format(
            result.errors)

    def test_zero_mae_detected(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = run_error_checks(y_true, y_pred)
        assert not result.passed, "Zero MAE should fail"

    def test_zero_rmse_detected(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = run_error_checks(y_true, y_pred)
        assert not result.passed, "Zero RMSE should fail"


class TestConsistencyChecks:
    """Tests for consistency.py"""

    def test_valid_consistency_passes(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(500)
        y_pred = y_true * 0.5 + rng.randn(500) * 0.5
        result = run_consistency_checks(y_true, y_pred, model_type="vol")
        assert result.passed, "Consistent data should pass: {}".format(
            result.errors)

    def test_random_predictions_pass_consistency(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(500)
        y_pred = rng.randn(500)  # uncorrelated
        result = run_consistency_checks(y_true, y_pred, model_type="vol")
        # Random should pass consistency (low corr, ~50% dir acc)
        assert result.passed, "Random should pass consistency: {}".format(
            result.errors)
