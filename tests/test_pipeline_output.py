# ============================================================
# test_pipeline_output.py — Integration test
# Loads predictions.parquet and runs full QVG validation
# Skipped if file does not exist (CI without data)
# FIXED: horizon=1 to avoid false embargo violations in tests
# ============================================================

import numpy as np
import pandas as pd
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quant_gate.validator import validate, load_predictions


PRED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "predictions.parquet")


class TestLoadPredictions:

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_predictions("nonexistent_file.parquet")

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            df.to_parquet(f.name)
            with pytest.raises(ValueError, match="Missing required"):
                load_predictions(f.name)

    def test_valid_file_loads(self):
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            "y_true": rng.randn(n),
            "y_pred": rng.randn(n),
            "fold": np.repeat([0, 1], n // 2),
            "horizon": np.repeat([1], n),
        })
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            df.to_parquet(f.name)
            result = load_predictions(f.name)
            assert len(result) == n


class TestFullValidation:

    def test_good_signal_passes(self):
        """Synthetic data with real signal should pass QVG."""
        rng = np.random.RandomState(42)
        n_per_fold = 300
        rows = []
        for fold in range(4):
            y_true = rng.randn(n_per_fold) * 0.1
            y_pred = y_true * 0.4 + rng.randn(n_per_fold) * 0.06
            for i in range(n_per_fold):
                rows.append({
                    "y_true": y_true[i],
                    "y_pred": y_pred[i],
                    "fold": fold,
                    "horizon": 1,
                })
        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            df.to_parquet(f.name)
            result = validate(f.name, verbose=False, model_type="vol")
            assert result is True

    def test_random_predictions_fail(self):
        """Pure random predictions should fail QVG."""
        rng = np.random.RandomState(42)
        n_per_fold = 300
        rows = []
        for fold in range(4):
            y_true = rng.randn(n_per_fold)
            y_pred = rng.randn(n_per_fold)
            for i in range(n_per_fold):
                rows.append({
                    "y_true": y_true[i],
                    "y_pred": y_pred[i],
                    "fold": fold,
                    "horizon": 1,
                })
        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            df.to_parquet(f.name)
            with pytest.raises(AssertionError):
                validate(f.name, verbose=False, model_type="vol")

    def test_leakage_data_fails(self):
        """y_pred == y_true should fail QVG."""
        rng = np.random.RandomState(42)
        n_per_fold = 300
        rows = []
        for fold in range(4):
            y_true = rng.randn(n_per_fold)
            for i in range(n_per_fold):
                rows.append({
                    "y_true": y_true[i],
                    "y_pred": y_true[i],
                    "fold": fold,
                    "horizon": 1,
                })
        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            df.to_parquet(f.name)
            with pytest.raises(AssertionError):
                validate(f.name, verbose=False, model_type="vol")

    @pytest.mark.skipif(
        not os.path.exists(PRED_PATH),
        reason="No predictions.parquet found")
    def test_real_pipeline_output(self):
        """Run QVG on actual pipeline output if available."""
        validate(PRED_PATH, verbose=True, model_type="vol")
