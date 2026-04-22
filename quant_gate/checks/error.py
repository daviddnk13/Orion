# ============================================================
# error.py — Error metric validation
# Rejects zero-error results (impossible in real data)
# ============================================================

import numpy as np
from ..metrics import mae, rmse


class ErrorCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.mae_value = None
        self.rmse_value = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["ErrorCheck: {} (MAE={:.6f}, RMSE={:.6f})".format(
            status, self.mae_value or 0, self.rmse_value or 0)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def run_error_checks(y_true, y_pred, min_mae=1e-10, min_rmse=1e-10):
    """Validate that error metrics are non-zero and reasonable."""
    result = ErrorCheckResult()

    # Compute metrics manually (no sklearn)
    result.mae_value = mae(y_true, y_pred)
    result.rmse_value = rmse(y_true, y_pred)

    # 1. Zero MAE = impossible with real predictions
    if result.mae_value < min_mae:
        result.fail(
            "MAE is effectively zero ({:.2e}). "
            "Likely leakage or evaluation bug.".format(result.mae_value)
        )

    # 2. Zero RMSE = impossible with real predictions
    if result.rmse_value < min_rmse:
        result.fail(
            "RMSE is effectively zero ({:.2e}). "
            "Likely leakage or evaluation bug.".format(result.rmse_value)
        )

    # 3. RMSE < MAE is mathematically impossible
    if result.rmse_value < result.mae_value - 1e-12:
        result.fail(
            "RMSE ({:.6f}) < MAE ({:.6f}): mathematically impossible".format(
                result.rmse_value, result.mae_value)
        )

    # 4. Suspiciously low error relative to target range
    target_range = np.max(y_true) - np.min(y_true)
    if target_range > 0:
        relative_mae = result.mae_value / target_range
        if relative_mae < 1e-6:
            result.fail(
                "MAE/range ratio suspiciously low: {:.2e}".format(relative_mae)
            )

    return result
