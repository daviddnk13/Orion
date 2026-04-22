# ============================================================
# baseline.py — QVG v1.1b: Model-type-aware baseline comparison
# Lag-1 MAE = warning, lag-1 IC = hard requirement
# ============================================================

import numpy as np
from ..metrics import mae, pearson_corr, spearman_corr


class BaselineCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
        self.model_mae = None
        self.zero_mae = None
        self.random_mae = None
        self.lag1_mae = None
        self.rolling_mae = None
        self.model_ic = None
        self.lag1_ic = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["BaselineCheck: {} (model_MAE={:.6f}, model_IC={:.4f})".format(
            status, self.model_mae or 0, self.model_ic or 0)]
        for e in self.errors:
            lines.append("  - " + e)
        for w in self.warnings:
            lines.append("  ~ " + w)
        return "\n".join(lines)


def run_baseline_checks(y_true, y_pred, seed=42, rolling_window=20,
                        model_type="vol"):
    """Compare model vs multiple baselines.

    For vol models: lag-1 MAE is a WARNING (persistence is strong),
    but lag-1 IC is a HARD requirement (must beat in rank correlation).
    """
    result = BaselineCheckResult()
    result.model_mae = mae(y_true, y_pred)
    result.model_ic, _ = spearman_corr(y_true, y_pred)

    # Baseline 1: Zero predictor (predict mean)
    zero_pred = np.full_like(y_true, np.mean(y_true))
    result.zero_mae = mae(y_true, zero_pred)

    # Baseline 2: Random predictor
    rng = np.random.RandomState(seed)
    random_pred = rng.permutation(y_true)
    result.random_mae = mae(y_true, random_pred)

    # Baseline 3: Lag-1 persistence
    lag1_pred = None
    if len(y_true) > 1:
        lag1_pred = np.concatenate([[y_true[0]], y_true[:-1]])
        result.lag1_mae = mae(y_true, lag1_pred)
        result.lag1_ic, _ = spearman_corr(y_true, lag1_pred)

    # Baseline 4: Rolling mean
    if len(y_true) > rolling_window:
        roll_pred = np.full_like(y_true, np.mean(y_true))
        for i in range(rolling_window, len(y_true)):
            roll_pred[i] = np.mean(y_true[max(0, i - rolling_window):i])
        valid = slice(rolling_window, None)
        result.rolling_mae = mae(y_true[valid], roll_pred[valid])

    # --- Hard checks ---

    # Must beat zero predictor (always required)
    if result.model_mae >= result.zero_mae:
        result.fail(
            "Model MAE ({:.6f}) >= mean predictor ({:.6f}).".format(
                result.model_mae, result.zero_mae))

    # Must beat random predictor (always required)
    if result.model_mae >= result.random_mae:
        result.fail(
            "Model MAE ({:.6f}) >= random predictor ({:.6f}).".format(
                result.model_mae, result.random_mae))

    # Lag-1: behavior depends on model_type
    if lag1_pred is not None:
        if model_type == "vol":
            # Vol: lag-1 MAE is just a warning (persistence is strong)
            if result.model_mae >= result.lag1_mae:
                result.warn(
                    "Model MAE ({:.6f}) >= lag-1 MAE ({:.6f}). "
                    "Persistence is strong in vol.".format(
                        result.model_mae, result.lag1_mae))
            # Vol: lag-1 IC is HARD requirement
            if result.model_ic <= result.lag1_ic:
                result.fail(
                    "Model IC ({:.4f}) <= lag-1 IC ({:.4f}). "
                    "Must beat persistence in rank correlation.".format(
                        result.model_ic, result.lag1_ic))
        else:
            # Directional: lag-1 MAE is hard requirement
            if result.model_mae >= result.lag1_mae:
                result.fail(
                    "Model MAE ({:.6f}) >= lag-1 ({:.6f}).".format(
                        result.model_mae, result.lag1_mae))

    # Correlation must beat random
    random_corr, _ = spearman_corr(y_true, random_pred)
    if abs(result.model_ic) <= abs(random_corr) + 0.01:
        result.fail(
            "Model IC ({:.4f}) not better than random ({:.4f}).".format(
                result.model_ic, random_corr))

    return result
