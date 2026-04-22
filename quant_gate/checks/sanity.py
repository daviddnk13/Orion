# ============================================================
# sanity.py — Fundamental data integrity checks
# Detects leakage, constant arrays, length mismatches
# ============================================================

import numpy as np


class SanityCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        if self.passed:
            return "SanityCheck: PASS"
        return "SanityCheck: FAIL\n" + "\n".join("  - " + e for e in self.errors)


def run_sanity_checks(y_true, y_pred):
    """Run all sanity checks. Returns SanityCheckResult."""
    result = SanityCheckResult()

    # 1. Length check
    if len(y_true) != len(y_pred):
        result.fail(
            "Length mismatch: y_true={} y_pred={}".format(len(y_true), len(y_pred))
        )
        return result  # Cannot continue with mismatched lengths

    if len(y_true) < 10:
        result.fail("Insufficient samples: {}".format(len(y_true)))
        return result

    # 2. NaN/Inf check
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        result.fail("y_true contains NaN or Inf")

    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        result.fail("y_pred contains NaN or Inf")

    # 3. Leakage detection (predictions == target)
    if np.allclose(y_pred, y_true, atol=1e-10):
        result.fail(
            "LEAKAGE DETECTED: y_pred is identical to y_true "
            "(max_diff={:.2e})".format(np.max(np.abs(y_pred - y_true)))
        )

    # 4. Constant predictions
    if np.std(y_pred) < 1e-10:
        result.fail(
            "Predictions are CONSTANT: std={:.2e}, "
            "mean={:.6f}".format(np.std(y_pred), np.mean(y_pred))
        )

    # 5. Constant target
    if np.std(y_true) < 1e-10:
        result.fail(
            "Target is CONSTANT: std={:.2e}, "
            "mean={:.6f}".format(np.std(y_true), np.mean(y_true))
        )

    # 6. Near-perfect correlation (suspiciously high)
    if result.passed and not np.allclose(y_pred, y_true):
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        if abs(corr) > 0.99:
            result.fail(
                "Suspiciously high correlation: {:.6f} "
                "(possible near-leakage)".format(corr)
            )

    return result
