# ============================================================
# consistency.py — QVG v1.1b: Model-type-aware consistency
# Separates directional vs volatility validation logic
# ============================================================

import numpy as np
from ..metrics import pearson_corr, directional_accuracy


class ConsistencyCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
        self.pearson = None
        self.dir_acc = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["ConsistencyCheck: {} (Pearson={:.4f}, DirAcc={:.1f}%)".format(
            status, self.pearson or 0, self.dir_acc or 0)]
        for e in self.errors:
            lines.append("  - " + e)
        for w in self.warnings:
            lines.append("  ~ " + w)
        return "\n".join(lines)


def run_consistency_checks(y_true, y_pred, model_type="vol"):
    """Detect impossible metric combinations.

    Args:
        model_type: 'vol' for volatility prediction,
                    'directional' for return/direction prediction
    """
    result = ConsistencyCheckResult()

    r, _ = pearson_corr(y_true, y_pred)
    d = directional_accuracy(y_true, y_pred)
    result.pearson = r
    result.dir_acc = d

    if model_type == "directional":
        # Directional: high corr + bad dir acc = bug
        if abs(r) > 0.1 and d < 40.0:
            result.fail(
                "Inconsistent: Pearson={:.4f} but DirAcc={:.1f}%. "
                "These cannot coexist -- likely evaluation bug.".format(r, d))

        if d < 1.0 and abs(r) > 0.05:
            result.fail(
                "DirAcc is {:.1f}% with correlation {:.4f}. "
                "Metric implementation is broken.".format(d, r))

    elif model_type == "vol":
        # Vol prediction: DirAcc is informational only
        # Low DirAcc with high Pearson is EXPECTED (predicts level not changes)
        if d < 40.0 and abs(r) > 0.1:
            result.warn(
                "DirAcc={:.1f}% with Pearson={:.4f}. Expected for "
                "vol-level prediction (not vol-change).".format(d, r))

    # Universal: negative corr + high dir acc = sign convention error
    if r < -0.1 and d > 60.0:
        result.fail(
            "Negative correlation ({:.4f}) with high DirAcc ({:.1f}%). "
            "Check sign conventions.".format(r, d))

    return result
