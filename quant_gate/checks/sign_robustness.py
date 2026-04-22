# ============================================================
# sign_robustness.py — QVG v1.1: Sign robustness test
# Detects if model accidentally has inverted signal
# ============================================================

import numpy as np
from ..metrics import spearman_corr


class SignRobustnessCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.corr_normal = None
        self.corr_flipped = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["SignRobustnessCheck: {} (normal={:.4f}, flipped={:.4f})".format(
            status,
            self.corr_normal or 0,
            self.corr_flipped or 0)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def run_sign_robustness_checks(y_true, y_pred):
    """Test if flipping predictions improves correlation.

    If corr(y_true, -y_pred) > corr(y_true, y_pred):
    -> model has inverted signal (conceptual bug)
    """
    result = SignRobustnessCheckResult()

    corr_normal, _ = spearman_corr(y_true, y_pred)
    corr_flipped, _ = spearman_corr(y_true, -y_pred)

    result.corr_normal = corr_normal
    result.corr_flipped = corr_flipped

    # Flipped signal should NOT be better
    if corr_flipped > corr_normal and corr_flipped > 0.02:
        result.fail(
            "INVERTED SIGNAL: corr(y, -pred)={:.4f} > "
            "corr(y, pred)={:.4f}. Model predicts in the "
            "wrong direction.".format(corr_flipped, corr_normal))

    # Both directions near zero = no signal at all
    if abs(corr_normal) < 0.01 and abs(corr_flipped) < 0.01:
        result.fail(
            "No signal in either direction: "
            "normal={:.4f}, flipped={:.4f}".format(
                corr_normal, corr_flipped))

    return result
