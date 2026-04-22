# ============================================================
# overfitting.py — Overfitting detection
# Rejects models where OOS performance collapses vs IS
# ============================================================

import numpy as np


class OverfittingCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.is_oos_ratio = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        r = self.is_oos_ratio or 0
        lines = ["OverfittingCheck: {} (IS/OOS={:.2f}x)".format(status, r)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def run_overfitting_checks(is_metric, oos_metric, max_ratio=0.6,
                           metric_name="IC"):
    """Check IS vs OOS generalization.
    
    Args:
        is_metric: in-sample metric value
        oos_metric: out-of-sample metric value
        max_ratio: minimum OOS/IS ratio (reject below this)
        metric_name: label for reporting
    """
    result = OverfittingCheckResult()

    # Handle edge cases
    if is_metric is None or oos_metric is None:
        result.fail("Missing IS or OOS metric")
        return result

    if abs(is_metric) < 1e-10:
        # IS metric is zero — can't compute ratio
        result.is_oos_ratio = 0.0
        if abs(oos_metric) < 1e-10:
            result.fail("Both IS and OOS {} are zero".format(metric_name))
        return result

    ratio = oos_metric / is_metric
    result.is_oos_ratio = ratio

    # OOS/IS ratio below threshold = overfitting
    if ratio < max_ratio:
        result.fail(
            "{} OOS/IS ratio = {:.2f}x (threshold: {:.2f}x). "
            "IS={:.4f}, OOS={:.4f}. "
            "Model is overfitting.".format(
                metric_name, ratio, max_ratio, is_metric, oos_metric)
        )

    # Negative ratio = sign flip between IS and OOS
    if ratio < 0:
        result.fail(
            "{} sign flip: IS={:.4f} but OOS={:.4f}. "
            "Signal does not generalize.".format(
                metric_name, is_metric, oos_metric)
        )

    # Extremely high IS performance is suspicious
    if abs(is_metric) > 0.9:
        result.fail(
            "IS {} = {:.4f} is suspiciously high. "
            "Possible data snooping.".format(metric_name, is_metric)
        )

    return result
