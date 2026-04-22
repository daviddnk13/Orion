# ============================================================
# stability.py — QVG v1.1: Hardened cross-fold stability
# 75% positive ratio, median IC > 0.03, reject IC < -0.05
# ============================================================

import numpy as np
from ..metrics import spearman_corr


class StabilityCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.fold_ics = []
        self.positive_ratio = None
        self.median_ic = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["StabilityCheck: {} (pos_ratio={:.0%}, median_IC={:.4f})".format(
            status, self.positive_ratio or 0, self.median_ic or 0)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def run_stability_checks(fold_results, min_positive_ratio=0.75,
                         min_median_ic=0.03, min_fold_ic=-0.05):
    """Hardened cross-fold stability validation.

    Args:
        fold_results: list of dicts with 'y_true' and 'y_pred'
        min_positive_ratio: minimum fraction of folds with IC > 0
        min_median_ic: minimum median IC across folds
        min_fold_ic: reject if any fold IC below this
    """
    result = StabilityCheckResult()

    if len(fold_results) < 2:
        result.fail("Need at least 2 folds, got {}".format(len(fold_results)))
        return result

    ics = []
    for i, fold in enumerate(fold_results):
        ic, _ = spearman_corr(fold['y_true'], fold['y_pred'])
        ics.append(ic)

    result.fold_ics = ics
    n_positive = sum(1 for ic in ics if ic > 0)
    result.positive_ratio = n_positive / len(ics)
    result.median_ic = float(np.median(ics))

    # 1. At least 75% folds must have positive IC
    if result.positive_ratio < min_positive_ratio:
        result.fail(
            "Only {}/{} folds have positive IC ({:.0%} < {:.0%})".format(
                n_positive, len(ics), result.positive_ratio,
                min_positive_ratio))

    # 2. Median IC must exceed threshold
    if result.median_ic < min_median_ic:
        result.fail(
            "Median IC ({:.4f}) < threshold ({:.4f})".format(
                result.median_ic, min_median_ic))

    # 3. Reject any fold with catastrophic IC
    for i, ic in enumerate(ics):
        if ic < min_fold_ic:
            result.fail(
                "Fold {} IC = {:.4f} < floor ({:.4f}). "
                "Catastrophic fold.".format(i, ic, min_fold_ic))

    # 4. Check for extreme variance across folds
    ic_std = np.std(ics)
    ic_mean = np.mean(ics)
    if ic_mean > 0 and ic_std > 3 * abs(ic_mean):
        result.fail(
            "IC extremely unstable: mean={:.4f} std={:.4f} "
            "(CV={:.1f}x)".format(ic_mean, ic_std, ic_std / abs(ic_mean)))

    return result
