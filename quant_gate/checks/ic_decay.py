# ============================================================
# ic_decay.py — QVG v1.1: IC decay validation across horizons
# IC should decay (not explode) at longer horizons
# ============================================================

import numpy as np
from ..metrics import spearman_corr


class ICDecayCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.horizon_ics = {}

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        ic_str = ", ".join(
            "h{}={:.4f}".format(h, ic)
            for h, ic in sorted(self.horizon_ics.items()))
        lines = ["ICDecayCheck: {} ({})".format(status, ic_str)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def run_ic_decay_checks(horizon_data, max_increase_ratio=2.0):
    """Validate IC does not explode at longer horizons.

    Args:
        horizon_data: dict {horizon: {'y_true': array, 'y_pred': array}}
        max_increase_ratio: max allowed IC increase vs shortest horizon

    Principle: IC(short) >= IC(long) in well-specified models.
    If IC increases at longer horizons, likely overfitting or leakage.
    """
    result = ICDecayCheckResult()

    if len(horizon_data) < 2:
        # Only 1 horizon — skip check
        return result

    horizons = sorted(horizon_data.keys())

    for h in horizons:
        ic, _ = spearman_corr(
            horizon_data[h]['y_true'],
            horizon_data[h]['y_pred'])
        result.horizon_ics[h] = ic

    # Check for exploding IC at longer horizons
    shortest_h = horizons[0]
    shortest_ic = abs(result.horizon_ics[shortest_h])

    if shortest_ic > 0.01:  # Only check if base IC is meaningful
        for h in horizons[1:]:
            h_ic = abs(result.horizon_ics[h])
            ratio = h_ic / (shortest_ic + 1e-10)

            if ratio > max_increase_ratio:
                result.fail(
                    "IC INCREASES at longer horizon: "
                    "h{}={:.4f} vs h{}={:.4f} "
                    "(ratio={:.2f}x > {:.1f}x). "
                    "Likely overfitting or leakage.".format(
                        h, h_ic, shortest_h, shortest_ic,
                        ratio, max_increase_ratio))

    # Check for sign flip across horizons (inconsistency)
    ics = [result.horizon_ics[h] for h in horizons]
    signs = [np.sign(ic) for ic in ics if abs(ic) > 0.01]
    if len(signs) >= 2:
        if len(set(signs)) > 1:
            result.fail(
                "IC sign FLIPS across horizons: {}. "
                "Model direction inconsistent.".format(
                    {h: "{:.4f}".format(result.horizon_ics[h])
                     for h in horizons}))

    return result
