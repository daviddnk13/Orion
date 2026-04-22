# ============================================================
# significance.py — QVG v1.1c: Block permutation test
# FIXED: Spearman everywhere, horizon passthrough, block_size cap
# ============================================================

import numpy as np
from ..metrics import spearman_corr


class SignificanceCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.ic = None
        self.perm_p_value = None
        self.n_perms = None
        self.block_size_used = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["SignificanceCheck: {} (IC={:.4f}, perm_p={:.4f}, bs={})".format(
            status, self.ic or 0, self.perm_p_value or 1,
            self.block_size_used or 0)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def _block_permute(arr, block_size, rng):
    """Permute array in blocks to preserve autocorrelation."""
    n = len(arr)
    n_blocks = max(1, int(np.ceil(n / block_size)))
    padded_len = n_blocks * block_size
    padded = np.zeros(padded_len)
    padded[:n] = arr
    blocks = padded.reshape(n_blocks, block_size)
    idx = rng.permutation(n_blocks)
    shuffled = blocks[idx].ravel()
    return shuffled[:n]


def block_permutation_test(y_true, y_pred, n_perm=1000,
                           block_size=None, seed=42, horizon=6):
    """Block permutation test for IC significance (Spearman).

    Args:
        y_true: actual values
        y_pred: predicted values
        n_perm: number of permutations (1000 for production)
        block_size: size of blocks (auto-calibrated if None)
        seed: random seed for reproducibility
        horizon: prediction horizon in bars (used for block_size)

    Returns:
        observed_ic, p_value, block_size_used
    """
    n = len(y_true)

    # Auto-calibrate block_size with cap to prevent p-value bias
    if block_size is None:
        block_size = min(
            max(horizon * 2, int(2 * np.sqrt(n))),
            n // 5
        )
    block_size = max(block_size, 2)  # floor at 2

    rng = np.random.RandomState(seed)

    # Observed IC using Spearman (consistent with metrics.py)
    observed_ic, _ = spearman_corr(y_true, y_pred)

    # Count how many permuted ICs exceed observed
    n_exceed = 0
    for _ in range(n_perm):
        perm_pred = _block_permute(y_pred, block_size, rng)
        perm_ic, _ = spearman_corr(y_true, perm_pred)
        if abs(perm_ic) >= abs(observed_ic):
            n_exceed += 1

    p_value = (n_exceed + 1) / (n_perm + 1)
    return observed_ic, p_value, block_size


def run_significance_checks(y_true, y_pred, min_ic=None, max_p=0.05,
                            n_perm=1000, block_size=None,
                            model_type="vol", horizon=6):
    """Accept IC only if block permutation p < max_p AND IC > min_ic.

    Args:
        min_ic: minimum IC threshold (auto-set by model_type if None)
        n_perm: number of permutations (1000 for production)
        block_size: block size for permutation (auto if None)
        model_type: 'vol' (threshold 0.10) or 'directional' (threshold 0.03)
        horizon: prediction horizon in bars
    """
    if min_ic is None:
        min_ic = 0.10 if model_type == "vol" else 0.03

    result = SignificanceCheckResult()

    ic, perm_p, bs_used = block_permutation_test(
        y_true, y_pred, n_perm=n_perm, block_size=block_size,
        horizon=horizon)

    result.ic = ic
    result.perm_p_value = perm_p
    result.n_perms = n_perm
    result.block_size_used = bs_used

    if abs(ic) < min_ic:
        result.fail(
            "IC too low: {:.4f} (threshold: {:.4f})".format(abs(ic), min_ic))

    if perm_p > max_p:
        result.fail(
            "Not significant (block permutation): p={:.4f} "
            "(threshold: {:.4f}, n_perm={}, block_size={})".format(
                perm_p, max_p, n_perm, bs_used))

    return result
