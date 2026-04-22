# ============================================================
# metrics.py — Core metric computations for QVG v1
# Pure functions, no side effects, deterministic
# ============================================================

import numpy as np
from scipy import stats


def pearson_corr(y_true, y_pred):
    """Pearson correlation with p-value."""
    if len(y_true) < 10:
        return 0.0, 1.0
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r), float(p)


def spearman_corr(y_true, y_pred):
    """Spearman rank correlation with p-value."""
    if len(y_true) < 10:
        return 0.0, 1.0
    r, p = stats.spearmanr(y_true, y_pred)
    return float(r), float(p)


def mae(y_true, y_pred):
    """Mean Absolute Error — manual, no sklearn dependency."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error — manual."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true, y_pred):
    """Fraction of times sign(delta_pred) == sign(delta_true)."""
    if len(y_true) < 3:
        return 0.0
    dy = np.diff(y_true)
    dp = np.diff(y_pred)
    valid = (dy != 0) & (dp != 0)
    if valid.sum() < 5:
        return 0.0
    return float(np.mean(np.sign(dy[valid]) == np.sign(dp[valid])) * 100.0)


def information_coefficient(y_true, y_pred):
    """IC = Spearman rank correlation (standard cross-sectional quant)."""
    r, p = spearman_corr(y_true, y_pred)
    return r, p


def quantile_hit_rate(y_true, y_pred, n_quantiles=5):
    """Fraction of times both fall in the same quantile bucket."""
    if len(y_true) < n_quantiles * 5:
        return 0.0
    try:
        bins = np.percentile(y_true, np.linspace(0, 100, n_quantiles + 1))
        bins[0] -= 1e-8
        bins[-1] += 1e-8
        true_q = np.digitize(y_true, bins)
        pred_q = np.digitize(y_pred, bins)
        return float(np.mean(true_q == pred_q) * 100.0)
    except Exception:
        return 0.0


def regime_accuracy(y_true, y_pred, n_regimes=3):
    """Accuracy of regime classification using terciles."""
    if len(y_true) < n_regimes * 10:
        return 0.0
    try:
        cuts = np.percentile(y_true, np.linspace(0, 100, n_regimes + 1))
        cuts[0] -= 1e-8
        cuts[-1] += 1e-8
        true_reg = np.digitize(y_true, cuts)
        pred_reg = np.digitize(y_pred, cuts)
        return float(np.mean(true_reg == pred_reg) * 100.0)
    except Exception:
        return 0.0
