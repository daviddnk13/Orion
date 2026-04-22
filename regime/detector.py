# ============================================================
# regime/detector.py — V19.1 Rule-Based Regime Detection
# Statistical rules only, no ML, no future labels
# Output: integer 0-3 per sample
#   0 = trend, 1 = mean reversion, 2 = high vol, 3 = low vol
# ============================================================

import numpy as np


def compute_rolling_vol(returns, window=20):
    """Rolling standard deviation of returns."""
    n = len(returns)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = np.std(returns[i - window:i])
    # Fill initial NaNs with first valid value
    first_valid = vol[window] if window < n else 0.0
    vol[:window] = first_valid
    return vol


def compute_trend_strength(returns, window=20):
    """Trend strength = abs(cumulative return) / sum(abs returns).

    Range: 0 (choppy) to 1 (perfect trend).
    Known as efficiency ratio (Kaufman).
    """
    n = len(returns)
    strength = np.full(n, 0.5)
    for i in range(window, n):
        chunk = returns[i - window:i]
        net_move = abs(np.sum(chunk))
        total_move = np.sum(np.abs(chunk))
        if total_move > 1e-12:
            strength[i] = net_move / total_move
    # Fill initial values
    strength[:window] = strength[window] if window < n else 0.5
    return strength


def compute_mean_reversion_score(returns, window=20):
    """Mean reversion score = fraction of sign changes in returns.

    High score = choppy / mean reverting.
    """
    n = len(returns)
    score = np.full(n, 0.5)
    for i in range(window, n):
        chunk = returns[i - window:i]
        signs = np.sign(chunk)
        nonzero = signs[signs != 0]
        if len(nonzero) > 1:
            changes = np.sum(np.diff(nonzero) != 0)
            score[i] = changes / (len(nonzero) - 1)
    score[:window] = score[window] if window < n else 0.5
    return score


def detect_regimes(returns, vol_window=20, trend_window=20,
                   vol_high_pct=80, vol_low_pct=20,
                   trend_threshold=0.4, mr_threshold=0.6):
    """Detect market regime per sample using statistical rules.

    Priority order (first match wins):
        1. High vol (rolling vol > 80th percentile of history)
        2. Low vol (rolling vol < 20th percentile of history)
        3. Trend (efficiency ratio > threshold)
        4. Mean reversion (sign change ratio > threshold)
        Fallback: trend (0)

    Args:
        returns: (n_samples,) array of log returns
        vol_window: window for rolling volatility
        trend_window: window for trend/MR detection
        vol_high_pct: percentile for high vol classification
        vol_low_pct: percentile for low vol classification
        trend_threshold: efficiency ratio above this = trend
        mr_threshold: sign change ratio above this = mean reversion

    Returns:
        regimes: (n_samples,) integer array (0-3)
        regime_report: dict with diagnostics
    """
    n = len(returns)
    regimes = np.zeros(n, dtype=int)

    # Compute indicators
    vol = compute_rolling_vol(returns, vol_window)
    trend = compute_trend_strength(returns, trend_window)
    mr = compute_mean_reversion_score(returns, trend_window)

    # Expanding percentiles for vol thresholds (no lookahead)
    for i in range(vol_window, n):
        hist_vol = vol[vol_window:i + 1]
        high_thresh = np.percentile(hist_vol, vol_high_pct)
        low_thresh = np.percentile(hist_vol, vol_low_pct)

        if vol[i] >= high_thresh:
            regimes[i] = 2  # high vol
        elif vol[i] <= low_thresh:
            regimes[i] = 3  # low vol
        elif trend[i] >= trend_threshold:
            regimes[i] = 0  # trend
        elif mr[i] >= mr_threshold:
            regimes[i] = 1  # mean reversion
        else:
            regimes[i] = 0  # default: trend

    # Fill initial period
    regimes[:vol_window] = 0

    # Report
    counts = {
        'trend': int(np.sum(regimes == 0)),
        'mean_reversion': int(np.sum(regimes == 1)),
        'high_vol': int(np.sum(regimes == 2)),
        'low_vol': int(np.sum(regimes == 3)),
    }
    regime_report = {
        'counts': counts,
        'fractions': {k: v / n for k, v in counts.items()},
        'vol_window': vol_window,
        'trend_window': trend_window,
    }

    return regimes, regime_report
