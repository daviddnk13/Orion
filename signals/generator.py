# ============================================================
# signals/generator.py — V19.1 Signal Generator
# IC-weighted linear combination — deterministic, no ML
# ============================================================

import numpy as np
from scipy.stats import spearmanr, zscore


def compute_ic_weights(X_train, y_train, feature_names, min_ic=0.02):
    """Compute Spearman IC weights from training data only.

    Args:
        X_train: (n_train, n_features) training features
        y_train: (n_train,) training target
        feature_names: list of feature names
        min_ic: minimum absolute IC to include feature

    Returns:
        weights: (n_features,) array of IC-based weights
        weight_report: dict with per-feature IC details
    """
    n_feat = X_train.shape[1]
    raw_ics = np.zeros(n_feat)
    weight_report = {}

    for j in range(n_feat):
        r, _ = spearmanr(X_train[:, j], y_train)
        if np.isnan(r):
            r = 0.0
        raw_ics[j] = r
        weight_report[feature_names[j]] = float(r)

    # Zero out features below minimum IC
    weights = np.where(np.abs(raw_ics) >= min_ic, raw_ics, 0.0)

    # Normalize to sum of absolute weights = 1
    total = np.sum(np.abs(weights))
    if total > 1e-10:
        weights = weights / total
    else:
        # Fallback: equal weights for all features
        weights = np.ones(n_feat) / n_feat

    return weights, weight_report


def generate_signal_raw(X, weights):
    """Generate raw signal as weighted linear combination of z-scored features.

    Args:
        X: (n_samples, n_features) feature matrix
        weights: (n_features,) IC-based weights

    Returns:
        y_pred_raw: (n_samples,) continuous signal
    """
    # Z-score each feature column (zero mean, unit variance)
    X_z = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        std = np.std(col)
        if std > 1e-10:
            X_z[:, j] = (col - np.mean(col)) / std
        else:
            X_z[:, j] = 0.0

    # Weighted linear combination
    y_pred_raw = X_z @ weights

    return y_pred_raw.astype(float)


def smooth_signal(y_pred_raw, window=5):
    """Apply simple moving average smoothing to reduce noise.

    Args:
        y_pred_raw: (n_samples,) raw signal
        window: smoothing window size

    Returns:
        y_pred_smooth: (n_samples,) smoothed signal
    """
    if window <= 1:
        return y_pred_raw.copy()

    n = len(y_pred_raw)
    y_smooth = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)
        y_smooth[i] = np.mean(y_pred_raw[start:i + 1])

    return y_smooth


def clip_signal(y_pred, clip_std=3.0):
    """Clip extreme predictions to reduce outlier impact.

    Args:
        y_pred: (n_samples,) signal
        clip_std: number of standard deviations for clipping

    Returns:
        y_pred_clipped: (n_samples,) clipped signal
    """
    mean = np.mean(y_pred)
    std = np.std(y_pred)
    if std < 1e-10:
        return y_pred.copy()

    lower = mean - clip_std * std
    upper = mean + clip_std * std
    return np.clip(y_pred, lower, upper)


def generate_signal(X_train, y_train, X_test, feature_names,
                    min_ic=0.02, smooth_window=5, clip_std=3.0):
    """Full signal generation pipeline: weights -> raw -> smooth -> clip.

    Uses ONLY training data for weight computation (no lookahead).

    Args:
        X_train: (n_train, n_features) training features
        y_train: (n_train,) training target
        X_test: (n_test, n_features) test features
        feature_names: list of feature names
        min_ic: minimum IC for feature inclusion
        smooth_window: smoothing window size
        clip_std: clipping threshold in std

    Returns:
        y_pred_raw: (n_test,) raw signal
        y_pred_final: (n_test,) filtered signal (goes to QVG)
        signal_report: dict with weights and diagnostics
    """
    # Step 1: Compute weights from training data ONLY
    weights, weight_report = compute_ic_weights(
        X_train, y_train, feature_names, min_ic)

    # Step 2: Generate raw signal on test data
    y_pred_raw = generate_signal_raw(X_test, weights)

    # Step 3: Smooth
    y_pred_smooth = smooth_signal(y_pred_raw, smooth_window)

    # Step 4: Clip extremes
    y_pred_final = clip_signal(y_pred_smooth, clip_std)

    signal_report = {
        'weights': dict(zip(feature_names, weights.tolist())),
        'ic_per_feature': weight_report,
        'n_active_features': int(np.sum(np.abs(weights) > 1e-10)),
        'n_total_features': len(feature_names),
        'raw_std': float(np.std(y_pred_raw)),
        'final_std': float(np.std(y_pred_final)),
    }

    return y_pred_raw, y_pred_final, signal_report
