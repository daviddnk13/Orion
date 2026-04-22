# ============================================================
# signals/filters.py — V19.1 Feature Filtering
# Removes noise, redundancy, and instability BEFORE signal gen
# Deterministic, no ML, no lookahead
# ============================================================

import numpy as np
from scipy.stats import spearmanr


def filter_low_variance(X, feature_names, min_std=1e-6):
    """Remove features with near-zero variance.

    Args:
        X: (n_samples, n_features) array
        feature_names: list of feature name strings
        min_std: minimum standard deviation threshold

    Returns:
        X_filtered, kept_names, removed_names
    """
    stds = np.std(X, axis=0)
    mask = stds > min_std
    kept = [f for f, m in zip(feature_names, mask) if m]
    removed = [f for f, m in zip(feature_names, mask) if not m]
    return X[:, mask], kept, removed


def filter_high_correlation(X, feature_names, max_corr=0.95):
    """Remove redundant features (pairwise Spearman > threshold).

    Keeps the first feature in each correlated pair (order matters).

    Args:
        X: (n_samples, n_features) array
        feature_names: list of feature name strings
        max_corr: maximum allowed pairwise correlation

    Returns:
        X_filtered, kept_names, removed_names
    """
    n_feat = X.shape[1]
    if n_feat < 2:
        return X, list(feature_names), []

    # Compute Spearman correlation matrix
    corr_matrix = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r, _ = spearmanr(X[:, i], X[:, j])
            corr_matrix[i, j] = abs(r)
            corr_matrix[j, i] = abs(r)

    # Greedily remove features with highest correlation
    to_remove = set()
    for i in range(n_feat):
        if i in to_remove:
            continue
        for j in range(i + 1, n_feat):
            if j in to_remove:
                continue
            if corr_matrix[i, j] > max_corr:
                to_remove.add(j)

    mask = [i not in to_remove for i in range(n_feat)]
    kept = [f for f, m in zip(feature_names, mask) if m]
    removed = [f for f, m in zip(feature_names, mask) if not m]
    return X[:, mask], kept, removed


def filter_unstable_ic(X, y, feature_names, fold_indices,
                       max_ic_std=0.15, min_positive_folds=0.5):
    """Remove features with unstable IC across folds.

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) target array
        feature_names: list of feature name strings
        fold_indices: list of (train_idx, test_idx) tuples
        max_ic_std: maximum IC standard deviation across folds
        min_positive_folds: minimum fraction of folds with IC > 0

    Returns:
        X_filtered, kept_names, removed_names, ic_report
    """
    n_feat = X.shape[1]
    ic_report = {}

    for j in range(n_feat):
        fold_ics = []
        for train_idx, test_idx in fold_indices:
            x_test = X[test_idx, j]
            y_test = y[test_idx]
            if len(x_test) < 20:
                continue
            r, _ = spearmanr(x_test, y_test)
            if not np.isnan(r):
                fold_ics.append(r)

        fname = feature_names[j]
        if len(fold_ics) < 2:
            ic_report[fname] = {
                'mean_ic': 0.0, 'std_ic': 1.0,
                'pos_ratio': 0.0, 'stable': False
            }
            continue

        mean_ic = np.mean(fold_ics)
        std_ic = np.std(fold_ics)
        pos_ratio = sum(1 for ic in fold_ics if ic > 0) / len(fold_ics)

        stable = std_ic <= max_ic_std and pos_ratio >= min_positive_folds
        ic_report[fname] = {
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            'pos_ratio': float(pos_ratio),
            'fold_ics': fold_ics,
            'stable': stable,
        }

    # Filter
    mask = []
    for j, fname in enumerate(feature_names):
        mask.append(ic_report.get(fname, {}).get('stable', False))

    kept = [f for f, m in zip(feature_names, mask) if m]
    removed = [f for f, m in zip(feature_names, mask) if not m]
    return X[:, mask], kept, removed, ic_report


def apply_all_filters(X, y, feature_names, fold_indices,
                      min_std=1e-6, max_corr=0.95,
                      max_ic_std=0.15, min_positive_folds=0.5):
    """Apply all filters in sequence: variance -> correlation -> IC stability.

    Returns:
        X_filtered, kept_names, filter_report
    """
    report = {'original_count': len(feature_names)}

    # Step 1: Variance filter
    X, names, removed_var = filter_low_variance(X, feature_names, min_std)
    report['removed_variance'] = removed_var
    report['after_variance'] = len(names)

    # Step 2: Correlation filter
    X, names, removed_corr = filter_high_correlation(X, names, max_corr)
    report['removed_correlation'] = removed_corr
    report['after_correlation'] = len(names)

    # Step 3: IC stability filter
    X, names, removed_ic, ic_report = filter_unstable_ic(
        X, y, names, fold_indices, max_ic_std, min_positive_folds)
    report['removed_ic_stability'] = removed_ic
    report['ic_report'] = ic_report
    report['final_count'] = len(names)

    return X, names, report
