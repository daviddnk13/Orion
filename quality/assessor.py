# ============================================================
# quality/assessor.py — V19.1 Feature Quality Assessment
# IC = Spearman, per-horizon, cross-fold stability
# Output: descending feature ranking + full report
# ============================================================

import numpy as np
from scipy.stats import spearmanr


def compute_feature_ic(X, y, feature_names):
    """Compute Spearman IC for each feature against target.

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) target array
        feature_names: list of feature name strings

    Returns:
        ic_dict: {feature_name: (ic_value, p_value)}
    """
    ic_dict = {}
    for j, fname in enumerate(feature_names):
        if len(X[:, j]) < 20:
            ic_dict[fname] = (0.0, 1.0)
            continue
        r, p = spearmanr(X[:, j], y)
        if np.isnan(r):
            r, p = 0.0, 1.0
        ic_dict[fname] = (float(r), float(p))
    return ic_dict


def compute_cross_fold_quality(X, y, feature_names, fold_indices):
    """Compute IC stability across folds for each feature.

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) target array
        feature_names: list of feature name strings
        fold_indices: list of (train_idx, test_idx) tuples

    Returns:
        quality_report: dict with per-feature quality metrics
    """
    n_feat = len(feature_names)
    n_folds = len(fold_indices)

    # Collect IC per fold per feature
    ic_matrix = np.zeros((n_folds, n_feat))

    for fold_i, (train_idx, test_idx) in enumerate(fold_indices):
        for j in range(n_feat):
            x_test = X[test_idx, j]
            y_test = y[test_idx]
            if len(x_test) < 20:
                ic_matrix[fold_i, j] = 0.0
                continue
            r, _ = spearmanr(x_test, y_test)
            ic_matrix[fold_i, j] = r if not np.isnan(r) else 0.0

    # Aggregate per feature
    quality_report = {}
    for j, fname in enumerate(feature_names):
        fold_ics = ic_matrix[:, j]
        mean_ic = float(np.mean(fold_ics))
        std_ic = float(np.std(fold_ics))
        min_ic = float(np.min(fold_ics))
        max_ic = float(np.max(fold_ics))
        pos_ratio = float(np.mean(fold_ics > 0))

        # Stability score: higher = more stable
        # Penalizes high variance and inconsistent sign
        if std_ic < 1e-10:
            stability = 1.0 if mean_ic > 0 else 0.0
        else:
            stability = max(0.0, mean_ic / (std_ic + 1e-8))

        quality_report[fname] = {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'min_ic': min_ic,
            'max_ic': max_ic,
            'pos_ratio': pos_ratio,
            'stability_score': float(stability),
            'fold_ics': fold_ics.tolist(),
        }

    return quality_report


def rank_features(quality_report, sort_by='stability_score'):
    """Rank features by quality metric (descending).

    Args:
        quality_report: dict from compute_cross_fold_quality
        sort_by: metric to sort by (default: stability_score)

    Returns:
        ranked: list of (feature_name, metrics_dict) sorted descending
    """
    ranked = sorted(
        quality_report.items(),
        key=lambda x: x[1].get(sort_by, 0.0),
        reverse=True
    )
    return ranked


def assess_quality(X, y, feature_names, fold_indices,
                   horizons=None, sort_by='stability_score'):
    """Full feature quality assessment pipeline.

    If horizons provided, evaluates per horizon separately.

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) target array OR dict {horizon: y_array}
        feature_names: list of feature name strings
        fold_indices: list of (train_idx, test_idx) tuples
        horizons: list of horizon values (optional)
        sort_by: ranking metric

    Returns:
        full_report: {
            'overall': quality_report + ranking,
            'per_horizon': {h: quality_report + ranking} (if horizons)
        }
    """
    full_report = {}

    # Overall assessment
    if isinstance(y, dict):
        # Use first available horizon for overall
        first_h = list(y.keys())[0]
        y_overall = y[first_h]
    else:
        y_overall = y

    overall_quality = compute_cross_fold_quality(
        X, y_overall, feature_names, fold_indices)
    overall_ranking = rank_features(overall_quality, sort_by)

    full_report['overall'] = {
        'quality': overall_quality,
        'ranking': [(name, metrics) for name, metrics in overall_ranking],
        'top5': [name for name, _ in overall_ranking[:5]],
    }

    # Per-horizon assessment
    if horizons and isinstance(y, dict):
        full_report['per_horizon'] = {}
        for h in horizons:
            if h not in y:
                continue
            h_quality = compute_cross_fold_quality(
                X, y[h], feature_names, fold_indices)
            h_ranking = rank_features(h_quality, sort_by)
            full_report['per_horizon'][h] = {
                'quality': h_quality,
                'ranking': [(name, m) for name, m in h_ranking],
                'top5': [name for name, _ in h_ranking[:5]],
            }

    return full_report


def print_quality_report(full_report, max_features=10):
    """Print human-readable quality report.

    Args:
        full_report: output of assess_quality
        max_features: max features to display per section
    """
    lines = []
    lines.append("=" * 60)
    lines.append("FEATURE QUALITY REPORT — V19.1")
    lines.append("=" * 60)

    overall = full_report.get('overall', {})
    ranking = overall.get('ranking', [])

    lines.append(f"\nTop {min(max_features, len(ranking))} features "
                 f"(by stability score):")
    lines.append(f"{'Rank':<5} {'Feature':<30} {'MeanIC':<10} "
                 f"{'StdIC':<10} {'Pos%':<8} {'Stab':<8}")
    lines.append("-" * 71)

    for i, (name, m) in enumerate(ranking[:max_features]):
        lines.append(
            f"{i+1:<5} {name:<30} {m['mean_ic']:>+.4f}    "
            f"{m['std_ic']:.4f}    {m['pos_ratio']*100:>5.1f}%  "
            f"{m['stability_score']:>.3f}"
        )

    # Per horizon if available
    per_h = full_report.get('per_horizon', {})
    for h in sorted(per_h.keys()):
        h_data = per_h[h]
        h_rank = h_data.get('ranking', [])
        lines.append(f"\n--- Horizon {h} ---")
        lines.append(f"  Top 5: {h_data.get('top5', [])}")

    report_text = "\n".join(lines)
    print(report_text)
    return report_text
