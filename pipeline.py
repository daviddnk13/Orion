# ============================================================
# pipeline.py — V19.1 Orchestrator
# Produces QVG-compatible output: y_true, y_pred, fold, horizon
# Integrates: filters -> quality -> regime -> signal -> output
# ============================================================

import numpy as np
import os
import json

# Local imports (relative to orion/)
from signals.filters import apply_all_filters
from signals.generator import generate_signal
from regime.detector import detect_regimes
from quality.assessor import assess_quality, print_quality_report


def build_fold_indices(n_samples, n_folds=4, test_size=1250, embargo=180):
    """Build walk-forward fold indices (anchored expanding window).

    Args:
        n_samples: total number of samples
        n_folds: number of folds
        test_size: test set size per fold
        embargo: gap between train and test

    Returns:
        folds: list of (train_indices, test_indices) arrays
    """
    folds = []
    for fold_i in range(n_folds):
        test_end = n_samples - (n_folds - fold_i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - embargo

        if train_end < 100:
            raise ValueError(
                f"Fold {fold_i}: insufficient training data "
                f"(train_end={train_end})")

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        folds.append((train_idx, test_idx))

    return folds


def run_pipeline(X, y, feature_names, returns,
                 n_folds=4, test_size=1250, embargo=180,
                 horizon=6, seed=42,
                 min_std=1e-6, max_corr=0.95,
                 max_ic_std=0.15, min_positive_folds=0.5,
                 min_ic=0.02, smooth_window=5, clip_std=3.0,
                 output_dir="outputs"):
    """Run full V19.1 pipeline.

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) target (realized volatility)
        feature_names: list of feature name strings
        returns: (n_samples,) log returns for regime detection
        n_folds: number of walk-forward folds
        test_size: test samples per fold
        embargo: gap between train/test
        horizon: forecast horizon in bars
        seed: random seed
        min_std: variance filter threshold
        max_corr: correlation filter threshold
        max_ic_std: IC stability filter threshold
        min_positive_folds: min fraction of positive IC folds
        min_ic: minimum IC for signal weight inclusion
        smooth_window: signal smoothing window
        clip_std: signal clipping threshold
        output_dir: directory for output files

    Returns:
        results: dict with all outputs and reports
    """
    np.random.seed(seed)
    n_samples = X.shape[0]

    print("=" * 60)
    print("ORION V19.1 — SIGNAL QUALITY PIPELINE")
    print("=" * 60)
    print(f"Samples: {n_samples}, Features: {len(feature_names)}")
    print(f"Folds: {n_folds}, Test: {test_size}, Embargo: {embargo}")
    print(f"Horizon: {horizon}")
    print()

    # ---- Step 0: Build fold indices ----
    folds = build_fold_indices(n_samples, n_folds, test_size, embargo)
    print(f"[OK] Built {len(folds)} walk-forward folds")
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i}: train=[0:{len(tr)}], "
              f"test=[{te[0]}:{te[-1]+1}] ({len(te)} samples)")

    # ---- Step 1: Feature filtering ----
    print("\n--- STEP 1: Feature Filtering ---")
    X_filtered, kept_names, filter_report = apply_all_filters(
        X, y, feature_names, folds,
        min_std=min_std, max_corr=max_corr,
        max_ic_std=max_ic_std, min_positive_folds=min_positive_folds)

    print(f"  Original features:  {filter_report['original_count']}")
    print(f"  After variance:     {filter_report['after_variance']} "
          f"(removed: {filter_report['removed_variance']})")
    print(f"  After correlation:  {filter_report['after_correlation']} "
          f"(removed: {filter_report['removed_correlation']})")
    print(f"  After IC stability: {filter_report['final_count']} "
          f"(removed: {filter_report['removed_ic_stability']})")

    if filter_report['final_count'] < 2:
        print("\n[ABORT] Fewer than 2 features survived filtering!")
        print("Falling back to top features by mean IC...")
        # Fallback: use top features by absolute mean IC
        ic_report = filter_report.get('ic_report', {})
        sorted_feats = sorted(
            ic_report.items(),
            key=lambda x: abs(x[1].get('mean_ic', 0)),
            reverse=True)
        fallback_names = [f for f, _ in sorted_feats[:5]]
        fallback_idx = [feature_names.index(f) for f in fallback_names
                        if f in feature_names]
        X_filtered = X[:, fallback_idx]
        kept_names = fallback_names[:len(fallback_idx)]
        print(f"  Fallback features: {kept_names}")

    # ---- Step 2: Feature quality assessment ----
    print("\n--- STEP 2: Feature Quality Assessment ---")
    quality_report = assess_quality(
        X_filtered, y, kept_names, folds, sort_by='stability_score')
    report_text = print_quality_report(quality_report)

    # ---- Step 3: Regime detection ----
    print("\n--- STEP 3: Regime Detection ---")
    regimes, regime_report = detect_regimes(returns)
    print(f"  Regime distribution: {regime_report['fractions']}")

    # ---- Step 4: Per-fold signal generation ----
    print("\n--- STEP 4: Signal Generation (per fold) ---")
    all_y_true = []
    all_y_pred_raw = []
    all_y_pred_final = []
    all_folds = []
    all_horizons = []
    all_regimes = []
    fold_reports = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train = X_filtered[train_idx]
        y_train = y[train_idx]
        X_test = X_filtered[test_idx]
        y_test = y[test_idx]

        y_pred_raw, y_pred_final, signal_report = generate_signal(
            X_train, y_train, X_test, kept_names,
            min_ic=min_ic, smooth_window=smooth_window,
            clip_std=clip_std)

        # Sanity checks
        assert len(y_pred_final) == len(y_test), \
            f"Fold {fold_i}: length mismatch {len(y_pred_final)} vs {len(y_test)}"
        assert np.std(y_pred_final) > 1e-10, \
            f"Fold {fold_i}: predictions are constant"
        assert not np.any(np.isnan(y_pred_final)), \
            f"Fold {fold_i}: NaN in predictions"

        all_y_true.append(y_test)
        all_y_pred_raw.append(y_pred_raw)
        all_y_pred_final.append(y_pred_final)
        all_folds.append(np.full(len(y_test), fold_i, dtype=int))
        all_horizons.append(np.full(len(y_test), horizon, dtype=int))
        all_regimes.append(regimes[test_idx])

        fold_reports.append(signal_report)

        print(f"  Fold {fold_i}: active_features="
              f"{signal_report['n_active_features']}, "
              f"raw_std={signal_report['raw_std']:.6f}, "
              f"final_std={signal_report['final_std']:.6f}")
        top_w = sorted(signal_report['weights'].items(),
                       key=lambda x: abs(x[1]), reverse=True)[:3]
        top_str = ", ".join(f"{n}={w:+.3f}" for n, w in top_w)
        print(f"         top_weights: {top_str}")

    # ---- Step 5: Assemble QVG-compatible output ----
    print("\n--- STEP 5: Assemble Output ---")
    y_true_all = np.concatenate(all_y_true)
    y_pred_raw_all = np.concatenate(all_y_pred_raw)
    y_pred_final_all = np.concatenate(all_y_pred_final)
    fold_all = np.concatenate(all_folds)
    horizon_all = np.concatenate(all_horizons)
    regime_all = np.concatenate(all_regimes)

    # Final sanity
    assert not np.allclose(y_pred_final_all, y_true_all), \
        "LEAKAGE: predictions identical to target"
    assert np.std(y_pred_final_all) > 1e-8, \
        "Predictions are constant across all folds"
    assert np.std(y_true_all) > 1e-8, \
        "Target is constant across all folds"
    assert len(y_true_all) == len(y_pred_final_all) == len(fold_all), \
        "Length mismatch in final arrays"

    print(f"  Total samples: {len(y_true_all)}")
    print(f"  y_true  range: [{y_true_all.min():.6f}, {y_true_all.max():.6f}]")
    print(f"  y_pred  range: [{y_pred_final_all.min():.6f}, "
          f"{y_pred_final_all.max():.6f}]")
    print(f"  Folds:  {np.unique(fold_all).tolist()}")

    # ---- Save outputs ----
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions.parquet (QVG input)
    try:
        import pandas as pd
        df = pd.DataFrame({
            'y_true': y_true_all,
            'y_pred': y_pred_final_all,
            'fold': fold_all,
            'horizon': horizon_all,
        })
        parquet_path = os.path.join(output_dir, "predictions.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"\n  [OK] Saved: {parquet_path}")
    except ImportError:
        # Fallback: save as CSV
        csv_path = os.path.join(output_dir, "predictions.csv")
        header = "y_true,y_pred,fold,horizon"
        data = np.column_stack([y_true_all, y_pred_final_all,
                                fold_all, horizon_all])
        np.savetxt(csv_path, data, delimiter=",", header=header,
                   comments="")
        print(f"\n  [OK] Saved: {csv_path}")

    # Save full report as JSON
    report_path = os.path.join(output_dir, "v19_1_report.json")
    full_results = {
        'version': 'V19.1',
        'n_samples': int(len(y_true_all)),
        'n_folds': n_folds,
        'horizon': horizon,
        'features_original': len(feature_names),
        'features_kept': len(kept_names),
        'kept_features': kept_names,
        'filter_report': {
            'original_count': filter_report['original_count'],
            'after_variance': filter_report['after_variance'],
            'after_correlation': filter_report['after_correlation'],
            'final_count': filter_report['final_count'],
            'removed_variance': filter_report['removed_variance'],
            'removed_correlation': filter_report['removed_correlation'],
            'removed_ic_stability': filter_report['removed_ic_stability'],
        },
        'regime_distribution': regime_report['fractions'],
        'fold_signal_reports': fold_reports,
        'quality_top5': quality_report.get('overall', {}).get('top5', []),
    }

    with open(report_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"  [OK] Saved: {report_path}")

    print("\n" + "=" * 60)
    print("V19.1 PIPELINE COMPLETE")
    print(f"Output ready for QVG: {output_dir}/predictions.parquet")
    print("=" * 60)

    return {
        'y_true': y_true_all,
        'y_pred_raw': y_pred_raw_all,
        'y_pred_final': y_pred_final_all,
        'fold': fold_all,
        'horizon': horizon_all,
        'regime': regime_all,
        'kept_features': kept_names,
        'filter_report': filter_report,
        'quality_report': quality_report,
        'regime_report': regime_report,
        'fold_signal_reports': fold_reports,
    }
