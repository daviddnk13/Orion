# ============================================================
# validator.py — QVG v1.1c Core Orchestrator
# Loads predictions, runs ALL checks, blocks on failure
# v1.1c FIXES: Spearman unified, horizon passthrough,
#   block_size capped, overfitting check WIRED
# ============================================================

import numpy as np
import pandas as pd
import os

from .checks.sanity import run_sanity_checks
from .checks.error import run_error_checks
from .checks.consistency import run_consistency_checks
from .checks.significance import run_significance_checks
from .checks.stability import run_stability_checks
from .checks.overfitting import run_overfitting_checks
from .checks.baseline import run_baseline_checks
from .checks.temporal import run_temporal_checks
from .checks.ic_decay import run_ic_decay_checks
from .checks.sign_robustness import run_sign_robustness_checks
from .metrics import spearman_corr


REQUIRED_COLUMNS = ['y_true', 'y_pred', 'fold', 'horizon']


def load_predictions(path="outputs/predictions.parquet"):
    """Load and validate the predictions file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Predictions file not found: {}".format(path))

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: {}".format(missing))

    for col in ['y_true', 'y_pred']:
        if not np.issubdtype(df[col].dtype, np.floating):
            df[col] = df[col].astype(float)

    for col in ['fold', 'horizon']:
        if not np.issubdtype(df[col].dtype, np.integer):
            df[col] = df[col].astype(int)

    return df


def validate_group(y_true, y_pred, label="", model_type="vol",
                   horizon=6):
    """Run all per-group checks. Returns list of results."""
    results = []

    # 1. Sanity
    r = run_sanity_checks(y_true, y_pred)
    results.append(("Sanity", r))

    # 2. Error metrics
    r = run_error_checks(y_true, y_pred)
    results.append(("Error", r))

    # 3. Consistency (model_type-aware)
    r = run_consistency_checks(y_true, y_pred, model_type=model_type)
    results.append(("Consistency", r))

    # 4. Significance (block permutation, Spearman, horizon-aware)
    r = run_significance_checks(y_true, y_pred,
                                model_type=model_type,
                                horizon=horizon)
    results.append(("Significance", r))

    # 5. Baseline (model_type-aware lag-1 logic)
    r = run_baseline_checks(y_true, y_pred, model_type=model_type)
    results.append(("Baseline", r))

    # 6. Sign robustness (detect inverted signal)
    r = run_sign_robustness_checks(y_true, y_pred)
    results.append(("SignRobustness", r))

    return results


def _run_overfitting_for_fold(df_fold, model_type="vol", verbose=True):
    """Run overfitting check for a single fold if split column exists.

    Requires 'split' column with values 'is' and 'oos'.
    Returns (check_name, result) or None if not applicable.
    """
    if 'split' not in df_fold.columns:
        return None

    is_data = df_fold[df_fold['split'] == 'is']
    oos_data = df_fold[df_fold['split'] == 'oos']

    if len(is_data) < 10 or len(oos_data) < 10:
        return None

    is_ic, _ = spearman_corr(is_data['y_true'].values,
                             is_data['y_pred'].values)
    oos_ic, _ = spearman_corr(oos_data['y_true'].values,
                              oos_data['y_pred'].values)

    r = run_overfitting_checks(is_ic, oos_ic, max_ratio=0.6,
                               metric_name="IC")
    return ("Overfitting", r)


def validate(path="outputs/predictions.parquet", verbose=True,
             model_type="vol"):
    """Full validation pipeline. Raises AssertionError on failure.

    Args:
        model_type: 'vol' for volatility prediction,
                    'directional' for return/direction prediction
    """
    df = load_predictions(path)

    has_split = 'split' in df.columns
    horizons = sorted(df['horizon'].unique())
    all_failures = []
    all_warnings = []
    total_checks = 0
    passed_checks = 0

    if verbose:
        print("QVG v1.1c | model_type={} | {} horizons | {} rows{}".format(
            model_type, len(horizons), len(df),
            " | split=yes" if has_split else ""))

    # ---- Temporal integrity (whole DataFrame) ----
    for h in horizons:
        df_h = df[df['horizon'] == h].copy()
        r = run_temporal_checks(df_h, horizon=h)
        total_checks += 1
        label = "h={}".format(h)
        if r.passed:
            passed_checks += 1
            if verbose:
                print("  [PASS] {} | Temporal".format(label))
        else:
            all_failures.append((label, "Temporal", r.errors))
            if verbose:
                print("  [FAIL] {} | Temporal".format(label))
                for e in r.errors:
                    print("         {}".format(e))

    # ---- Per-horizon, per-fold checks ----
    horizon_agg = {}

    for h in horizons:
        df_h = df[df['horizon'] == h]
        folds = sorted(df_h['fold'].unique())

        if verbose:
            print("\n" + "=" * 50)
            print("HORIZON {} | {} folds | {} samples".format(
                h, len(folds), len(df_h)))
            print("=" * 50)

        fold_data = []
        for f in folds:
            df_f = df_h[df_h['fold'] == f]
            # For per-fold checks, use OOS data only (or all if no split)
            if has_split:
                df_oos = df_f[df_f['split'] == 'oos']
                if len(df_oos) < 10:
                    df_oos = df_f  # fallback
            else:
                df_oos = df_f

            y_true = df_oos['y_true'].values
            y_pred = df_oos['y_pred'].values

            label = "h={} fold={}".format(h, f)
            results = validate_group(y_true, y_pred, label,
                                     model_type=model_type,
                                     horizon=h)

            for name, r in results:
                total_checks += 1
                if r.passed:
                    passed_checks += 1
                    if verbose:
                        print("  [PASS] {} | {}".format(label, name))
                else:
                    all_failures.append((label, name, r.errors))
                    if verbose:
                        print("  [FAIL] {} | {}".format(label, name))
                        for e in r.errors:
                            print("         {}".format(e))

                if hasattr(r, 'warnings'):
                    for w in r.warnings:
                        all_warnings.append((label, name, w))
                        if verbose:
                            print("  [WARN] {} | {}: {}".format(
                                label, name, w))

            fold_data.append({
                'y_true': y_true,
                'y_pred': y_pred,
            })

            # Overfitting check per fold (if split column exists)
            ov = _run_overfitting_for_fold(df_f, model_type, verbose)
            if ov is not None:
                ov_name, ov_r = ov
                total_checks += 1
                if ov_r.passed:
                    passed_checks += 1
                    if verbose:
                        print("  [PASS] {} | {}".format(label, ov_name))
                else:
                    all_failures.append((label, ov_name, ov_r.errors))
                    if verbose:
                        print("  [FAIL] {} | {}".format(label, ov_name))
                        for e in ov_r.errors:
                            print("         {}".format(e))

        # Cross-fold stability check (hardened v1.1)
        if len(fold_data) >= 2:
            r = run_stability_checks(fold_data)
            total_checks += 1
            label = "h={}".format(h)
            if r.passed:
                passed_checks += 1
                if verbose:
                    print("  [PASS] {} | Stability".format(label))
            else:
                all_failures.append((label, "Stability", r.errors))
                if verbose:
                    print("  [FAIL] {} | Stability".format(label))
                    for e in r.errors:
                        print("         {}".format(e))

        # Aggregate for IC decay
        all_y_true = df_h['y_true'].values
        all_y_pred = df_h['y_pred'].values
        horizon_agg[h] = {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
        }

    # ---- IC decay across horizons ----
    if len(horizon_agg) >= 2:
        r = run_ic_decay_checks(horizon_agg)
        total_checks += 1
        label = "cross-horizon"
        if r.passed:
            passed_checks += 1
            if verbose:
                print("  [PASS] {} | ICDecay".format(label))
        else:
            all_failures.append((label, "ICDecay", r.errors))
            if verbose:
                print("  [FAIL] {} | ICDecay".format(label))
                for e in r.errors:
                    print("         {}".format(e))

    # ---- Final report ----
    if verbose:
        print("\n" + "=" * 50)
        print("QVG v1.1c SUMMARY: {}/{} checks passed".format(
            passed_checks, total_checks))
        if all_warnings:
            print("  Warnings: {}".format(len(all_warnings)))
        if not has_split:
            print("  Note: No 'split' column -- overfitting check skipped")
        print("=" * 50)

    if all_failures:
        msg_lines = ["QVG VALIDATION FAILED ({} failures):".format(
            len(all_failures))]
        for label, name, errors in all_failures:
            msg_lines.append("  {} | {}: {}".format(
                label, name, "; ".join(errors)))
        msg = "\n".join(msg_lines)

        if verbose:
            print("\n" + msg)

        raise AssertionError(msg)

    if verbose:
        print("\nQVG v1.1c VALIDATION PASSED -- all checks green.")

    return True
