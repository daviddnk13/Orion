# v20_3_validate.py — Orion V20.3 Risk Engine Validation
# End-to-end walk-forward: LightGBM binary classifier + RiskEngine modulation

import numpy as np
import pandas as pd
import os
import json
from data_loader import fetch_okx_ohlcv, fetch_macro_daily, align_daily_to_4h, load_derivatives_data
from features import build_features, validate_features
from targets import build_targets
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from risk_engine import RiskEngine
from config import CONFIG


def build_fold_indices(n_samples, n_folds=4, test_size=1250, embargo=180):
    """Build walk-forward fold indices (anchored expanding window)."""
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


def compute_fold_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def portfolio_stats(simple_returns, periods_per_year=365):
    """Compute total return, annualized Sharpe, and max drawdown from simple returns."""
    if len(simple_returns) == 0:
        return {'total_return': 0.0, 'sharpe': 0.0, 'max_dd': 0.0}
    # Total return (compounded)
    total_ret = np.prod(1 + simple_returns) - 1
    # Mean and std per period
    mean = simple_returns.mean()
    std = simple_returns.std()
    sharpe = mean / std * np.sqrt(periods_per_year) if std > 1e-10 else 0.0
    # Max drawdown
    wealth = np.cumprod(1 + simple_returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    max_dd = drawdown.min()
    return {'total_return': float(total_ret), 'sharpe': float(sharpe), 'max_dd': float(max_dd)}


def load_ohlcv_data():
    """Try to load OHLCV from CSV, else fetch from OKX API."""
    possible_paths = [
        'eth_usdt_4h.csv',
        'ETH-USDT_4h.csv',
        'data/eth_4h.csv',
        '/kaggle/input/eth-4h/eth_usdt_4h.csv',
        '/kaggle/input/okx-eth-4h/ohlcv.csv',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'timestamp' in df.columns and 'close' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    print(f"[DATA] Loaded OHLCV from {path}")
                    return df
            except Exception as e:
                print(f"[WARN] Failed to read {path}: {e}")
                continue
    # Fallback: fetch from OKX
    print("[DATA] Fetching OHLCV from OKX API...")
    df = fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4H', n_candles=CONFIG['n_candles'])
    return df


def main():
    np.random.seed(42)

    print("=" * 70)
    print("ORION V20.3 — RISK ENGINE VALIDATION")
    print("=" * 70)

    # ---- Step 1: Load data ----
    df = load_ohlcv_data()

    # ---- Step 2: Macro and derivatives ----
    macro = fetch_macro_daily()
    df = align_daily_to_4h(df, macro)
    df, has_funding, has_oi = load_derivatives_data(df)

    # ---- Step 3: Build features and targets ----
    print("\n[FEATURES] Building features...")
    df, raw_features = build_features(df, has_funding, has_oi)
    n_issues = validate_features(df, raw_features)
    if n_issues > 0:
        print(f"[WARN] {n_issues} feature issues detected")

    print("\n[TARGETS] Building targets...")
    df = build_targets(df)

    # Compute 24h forward cumulative log return for simulation
    print("[TARGETS] Computing 24h forward returns...")
    df['return_24h'] = df['log_ret'].rolling(window=6).sum().shift(-6)

    # ---- Step 4: Cleanup ----
    warmup = 200
    df = df.iloc[warmup:].reset_index(drop=True)
    print(f"[CLEAN] After warmup drop: {len(df)} rows")

    valid_mask = df['vol_24h_future'].notna()
    df = df[valid_mask].reset_index(drop=True)
    print(f"[CLEAN] After target NaN drop: {len(df)} rows")

    # Ensure no NaNs in features (build_features should fill, but enforce)
    # Drop rows with any NaN in raw_features just in case
    # But build_features fills with 0, so should be safe. We'll skip for speed.

    if len(df) < 5000:
        print(f"[ERROR] Insufficient data after cleaning: {len(df)} < 5000")
        return

    # Prepare arrays
    X = df[raw_features].values.astype(np.float64)
    y_vol = df['vol_24h_future'].values
    returns_24h_log = df['return_24h'].values

    n_samples = len(df)
    print(f"\n[CONFIG] Samples: {n_samples}, Features: {len(raw_features)}")
    print(f"  Walk-forward: 4 folds, test_size=1250, embargo=180, horizon=6")

    # Build walk-forward folds
    folds = build_fold_indices(n_samples, n_folds=4, test_size=1250, embargo=180)
    print(f"[FOLDS] Built {len(folds)} folds:")
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i}: train=[0:{len(tr)}], test=[{te[0]}:{te[-1]+1}] ({len(te)} samples)")

    # LightGBM parameters (V20.1 exact)
    lgbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": True,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "seed": 42,
        "verbose": -1,
    }
    num_boost_round = 300

    # RiskEngine default params
    engine_defaults = {'sensitivity': 1.0, 'max_reduction': 0.7, 'target_recall': 0.70}

    # Storage
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_fold_idx = []
    all_position_scale = []
    all_asset_returns_log = []

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n[FOLD {fold_i}]")

        # Split
        X_train, X_test = X[train_idx], X[test_idx]
        y_vol_train, y_vol_test = y_vol[train_idx], y_vol[test_idx]
        returns_24h_test = returns_24h_log[test_idx]

        # Define binary target: HIGH_VOL = top 25% of y_vol_train (75th percentile)
        threshold_vol = np.percentile(y_vol_train, 75)
        y_train_binary = (y_vol_train >= threshold_vol).astype(int)
        y_test_binary = (y_vol_test >= threshold_vol).astype(int)

        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train_binary)
        model = lgb.train(lgbm_params, train_data, num_boost_round=num_boost_round)

        # Predict probabilities
        y_train_prob = model.predict(X_train)
        y_test_prob = model.predict(X_test)

        # Fit RiskEngine on training data
        engine = RiskEngine(**engine_defaults)
        engine.fit(y_train_binary, y_train_prob)

        # Predict on test
        y_test_pred_signal = engine.predict_signal(y_test_prob)
        position_scale = engine.predict_scale(y_test_prob)

        # Metrics
        fold_metrics = compute_fold_metrics(y_test_binary, y_test_pred_signal)
        fold_metrics['auc'] = roc_auc_score(y_test_binary, y_test_prob)
        fold_metrics['signal_rate'] = np.mean(y_test_pred_signal)
        fold_metrics['position_scale_mean'] = float(np.mean(position_scale))
        fold_metrics['position_scale_min'] = float(np.min(position_scale))
        fold_metrics['position_scale_max'] = float(np.max(position_scale))

        fold_results.append(fold_metrics)

        # Print fold summary
        print(f"  Threshold (vol): {threshold_vol:.6f}")
        print(f"  Signal rate: {fold_metrics['signal_rate']:.1%}")
        print(f"  Precision: {fold_metrics['precision']:.3f}")
        print(f"  Recall:    {fold_metrics['recall']:.3f}")
        print(f"  F1:        {fold_metrics['f1']:.3f}")
        print(f"  AUC:       {fold_metrics['auc']:.3f}")
        print(f"  Pos scale: mean={fold_metrics['position_scale_mean']:.3f}, "
              f"min={fold_metrics['position_scale_min']:.3f}, max={fold_metrics['position_scale_max']:.3f}")

        # Store for aggregated analysis
        all_y_true.extend(y_test_binary)
        all_y_pred.extend(y_test_pred_signal)
        all_y_prob.extend(y_test_prob)
        all_fold_idx.extend([fold_i] * len(y_test_binary))
        all_position_scale.extend(position_scale)
        all_asset_returns_log.extend(returns_24h_test)

    # ---- Overall metrics ----
    overall_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    overall_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    overall_auc = roc_auc_score(all_y_true, all_y_prob)
    overall_signal_rate = np.mean(all_y_pred)

    print("\n" + "=" * 60)
    print("OVERALL SIGNAL QUALITY")
    print("=" * 60)
    print(f"Recall:    {overall_recall:.3f}  (target >= 0.70)")
    print(f"Precision: {overall_precision:.3f}")
    print(f"F1:        {overall_f1:.3f}  (target >= 0.40)")
    print(f"AUC:       {overall_auc:.3f}  (target > 0.60)")
    print(f"Signal rate: {overall_signal_rate:.1%}")

    # ---- Sizing impact simulation ----
    # Convert log returns to simple returns
    all_asset_returns_simple = np.exp(all_asset_returns_log) - 1
    all_position_scale_arr = np.array(all_position_scale)
    scaled_returns_simple = all_position_scale_arr * all_asset_returns_simple
    unscaled_returns_simple = all_asset_returns_simple  # full 1.0 scale

    scaled_stats = portfolio_stats(scaled_returns_simple)
    unscaled_stats = portfolio_stats(unscaled_returns_simple)

    print("\n" + "=" * 60)
    print("SIZING IMPACT")
    print("=" * 60)
    print("Unscaled (buy-and-hold):")
    print(f"  Total return: {unscaled_stats['total_return']:.2%}")
    print(f"  Annualized Sharpe: {unscaled_stats['sharpe']:.2f}")
    print(f"  Max drawdown: {unscaled_stats['max_dd']:.2%}")
    print("Scaled (with RiskEngine):")
    print(f"  Total return: {scaled_stats['total_return']:.2%}")
    print(f"  Annualized Sharpe: {scaled_stats['sharpe']:.2f}")
    print(f"  Max drawdown: {scaled_stats['max_dd']:.2%}")

    sharpe_diff = scaled_stats['sharpe'] - unscaled_stats['sharpe']
    dd_diff = unscaled_stats['max_dd'] - scaled_stats['max_dd']  # positive means reduction
    ret_diff = scaled_stats['total_return'] - unscaled_stats['total_return']

    print("\nDifferences (scaled - unscaled):")
    print(f"  Sharpe diff: {sharpe_diff:+.3f}")
    print(f"  MaxDD reduction: {dd_diff:+.2%}")
    print(f"  Total return diff: {ret_diff:+.2%}")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    if overall_recall >= 0.70 and overall_f1 >= 0.40:
        if scaled_stats['sharpe'] > unscaled_stats['sharpe']:
            verdict = "RISK ENGINE OPERATIONAL"
        else:
            verdict = "SIGNAL OK, SIZING NEEDS TUNING"
    else:
        verdict = "THRESHOLD OPTIMIZATION FAILED"
    print(f"FINAL VERDICT: {verdict}")
    print("=" * 60)

    # ---- Save results JSON ----
    results = {
        'version': 'V20.3',
        'overall': {
            'recall': float(overall_recall),
            'precision': float(overall_precision),
            'f1': float(overall_f1),
            'auc': float(overall_auc),
            'signal_rate': float(overall_signal_rate),
        },
        'folds': fold_results,
        'sizing': {
            'unscaled': unscaled_stats,
            'scaled': scaled_stats,
            'sharpe_diff': float(sharpe_diff),
            'maxdd_reduction': float(dd_diff),
            'return_diff': float(ret_diff),
        },
        'verdict': verdict,
    }
    out_path = 'v20_3_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results JSON: {out_path}")


if __name__ == '__main__':
    main()
