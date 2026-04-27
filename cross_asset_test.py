#!/home/ubuntu/orion/venv/bin/python3
# -*- coding: utf-8 -*-
"""
ORION V20.7 — Cross-Asset Generalization Test
Evaluates if the ETH-trained model (model_v20_6_1.pkl) generalizes to BTC and SOL
without retraining.

Author: Claude Code (Anthropic)
Date: 2026-04-26
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import hashlib
import requests
from datetime import datetime, timedelta, timezone
from scipy.stats import spearmanr, pearsonr
import joblib
import lightgbm as lgb

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = '/home/ubuntu/orion/model_v20_6_1.pkl'

ASSETS = ['ETH-USDT', 'BTC-USDT', 'SOL-USDT']
TIMEFRAME = '4H'
WARMUP_BARS = 200  # Drop first N bars after feature calculation

# Walk-forward parameters
N_FOLDS = 4
TEST_SIZE = 1250
EMBARGO = 180
INITIAL_TRAIN_SIZE = 2000  # Minimum initial training size

# Telegram (optional, for alerts)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '-1003505760554')

# ============================================================
# DATA FETCHING (OKX) — máximo histórico con paginación
# ============================================================
def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4h', n_candles=None):
    """
    Fetch OHLCV from OKX public API.
    If n_candles is None, fetch all available history by pagination.
    Retries up to 3 times on failure.
    """
    base_url = "https://www.okx.com"
    endpoint = "/api/v5/market/history-candles"
    params = {
        'instId': symbol,
        'bar': timeframe,
        'limit': '100'  # OKX max per request
    }

    all_candles = []
    # If n_candles is specified, fetch that many. Otherwise, fetch all possible.
    remaining = n_candles if n_candles is not None else float('inf')
    retries = 3

    while remaining > 0:
        attempt = 0
        success = False
        while attempt < retries and not success:
            try:
                limit = min(100, remaining) if n_candles is not None else 100
                params['limit'] = str(limit)
                r = requests.get(base_url + endpoint, params=params, timeout=10)
                data = r.json()
                if data.get('code') != '0':
                    print(f"[OKX] API error for {symbol}: {data.get('msg', 'unknown')}")
                    attempt += 1
                    time.sleep(1)
                    continue
                candles = data.get('data', [])
                if not candles:
                    print(f"[OKX] No more data for {symbol}")
                    # No more data available, exit both loops
                    remaining = 0
                    success = True
                    break
                all_candles.extend(candles)
                remaining -= len(candles)
                if candles:
                    # pagination: use 'after' to get older candles
                    params['after'] = candles[-1][0]
                time.sleep(0.1)  # Rate limit
                success = True
            except Exception as e:
                print(f"[OKX] Fetch error for {symbol} (attempt {attempt+1}/{retries}): {e}")
                attempt += 1
                time.sleep(1)
        if not success:
            print(f"[OKX] Failed after {retries} attempts for {symbol}")
            break

    if not all_candles:
        raise RuntimeError(f"No data fetched from OKX for {symbol}")

    df = pd.DataFrame(all_candles, columns=[
        'ts', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote', 'confirm'
    ])
    df['ts'] = pd.to_datetime(df['ts'].astype(np.int64), unit='ms', utc=True)
    df = df.sort_values('ts').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col])

    return df[['ts', 'open', 'high', 'low', 'close', 'vol']].rename(columns={'ts': 'timestamp'})

# ============================================================
# FEATURE ENGINEERING (23 features — EXACT COPY FROM paper_trading_v20_7.py)
# ============================================================
def build_features(df):
    """Build exactly 23 features as per V20.6.1 spec."""
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret
    eps = 1e-8

    # --- BASE (16) ---
    # 1. ret_4h (6-bar sum)
    df['ret_4h'] = log_ret.rolling(6).sum().shift(1)

    # 2. rsi_norm (14-period RSI normalized to [0,1])
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + eps)
    rsi = 100 - 100 / (1 + rs)
    df['rsi_norm'] = (rsi / 100).fillna(0.5).shift(1)

    # 3. bb_position (Bollinger Bands position)
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_pos = (df['close'] - lower) / (upper - lower + eps)
    df['bb_position'] = bb_pos.fillna(0.5).shift(1)

    # 4. macd_norm (MACD normalized)
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9).mean()
    macd_norm = (macd - macd_sig) / (df['close'] * 0.01 + eps)
    df['macd_norm'] = macd_norm.fillna(0).shift(1)

    # 5-6. ret_4h_lag1, ret_4h_lag2
    df['ret_4h_lag1'] = df['ret_4h'].shift(1)
    df['ret_4h_lag2'] = df['ret_4h'].shift(2)

    # 7. atr_norm (14-period ATR normalized by close)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_norm'] = (atr / df['close']).fillna(0).shift(1)

    # 8. vol_zscore (rolling 180-bar z-score of 6-bar realized vol)
    rv_6 = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_mean = rv_6.rolling(180).mean().shift(1)
    rv_std = rv_6.rolling(180).std().shift(1)
    df['vol_zscore'] = ((rv_6 - rv_mean) / (rv_std + eps)).fillna(0).shift(1)

    # 9. vol_regime (categorical: 0=low, 1=normal, 2=high)
    vol_rank = rv_6.rolling(180).rank(pct=True).shift(1)
    df['vol_regime'] = pd.cut(vol_rank, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2]).astype(float).fillna(1).shift(1)

    # 10. ret_8h (12-bar sum)
    df['ret_8h'] = log_ret.rolling(12).sum().shift(1)

    # 11. ret_24h (24-bar sum)
    df['ret_24h'] = log_ret.rolling(24).sum().shift(1)

    # 12. ema_slope (slope of EMA20)
    ema_20 = df['close'].ewm(span=20).mean()
    slope = (ema_20 - ema_20.shift(1)) / (ema_20.shift(1) + eps)
    df['ema_slope'] = slope.fillna(0).shift(1)

    # 13. trend_strength (ADX-like)
    plus_dm = df['high'] - df['high'].shift(1)
    minus_dm = df['low'].shift(1) - df['low']
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / (tr14 + eps))
    minus_di = 100 * (minus_dm.rolling(14).sum() / (tr14 + eps))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + eps)
    adx = dx.rolling(14).mean()
    df['trend_strength'] = (adx / 100).fillna(0.5).shift(1)

    # 14. drawdown_market (rolling max DD)
    roll_max = df['close'].rolling(100).max()
    dd = (df['close'] - roll_max) / (roll_max + eps)
    df['drawdown_market'] = dd.fillna(0).shift(1)

    # 15. tf_coherence (momentum coherence: short vs long momentum agreement)
    mom_short = df['close'] / df['close'].shift(6) - 1.0
    mom_long = df['close'] / df['close'].shift(24) - 1.0
    df['tf_coherence'] = (np.sign(mom_short) * np.sign(mom_long) *
                          np.minimum(np.abs(mom_short), np.abs(mom_long))).fillna(0)

    # 16. dist_ema200 (distance to EMA200 in ATR units)
    ema_200 = df['close'].ewm(span=200).mean()
    df['dist_ema200'] = ((df['close'] - ema_200) / (atr + eps)).fillna(0).shift(1)

    # --- VOL (7) ---
    # 17. parkinson_vol (Parkinson 42-bar)
    hl_ratio = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt(
        (hl_ratio ** 2).rolling(42).mean().shift(1) / (4 * np.log(2))
    ).fillna(0)

    # 18. vol_compression = realized_vol_1d / (realized_vol_7d + eps)
    rv_1d = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_7d = log_ret.rolling(42).std().shift(1) * np.sqrt(6)
    df['realized_vol_1d'] = rv_1d
    df['realized_vol_7d'] = rv_7d
    df['vol_compression'] = rv_1d / (rv_7d + eps)

    # 19. garman_klass_vol
    gk = 0.5 * (np.log(df['high'] / df['low'])) ** 2 - (2*np.log(2)-1) * (np.log(df['close']/df['open']))**2
    df['garman_klass_vol'] = np.sqrt(gk.rolling(6).mean().shift(1).clip(lower=0))

    # 20. vol_regime_rank (pct rank of 7d vol over 180)
    vol_rank_pct = rv_7d.rolling(180).rank(pct=True).shift(1)
    df['vol_regime_rank'] = vol_rank_pct.fillna(0.5)

    # 21. trend_efficiency (close_diff_24 / sum_abs_diff_24)
    close_diff_24 = (df['close'] - df['close'].shift(24)).abs()
    sum_abs_diff = df['close'].diff().abs().rolling(24).sum()
    df['trend_efficiency'] = (close_diff_24 / (sum_abs_diff + eps)).fillna(0).shift(1)

    # 22-23. realized_vol_7d already created, realized_vol_1d already created

    raw_features = [
        'ret_4h', 'rsi_norm', 'bb_position', 'macd_norm',
        'ret_4h_lag1', 'ret_4h_lag2',
        'atr_norm', 'vol_zscore', 'vol_regime', 'ret_8h', 'ret_24h',
        'ema_slope', 'drawdown_market', 'tf_coherence', 'dist_ema200', 'trend_strength',
        'parkinson_vol', 'vol_compression', 'garman_klass_vol',
        'vol_regime_rank', 'realized_vol_7d', 'trend_efficiency', 'realized_vol_1d'
    ]

    # Fill any remaining NaNs
    for f in raw_features:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].fillna(0)

    return df, raw_features

# ============================================================
# TARGET: Forward realized volatility (24h = 6 bars)
# ============================================================
def add_target(df):
    """Add vol_24h_future: 6-bar forward realized volatility."""
    log_ret = df['log_ret'].values
    n = len(log_ret)
    vol_24h = np.full(n, np.nan)
    for t in range(n - 6):
        window = log_ret[t + 1: t + 7]
        if len(window) == 6:
            vol_24h[t] = np.std(window, ddof=1) * np.sqrt(6)
    df['vol_24h_future'] = vol_24h
    return df

# ============================================================
# WALK-FORWARD SPLITS
# ============================================================
def generate_walk_forward_splits(n_samples, initial_train_size=INITIAL_TRAIN_SIZE,
                                 test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS):
    """
    Generate expanding walk-forward splits with embargo.
    Yields (train_idx, test_idx) for each fold.
    """
    # Ensure we have enough data
    total_needed = initial_train_size + n_folds * test_size + n_folds * embargo
    if n_samples < total_needed:
        print(f"[WARN] Only {n_samples} samples, need ~{total_needed}. Reducing parameters.")
        # Adjust: reduce initial_train_size or n_folds
        max_possible_folds = (n_samples - initial_train_size) // (test_size + embargo)
        if max_possible_folds < 1:
            raise ValueError(f"Insufficient data: {n_samples} < required minimum")
        n_folds = min(n_folds, max_possible_folds)
        print(f"[ADJUST] Using {n_folds} folds")

    splits = []
    train_end = initial_train_size

    for fold in range(n_folds):
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_samples:
            print(f"[WARN] Fold {fold}: test_end {test_end} exceeds n_samples {n_samples}. Stopping.")
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

        # Expand training set to just before next test
        train_end = test_end

    return splits

# ============================================================
# EVALUATION METRICS
# ============================================================
def evaluate_fold(model, X_test, y_test_vol, calib_data=None):
    """
    Evaluate one fold.
    X_test: DataFrame with 23 columns named Column_0..Column_22
    y_test_vol: series of realized vol (vol_24h_future)
    calib_data: tuple (X_cal, y_cal_vol) used to compute thresholds
    Returns dict with metrics, and arrays for diagnostic.
    """
    X_test_values = X_test.values.astype(np.float64)
    X_test_values = np.clip(X_test_values, -5, 5)
    proba = model.predict(X_test_values)  # shape (n_test,)

    # Máscara de valores finitos en y_test_vol
    valid_mask = np.isfinite(y_test_vol.values)
    if valid_mask.sum() == 0:
        metrics = {
            'ic': np.nan, 'ic_pvalue': np.nan, 'monotonicity': np.nan,
            'accuracy': np.nan, 'n_samples': 0,
            'p20_proba': np.nan, 'p80_proba': np.nan,
            'p20_vol': np.nan, 'p80_vol': np.nan
        }
        return metrics, proba, None, None

    proba_valid = proba[valid_mask]
    y_vol_valid = y_test_vol.values[valid_mask]

    # Calibración usando calib_data (training set)
    if calib_data is not None:
        X_cal, y_cal_vol = calib_data
        X_cal_values = X_cal.values.astype(np.float64)
        X_cal_values = np.clip(X_cal_values, -5, 5)
        proba_cal = model.predict(X_cal_values)
        # Thresholds de proba
        p20_proba = np.percentile(proba_cal, 20)
        p80_proba = np.percentile(proba_cal, 80)
        # Thresholds de vol (solo finitos)
        y_cal_finite = y_cal_vol.dropna().values
        if len(y_cal_finite) > 0:
            p20_vol = np.percentile(y_cal_finite, 20)
            p80_vol = np.percentile(y_cal_finite, 80)
        else:
            p20_vol = p80_vol = np.nan
    else:
        # Fallback: usar test data
        p20_proba = np.percentile(proba_valid, 20)
        p80_proba = np.percentile(proba_valid, 80)
        p20_vol = np.percentile(y_vol_valid, 20)
        p80_vol = np.percentile(y_vol_valid, 80)

    # Regímenes
    actual_regime = np.select(
        [y_vol_valid <= p20_vol, y_vol_valid >= p80_vol],
        [0, 2],
        default=1
    )
    pred_regime = np.select(
        [proba_valid <= p20_proba, proba_valid >= p80_proba],
        [0, 2],
        default=1
    )

    # 1. IC
    if len(proba_valid) >= 10:
        ic, p_val = spearmanr(proba_valid, y_vol_valid)
        ic = ic if np.isfinite(ic) else 0.0
        p_val = p_val if np.isfinite(p_val) else 1.0
    else:
        ic, p_val = 0.0, 1.0

    # 2. Monotonicity: quintiles sobre proba_valid usando boundaries de calib o test
    if calib_data is not None:
        quintile_bounds = np.percentile(proba_cal, [20, 40, 60, 80])
    else:
        quintile_bounds = np.percentile(proba_valid, [20, 40, 60, 80])
    quintile = np.digitize(proba_valid, quintile_bounds) + 1

    quintile_means = []
    for q in range(1, 6):
        mask_q = quintile == q
        if mask_q.sum() > 0:
            quintile_means.append(np.mean(y_vol_valid[mask_q]))
        else:
            quintile_means.append(np.nan)

    valid_q = ~np.isnan(quintile_means)
    if valid_q.sum() >= 3:
        mono_score, _ = spearmanr(np.arange(1, 6)[valid_q], np.array(quintile_means)[valid_q])
        mono_score = mono_score if np.isfinite(mono_score) else 0.0
    else:
        mono_score = 0.0

    # 3. Accuracy
    acc = np.mean(actual_regime == pred_regime) if len(actual_regime) > 0 else np.nan

    metrics = {
        'ic': ic,
        'ic_pvalue': p_val,
        'monotonicity': mono_score,
        'accuracy': acc,
        'n_samples': len(proba_valid),
        'p20_proba': float(p20_proba),
        'p80_proba': float(p80_proba),
        'p20_vol': float(p20_vol),
        'p80_vol': float(p80_vol)
    }
    return metrics, proba, actual_regime, pred_regime


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("ORION CROSS-ASSET GENERALIZATION TEST")
    print("Model: model_v20_6_1.pkl (trained on ETH)")
    print("Assets: ETH, BTC, SOL")
    print("=" * 80)

    # Load model (LightGBM Booster)
    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[MODEL] Loading model...")
    model = joblib.load(MODEL_PATH)
    print(f"[MODEL] Loaded: {type(model)}")

    # Verify model input feature names? The model expects Column_0..Column_22.
    # We'll rename feature columns later.

    results = []

    for asset in ASSETS:
        print(f"\n{'='*80}")
        print(f"[ASSET] {asset}")
        print(f"{'='*80}")

        # 1. Fetch data
        print("[DATA] Fetching historical 4H candles...")
        try:
            df = fetch_okx_ohlcv(symbol=asset, timeframe=TIMEFRAME, n_candles=8000)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {asset}: {e}")
            continue

        print(f"[DATA] Retrieved {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        # 2. Build features
        print("[FEAT] Building features...")
        df, raw_features = build_features(df)
        print(f"[FEAT] Raw features: {len(raw_features)}")

        # 3. Add target (vol_24h_future)
        df = add_target(df)

        # 4. Drop warmup rows (first 200 bars after feature calculation)
        df = df.iloc[WARMUP_BARS:].reset_index(drop=True)
        print(f"[DATA] After warmup drop: {len(df)} rows")

        # 5. Prepare feature matrix with Column_0..Column_22 names
        # The model expects 23 columns named Column_0...Column_22
        feature_data = df[raw_features].copy()
        feature_data.columns = [f'Column_{i}' for i in range(23)]

        # 6. Clip features to ±5 std (per spec)
        feature_data = feature_data.clip(-5, 5)

        # 7. Target series
        y_vol = df['vol_24h_future']

        # 8. Generate walk-forward splits
        n = len(feature_data)
        splits = generate_walk_forward_splits(n, initial_train_size=INITIAL_TRAIN_SIZE,
                                              test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS)
        print(f"[SPLITS] Generated {len(splits)} folds")

        fold_metrics = []
        fold_probas = []
        all_ic = []
        all_mono = []
        all_acc = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

            X_train = feature_data.iloc[train_idx]
            y_train_vol = y_vol.iloc[train_idx]
            X_test = feature_data.iloc[test_idx]
            y_test_vol = y_vol.iloc[test_idx]

            # Calibration set = training set (to avoid lookahead)
            calib_data = (X_train, y_train_vol)

            metrics, proba, _, _ = evaluate_fold(model, X_test, y_test_vol, calib_data=calib_data)

            fold_metrics.append(metrics)
            fold_probas.append((test_idx, proba))
            all_ic.append(metrics['ic'])
            all_mono.append(metrics['monotonicity'])
            all_acc.append(metrics['accuracy'])

            print(f"    IC = {metrics['ic']:.4f} (p={metrics['ic_pvalue']:.4f}), "
                  f"mono = {metrics['monotonicity']:.4f}, acc = {metrics['accuracy']:.2%}")

        # 9. Compute Sharpe ratios over all test folds combined, handling gaps
        all_ret_strategy = []
        all_ret_bh = []
        for fold, (test_idx, proba) in enumerate(fold_probas):
            close_fold = df['close'].iloc[test_idx].values
            if len(proba) != len(close_fold):
                print(f"[WARN] Fold {fold}: length mismatch proba {len(proba)} vs close {len(close_fold)} — skipping Sharpe")
                continue
            if len(close_fold) < 2:
                continue
            # Simple returns
            ret_simple = np.diff(close_fold) / close_fold[:-1]
            # Position determined at t (proba[i]) used for return from t to t+1
            position = 1.0 - proba[:-1]
            # Align
            ret_strat = position * ret_simple
            all_ret_strategy.append(ret_strat)
            all_ret_bh.append(ret_simple)

        # Concatenate all fold returns
        if all_ret_strategy:
            all_ret_strategy_concat = np.concatenate(all_ret_strategy)
            all_ret_bh_concat = np.concatenate(all_ret_bh)
            # Annualized Sharpe
            if np.std(all_ret_strategy_concat) > 1e-8:
                mean_ret = np.mean(all_ret_strategy_concat)
                std_ret = np.std(all_ret_strategy_concat)
                sharpe_strategy = mean_ret / std_ret * np.sqrt(6 * 365)
            else:
                sharpe_strategy = 0.0
            if np.std(all_ret_bh_concat) > 1e-8:
                mean_ret_bh = np.mean(all_ret_bh_concat)
                std_ret_bh = np.std(all_ret_bh_concat)
                sharpe_bh = mean_ret_bh / std_ret_bh * np.sqrt(6 * 365)
            else:
                sharpe_bh = 0.0
        else:
            sharpe_strategy = np.nan
            sharpe_bh = np.nan

        #汇总 statistics
        mean_ic = np.mean(all_ic)
        std_ic = np.std(all_ic)
        ic_sig_count = sum(1 for ic, p in [(m['ic'], m['ic_pvalue']) for m in fold_metrics] if p < 0.05)
        mean_mono = np.mean(all_mono)
        mean_acc = np.nanmean(all_acc)

        asset_result = {
            'asset': asset,
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'ic_sig_folds': ic_sig_count,
            'mean_monotonicity': mean_mono,
            'mean_accuracy': mean_acc,
            'sharpe_strategy': sharpe_strategy,
            'sharpe_bh': sharpe_bh
        }
        results.append(asset_result)

        print(f"[RESULT] Mean IC = {mean_ic:.4f} ± {std_ic:.4f}, sig folds = {ic_sig_count}/{len(splits)}")
        print(f"[RESULT] Monotonicity = {mean_mono:.4f}, Accuracy = {mean_acc:.2%}")
        print(f"[RESULT] Sharpe (strategy) = {sharpe_strategy:.2f}, Sharpe (B&H) = {sharpe_bh:.2f}")

    # ============================================================
    # OUTPUT COMPARATIVE TABLE
    # ============================================================
    print("\n" + "=" * 80)
    print("CROSS-ASSET COMPARISON")
    print("=" * 80)

    header = f"{'Asset':<10} {'Mean IC':>10} {'IC std':>10} {'IC sig':>8} {'Monotonic':>10} {'Acc':>8} {'Sharpe Strat':>13} {'Sharpe B&H':>11}"
    print(header)
    print("-" * 80)

    for r in results:
        line = f"{r['asset']:<10} {r['mean_ic']:>10.4f} {r['std_ic']:>10.4f} {r['ic_sig_folds']:>8} {r['mean_monotonicity']:>10.4f} {r['mean_accuracy']:>8.2%} {r['sharpe_strategy']:>13.2f} {r['sharpe_bh']:>11.2f}"
        print(line)

    print("=" * 80)

    # ============================================================
    # VEREDICTO
    # ============================================================
    # Verificar si BTC ySOL generalizan
    btc_res = next(r for r in results if r['asset'] == 'BTC-USDT')
    sol_res = next(r for r in results if r['asset'] == 'SOL-USDT')

    # Regla para verificar generalización: Mean IC > 0.02 y Monotonicity > 0.5
    btc_ok = btc_res['mean_ic'] > 0.02 and btc_res['mean_monotonicity'] > 0.5
    sol_ok = sol_res['mean_ic'] > 0.02 and sol_res['mean_monotonicity'] > 0.5

    print("\nVEREDICTO:")
    if btc_ok and sol_ok:
        print("✅ CASO A: BTC Y SOL GENERALIZAN — modelo único viable")
        print("   El modelo entrenado en ETH muestra capacidad de generalización")
        print("   a otras cripto activos.IC > 0.02 y monotonicidad > 0.5 en ambos.")
    else:
        print("❌ CASO B: NO GENERALIZA — entrenar modelos separados")
        if not btc_ok:
            print(f"   BTC: IC={btc_res['mean_ic']:.4f}, mono={btc_res['mean_monotonicity']:.4f} — falla criterio")
        if not sol_ok:
            print(f"   SOL: IC={sol_res['mean_ic']:.4f}, mono={sol_res['mean_monotonicity']:.4f} — falla criterio")

    print("\nNota: ETH se incluye como control (baseline) — no se evalúa generalización sobre sí mismo.")
    print("=" * 80)

    return results

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] User interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
