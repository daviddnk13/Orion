#!/home/ubuntu/orion/venv/bin/python3
# -*- coding: utf-8 -*-
"""
CASH THRESHOLD CALIBRATION
Tests 5 cash thresholds to find optimal risk-adjusted performance
No triple quotes allowed — comments only with #
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta, timezone
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = 'model_v20_6_1.pkl'

# Telegram (hardcoded for topic 972 — Modelos)
TELEGRAM_BOT_TOKEN = '8723893197:AAFfIORXd2Y-qQ8TclOq23afEPt_knr7xrU'
TELEGRAM_CHAT_ID = '-1003505760554'
TOPIC_MODELOS = 972

# Data fetching
ETH_SYMBOL = 'ETH-USDT'
TIMEFRAME = '4H'
N_CANDLES = 8000

# Walk-forward
N_FOLDS = 4
TEST_SIZE = 1250
EMBARGO = 180
INITIAL_TRAIN_SIZE = 2000
SEED = 42
np.random.seed(SEED)

# Risk parameters
TARGET_VOL = 0.15
SMOOTHING_ALPHA = 0.7  # current_weight
FEE_BPS = 5
SLIP_BPS = 5
TOTAL_FRICTION = (FEE_BPS + SLIP_BPS) / 10000.0

# Thresholds to test
# None = baseline without cash cutoff
THRESHOLDS = [None, 0.50, 0.55, 0.60, 0.65, 0.70]

# ============================================================
# DATA FETCHING (OKX)
# ============================================================
def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4h', n_candles=None):
    base_url = "https://www.okx.com"
    endpoint = "/api/v5/market/history-candles"
    params = {
        'instId': symbol,
        'bar': timeframe,
        'limit': '100'
    }

    all_candles = []
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
                    remaining = 0
                    success = True
                    break
                all_candles.extend(candles)
                remaining -= len(candles)
                if candles:
                    params['after'] = candles[-1][0]
                time.sleep(0.1)
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
# FEATURE ENGINEERING (EXACT COPY FROM paper_trading_v20_8.py)
# ============================================================
def build_features(df):
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret
    eps = 1e-8

    # ret_4h (6-bar sum)
    df['ret_4h'] = log_ret.rolling(6).sum().shift(1)

    # rsi_norm (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + eps)
    rsi = 100 - 100 / (1 + rs)
    df['rsi_norm'] = (rsi / 100).fillna(0.5).shift(1)

    # bb_position
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_pos = (df['close'] - lower) / (upper - lower + eps)
    df['bb_position'] = bb_pos.fillna(0.5).shift(1)

    # macd_norm
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9).mean()
    macd_norm = (macd - macd_sig) / (df['close'] * 0.01 + eps)
    df['macd_norm'] = macd_norm.fillna(0).shift(1)

    # ret_4h_lag1, ret_4h_lag2
    df['ret_4h_lag1'] = df['ret_4h'].shift(1)
    df['ret_4h_lag2'] = df['ret_4h'].shift(2)

    # atr_norm (14-period)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_norm'] = (atr / df['close']).fillna(0).shift(1)

    # vol_zscore
    rv_6 = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_mean = rv_6.rolling(180).mean().shift(1)
    rv_std = rv_6.rolling(180).std().shift(1)
    df['vol_zscore'] = ((rv_6 - rv_mean) / (rv_std + eps)).fillna(0).shift(1)

    # vol_regime (categorical)
    vol_rank = rv_6.rolling(180).rank(pct=True).shift(1)
    df['vol_regime'] = pd.cut(vol_rank, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2]).astype(float).fillna(1).shift(1)

    # ret_8h (12-bar sum)
    df['ret_8h'] = log_ret.rolling(12).sum().shift(1)

    # ret_24h (24-bar sum)
    df['ret_24h'] = log_ret.rolling(24).sum().shift(1)

    # ema_slope
    ema_20 = df['close'].ewm(span=20).mean()
    slope = (ema_20 - ema_20.shift(1)) / (ema_20.shift(1) + eps)
    df['ema_slope'] = slope.fillna(0).shift(1)

    # trend_strength (ADX-like)
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

    # drawdown_market
    roll_max = df['close'].rolling(100).max()
    dd = (df['close'] - roll_max) / (roll_max + eps)
    df['drawdown_market'] = dd.fillna(0).shift(1)

    # tf_coherence
    mom_short = df['close'] / df['close'].shift(6) - 1.0
    mom_long = df['close'] / df['close'].shift(24) - 1.0
    df['tf_coherence'] = (np.sign(mom_short) * np.sign(mom_long) *
                          np.minimum(np.abs(mom_short), np.abs(mom_long))).fillna(0)

    # dist_ema200
    ema_200 = df['close'].ewm(span=200).mean()
    df['dist_ema200'] = ((df['close'] - ema_200) / (atr + eps)).fillna(0).shift(1)

    # parkinson_vol (42-bar)
    hl_ratio = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt(
        (hl_ratio ** 2).rolling(42).mean().shift(1) / (4 * np.log(2))
    ).fillna(0)

    # vol_compression
    rv_1d = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_7d = log_ret.rolling(42).std().shift(1) * np.sqrt(6)
    df['realized_vol_1d'] = rv_1d
    df['realized_vol_7d'] = rv_7d
    df['vol_compression'] = rv_1d / (rv_7d + eps)

    # garman_klass_vol
    gk = 0.5 * (np.log(df['high'] / df['low'])) ** 2 - (2*np.log(2)-1) * (np.log(df['close']/df['open']))**2
    df['garman_klass_vol'] = np.sqrt(gk.rolling(6).mean().shift(1).clip(lower=0))

    # vol_regime_rank
    vol_rank_pct = rv_7d.rolling(180).rank(pct=True).shift(1)
    df['vol_regime_rank'] = vol_rank_pct.fillna(0.5)

    # trend_efficiency
    close_diff_24 = (df['close'] - df['close'].shift(24)).abs()
    sum_abs_diff = df['close'].diff().abs().rolling(24).sum()
    df['trend_efficiency'] = (close_diff_24 / (sum_abs_diff + eps)).fillna(0).shift(1)

    raw_features = [
        'ret_4h', 'rsi_norm', 'bb_position', 'macd_norm',
        'ret_4h_lag1', 'ret_4h_lag2',
        'atr_norm', 'vol_zscore', 'vol_regime', 'ret_8h', 'ret_24h',
        'ema_slope', 'drawdown_market', 'tf_coherence', 'dist_ema200', 'trend_strength',
        'parkinson_vol', 'vol_compression', 'garman_klass_vol',
        'vol_regime_rank', 'realized_vol_7d', 'trend_efficiency', 'realized_vol_1d'
    ]

    for f in raw_features:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].fillna(0)

    return df, raw_features

# ============================================================
# WALK-FORWARD SPLITS
# ============================================================
def generate_walk_forward_splits(n_samples, initial_train_size=INITIAL_TRAIN_SIZE,
                                 test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS):
    total_needed = initial_train_size + n_folds * test_size + n_folds * embargo
    if n_samples < total_needed:
        print(f"[WARN] Only {n_samples} samples, need ~{total_needed}. Adjusting.")
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

        train_end = test_end

    return splits

# ============================================================
# TELEGRAM NOTIFICATION
# ============================================================
def tg_send(text, topic_id=None, parse_mode='HTML'):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': parse_mode}
        if topic_id is not None:
            payload['message_thread_id'] = topic_id
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"[TELEGRAM] HTTP {r.status_code}: {r.text}")
        return r.status_code == 200
    except Exception as e:
        print(f"[TELEGRAM] Failed: {e}")
        return False

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("CASH THRESHOLD CALIBRATION")
    print(f"Model: {MODEL_PATH}")
    print(f"Asset: {ETH_SYMBOL} ({TIMEFRAME})")
    print(f"Folds: {N_FOLDS}, test={TEST_SIZE}, embargo={EMBARGO}, initial_train={INITIAL_TRAIN_SIZE}")
    print(f"Thresholds: {THRESHOLDS}")
    print("=" * 80)

    # 1. Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[MODEL] Loading LightGBM...")
    model = joblib.load(MODEL_PATH)
    print(f"[MODEL] Loaded: {type(model)}")

    # 2. Fetch data
    print(f"[DATA] Fetching {N_CANDLES} candles for {ETH_SYMBOL}...")
    try:
        df = fetch_okx_ohlcv(symbol=ETH_SYMBOL, timeframe=TIMEFRAME, n_candles=N_CANDLES)
    except Exception as e:
        print(f"[FATAL] Failed to fetch data: {e}")
        sys.exit(1)

    print(f"[DATA] Retrieved {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # 3. Build features
    print("[FEAT] Building 23 features...")
    df, raw_features = build_features(df)
    print(f"[FEAT] Features: {len(raw_features)}")

    # 4. Prepare feature matrix with Column_0..Column_22 names (matching model training)
    feature_data = df[raw_features].copy()
    feature_data.columns = [f'Column_{i}' for i in range(23)]
    feature_data = feature_data.clip(-5, 5)

    # 5. Generate splits
    n = len(feature_data)
    splits = generate_walk_forward_splits(n, initial_train_size=INITIAL_TRAIN_SIZE,
                                          test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS)
    print(f"[SPLITS] Generated {len(splits)} folds")

    # 6. Run calibration for each threshold
    threshold_results = []

    for threshold in THRESHOLDS:
        print(f"\n{'='*80}")
        print(f"[THRESHOLD] Testing threshold = {threshold if threshold is None else f'{threshold:.2f}'}")
        print("=" * 80)

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

            # Get test data
            X_test = feature_data.iloc[test_idx].values.astype(np.float64)
            close_prices = df['close'].iloc[test_idx].values
            log_returns = df['log_ret'].iloc[test_idx]

            # Predict proba
            proba_raw = model.predict(X_test)  # shape (n_test,)

            # Pipeline
            # a) Apply cash threshold logic
            # If threshold is None: baseline (no cash cutoff)
            # If threshold is set: if proba > threshold -> cash (0), else baseline (1-proba)
            if threshold is None:
                raw_position = 1.0 - proba_raw
            else:
                raw_position = np.where(proba_raw > threshold, 0.0, 1.0 - proba_raw)

            # b) vol_ratio = target_vol / (realized_vol + 1e-8), clip(0.5, 2.0)
            realized_vol = log_returns.rolling(window=42).std().values * np.sqrt(6 * 365)
            realized_vol = np.nan_to_num(realized_vol, nan=TARGET_VOL)
            vol_ratio = TARGET_VOL / (realized_vol + 1e-8)
            vol_ratio = np.clip(vol_ratio, 0.5, 2.0)

            # c) position = raw_position * vol_ratio
            position = raw_position * vol_ratio

            # d) position = clip(position, 0, 0.5)
            position = np.clip(position, 0, 0.5)

            # e) position = 0.7 * position_prev + 0.3 * position (smoothing)
            position_smooth = np.zeros_like(position)
            prev_pos = 0.0
            for i in range(len(position)):
                position_smooth[i] = SMOOTHING_ALPHA * position[i] + (1 - SMOOTHING_ALPHA) * prev_pos
                prev_pos = position_smooth[i]
            position = position_smooth

            # f) DD scalar: dd_scalar = clip(1 + dd/0.50, 0.1, 1.0)
            # Calculate running DD
            running_balance = 1.0
            peak_balance = 1.0
            dd_history = []
            for i in range(len(position)-1):
                # Compute return for this step
                if i < len(close_prices) - 1:
                    price_change = (close_prices[i+1] - close_prices[i]) / close_prices[i]
                    pos_prev = position[i]
                    pnl = pos_prev * price_change
                    running_balance *= (1 + pnl)
                    peak_balance = max(peak_balance, running_balance)
                    dd = (running_balance - peak_balance) / peak_balance if peak_balance > 0 else 0.0
                    dd_history.append(dd)

            current_dd = dd_history[-1] if dd_history else 0.0
            dd_scalar = np.clip(1.0 + current_dd / 0.50, 0.1, 1.0)

            # Apply DD scalar to all positions
            position = position * dd_scalar

            # g) Calculate PnL with fees
            returns = []
            turnovers = []
            running_balance = 1.0
            for i in range(len(position)-1):
                pos_prev = position[i]
                pos_curr = position[i+1]
                price_change = (close_prices[i+1] - close_prices[i]) / close_prices[i]

                # Fees only if there is turnover (entry or exit)
                position_change = abs(pos_curr - pos_prev)
                fees = TOTAL_FRICTION * position_change if position_change > 1e-8 else 0.0

                pnl = pos_prev * price_change - fees
                returns.append(pnl)
                turnovers.append(position_change)

            returns = np.array(returns)
            turnovers = np.array(turnovers)
            positions_for_metrics = position[:-1] if len(position) > 1 else np.array([])
            n_bars = len(positions_for_metrics)

            # Initialize all metrics
            sharpe = 0.0
            max_dd = 0.0
            exposure = 0.0
            mean_turnover = 0.0
            time_in_cash = 0.0
            exposure_real = 0.0
            sharpe_in_market = 0.0

            # Compute time_in_cash and exposure_real
            if n_bars > 0:
                # Exposure: % bars with position > 0
                exposure = np.mean(positions_for_metrics > 0) * 100.0
                # time_in_cash: % bars with position == 0
                time_in_cash = np.mean(positions_for_metrics == 0) * 100.0
                # exposure_real: mean position size when > 0
                pos_positive = positions_for_metrics[positions_for_metrics > 1e-8]
                exposure_real = np.mean(pos_positive) if len(pos_positive) > 0 else 0.0

            # Compute overall Sharpe, Max DD, Turnover, and Sharpe_in_market
            if len(returns) > 0 and n_bars == len(returns):
                # Overall Sharpe
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = mean_ret / std_ret * np.sqrt(6 * 365) if std_ret > 1e-8 else 0.0

                # Max DD
                equity = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(equity)
                dd_curve = (equity - running_max) / running_max
                max_dd = np.min(dd_curve) if len(dd_curve) > 0 else 0.0

                # Turnover
                mean_turnover = np.mean(turnovers) if len(turnovers) > 0 else 0.0

                # Sharpe in market (only bars with position > 0)
                market_mask = positions_for_metrics > 1e-8
                returns_in_market = returns[market_mask] if np.any(market_mask) else np.array([])
                if len(returns_in_market) > 0:
                    mean_ret_mkt = np.mean(returns_in_market)
                    std_ret_mkt = np.std(returns_in_market)
                    sharpe_in_market = mean_ret_mkt / std_ret_mkt * np.sqrt(6 * 365) if std_ret_mkt > 1e-8 else 0.0
                else:
                    sharpe_in_market = 0.0

            fold_metric = {
                'fold': fold,
                'sharpe': float(sharpe),
                'max_dd': float(max_dd),
                'exposure': float(exposure),
                'turnover': float(mean_turnover),
                'time_in_cash': float(time_in_cash),
                'exposure_real': float(exposure_real),
                'sharpe_in_market': float(sharpe_in_market),
                'n_test': len(test_idx)
            }
            fold_metrics.append(fold_metric)

            print(f"    Sharpe={sharpe:.2f}, Sharpe_mkt={sharpe_in_market:.2f}, DD={max_dd*100:.1f}%, exposure={exposure:.1f}%, cash={time_in_cash:.1f}%, turnover={mean_turnover:.4f}")

        # Aggregate across folds
        sharpe_vals = [fm['sharpe'] for fm in fold_metrics]
        dd_vals = [fm['max_dd'] for fm in fold_metrics]
        exposure_vals = [fm['exposure'] for fm in fold_metrics]
        turnover_vals = [fm['turnover'] for fm in fold_metrics]
        sharpe_mkt_vals = [fm['sharpe_in_market'] for fm in fold_metrics]
        time_in_cash_vals = [fm['time_in_cash'] for fm in fold_metrics]
        exposure_real_vals = [fm['exposure_real'] for fm in fold_metrics]

        mean_sharpe = np.mean(sharpe_vals)
        mean_dd = np.mean(dd_vals)
        mean_exposure = np.mean(exposure_vals)
        mean_turnover = np.mean(turnover_vals)
        mean_sharpe_in_market = np.mean(sharpe_mkt_vals)
        mean_time_in_cash = np.mean(time_in_cash_vals)
        mean_exposure_real = np.mean(exposure_real_vals) if exposure_real_vals else 0.0

        # Count folds with Sharpe > 0
        folds_sharpe_pos = sum(1 for s in sharpe_vals if s > 0)

        result = {
            'threshold': threshold,
            'folds': fold_metrics,
            'aggregate': {
                'mean_sharpe': float(mean_sharpe),
                'mean_max_dd': float(mean_dd),
                'mean_exposure': float(mean_exposure),
                'mean_turnover': float(mean_turnover),
                'mean_sharpe_in_market': float(mean_sharpe_in_market),
                'mean_time_in_cash': float(mean_time_in_cash),
                'mean_exposure_real': float(mean_exposure_real),
                'folds_sharpe_positive': int(folds_sharpe_pos)
            }
        }
        threshold_results.append(result)

        print(f"[AGGREGATE] Sharpe={mean_sharpe:.2f}, DD={mean_dd*100:.1f}%, "
              f"exposure={mean_exposure:.1f}%, turnover={mean_turnover:.4f}, folds+={folds_sharpe_pos}/{len(fold_metrics)}")

    # 7. Print comparative table
    print("\n" + "=" * 120)
    print("COMPARATIVE RESULTS")
    print("=" * 120)
    header = f"{'Threshold':>10} {'Sharpe':>10} {'Sharpe_mkt':>11} {'Max DD':>10} {'Turnover':>10} {'Exposure':>10} {'Cash%':>8} {'Folds+':>10}"
    print(header)
    print("-" * 120)

    for res in threshold_results:
        agg = res['aggregate']
        threshold_display = "None" if res['threshold'] is None else f"{res['threshold']:.2f}"
        line = (f"{threshold_display:>10} "
                f"{agg['mean_sharpe']:>10.2f} "
                f"{agg['mean_sharpe_in_market']:>11.2f} "
                f"{agg['mean_max_dd']*100:>9.1f}% "
                f"{agg['mean_turnover']:>10.4f} "
                f"{agg['mean_exposure']:>9.1f}% "
                f"{agg['mean_time_in_cash']:>7.1f}% "
                f"{agg['folds_sharpe_positive']:>10}/{len(res['folds'])}")
        print(line)

    print("-" * 120)

    # Validation logic (vs baseline None)
    baseline_result = None
    for r in threshold_results:
        if r['threshold'] is None:
            baseline_result = r
            break

    if baseline_result:
        print("\n" + "=" * 120)
        print("VALIDATION (vs baseline None)")
        print("=" * 120)
        header_val = f"{'Threshold':>10} {'Valid?':>20} {'Sharpe Δ':>12} {'DD Δ (%)':>12} {'Expo Δ (%)':>12} {'Reason':>30}"
        print(header_val)
        print("-" * 120)

        b_agg = baseline_result['aggregate']
        b_sharpe = b_agg['mean_sharpe']
        b_dd = b_agg['mean_max_dd']
        b_exposure = b_agg['mean_exposure']

        for r in threshold_results:
            t = r['threshold']
            if t is None:
                continue
            a = r['aggregate']
            sharpe = a['mean_sharpe']
            dd = a['mean_max_dd']
            exposure = a['mean_exposure']

            sharpe_ok = sharpe >= b_sharpe
            dd_ok = dd <= b_dd
            exposure_ok = exposure < 0.8 * b_exposure

            valid = sharpe_ok and dd_ok and exposure_ok

            reasons = []
            if not sharpe_ok:
                reasons.append("Sharpe↓")
            if not dd_ok:
                reasons.append("DD↑")
            if not exposure_ok:
                reasons.append("Expo≥80%")
            reason = ", ".join(reasons) if reasons else "OK"

            valid_str = "✅ VÁLIDO" if valid else "❌ NO VÁLIDO"
            sharpe_delta = sharpe - b_sharpe
            dd_delta = dd - b_dd
            exp_delta = exposure - b_exposure

            print(f"{t:>10.2f} {valid_str:>20} {sharpe_delta:>12.2f} {dd_delta*100:>11.1f}% {exp_delta:>11.1f}% {reason:>30}")

        print("-" * 120)
    else:
        print("\n[VALIDATION] No baseline (None) found. Skipping validation.")

    # 8. Identify best threshold
    # Criteria: Sharpe > 0
    valid_results = [r for r in threshold_results if r['aggregate']['mean_sharpe'] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x['aggregate']['mean_sharpe'])
        print(f"\n🏆 BEST THRESHOLD: {best['threshold']:.2f} "
              f"(Sharpe={best['aggregate']['mean_sharpe']:.2f}, "
              f"DD={best['aggregate']['mean_max_dd']*100:.1f}%, "
              f"Exposure={best['aggregate']['mean_exposure']:.1f}%)")
    else:
        print("\n⚠️ NO VALID THRESHOLD: All have Sharpe <= 0")

    print("=" * 120)

    # 9. Save results to JSON
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': MODEL_PATH,
        'asset': ETH_SYMBOL,
        'timeframe': TIMEFRAME,
        'n_candles': len(df),
        'n_folds': len(splits),
        'test_size': TEST_SIZE,
        'embargo': EMBARGO,
        'thresholds_tested': THRESHOLDS,
        'results': threshold_results
    }

    json_path = 'cash_threshold_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[OUTPUT] Saved results to {json_path}")

    # 10. Generate plots
    # a) Sharpe vs Threshold
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(threshold_results))
    threshold_labels = [str(r['threshold']) if r['threshold'] is not None else "None" for r in threshold_results]
    sharpes = [r['aggregate']['mean_sharpe'] for r in threshold_results]
    plt.plot(x_pos, sharpes, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Cash Threshold')
    plt.ylabel('Mean Sharpe')
    plt.title('Sharpe Ratio vs Cash Threshold')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, threshold_labels)
    plot1_path = 'cash_threshold_sharpe.png'
    plt.savefig(plot1_path, dpi=100, bbox_inches='tight')
    print(f"[PLOT] Saved Sharpe plot to {plot1_path}")
    plt.close()

    # b) Exposure vs Threshold
    plt.figure(figsize=(10, 6))
    exposures = [r['aggregate']['mean_exposure'] for r in threshold_results]
    plt.plot(x_pos, exposures, marker='s', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Cash Threshold')
    plt.ylabel('Mean Exposure (%)')
    plt.title('Exposure vs Cash Threshold')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, threshold_labels)
    plot2_path = 'cash_threshold_exposure.png'
    plt.savefig(plot2_path, dpi=100, bbox_inches='tight')
    print(f"[PLOT] Saved Exposure plot to {plot2_path}")
    plt.close()

    # 11. Send summary to Telegram topic 972
    tg_msg = [
        "💰 CASH THRESHOLD CALIBRATION",
        f"UTC: {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M')}",
        f"Model: {MODEL_PATH}",
        f"Data: {len(df)} {ETH_SYMBOL} {TIMEFRAME} candles",
        f"Folds: {len(splits)} (test_size={TEST_SIZE}, embargo={EMBARGO})",
        "",
        "<b>Comparative Table:</b>"
    ]

    for res in threshold_results:
        agg = res['aggregate']
        line = f"<code>Thresh={res['threshold']:.2f}  Sharpe={agg['mean_sharpe']:>5.2f}  Sharpe_mkt={agg['mean_sharpe_in_market']:>5.2f}  DD={agg['mean_max_dd']*100:>5.1f}%  Expo={agg['mean_exposure']:>5.1f}%  Cash={agg['mean_time_in_cash']:>4.1f}%  Turn={agg['mean_turnover']:.4f}  Folds+={agg['folds_sharpe_positive']}/{len(res['folds'])}</code>"
        tg_msg.append(line)

    # Validation section in Telegram
    baseline_result_tg = None
    for r in threshold_results:
        if r['threshold'] is None:
            baseline_result_tg = r
            break

    if baseline_result_tg:
        tg_msg.append("")
        tg_msg.append("<b>VALIDATION (vs baseline None)</b>")
        b_agg = baseline_result_tg['aggregate']
        b_sharpe = b_agg['mean_sharpe']
        b_dd = b_agg['mean_max_dd']
        b_exposure = b_agg['mean_exposure']

        for r in threshold_results:
            t = r['threshold']
            if t is None:
                continue
            a = r['aggregate']
            sharpe = a['mean_sharpe']
            dd = a['mean_max_dd']
            exposure = a['mean_exposure']

            sharpe_ok = sharpe >= b_sharpe
            dd_ok = dd <= b_dd
            exposure_ok = exposure < 0.8 * b_exposure
            valid = sharpe_ok and dd_ok and exposure_ok
            valid_str = "✅ VÁLIDO" if valid else "❌ NO VÁLIDO"
            sharpe_delta = sharpe - b_sharpe
            dd_delta = dd - b_dd
            exp_delta = exposure - b_exposure

            reasons = []
            if not sharpe_ok:
                reasons.append("Sharpe↓")
            if not dd_ok:
                reasons.append("DD↑")
            if not exposure_ok:
                reasons.append("Expo≥80%")
            reason = ", ".join(reasons) if reasons else "OK"

            line_val = f"<code>Thresh={t:.2f}: {valid_str}  SharpeΔ={sharpe_delta:+.2f}  DDΔ={dd_delta*100:+.1f}%  ExpoΔ={exp_delta:+.1f}%  {reason}</code>"
            tg_msg.append(line_val)

    if valid_results:
        best = max(valid_results, key=lambda x: x['aggregate']['mean_sharpe'])
        tg_msg.append("")
        tg_msg.append(f"🏆 <b>BEST: {best['threshold']:.2f}</b> (Sharpe={best['aggregate']['mean_sharpe']:.2f})")
    else:
        tg_msg.append("")
        tg_msg.append("⚠️ <b>NO VALID THRESHOLD</b> — all Sharpe <= 0")

    tg_text = "\n".join(tg_msg)

    if tg_send(tg_text, topic_id=TOPIC_MODELOS):
        print("[TELEGRAM] Summary sent to topic 972 (Modelos)")
    else:
        print("[TELEGRAM] Failed to send message")

    print("\n✅ CALIBRATION COMPLETE")
    print("=" * 80)

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
