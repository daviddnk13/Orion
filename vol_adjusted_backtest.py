#!/home/ubuntu/orion/venv/bin/python3
# -*- coding: utf-8 -*-
"""
ORION V20.8 — Volatility-Adjusted Sizing Backtest
Tests vol-adjusted position sizing on BTC and SOL vs naive sizing and buy&hold.
Uses frozen ETH-trained model (model_v20_6_1.pkl) without retraining.

Author: Claude Code (Anthropic)
Date: 2026-04-26
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta, timezone
from scipy.stats import skew as scipy_skew
import joblib
import lightgbm as lgb

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = '/home/ubuntu/orion/model_v20_6_1.pkl'

ASSETS = ['ETH-USDT', 'BTC-USDT', 'SOL-USDT']
TIMEFRAME = '4H'
WARMUP_BARS = 200

# Walk-forward parameters
N_FOLDS = 4
TEST_SIZE = 1250
EMBARGO = 180
INITIAL_TRAIN_SIZE = 2000
SEED = 42

# Fee and slippage
FEE_BPS = 5
SLIP_BPS = 5
TOTAL_COST = (FEE_BPS + SLIP_BPS) / 10000.0

# Asset-specific vol targeting parameters
ASSET_CONFIG = {
    'ETH-USDT': {'target_vol': 0.15, 'vol_window': 168},
    'BTC-USDT': {'target_vol': 0.12, 'vol_window': 252},
    'SOL-USDT': {'target_vol': 0.20, 'vol_window': 168}
}

# ============================================================
# DATA FETCHING (OKX) — máximo histórico con paginación
# ============================================================
def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4h', n_candles=None):
    """
    Fetch OHLCV from OKX public API.
    If n_candles is None, fetch all available history by pagination.
    Retries up to 3 times on failure with 2s sleep.
    """
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
                r = requests.get(base_url + endpoint, params=params, timeout=30)
                data = r.json()
                if data.get('code') != '0':
                    print(f"[OKX] API error for {symbol}: {data.get('msg', 'unknown')}")
                    attempt += 1
                    time.sleep(2)
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
                time.sleep(2)
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
# WALK-FORWARD SPLITS
# ============================================================
def generate_walk_forward_splits(n_samples, initial_train_size=INITIAL_TRAIN_SIZE,
                                 test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS):
    """Generate expanding walk-forward splits with embargo."""
    total_needed = initial_train_size + n_folds * test_size + n_folds * embargo
    if n_samples < total_needed:
        print(f"[WARN] Only {n_samples} samples, need ~{total_needed}. Reducing parameters.")
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
# METRICS CALCULATION
# ============================================================
def calculate_metrics(returns, positions, freq_multiplier=np.sqrt(6*365)):
    """
    Calculate trading metrics.
    returns: array of net returns per bar
    positions: array of position sizes per bar (used for turnover)
    """
    if len(returns) == 0:
        return {}

    # Sharpe
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = mean_ret / std_ret * freq_multiplier if std_ret > 1e-8 else 0.0

    # Max Drawdown
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - running_max) / running_max
    max_dd = dd.min()

    # Total Return
    total_ret = equity_curve[-1] - 1.0

    # Turnover (mean absolute position change)
    position_diff = np.diff(positions)
    turnover = np.mean(np.abs(position_diff)) if len(position_diff) > 0 else 0.0

    # Tail Risk (5th percentile)
    tail_5 = np.percentile(returns, 5)

    # Skew
    try:
        skew = scipy_skew(returns)
    except:
        skew = 0.0

    # Mean Position
    mean_pos = np.mean(positions)
    # Sharpe using non-overlapping returns (daily-ish)
    if len(returns) >= 6:
        daily_returns = returns[::6]  # every 6th bar = daily
        if len(daily_returns) > 1:
            sharpe_daily = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
        else:
            sharpe_daily = np.nan
    else:
        sharpe_daily = np.nan

    return {
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_return': total_ret,
        'turnover': turnover,
        'tail_5': tail_5,
        'skew': skew,
        'mean_pos': mean_pos,
        'sharpe_daily': sharpe_daily
}

# ============================================================
# STRATEGIES
# ============================================================
def strategy_naive(proba):
    """Strategy A: Naive 1 - proba."""
    position = 1.0 - proba
    position = np.clip(position, 0, 0.5)
    return position

def strategy_vol_adjusted(proba, realized_vol, target_vol, vol_window, prev_position=None):
    """
    Strategy B: Vol-adjusted with smoothing.

    realized_vol: array of rolling realized vol (aligned with proba)
    target_vol: target annualized volatility
    vol_window: lookback window for realized vol (in bars)
    prev_position: previous position for smoothing (array, len = len(proba)+1 with prev_position[0] = 0)
    """
    vol_ratio = target_vol / (realized_vol + 1e-8)
    vol_ratio = np.clip(vol_ratio, 0.5, 2.0)

    raw_position = (1.0 - proba) * vol_ratio
    raw_position = np.clip(raw_position, 0, 0.5)

    if prev_position is not None:
        # Exponential smoothing: 0.7 * current + 0.3 * previous
        position = 0.7 * raw_position + 0.3 * prev_position[1:]  # align
    else:
        position = raw_position

    return position

def strategy_bh(n_bars):
    """Strategy C: Buy & Hold."""
    return np.ones(n_bars)

# ============================================================
# WALK-FORWARD EVALUATION PER ASSET
# ============================================================
def evaluate_asset(asset_symbol, model):
    """
    Run walk-forward backtest for a single asset.
    Returns per-fold results and aggregated results for all 3 strategies.
    """
    print(f"\n{'='*80}")
    print(f"[ASSET] {asset_symbol}")
    print(f"{'='*80}")

    config = ASSET_CONFIG[asset_symbol]
    target_vol = config['target_vol']
    vol_window = config['vol_window']

    # 1. Fetch data
    print("[DATA] Fetching historical 4H candles...")
    try:
        df = fetch_okx_ohlcv(symbol=asset_symbol, timeframe=TIMEFRAME, n_candles=None)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {asset_symbol}: {e}")
        return None

    print(f"[DATA] Retrieved {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # 2. Build features
    print("[FEAT] Building features...")
    df, raw_features = build_features(df)
    print(f"[FEAT] Raw features: {len(raw_features)}")

    # 3. Drop warmup
    df = df.iloc[WARMUP_BARS:].reset_index(drop=True)
    print(f"[DATA] After warmup drop: {len(df)} rows")

    # 4. Prepare feature matrix
    feature_data = df[raw_features].copy()
    feature_data.columns = [f'Column_{i}' for i in range(23)]
    feature_data = feature_data.clip(-5, 5)

    # 5. Compute realized vol for vol-adjust (using log_ret)
    log_ret = df['log_ret'].values
    realized_vol = pd.Series(log_ret).rolling(vol_window).std().shift(1).values * np.sqrt(6*365)
    realized_vol = np.nan_to_num(realized_vol, nan=target_vol)

    # 6. Close prices and simple returns
    close = df['close'].values
    ret_simple = np.diff(close) / close[:-1]

    # 7. Generate walk-forward splits
    n = len(feature_data)
    splits = generate_walk_forward_splits(n, initial_train_size=INITIAL_TRAIN_SIZE,
                                          test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS)
    print(f"[SPLITS] Generated {len(splits)} folds")

    # Store results per fold
    fold_results = []
    lookahead_ok = True
    lookahead_reason = ""

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        # Model prediction on test set
        X_test = feature_data.iloc[test_idx].values.astype(np.float64)
        proba = model.predict(X_test)

        # Align returns: position at t used for return t to t+1
        # So we need close prices at test_idx, returns from test_idx[:-1]
        close_test = close[test_idx]
        if len(close_test) < 2:
            print(f"    [WARN] Fold {fold}: insufficient bars (<2)")
            continue

        ret_fold = (close_test[1:] - close_test[:-1]) / close_test[:-1]
        proba_fold = proba[:-1]  # position at t uses proba[t] for return[t->t+1]
        rv_fold = realized_vol[test_idx][:-1]

        if len(ret_fold) != len(proba_fold):
            lookahead_ok = False
            lookahead_reason = f"proba length {len(proba_fold)} != close-1 length {len(close_fold)-1}"
            print(f"    [WARN] Fold {fold}: length mismatch ret {len(ret_fold)} vs proba {len(proba_fold)}")
            continue

        n_test_bars = len(ret_fold)

        # ========== STRATEGY A: NAIVE ==========
        pos_a = strategy_naive(proba_fold)
        gross_ret_a = pos_a * ret_fold
        # Transaction cost: apply on position change from previous bar (prev=0 for first bar)
        pos_prev_a = np.zeros_like(pos_a)
        if len(pos_a) > 1:
            pos_prev_a[1:] = pos_a[:-1]
        cost_a = TOTAL_COST * np.abs(pos_a - pos_prev_a)
        net_ret_a = gross_ret_a - cost_a
        metrics_a = calculate_metrics(net_ret_a, pos_a)

        # ========== STRATEGY B: VOL-ADJUSTED ==========
        # Need prev_position for smoothing: start with pos=0
        pos_b_raw = strategy_vol_adjusted(proba_fold, rv_fold, target_vol, vol_window, prev_position=None)
        # Apply smoothing manually with prev = 0 initially
        pos_b = np.zeros_like(pos_b_raw)
        prev = 0.0
        for i in range(len(pos_b_raw)):
            pos_b[i] = 0.7 * pos_b_raw[i] + 0.3 * prev
            prev = pos_b[i]
        gross_ret_b = pos_b * ret_fold
        # Transaction cost: apply on position change from previous bar (prev=0 for first bar)
        pos_prev_b = np.zeros_like(pos_b)
        if len(pos_b) > 1:
            pos_prev_b[1:] = pos_b[:-1]
        cost_b = TOTAL_COST * np.abs(pos_b - pos_prev_b)
        net_ret_b = gross_ret_b - cost_b
        metrics_b = calculate_metrics(net_ret_b, pos_b)

        # ========== STRATEGY C: BUY & HOLD ==========
        pos_c = strategy_bh(len(ret_fold))
        gross_ret_c = pos_c * ret_fold
        # Transaction cost: apply on position change from previous bar (prev=0 for first bar)
        pos_prev_c = np.zeros_like(pos_c)
        if len(pos_c) > 1:
            pos_prev_c[1:] = pos_c[:-1]
        cost_c = TOTAL_COST * np.abs(pos_c - pos_prev_c)
        net_ret_c = gross_ret_c - cost_c
        metrics_c = calculate_metrics(net_ret_c, pos_c)

        # Store fold results
        fold_data = {
            'fold': fold,
            'naive': metrics_a,
            'vol_adj': metrics_b,
            'bh': metrics_c
        }
        fold_results.append(fold_data)

        # Print fold summary
        print(f"    Naive:     Sharpe={metrics_a['sharpe']:.2f}, DD={metrics_a['max_dd']:.1%}")
        print(f"    Vol-Adj:   Sharpe={metrics_b['sharpe']:.2f}, DD={metrics_b['max_dd']:.1%}")
        print(f"    B&H:       Sharpe={metrics_c['sharpe']:.2f}, DD={metrics_c['max_dd']:.1%}")

    # Lookahead check summary
    if lookahead_ok:
        print("LOOKAHEAD CHECK: OK")
    else:
        print(f"LOOKAHEAD CHECK: FAIL — {lookahead_reason}")
    return {
        'asset': asset_symbol,
        'target_vol': target_vol,
        'vol_window': vol_window,
        'folds': fold_results
    }

# ============================================================
# OUTPUT TABLES
# ============================================================
def print_table1(fold_results_dict):
    """TABLA 1: RESULTADOS POR FOLD."""
    print("\n" + "=" * 80)
    print("TABLA 1: RESULTADOS POR FOLD")
    print("=" * 80)
    header = f"{'Fold':<4} {'Naive (A)':<20} {'Vol-Adjusted (B)':<22} {'Buy&Hold (C)':<16}"
    print(header)
    print("-" * 80)

    for asset_data in fold_results_dict:
        asset = asset_data['asset']
        print(f"\n{asset}:")
        for fold_data in asset_data['folds']:
            fold = fold_data['fold']
            na = fold_data['naive']
            vb = fold_data['vol_adj']
            bh = fold_data['bh']
            line = f"{fold:<4} Sh={na['sharpe']:5.2f} DD={na['max_dd']*100:4.1f}%   Sh={vb['sharpe']:5.2f} DD={vb['max_dd']*100:4.1f}%   Sh={bh['sharpe']:5.2f} DD={bh['max_dd']*100:4.1f}%"
            print(line)

def print_table2(aggregated_results):
    """TABLA 2: RESUMEN AGREGADO."""
    print("\n" + "=" * 100)
    print("TABLA 2: RESUMEN AGREGADO")
    print("=" * 100)
    header = f"{'Asset':<10} {'Strategy':<12} {'Mean Sharpe':>13} {'Sharpe Daily':>14} {'Max DD':>10} {'Turnover':>10} {'Tail 5%':>10} {'Skew':>10} {'MeanPos':>10}"
    print(header)
    print("-" * 100)

    for asset_data in aggregated_results:
        asset = asset_data['asset']
        for strat in ['naive', 'vol_adj', 'bh']:
            strat_name = 'Naive' if strat == 'naive' else ('Vol-Adj' if strat == 'vol_adj' else 'B&H')
            m = asset_data['aggregated'][strat]
            line = f"{asset:<10} {strat_name:<12} {m['mean_sharpe']:>13.2f} {m['mean_sharpe_daily']:>14.2f} {m['max_dd']*100:>10.1f}% {m['mean_turnover']:>10.4f} {m['mean_tail_5']*100:>10.1f}% {m['mean_skew']:>10.3f} {m['mean_pos']:>10.3f}"
            print(line)

def print_table3(delta_results):
    """TABLA 3: DELTA VOL-ADJUSTED vs NAIVE."""
    print("\n" + "=" * 80)
    print("TABLA 3: DELTA VOL-ADJUSTED vs NAIVE")
    print("=" * 80)
    header = f"{'Asset':<10} {'Sharpe Delta':>14} {'DD Delta':>12} {'Tail Delta':>14}"
    print(header)
    print("-" * 80)

    for delta in delta_results:
        line = f"{delta['asset']:<10} {delta['sharpe_delta']:>+14.2f} {delta['dd_delta']*100:>+12.1f}% {delta['tail_delta']*100:>+14.1f}%"
        print(line)

def print_verdict(aggregated_results):
    """VEREDICTO FINAL y RECOMENDACIÓN."""
    print("\n" + "=" * 80)
    print("VEREDICTO FINAL")
    print("=" * 80)

    verdicts = []
    for asset_data in aggregated_results:
        asset = asset_data['asset']
        mean_sharpe_vol = asset_data['aggregated']['vol_adj']['mean_sharpe']
        mean_sharpe_naive = asset_data['aggregated']['naive']['mean_sharpe']
        max_dd_vol = asset_data['aggregated']['vol_adj']['max_dd']

        # PASS criteria: mean Sharpe > 0.15 AND max DD < 50%
        pass_criteria = mean_sharpe_vol > 0.15 and abs(max_dd_vol) < 0.50
        improvement = mean_sharpe_vol > mean_sharpe_naive

        if pass_criteria and improvement:
            verdict = "PASS+MEJORA"
        elif pass_criteria:
            verdict = "PASS (sin mejora)"
        else:
            verdict = "FAIL"

        asset_data['verdict'] = verdict
        asset_data['pass'] = pass_criteria
        asset_data['improvement'] = improvement

        print(f"{asset}: {verdict} (Vol-Adj Sharpe={mean_sharpe_vol:.2f} vs Naive={mean_sharpe_naive:.2f}, DD={max_dd_vol:.1%})")

    # Recomendación automática
    print("\n" + "-" * 80)
    print("RECOMENDACIÓN:")
    eth_pass = aggregated_results[0]['verdict'] in ['PASS+MEJORA', 'PASS (sin mejora)']
    btc_pass = aggregated_results[1]['verdict'] in ['PASS+MEJORA', 'PASS (sin mejora)']
    sol_pass = aggregated_results[2]['verdict'] in ['PASS+MEJORA', 'PASS (sin mejora)']

    eth_improve = aggregated_results[0]['improvement']
    btc_improve = aggregated_results[1]['improvement']
    sol_improve = aggregated_results[2]['improvement']

    all_pass = eth_pass and btc_pass and sol_pass
    all_improve = eth_improve and btc_improve and sol_improve

    if all_pass and all_improve:
        print("✅ PROCEDER a V20.8 multi-asset con 1 modelo — ETH+BTC+SOL todos PASS+MEJORA")
    elif eth_pass and btc_pass and not sol_pass:
        print("⚠️ V20.8 con ETH+BTC, modelo separado para SOL (SOL no pasa)")
    elif eth_pass and not btc_pass:
        print("⚠️ Solo ETH PASS — mantener V20.7 solo ETH, investigar sizing para BTC/SOL")
    elif not eth_pass:
        print("❌ ETH no PASS — revisar modelo o features antes de multi-asset")
    else:
        print("❓ Resultados mixtos — revisar configuración por asset")

    print("=" * 80)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("ORION V20.8 — VOLATILITY-ADJUSTED SIZING BACKTEST")
    print(f"Assets: {', '.join(ASSETS)}")
    print(f"Model: {MODEL_PATH} (frozen, ETH-trained)")
    print("Strategies: Naive (A) | Vol-Adjusted (B) | Buy&Hold (C)")
    print("=" * 80)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[MODEL] Loading model...")
    model = joblib.load(MODEL_PATH)
    print(f"[MODEL] Loaded: {type(model)}")

    # Seed for reproducibility
    np.random.seed(SEED)

    # Evaluate each asset independently (try/except)
    all_results = []
    for asset in ASSETS:
        try:
            asset_res = evaluate_asset(asset, model)
            if asset_res is not None:
                all_results.append(asset_res)
        except Exception as e:
            print(f"[ERROR] Asset {asset} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) == 0:
        print("[FATAL] No assets completed successfully")
        sys.exit(1)

    # Aggregate fold metrics into portfolio-level statistics
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS...")
    print("=" * 80)

    aggregated_results = []
    for asset_data in all_results:
        asset = asset_data['asset']
        folds = asset_data['folds']

        # Collect metrics across folds for each strategy
        strategies = ['naive', 'vol_adj', 'bh']
        agg = {}

        for strat in strategies:
            sharpe_vals = []
            dd_vals = []
            ret_vals = []
            turnover_vals = []
            tail_vals = []
            skew_vals = []
            pos_vals = []
            sharpe_daily_vals = []

            for fold_data in folds:
                m = fold_data[strat]
                sharpe_vals.append(m['sharpe'])
                dd_vals.append(m['max_dd'])
                ret_vals.append(m['total_return'])
                turnover_vals.append(m['turnover'])
                tail_vals.append(m['tail_5'])
                skew_vals.append(m['skew'])
                pos_vals.append(m['mean_pos'])
                sharpe_daily_vals.append(m['sharpe_daily'])

            valid_sharpe_daily = [v for v in sharpe_daily_vals if not np.isnan(v)]
            mean_sharpe_daily = np.mean(valid_sharpe_daily) if valid_sharpe_daily else np.nan
            agg[strat] = {
                'mean_sharpe': np.mean(sharpe_vals),
                'std_sharpe': np.std(sharpe_vals),
                'max_dd': np.min(dd_vals),  # worst DD across folds
                'mean_total_return': np.mean(ret_vals),
                'mean_turnover': np.mean(turnover_vals),
                'mean_tail_5': np.mean(tail_vals),
                'mean_skew': np.mean(skew_vals),
                'mean_pos': np.mean(pos_vals),
                'mean_sharpe_daily': mean_sharpe_daily
            }

        asset_data['aggregated'] = agg
        aggregated_results.append(asset_data)

    # Print tables
    print_table1(all_results)
    print_table2(aggregated_results)

    # Compute deltas
    delta_results = []
    for asset_data in aggregated_results:
        asset = asset_data['asset']
        vol_adj = asset_data['aggregated']['vol_adj']
        naive = asset_data['aggregated']['naive']

        delta_results.append({
            'asset': asset,
            'sharpe_delta': vol_adj['mean_sharpe'] - naive['mean_sharpe'],
            'dd_delta': vol_adj['max_dd'] - naive['max_dd'],
            'tail_delta': vol_adj['mean_tail_5'] - naive['mean_tail_5']
        })

    print_table3(delta_results)
    print_verdict(aggregated_results)

    print("\n[COMPLETE] Backtest finished successfully.")
    return aggregated_results

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
