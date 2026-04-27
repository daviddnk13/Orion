#!/home/ubuntu/orion/venv/bin/python3
# -*- coding: utf-8 -*-
"""
ORION PHASE 0 — SOL MAPPING TEST (KAGGLE VERSION)
Tests 6 position mappings on SOL using ETH-trained model
No triple quotes allowed — comments only with #
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
from scipy.stats import spearmanr
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION (KAGGLE)
# ============================================================
MODEL_PATH = 'model_v20_6_1.pkl'

# Telegram (hardcoded for topic 972 — Modelos)
TELEGRAM_BOT_TOKEN = '8723893197:AAFfIORXd2Y-qQ8TclOq23afEPt_knr7xrU'
TELEGRAM_CHAT_ID = '-1003505760554'
TOPIC_MODELOS = 972

# Data fetching
SOL_SYMBOL = 'SOL-USDT'
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
TARGET_VOL = 0.20
SMOOTHING_ALPHA = 0.7  # current_weight
FEE_BPS = 5
SLIP_BPS = 5
TOTAL_FRICTION = (FEE_BPS + SLIP_BPS) / 10000.0

# Mappings definitions
MAPPINGS = {
    'baseline': {
        'func': lambda p: 1.0 - p,
        'desc': 'Current ETH logic: pos = 1 - proba'
    },
    'linear_clipped': {
        'func': lambda p: max(0.0, 2.0 * (p - 0.5)),
        'desc': 'Inverted: high vol → more position'
    },
    'confidence_weighted': {
        'func': lambda p: max(0.0, (p - 0.5)) * abs(p - 0.5) * 4.0,
        'desc': 'Quadratic confidence weighting'
    },
    'threshold': {
        'func': lambda p: 0.5 if p > 0.55 else 0.0,
        'desc': 'Threshold at 0.55, full 0.5 allocation'
    },
    'convex': {
        'func': lambda p: max(0.0, (p - 0.5)) ** 2 * 4.0,
        'desc': 'Extreme-focused convex function'
    },
    'abs_confidence': {
        'func': lambda p: abs(2.0 * (p - 0.5)),
        'desc': 'Confidence regardless of direction'
    }
}

# ============================================================
# DATA FETCHING (OKX)
# ============================================================
def fetch_okx_ohlcv(symbol='SOL-USDT', timeframe='4h', n_candles=None):
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
# METRICS CALCULATION
# ============================================================
def calculate_metrics(returns, positions, turnovers):
    """Sharpe, Max DD, Turnover, Hit rate, Win/Loss ratio"""
    if len(returns) == 0:
        return {'sharpe': 0.0, 'max_dd': 0.0, 'turnover': 0.0, 'hit_rate': 0.0, 'win_loss_ratio': 0.0}

    # Sharpe (annualized, 6 periods per day)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = mean_ret / std_ret * np.sqrt(6 * 365) if std_ret > 1e-8 else 0.0

    # Max DD (equity curve)
    cum_ret = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - running_max) / running_max
    max_dd = np.min(dd) if len(dd) > 0 else 0.0

    # Turnover (mean absolute position change)
    turnover = np.mean(turnovers) if len(turnovers) > 0 else 0.0

    # Hit rate and Win/Loss
    hits = returns[returns > 0]
    losses = returns[returns < 0]
    hit_rate = len(hits) / len(returns) if len(returns) > 0 else 0.0
    mean_win = np.mean(hits) if len(hits) > 0 else 0.0
    mean_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0.0
    win_loss_ratio = mean_win / mean_loss if mean_loss > 1e-8 else np.inf if mean_win > 0 else 0.0

    return {
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'turnover': float(turnover),
        'hit_rate': float(hit_rate),
        'win_loss_ratio': float(win_loss_ratio)
    }

def calculate_monotonicity(proba, returns, n_quintiles=5):
    """Return monotonicity score and quintile means"""
    if len(proba) < 10:
        return 0.0, [np.nan] * n_quintiles

    # Bin by proba
    bounds = np.percentile(proba, np.linspace(0, 100, n_quintiles+1)[1:-1])
    quintile = np.digitize(proba, bounds) + 1

    quintile_means = []
    for q in range(1, n_quintiles+1):
        mask = quintile == q
        if mask.sum() > 0:
            quintile_means.append(np.mean(returns[mask]))
        else:
            quintile_means.append(np.nan)

    # Spearman correlation
    valid = ~np.isnan(quintile_means)
    if valid.sum() >= 3:
        mono_score, _ = spearmanr(np.arange(1, n_quintiles+1)[valid], np.array(quintile_means)[valid])
        mono_score = mono_score if np.isfinite(mono_score) else 0.0
    else:
        mono_score = 0.0

    return float(mono_score), quintile_means

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
    print("ORION PHASE 0 — SOL MAPPING TEST (KAGGLE)")
    print(f"Model: {MODEL_PATH}")
    print(f"Asset: {SOL_SYMBOL} ({TIMEFRAME})")
    print(f"Folds: {N_FOLDS}, test={TEST_SIZE}, embargo={EMBARGO}, initial_train={INITIAL_TRAIN_SIZE}")
    print(f"Mappings: {len(MAPPINGS)}")
    print("=" * 80)

    # 1. Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[MODEL] Loading LightGBM...")
    model = joblib.load(MODEL_PATH)
    print(f"[MODEL] Loaded: {type(model)}")

    # 2. Fetch data
    print(f"[DATA] Fetching {N_CANDLES} candles for {SOL_SYMBOL}...")
    try:
        df = fetch_okx_ohlcv(symbol=SOL_SYMBOL, timeframe=TIMEFRAME, n_candles=N_CANDLES)
    except Exception as e:
        print(f"[FATAL] Failed to fetch data: {e}")
        sys.exit(1)

    print(f"[DATA] Retrieved {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # 3. Build features
    print("[FEAT] Building 23 features...")
    df, raw_features = build_features(df)
    print(f"[FEAT] Features: {len(raw_features)}")

    # 4. Prepare feature matrix with Column_0..Column_22 names
    feature_data = df[raw_features].copy()
    feature_data.columns = [f'Column_{i}' for i in range(23)]
    feature_data = feature_data.clip(-5, 5)

    # 5. Generate splits
    n = len(feature_data)
    splits = generate_walk_forward_splits(n, initial_train_size=INITIAL_TRAIN_SIZE,
                                          test_size=TEST_SIZE, embargo=EMBARGO, n_folds=N_FOLDS)
    print(f"[SPLITS] Generated {len(splits)} folds")

    # 6. Run tests for each mapping
    all_results = []

    for map_name, map_def in MAPPINGS.items():
        print(f"\n{'='*80}")
        print(f"[MAPPING] {map_name}: {map_def['desc']}")
        print("=" * 80)

        mapping_func = map_def['func']

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

            # Get test data
            X_test = feature_data.iloc[test_idx].values.astype(np.float64)
            y_test_vol = df['vol_24h_future'].iloc[test_idx] if 'vol_24h_future' in df.columns else None

            # Predict proba
            proba_raw = model.predict(X_test)  # shape (n_test,)
            # Remove any calibration — use raw predictions directly

            # Apply mapping and clipping
            positions_raw = np.array([mapping_func(p) for p in proba_raw])
            positions = np.clip(positions_raw, 0, 0.5)

            # Smoothing: 70% current, 30% previous
            positions_smooth = np.zeros_like(positions)
            prev_pos = 0.0
            for i in range(len(positions)):
                positions_smooth[i] = SMOOTHING_ALPHA * positions[i] + (1 - SMOOTHING_ALPHA) * prev_pos
                prev_pos = positions_smooth[i]
            positions = positions_smooth

            # Vol-adjustment using realized vol
            log_returns = df['log_ret'].iloc[test_idx]
            realized_vol = log_returns.rolling(window=42).std().values * np.sqrt(6 * 365)
            realized_vol = np.nan_to_num(realized_vol, nan=TARGET_VOL)
            vol_ratio = TARGET_VOL / (realized_vol + 1e-8)
            vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
            positions = positions * vol_ratio

            # DD scalar (walking DD based on running balance)
            close_prices = df['close'].iloc[test_idx].values
            returns = np.diff(close_prices) / close_prices[:-1] if len(close_prices) > 1 else []
            pos_for_returns = positions[:-1] if len(positions) > 1 else []
            strategy_returns = pos_for_returns * returns if len(returns) > 0 else []

            running_balance = 1.0
            peak_balance = 1.0
            dd_history = []
            for r in strategy_returns:
                running_balance *= (1 + r)
                peak_balance = max(peak_balance, running_balance)
                dd = (running_balance - peak_balance) / peak_balance if peak_balance > 0 else 0.0
                dd_history.append(dd)

            # Apply DD scalar to positions (use latest DD)
            current_dd = dd_history[-1] if dd_history else 0.0
            dd_scalar = np.clip(1.0 + current_dd / 0.50, 0.1, 1.0)
            positions = positions * dd_scalar

            # Recalculate strategy returns with final positions
            if len(positions[:-1]) > 0 and len(returns) > 0:
                strategy_returns = positions[:-1] * returns
            else:
                strategy_returns = []

            # Calculate turnover (absolute position changes)
            pos_changes = np.abs(np.diff(positions))
            turnover = np.mean(pos_changes) if len(pos_changes) > 0 else 0.0

            # Calculate metrics
            if len(strategy_returns) > 0:
                mean_ret = np.mean(strategy_returns)
                std_ret = np.std(strategy_returns)
                sharpe = mean_ret / std_ret * np.sqrt(6 * 365) if std_ret > 1e-8 else 0.0

                # Max DD with final positions
                equity = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(equity)
                dd_curve = (equity - running_max) / running_max
                max_dd = np.min(dd_curve) if len(dd_curve) > 0 else 0.0

                # Hit rate and win/loss
                hits = strategy_returns[strategy_returns > 0]
                losses = strategy_returns[strategy_returns < 0]
                hit_rate = len(hits) / len(strategy_returns)
                mean_win = np.mean(hits) if len(hits) > 0 else 0.0
                mean_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0.0
                win_loss = mean_win / mean_loss if mean_loss > 1e-8 else np.inf

                # Monotonicity
                mono_score, quintile_means = calculate_monotonicity(proba_raw[:-1], strategy_returns, n_quintiles=5)
            else:
                sharpe = 0.0
                max_dd = 0.0
                hit_rate = 0.0
                win_loss = 0.0
                mono_score = 0.0
                quintile_means = [np.nan] * 5

            fold_metric = {
                'fold': fold,
                'sharpe': float(sharpe),
                'max_dd': float(max_dd),
                'turnover': float(turnover),
                'hit_rate': float(hit_rate),
                'win_loss_ratio': float(win_loss),
                'monotonicity': float(mono_score),
                'quintile_means': quintile_means,
                'n_test': len(test_idx)
            }
            fold_metrics.append(fold_metric)

            print(f"    Sharpe={sharpe:.2f}, DD={max_dd*100:.1f}%, turnover={turnover:.4f}, "
                  f"hit={hit_rate:.2%}, W/L={win_loss:.2f}, mono={mono_score:.3f}")

        # Aggregate over folds
        sharpe_vals = [fm['sharpe'] for fm in fold_metrics]
        dd_vals = [fm['max_dd'] for fm in fold_metrics]
        turnover_vals = [fm['turnover'] for fm in fold_metrics]

        mean_sharpe = np.mean(sharpe_vals)
        mean_dd = np.mean(dd_vals)
        mean_turnover = np.mean(turnover_vals)

        result = {
            'mapping': map_name,
            'description': map_def['desc'],
            'folds': fold_metrics,
            'aggregate': {
                'mean_sharpe': float(mean_sharpe),
                'mean_max_dd': float(mean_dd),
                'mean_turnover': float(mean_turnover)
            }
        }
        all_results.append(result)

        print(f"[AGGREGATE] Sharpe={mean_sharpe:.2f}, DD={mean_dd*100:.1f}%, turnover={mean_turnover:.4f}")

    # 7. Print comparative table
    print("\n" + "=" * 120)
    print("COMPARATIVE RESULTS")
    print("=" * 120)
    header = f"{'Mapping':<25} {'Sharpe':>10} {'Max DD':>10} {'Turnover':>10} {'Hit Rate':>10} {'W/L Ratio':>12} {'Monotonicity':>12}"
    print(header)
    print("-" * 120)

    for res in all_results:
        # Calculate means across folds for hit_rate and win_loss
        hit_vals = [fm['hit_rate'] for fm in res['folds']]
        wl_vals = [fm['win_loss_ratio'] for fm in res['folds'] if np.isfinite(fm['win_loss_ratio'])]
        mono_vals = [fm['monotonicity'] for fm in res['folds']]

        mean_hit = np.mean(hit_vals)
        mean_wl = np.mean(wl_vals) if len(wl_vals) > 0 else 0.0
        mean_mono = np.mean(mono_vals)

        line = (f"{res['mapping']:<25} {res['aggregate']['mean_sharpe']:>10.2f} "
                f"{res['aggregate']['mean_max_dd']*100:>9.1f}% {res['aggregate']['mean_turnover']:>10.4f} "
                f"{mean_hit:>10.2%} {mean_wl:>11.2f} {mean_mono:>12.3f}")
        print(line)

    print("=" * 120)

    # 8. Identify best mapping
    valid_results = [r for r in all_results if r['aggregate']['mean_sharpe'] > 0 and r['aggregate']['mean_max_dd'] < 0.30 and r['aggregate']['mean_turnover'] < 0.05]
    if valid_results:
        best = max(valid_results, key=lambda x: x['aggregate']['mean_sharpe'])
        print(f"\n🏆 BEST MAPPING: {best['mapping']} (Sharpe={best['aggregate']['mean_sharpe']:.2f}, DD={best['aggregate']['mean_max_dd']*100:.1f}%)")
    else:
        print("\n⚠️ NO VALID MAPPING: All fail acceptance criteria (Sharpe>0, DD<30%, turnover<0.05)")

    # 9. Save results to JSON
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': MODEL_PATH,
        'asset': SOL_SYMBOL,
        'timeframe': TIMEFRAME,
        'n_candles': len(df),
        'n_folds': len(splits),
        'test_size': TEST_SIZE,
        'embargo': EMBARGO,
        'mappings': all_results
    }

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sol_phase0_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[OUTPUT] Saved results to {json_path}")

    # 10. Generate monotonicity plots for each mapping
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, res in enumerate(all_results):
        ax = axes[idx]
        # Average quintile means across folds
        quintile_matrix = np.array([fm['quintile_means'] for fm in res['folds']])
        avg_quintiles = np.nanmean(quintile_matrix, axis=0)

        x = np.arange(1, 6)
        ax.bar(x, avg_quintiles, color='steelblue', alpha=0.7)
        ax.set_xlabel('Proba Quintile (1=lowest, 5=highest)')
        ax.set_ylabel('Mean Return')
        ax.set_title(f"{res['mapping']} (mono={np.mean([fm['monotonicity'] for fm in res['folds']]):.3f})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sol_phase0_monotonicity.png')
    plt.savefig(plot_path, dpi=100)
    print(f"[PLOT] Saved monotonicity plot to {plot_path}")

    # 11. Send summary to Telegram topic 972
    tg_msg = [
        "📊 ORION PHASE 0 — SOL MAPPING TEST",
        f"UTC: {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M')}",
        f"Model: ETH model (model_v20_6_1.pkl)",
        f"Data: {len(df)} {SOL_SYMBOL} {TIMEFRAME} candles",
        f"Folds: {len(splits)} (test_size={TEST_SIZE}, embargo={EMBARGO})",
        "",
        "<b>Comparative Table:</b>"
    ]

    for res in all_results:
        agg = res['aggregate']
        mono_avg = np.mean([fm['monotonicity'] for fm in res['folds']])
        line = f"<code>{res['mapping']:25} Sharpe={agg['mean_sharpe']:>5.2f} DD={agg['mean_max_dd']*100:>5.1f}% mono={mono_avg:.3f}</code>"
        tg_msg.append(line)

    if valid_results:
        best = max(valid_results, key=lambda x: x['aggregate']['mean_sharpe'])
        tg_msg.append("")
        tg_msg.append(f"🏆 <b>BEST: {best['mapping']}</b> (Sharpe={best['aggregate']['mean_sharpe']:.2f})")
    else:
        tg_msg.append("")
        tg_msg.append("⚠️ <b>NO VALID MAPPING</b> — all fail criteria")

    tg_text = "\n".join(tg_msg)

    if tg_send(tg_text, topic_id=TOPIC_MODELOS):
        print("[TELEGRAM] Summary sent to topic 972 (Modelos)")
    else:
        print("[TELEGRAM] Failed to send message")

    print("\n✅ PHASE 0 COMPLETE")
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
