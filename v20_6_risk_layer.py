# v20_4_robustness.py — Orion V20.4 Risk Engine Robustness Validation
# Self-contained script for Kaggle
# 6 robustness tests on V20.3 Risk Engine (no model changes)

import numpy as np
import pandas as pd
import os
import json
import requests
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import lightgbm as lgb

# ============================================================
# TELEGRAM CONFIG
# ============================================================
TELEGRAM = {
    'bot_token': '8723893197:AAFfIORXd2Y-qQ8TclOq23afEPt_knr7xrU',
    'chat_id': '-1003505760554',
    'topic_971': 971,
}

def tg_send(text, topic_id=None):
    """Send message to Telegram topic."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM['bot_token']}/sendMessage"
        payload = {
            'chat_id': TELEGRAM['chat_id'],
            'text': text,
            'parse_mode': 'HTML'
        }
        if topic_id is not None:
            payload['message_thread_id'] = topic_id
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[TELEGRAM] Failed: {e}")
        return False

# ============================================================
# DATA FETCHING (OKX)
# ============================================================
def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4H', n_candles=12000):
    """Fetch OHLCV from OKX public API."""
    import time
    base_url = "https://www.okx.com"
    endpoint = f"/api/v5/market/history-candles"
    params = {
        'instId': symbol,
        'bar': timeframe,
        'limit': str(min(1000, n_candles))
    }
    all_candles = []
    remaining = n_candles
    offset = 0
    while remaining > 0:
        params['limit'] = str(min(1000, remaining))
        try:
            r = requests.get(base_url + endpoint, params=params, timeout=10)
            data = r.json()
            if data.get('code') != '0':
                break
            candles = data['data']
            if not candles:
                break
            all_candles.extend(candles)
            remaining -= len(candles)
            if candles:
                params['after'] = candles[-1][0]
            time.sleep(0.1)
        except Exception as e:
            print(f"[OKX] Error: {e}")
            break
    if not all_candles:
        raise RuntimeError("No data fetched from OKX")
    df = pd.DataFrame(all_candles, columns=[
        'ts', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote', 'confirm'
    ])
    df['ts'] = pd.to_datetime(df['ts'].astype(np.int64), unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col])
    return df[['ts', 'open', 'high', 'low', 'close', 'vol']].rename(columns={'ts': 'timestamp'})

def fetch_macro_daily():
    """Fetch daily macro data (placeholder — not used in V20.x)."""
    return pd.DataFrame()

def align_daily_to_4h(df_4h, df_daily):
    """No-op for V20.x (macro not used)."""
    return df_4h

def load_derivatives_data(df):
    """Placeholder — derivatives not used in V20.1+."""
    return df, False, False

# ============================================================
# FEATURE ENGINEERING (23 features exactas de la spec)
# ============================================================
def build_features(df, has_funding=False, has_oi=False):
    """Build exactly 23 features as per spec."""
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret
    clip_val = 1e6
    eps = 1e-8

    # --- BASE (16) ---
    # ret_4h (6-bar sum)
    df['ret_4h'] = log_ret.rolling(6).sum().shift(1)

    # rsi_norm (14-period RSI normalized to [0,1])
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + eps)
    rsi = 100 - 100 / (1 + rs)
    df['rsi_norm'] = (rsi / 100).fillna(0.5).shift(1)

    # bb_position (Bollinger Bands position)
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_pos = (df['close'] - lower) / (upper - lower + eps)
    df['bb_position'] = bb_pos.fillna(0.5).shift(1)

    # macd_norm (MACD normalized)
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9).mean()
    macd_norm = (macd - macd_sig) / (df['close'] * 0.01 + eps)
    df['macd_norm'] = macd_norm.fillna(0).shift(1)

    # ret_4h_lag1, ret_4h_lag2
    df['ret_4h_lag1'] = df['ret_4h'].shift(1)
    df['ret_4h_lag2'] = df['ret_4h'].shift(2)

    # atr_norm (14-period ATR normalized by close)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_norm'] = (atr / df['close']).fillna(0).shift(1)

    # vol_zscore (rolling 180-bar z-score of 6-bar realized vol)
    rv_6 = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_mean = rv_6.rolling(180).mean().shift(1)
    rv_std = rv_6.rolling(180).std().shift(1)
    df['vol_zscore'] = ((rv_6 - rv_mean) / (rv_std + eps)).fillna(0).shift(1)

    # vol_regime (categorical encoded as int: 0=low,1=normal,2=high)
    vol_rank = rv_6.rolling(180).rank(pct=True).shift(1)
    df['vol_regime'] = pd.cut(vol_rank, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2]).astype(float).fillna(1).shift(1)

    # ret_8h (12-bar sum)
    df['ret_8h'] = log_ret.rolling(12).sum().shift(1)

    # ret_24h (24-bar sum)
    df['ret_24h'] = log_ret.rolling(24).sum().shift(1)

    # ema_slope (slope of EMA20)
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

    # drawdown_market (rolling max DD)
    roll_max = df['close'].rolling(100).max()
    dd = (df['close'] - roll_max) / (roll_max + eps)
    df['drawdown_market'] = dd.fillna(0).shift(1)

    # tf_coherence (correlation BTC/ETH over 42 bars, proxy with BTC data if available)
    # Since we don't have BTC data, set to 0
    df['tf_coherence'] = 0.0

    # dist_ema200 (distance to EMA200 in ATR units)
    ema_200 = df['close'].ewm(span=200).mean()
    dist = (df['close'] - ema_200) / (atr + eps)
    df['dist_ema200'] = dist.fillna(0).shift(1)

    # --- VOL (7) ---
    # parkinson_vol (Parkinson 42-bar)
    hl_ratio = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt(
        (hl_ratio ** 2).rolling(42).mean().shift(1) / (4 * np.log(2))
    ).fillna(0)

    # vol_compression = realized_vol_1d / (realized_vol_7d + eps)
    rv_1d = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_7d = log_ret.rolling(42).std().shift(1) * np.sqrt(6)
    df['realized_vol_1d'] = rv_1d
    df['realized_vol_7d'] = rv_7d
    df['vol_compression'] = rv_1d / (rv_7d + eps)

    # garman_klass_vol
    gk = 0.5 * (np.log(df['high'] / df['low'])) ** 2 - (2*np.log(2)-1) * (np.log(df['close']/df['open']))**2
    df['garman_klass_vol'] = np.sqrt(gk.rolling(6).mean().shift(1).clip(lower=0))

    # vol_regime_rank (pct rank of 7d vol over 180)
    vol_rank_pct = rv_7d.rolling(180).rank(pct=True).shift(1)
    df['vol_regime_rank'] = vol_rank_pct.fillna(0.5)

    # trend_efficiency (close_diff_24 / sum_abs_diff_24)
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

    # Fill any remaining NaNs
    for f in raw_features:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].fillna(0)

    print(f"[FEATURES] Built {len(raw_features)} features")
    return df, raw_features

# ============================================================
# TARGET ENGINEERING
# ============================================================
def build_targets(df):
    """Build forward volatility targets."""
    log_ret = df['log_ret'].values
    n = len(log_ret)
    # vol_24h: 6-bar forward realized vol
    vol_24h = np.full(n, np.nan)
    for t in range(n - 6):
        window = log_ret[t + 1: t + 7]
        if len(window) == 6:
            vol_24h[t] = np.std(window, ddof=1) * np.sqrt(6)
    df['vol_24h_future'] = vol_24h
    print("[TARGETS] Built vol_24h_future")
    return df

# ============================================================
# RISK ENGINE (from risk_engine.py)
# ============================================================
class RiskEngine:
    def __init__(self, sensitivity=1.0, max_reduction=0.7, target_recall=0.70):
        self.sensitivity = np.clip(sensitivity, 0.5, 2.0)
        self.max_reduction = np.clip(max_reduction, 0.3, 0.8)
        self.target_recall = target_recall
        self.threshold = None

    def fit(self, y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        if len(np.unique(y_true)) < 2:
            self.threshold = 0.5
            return self
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        precision = precision[:-1]
        recall = recall[:-1]
        candidate_indices = np.where(recall >= self.target_recall)[0]
        if len(candidate_indices) > 0:
            best_idx = candidate_indices[np.argmax(precision[candidate_indices])]
            self.threshold = thresholds[best_idx]
        else:
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx]
        return self

    def predict_scale(self, prob_high):
        reduction = np.clip(prob_high * self.sensitivity, 0, self.max_reduction)
        position_scale = 1.0 - reduction
        return position_scale

    def predict_signal(self, prob_high):
        if self.threshold is None:
            raise RuntimeError("RiskEngine must be fit before calling predict_signal")
        return np.asarray(prob_high) >= self.threshold

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def build_fold_indices(n_samples, n_folds=4, test_size=1250, embargo=180):
    """Build walk-forward fold indices (anchored expanding window)."""
    folds = []
    for fold_i in range(n_folds):
        test_end = n_samples - (n_folds - fold_i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - embargo
        if train_end < 100:
            raise ValueError(f"Fold {fold_i}: insufficient training data (train_end={train_end})")
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
    """Compute total return, annualized Sharpe, and max drawdown."""
    if len(simple_returns) == 0:
        return {'total_return': 0.0, 'sharpe': 0.0, 'max_dd': 0.0}
    total_ret = np.prod(1 + simple_returns) - 1
    mean = simple_returns.mean()
    std = simple_returns.std()
    sharpe = mean / std * np.sqrt(periods_per_year) if std > 1e-10 else 0.0
    wealth = np.cumprod(1 + simple_returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    max_dd = drawdown.min()
    return {'total_return': float(total_ret), 'sharpe': float(sharpe), 'max_dd': float(max_dd)}

# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(42)

    print("=" * 70)
    print("ORION V20.4 — RISK ENGINE ROBUSTNESS VALIDATION")
    print("=" * 70)

    # ---- SECTION 1: Clone repo / fetch data ----
    print("\nSECTION 1: Clone Orion repo")
    print("=" * 70)
    # In Kaggle, repo already cloned? We'll try to fetch anyway
    # For this script, we assume we can fetch OKX directly

    # ---- SECTION 2: Fetch OKX data ----
    print("\nSECTION 2: Fetch OKX data")
    print("=" * 70)
    df = fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4H', n_candles=12000)
    print(f"[OK] {len(df)} candles: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    # ---- SECTION 3: Build 23 features ----
    print("\nSECTION 3: Features (23 total)")
    print("=" * 70)
    df, raw_features = build_features(df)
    print(f"[OK] {len(raw_features)} features built")

    # ---- SECTION 4: Target = binary (HIGH vol vs REST, p66.67) ----
    print("\nSECTION 4: Target = binary (HIGH vol vs REST, p66.67)")
    print("=" * 70)
    df = build_targets(df)
    warmup = 200
    df = df.iloc[warmup:].reset_index(drop=True)
    print(f"[CLEAN] After warmup drop: {len(df)} rows")

    valid_mask = df['vol_24h_future'].notna()
    df = df[valid_mask].reset_index(drop=True)
    print(f"[CLEAN] After target NaN drop: {len(df)} rows")

    if len(df) < 5000:
        print(f"[ERROR] Insufficient data: {len(df)} < 5000")
        return

    # Prepare arrays
    X = df[raw_features].values.astype(np.float64)
    y_vol = df['vol_24h_future'].values
    returns_24h_log = np.log(df['close'] / df['close'].shift(6)).values  # 6-bar forward return

    n_samples = len(df)
    print(f"[CONFIG] Samples: {n_samples}, Features: {len(raw_features)}")
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

    # Engine defaults
    engine_defaults = {'sensitivity': 1.0, 'max_reduction': 0.7, 'target_recall': 0.70}

    # Storage for robustness tests
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_fold_idx = []
    all_position_scale = []
    all_asset_returns_log = []

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION (V20.3 REPRODUCTION)")
    print("=" * 60)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n[FOLD {fold_i}]")

        X_train, X_test = X[train_idx], X[test_idx]
        y_vol_train, y_vol_test = y_vol[train_idx], y_vol[test_idx]
        returns_24h_test = returns_24h_log[test_idx]

        # Binary target: HIGH = top 1/3 of training distribution (p66.67)
        threshold_vol = np.percentile(y_vol_train, 100/3*2)  # 66.67 percentile
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

        # Store for aggregate analysis and robustness tests
        all_y_true.extend(y_test_binary)
        all_y_pred.extend(y_test_pred_signal)
        all_y_prob.extend(y_test_prob)
        all_fold_idx.extend([fold_i] * len(y_test_binary))
        all_position_scale.extend(position_scale)
        all_asset_returns_log.extend(returns_24h_test)

        # Store fold-level data for per-fold tests
        fold_results.append({
            'y_true': y_test_binary,
            'y_pred': y_test_pred_signal,
            'y_prob': y_test_prob,
            'position_scale': position_scale,
            'returns_log': returns_24h_test,
            'threshold_vol': threshold_vol
        })

        print(f"  Threshold (vol p66.67): {threshold_vol:.6f}")
        print(f"  Signal rate: {np.mean(y_test_pred_signal):.1%}")

    # ---- Overall sizing simulation (from V20.3) ----
    all_asset_returns_simple = np.exp(all_asset_returns_log) - 1
    all_position_scale_arr = np.array(all_position_scale)
    scaled_returns_simple = all_position_scale_arr * all_asset_returns_simple
    unscaled_returns_simple = all_asset_returns_simple

    scaled_stats = portfolio_stats(scaled_returns_simple)
    unscaled_stats = portfolio_stats(unscaled_returns_simple)

    print("\n" + "=" * 60)
    print("SIZING IMPACT (V20.3 BASELINE)")
    print("=" * 60)
    print("Unscaled (buy-and-hold):")
    print(f"  Total return: {unscaled_stats['total_return']:.2%}")
    print(f"  Annualized Sharpe: {unscaled_stats['sharpe']:.2f}")
    print(f"  Max drawdown: {unscaled_stats['max_dd']:.2%}")
    print("Scaled (with RiskEngine):")
    print(f"  Total return: {scaled_stats['total_return']:.2%}")
    print(f"  Annualized Sharpe: {scaled_stats['sharpe']:.2f}")
    print(f"  Max drawdown: {scaled_stats['max_dd']:.2%}")

    # ============================================================
    # ============================================================
    # ============================================================
    # SECTION 6: V20.6 RISK LAYER VALIDATION
    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 6: V20.6 RISK LAYER VALIDATION")
    print("=" * 70)

    # ---- RISK LAYER PARAMETERS ----
    TARGET_VOL = 0.15          # 15% annualized target volatility
    VOL_LOOKBACK = 42          # bars for realized vol (7 days of 4H)
    POSITION_CAP = 0.5         # hard cap: never above 50% exposure
    DD_TRIGGER = -0.30         # -30% drawdown triggers circuit breaker
    DD_RECOVERY = -0.15        # resume when DD recovers to -15%
    FEE = 0.0005
    SLIPPAGE = 0.0005
    TOTAL_FRICTION = FEE + SLIPPAGE

    print(f"  Config: target_vol={TARGET_VOL}, cap={POSITION_CAP}, DD_trigger={DD_TRIGGER}, DD_recovery={DD_RECOVERY}")
    print(f"  Costs: fee={FEE}, slippage={SLIPPAGE}")

    # Pre-compute realized vol on full series for vol targeting
    log_ret_full = df['log_ret'].values
    rv_full = pd.Series(log_ret_full).rolling(VOL_LOOKBACK).std() * np.sqrt(365 * 6)

    # ---- TEST 1: RAW ML (no risk layer) — reference ----
    print("\n--- TEST 1: RAW ML (reference) ---")
    for fold_i, fd in enumerate(fold_results):
        rt = np.exp(fd['returns_log']) - 1
        raw_ret = fd['position_scale'] * rt
        rs = portfolio_stats(raw_ret)
        print(f"  Fold {fold_i}: Sharpe={rs['sharpe']:.3f}, Return={rs['total_return']:+.2%}, DD={rs['max_dd']:.2%}")

    # ---- TEST 2: RISK LAYER APPLIED ----
    print("\n--- TEST 2: WITH RISK LAYER ---")

    risk_layer_results = []

    for fold_i, fd in enumerate(fold_results):
        test_idx = folds[fold_i][1]
        scale_ml = fd['position_scale']
        returns_simple = np.exp(fd['returns_log']) - 1

        # Component 1: Vol targeting
        rv_test = rv_full.values[test_idx]
        vol_scalar = TARGET_VOL / (rv_test + 1e-8)
        vol_scalar = np.clip(vol_scalar, 0.0, 2.0)  # cap at 2x leverage max
        # Shift by 1 (no look-ahead)
        vol_scalar_shifted = np.roll(vol_scalar, 1)
        vol_scalar_shifted[0] = 1.0  # first bar: neutral

        # Component 2: Combine ML scale with vol targeting
        scale_combined = scale_ml * vol_scalar_shifted

        # Component 3: Hard position cap
        scale_capped = np.clip(scale_combined, 0.0, POSITION_CAP)

        # Component 4: DD circuit breaker
        scale_final = np.copy(scale_capped)
        equity = 1.0
        peak = 1.0
        breaker_active = False
        breaker_bars = 0

        for t in range(len(scale_final)):
            # Update equity
            equity *= (1 + scale_final[t] * returns_simple[t])
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak

            # Circuit breaker logic
            if not breaker_active and dd < DD_TRIGGER:
                breaker_active = True
                breaker_bars = 0
            elif breaker_active and dd > DD_RECOVERY:
                breaker_active = False

            if breaker_active:
                scale_final[t] = 0.05  # minimal exposure during circuit break
                breaker_bars += 1

        # Compute returns with risk layer
        returns_risk = scale_final * returns_simple

        # Compute costs
        delta_scale = np.abs(np.diff(scale_final, prepend=scale_final[0]))
        cost_bar = delta_scale * TOTAL_FRICTION
        returns_net = returns_risk - cost_bar

        # Stats
        stats_raw = portfolio_stats(fd['position_scale'] * returns_simple)
        stats_risk = portfolio_stats(returns_risk)
        stats_net = portfolio_stats(returns_net)

        # Equity curve for DD calc
        equity_curve = np.cumprod(1 + returns_net)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd_actual = np.min(drawdowns)

        risk_layer_results.append({
            'sharpe_raw': stats_raw['sharpe'],
            'sharpe_risk': stats_risk['sharpe'],
            'sharpe_net': stats_net['sharpe'],
            'return_raw': stats_raw['total_return'],
            'return_net': stats_net['total_return'],
            'max_dd_raw': stats_raw['max_dd'],
            'max_dd_net': float(max_dd_actual),
            'breaker_bars': breaker_bars,
            'mean_scale': float(np.mean(scale_final)),
            'turnover': float(np.mean(np.abs(np.diff(scale_final)))),
        })

        print(f"  Fold {fold_i}:")
        print(f"    Raw ML:     Sharpe={stats_raw['sharpe']:.3f}, Return={stats_raw['total_return']:+.2%}, DD={stats_raw['max_dd']:.2%}")
        print(f"    Risk Layer: Sharpe={stats_risk['sharpe']:.3f}, Return={stats_risk['total_return']:+.2%}, DD={stats_risk['max_dd']:.2%}")
        print(f"    Net (costs): Sharpe={stats_net['sharpe']:.3f}, Return={stats_net['total_return']:+.2%}, DD={max_dd_actual:.2%}")
        print(f"    Breaker:    {breaker_bars} bars active, mean_scale={np.mean(scale_final):.3f}")

    # ---- VERDICTS ----
    print("\n--- VERDICTS ---")

    # V1: DD improvement
    dd_improved = sum(1 for r in risk_layer_results if abs(r['max_dd_net']) < abs(r['max_dd_raw'])) 
    dd_under_50 = sum(1 for r in risk_layer_results if abs(r['max_dd_net']) < 0.50)
    dd_pass = dd_under_50 >= 3

    # V2: Sharpe preservation
    sharpe_preserved = sum(1 for r in risk_layer_results if r['sharpe_net'] > 0)
    mean_sharpe_change = np.mean([(r['sharpe_net'] - r['sharpe_raw']) / (abs(r['sharpe_raw']) + 1e-8) * 100 for r in risk_layer_results])
    sharpe_pass = sharpe_preserved >= 3 and mean_sharpe_change > -30.0

    # V3: Cost survival (same as V20.5)
    cost_deg = np.mean([(r['sharpe_risk'] - r['sharpe_net']) / (abs(r['sharpe_risk']) + 1e-8) * 100 for r in risk_layer_results])
    cost_pass = sharpe_preserved >= 3 and cost_deg < 30.0

    verdicts = {
        'dd_control': dd_pass,
        'sharpe_preservation': sharpe_pass,
        'cost_survival': cost_pass,
    }

    print(f"  DD Control:         DD<50% in {dd_under_50}/4 folds (need >=3) -> {'PASS' if dd_pass else 'FAIL'}")
    print(f"  Sharpe Preservation: Sharpe_net>0 in {sharpe_preserved}/4, mean change={mean_sharpe_change:+.1f}% -> {'PASS' if sharpe_pass else 'FAIL'}")
    print(f"  Cost Survival:      degradation={cost_deg:.1f}% (need <30%) -> {'PASS' if cost_pass else 'FAIL'}")

    # ============================================================
    # SECTION 7: V20.6 FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 7: V20.6 FINAL VERDICT")
    print("=" * 70)

    pc = sum(1 for v in verdicts.values() if v)

    for tn, tv in verdicts.items():
        print(f"  {tn:25s} -> {'PASS' if tv else 'FAIL'}")

    if pc == 3:
        fv = "PAPER TRADING READY"
    elif pc == 2:
        fv = "CLOSE - NEEDS TUNING"
    elif pc == 1:
        fv = "PARTIAL - NEEDS WORK"
    else:
        fv = "NOT READY"

    print(f"\n  *** Passed {pc}/3 tests ***")
    print(f"  *** FINAL VERDICT: {fv} ***")

    print(f"\n  --- PER-FOLD COMPARISON ---")
    print(f"  {'Fold':<6} {'Sh_raw':<10} {'Sh_net':<10} {'DD_raw':<10} {'DD_net':<10} {'Breaker':<10}")
    for fi, r in enumerate(risk_layer_results):
        print(f"  {fi:<6} {r['sharpe_raw']:<10.3f} {r['sharpe_net']:<10.3f} {r['max_dd_raw']:<10.2%} {r['max_dd_net']:<10.2%} {r['breaker_bars']:<10d}")

    print("=" * 70)

    # Save JSON
    results = {
        'version': 'V20.6',
        'target_vol': TARGET_VOL,
        'position_cap': POSITION_CAP,
        'dd_trigger': DD_TRIGGER,
        'dd_recovery': DD_RECOVERY,
        'fee': FEE,
        'slippage': SLIPPAGE,
        'verdicts': {k: bool(v) for k, v in verdicts.items()},
        'risk_layer_results': risk_layer_results,
        'final_verdict': fv,
    }

    with open('v20_6_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] Results JSON: v20_6_results.json")

    # Telegram
    tg_msg = f"""<b>V20.6 Risk Layer Validation</b>

<b>Verdict:</b> <code>{fv}</code>

<b>Config:</b>
target_vol={TARGET_VOL}, cap={POSITION_CAP}
DD trigger={DD_TRIGGER}, recovery={DD_RECOVERY}

<b>Results:</b>
DD < 50%: {dd_under_50}/4 folds -> {'PASS' if dd_pass else 'FAIL'}
Sharpe_net > 0: {sharpe_preserved}/4, change={mean_sharpe_change:+.1f}% -> {'PASS' if sharpe_pass else 'FAIL'}
Cost survival: {cost_deg:.1f}% deg -> {'PASS' if cost_pass else 'FAIL'}

Passed {pc}/3"""

    tg_send(tg_msg, topic_id=TELEGRAM['topic_971'])
    print("[TELEGRAM] Results sent to topic 971")



if __name__ == '__main__':
    main()
