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
    # SECTION 6: V20.5 PRODUCTION VALIDATION
    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 6: V20.5 PRODUCTION VALIDATION")
    print("=" * 70)

    # ---- BLOQUE 1: EMA SMOOTHING ----
    print("\n--- BLOQUE 1: EMA SMOOTHING ---")

    EMA_ALPHA = 0.3

    smoothed_fold_results = []

    for fold_i, fold_data in enumerate(fold_results):
        scale_raw = fold_data['position_scale']

        # Apply EMA
        scale_smooth = np.zeros_like(scale_raw)
        scale_smooth[0] = scale_raw[0]
        for t in range(1, len(scale_raw)):
            scale_smooth[t] = EMA_ALPHA * scale_raw[t] + (1 - EMA_ALPHA) * scale_smooth[t - 1]

        # Turnover
        turnover_raw = np.mean(np.abs(np.diff(scale_raw)))
        turnover_smooth = np.mean(np.abs(np.diff(scale_smooth)))
        turnover_reduction = (turnover_raw - turnover_smooth) / (turnover_raw + 1e-8) * 100

        # Sharpe
        returns_test = np.exp(fold_data['returns_log']) - 1
        sharpe_raw = portfolio_stats(scale_raw * returns_test)['sharpe']
        sharpe_smooth = portfolio_stats(scale_smooth * returns_test)['sharpe']
        sharpe_change = (sharpe_smooth - sharpe_raw) / (abs(sharpe_raw) + 1e-8) * 100

        smoothed_fold_results.append({
            'scale_smooth': scale_smooth,
            'returns_log': fold_data['returns_log'],
        })

        print(f"  Fold {fold_i}:")
        print(f"    Turnover:  raw={turnover_raw:.4f} -> smooth={turnover_smooth:.4f} (down {turnover_reduction:.1f}%)")
        print(f"    Sharpe:    raw={sharpe_raw:.3f} -> smooth={sharpe_smooth:.3f} ({sharpe_change:+.1f}%)")
        print(f"    Scale:     mean={np.mean(scale_smooth):.3f}, std={np.std(scale_smooth):.3f}")

    # Aggregate smoothing verdict
    all_t_red = []
    all_s_chg = []
    for fold_i, fold_data in enumerate(fold_results):
        sr = fold_data['position_scale']
        ss = smoothed_fold_results[fold_i]['scale_smooth']
        rt = np.exp(fold_data['returns_log']) - 1
        tr = np.mean(np.abs(np.diff(sr)))
        ts = np.mean(np.abs(np.diff(ss)))
        all_t_red.append((tr - ts) / (tr + 1e-8) * 100)
        shr = portfolio_stats(sr * rt)['sharpe']
        shs = portfolio_stats(ss * rt)['sharpe']
        all_s_chg.append((shs - shr) / (abs(shr) + 1e-8) * 100)

    mean_t_red = np.mean(all_t_red)
    mean_s_chg = np.mean(all_s_chg)
    smoothing_pass = mean_t_red >= 20.0 and mean_s_chg > -10.0

    print(f"\n  [SMOOTHING SUMMARY]")
    print(f"    Mean turnover reduction: {mean_t_red:.1f}% (need >=20%)")
    print(f"    Mean Sharpe change:      {mean_s_chg:+.1f}% (need >-10%)")
    print(f"    SMOOTHING VERDICT: {'PASS' if smoothing_pass else 'FAIL'}")

    # ---- BLOQUE 2: COST-AWARE BACKTEST ----
    print("\n--- BLOQUE 2: COST-AWARE BACKTEST ---")

    FEE = 0.0005
    SLIPPAGE = 0.0005
    TOTAL_FRICTION = FEE + SLIPPAGE

    cost_results = []

    for fold_i in range(len(fold_results)):
        ss = smoothed_fold_results[fold_i]['scale_smooth']
        rl = smoothed_fold_results[fold_i]['returns_log']
        rs = np.exp(rl) - 1

        returns_gross = ss * rs
        delta_s = np.abs(np.diff(ss, prepend=ss[0]))
        cost_bar = delta_s * TOTAL_FRICTION
        returns_net = returns_gross - cost_bar

        sg = portfolio_stats(returns_gross)
        sn = portfolio_stats(returns_net)
        tc = np.sum(cost_bar)
        deg = (sg['sharpe'] - sn['sharpe']) / (abs(sg['sharpe']) + 1e-8) * 100

        cost_results.append({
            'sharpe_gross': sg['sharpe'],
            'sharpe_net': sn['sharpe'],
            'return_net': sn['total_return'],
            'max_dd_net': sn['max_dd'],
            'degradation': deg,
        })

        print(f"  Fold {fold_i}:")
        print(f"    Gross:  Return={sg['total_return']:+.2%}, Sharpe={sg['sharpe']:.3f}, DD={sg['max_dd']:.2%}")
        print(f"    Net:    Return={sn['total_return']:+.2%}, Sharpe={sn['sharpe']:.3f}, DD={sn['max_dd']:.2%}")
        print(f"    Cost:   total={tc:.6f}, Sharpe degradation={deg:.1f}%")

    sn_pos = sum(1 for r in cost_results if r['sharpe_net'] > 0)
    mean_deg = np.mean([r['degradation'] for r in cost_results])
    cost_pass = sn_pos >= 3 and mean_deg < 30.0

    print(f"\n  [COST-AWARE SUMMARY]")
    print(f"    Sharpe_net > 0 in: {sn_pos}/4 folds (need >=3)")
    print(f"    Mean Sharpe degradation: {mean_deg:.1f}% (need <30%)")
    print(f"    COST-AWARE VERDICT: {'PASS' if cost_pass else 'FAIL'}")

    # ============================================================
    # SECTION 7: V20.5 FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 7: V20.5 FINAL VERDICT")
    print("=" * 70)

    verdicts_v20_5 = {
        'ema_smoothing': smoothing_pass,
        'cost_aware': cost_pass,
    }

    pc = sum(1 for v in verdicts_v20_5.values() if v)

    for tn, tv in verdicts_v20_5.items():
        print(f"  {tn:20s} -> {'PASS' if tv else 'FAIL'}")

    if pc == 2:
        fv = "PAPER TRADING READY"
    elif pc == 1:
        fv = "PARTIAL - NEEDS ADJUSTMENT"
    else:
        fv = "NOT READY"

    print(f"\n  *** Passed {pc}/2 V20.5 tests ***")
    print(f"  *** FINAL VERDICT: {fv} ***")

    print(f"\n  --- PER-FOLD NET PERFORMANCE ---")
    print(f"  {'Fold':<6} {'Sharpe_net':<12} {'Return_net':<14} {'MaxDD_net':<12} {'Cost_drag':<12}")
    for fi, r in enumerate(cost_results):
        print(f"  {fi:<6} {r['sharpe_net']:<12.3f} {r['return_net']:<14.2%} {r['max_dd_net']:<12.2%} {r['degradation']:<12.1f}%")

    print("=" * 70)

    results = {
        'version': 'V20.5',
        'ema_alpha': EMA_ALPHA,
        'fee': FEE,
        'slippage': SLIPPAGE,
        'smoothing_pass': smoothing_pass,
        'cost_pass': cost_pass,
        'mean_turnover_reduction': float(mean_t_red),
        'mean_sharpe_change': float(mean_s_chg),
        'cost_results': cost_results,
        'final_verdict': fv,
    }

    with open('v20_5_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] Results JSON: v20_5_results.json")

    tg_msg = f"""<b>V20.5 Production Prep</b>

<b>Verdict:</b> <code>{fv}</code>

<b>EMA Smoothing (alpha={EMA_ALPHA}):</b>
Turnover reduction: {mean_t_red:.1f}%
Sharpe change: {mean_s_chg:+.1f}%
Result: {'PASS' if smoothing_pass else 'FAIL'}

<b>Cost-Aware (fee={FEE}, slip={SLIPPAGE}):</b>
Sharpe_net > 0: {sn_pos}/4 folds
Mean degradation: {mean_deg:.1f}%
Result: {'PASS' if cost_pass else 'FAIL'}"""

    tg_send(tg_msg, topic_id=TELEGRAM['topic_971'])
    print("[TELEGRAM] Results sent to topic 971")



if __name__ == '__main__':
    main()
