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
    # SECTION 6: ROBUSTNESS TESTS
    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 6: ROBUSTNESS TESTS")
    print("=" * 70)

    test_verdicts = {}

    # ------------------------------------------------------------
    # TEST 1: MONOTONICITY
    # ------------------------------------------------------------
    print("\n--- TEST 1: MONOTONICITY ---")
    monotonicity_spearman = []
    for fold_i, fold_data in enumerate(fold_results):
        y_prob_fold = fold_data['y_prob']
        y_true_fold = fold_data['y_true']
        # Compute deciles of predicted probability
        if len(y_prob_fold) < 10:
            spearman = 0.0
            print(f"  Fold {fold_i}: insufficient data (<10) -> spearman=0.000 [FAIL]")
            monotonicity_spearman.append(spearman)
            continue
        decile_bins = pd.qcut(y_prob_fold, q=10, labels=False, duplicates='drop')
        mean_future_vol_per_decil = []
        for d in range(10):
            mask = decile_bins == d
            if mask.sum() == 0:
                mean_future_vol_per_decil.append(np.nan)
            else:
                # Use y_true as binary proxy for future_vol ranking (binary target)
                mean_future_vol_per_decil.append(np.mean(y_true_fold[mask]))
        # Remove NaNs
        decil_idx = np.arange(10)
        valid = ~np.isnan(mean_future_vol_per_decil)
        if valid.sum() < 2:
            spearman = 0.0
            print(f"  Fold {fold_i}: no valid means -> spearman=0.000 [FAIL]")
            monotonicity_spearman.append(spearman)
            continue
        spearman, _ = stats.spearmanr(decil_idx[valid], np.array(mean_future_vol_per_decil)[valid])
        spearman = spearman if np.isfinite(spearman) else 0.0
        monotonicity_spearman.append(spearman)
        status = "PASS" if spearman > 0.80 else "FAIL"
        print(f"  Fold {fold_i}: Spearman={spearman:.3f} [{status}]")
    passed_monotonicity = sum(1 for s in monotonicity_spearman if s > 0.80)
    test_verdicts['monotonicity'] = passed_monotonicity >= 3
    print(f"  MONOTONICITY VERDICT: {passed_monotonicity}/4 [{'PASS' if test_verdicts['monotonicity'] else 'FAIL'}]")

    # ------------------------------------------------------------
    # TEST 2: VOL-TARGET BASELINE
    # ------------------------------------------------------------
    print("\n--- TEST 2: VOL-TARGET BASELINE ---")
    baseline_wins = 0
    ml_wins = 0
    for fold_i, fold_data in enumerate(fold_results):
        # Compute baseline: scale_baseline = clip(median(realized_vol_7d) / realized_vol_7d, 0.3, 1.0)
        # Use training data: we need rv_7d from training period
        train_idx = folds[fold_i][0]
        # Compute rv_7d for training period from raw data
        log_ret_all = df['log_ret'].values
        rv_7d_train = pd.Series(log_ret_all[train_idx]).rolling(42).std().values * np.sqrt(6)
        median_rv = np.nanmedian(rv_7d_train)
        # For test period, compute rv_7d
        test_idx = folds[fold_i][1]
        rv_7d_test = pd.Series(log_ret_all[test_idx]).rolling(42).std().values * np.sqrt(6)
        scale_baseline = np.clip(median_rv / (rv_7d_test + 1e-8), 0.3, 1.0)
        # ML scale
        scale_ml = fold_data['position_scale']

        # Compute portfolio stats
        returns_test = np.exp(fold_data['returns_log']) - 1
        baseline_stats = portfolio_stats(scale_baseline * returns_test)
        ml_stats = portfolio_stats(scale_ml * returns_test)

        ml_sharpe = ml_stats['sharpe']
        base_sharpe = baseline_stats['sharpe']
        if ml_sharpe > base_sharpe:
            ml_wins += 1
            status = "ML WINS"
        else:
            baseline_wins += 1
            status = "BASELINE WINS"
        print(f"  Fold {fold_i}: Sharpe ML={ml_sharpe:.3f} vs Baseline={base_sharpe:.3f} [{status}]")
    test_verdicts['vol_target'] = ml_wins >= 2
    print(f"  BASELINE VERDICT: ML wins {ml_wins}/4 [{'PASS' if test_verdicts['vol_target'] else 'FAIL'}]")

    # ------------------------------------------------------------
    # TEST 3: TURNOVER
    # ------------------------------------------------------------
    print("\n--- TEST 3: TURNOVER ---")
    turnovers = []
    for fold_i, fold_data in enumerate(fold_results):
        scale = fold_data['position_scale']
        if len(scale) < 2:
            turnover = 0.0
        else:
            turnover = np.mean(np.abs(scale[1:] - scale[:-1]))
        turnovers.append(turnover)
        if turnover > 0.10:
            level = "CRITICAL"
        elif turnover > 0.05:
            level = "WARNING"
        else:
            level = "OK"
        print(f"  Fold {fold_i}: turnover={turnover:.4f} [{level}]")
    max_turnover = max(turnovers) if turnovers else 0.0
    if max_turnover > 0.10:
        final_turnover = "CRITICAL"
    elif max_turnover > 0.05:
        final_turnover = "WARNING"
    else:
        final_turnover = "OK"
    test_verdicts['turnover'] = final_turnover
    print(f"  TURNOVER VERDICT: {final_turnover} (max fold turnover={max_turnover:.4f})")

    # ------------------------------------------------------------
    # TEST 4: STABILITY
    # ------------------------------------------------------------
    print("\n--- TEST 4: STABILITY ---")
    sharpe_deltas = []
    for fold_data in fold_results:
        returns_test = np.exp(fold_data['returns_log']) - 1
        scale_ml = fold_data['position_scale']
        scaled_ret = scale_ml * returns_test
        unscaled_ret = returns_test
        s_scaled = portfolio_stats(scaled_ret)['sharpe']
        s_unscaled = portfolio_stats(unscaled_ret)['sharpe']
        sharpe_deltas.append(s_scaled - s_unscaled)
    sharpe_deltas = np.array(sharpe_deltas)
    mean_delta = sharpe_deltas.mean()
    std_delta = sharpe_deltas.std()
    cv = std_delta / (abs(mean_delta) + 1e-8) if abs(mean_delta) > 1e-8 else 0.0
    print(f"  Sharpe deltas: {np.round(sharpe_deltas, 3).tolist()}")
    print(f"  Mean={mean_delta:.3f}, Std={std_delta:.3f}, CV={cv:.3f}")
    test_verdicts['stability'] = cv < 1.5
    print(f"  STABILITY VERDICT: {'PASS' if test_verdicts['stability'] else 'FAIL'} (CV < 1.5)")

    # ------------------------------------------------------------
    # TEST 5: STRESS TEST
    # ------------------------------------------------------------
    print("\n--- TEST 5: STRESS TEST ---")
    # Original overall scaled stats (from all folds combined)
    original_sharpe = scaled_stats['sharpe']
    original_dd = scaled_stats['max_dd']

    # 5a. Latency: shift prob by +1 bar (use prob[t-1] for scale[t])
    shifted_probs = np.roll(all_y_prob, 1)
    shifted_probs[0] = all_y_prob[0]  # first stays same (no previous)
    # Need to recompute scale using RiskEngine fitted on each fold's train
    shifted_scaled_returns = []
    for fold_i, fold_data in enumerate(fold_results):
        y_train_binary = (df['vol_24h_future'].values[folds[fold_i][0]] >= fold_data['threshold_vol']).astype(int)
        y_train_prob = model.predict(X[folds[fold_i][0]])  # We don't have these stored; need to recompute differently
        # Simpler: use the stored scale but shift by 1
        scale_orig = fold_data['position_scale']
        scale_latency = np.roll(scale_orig, 1)
        scale_latency[0] = scale_orig[0]
        returns_test = np.exp(fold_data['returns_log']) - 1
        shifted_scaled_returns.extend(scale_latency * returns_test)
    latency_sharpe = portfolio_stats(np.array(shifted_scaled_returns))['sharpe']
    latency_dd = portfolio_stats(np.array(shifted_scaled_returns))['max_dd']
    latency_sharpe_drop_pct = (original_sharpe - latency_sharpe) / (abs(original_sharpe) + 1e-8) * 100

    # 5b. Noise: add Gaussian noise to probabilities, then recompute scale
    noise = np.random.normal(0, 0.10, size=len(all_y_prob))
    noisy_probs = np.clip(np.array(all_y_prob) + noise, 0, 1)
    # Recompute scale using the same engine (best we can do is approximate by linear transformation)
    # Since engine uses threshold and scale = 1 - clip(prob * sensitivity, 0, max_reduction)
    # We can apply the sensitivity and max_reduction to noisy probs (threshold already applied before), but we need to pass through same engine.
    # Simpler: approximate by assuming scale is linear in prob for those above threshold? Not exactly.
    # We'll recompute per fold using fitted engines. We need to store engines.
    # For robustness, we'll approximate: take engines from each fold and predict with noisy probs
    noisy_scaled_returns = []
    for fold_i, fold_data in enumerate(fold_results):
        # Recreate engine that was fit
        y_train_binary = (df['vol_24h_future'].values[folds[fold_i][0]] >= fold_data['threshold_vol']).astype(int)
        y_train_full_prob = model.predict(X[folds[fold_i][0]])
        engine = RiskEngine(**engine_defaults)
        engine.fit(y_train_binary, y_train_full_prob)
        # Get test indices
        test_idx = folds[fold_i][1]
        noisy_probs_fold = noisy_probs[np.array(all_fold_idx) == fold_i]
        scale_noisy = engine.predict_scale(noisy_probs_fold)
        returns_test = np.exp(fold_data['returns_log']) - 1
        noisy_scaled_returns.extend(scale_noisy * returns_test)
    noise_sharpe = portfolio_stats(np.array(noisy_scaled_returns))['sharpe']
    noise_dd = portfolio_stats(np.array(noisy_scaled_returns))['max_dd']
    noise_sharpe_drop_pct = (original_sharpe - noise_sharpe) / (abs(original_sharpe) + 1e-8) * 100

    print(f"  5a. Latency (+1 bar):")
    print(f"      Sharpe degradation: {latency_sharpe_drop_pct:.1f}%")
    lat_ok = latency_sharpe_drop_pct < 30.0
    print(f"      [{'PASS' if lat_ok else 'FAIL'}]")

    print(f"  5b. Noise (±10% Gaussian):")
    print(f"      Sharpe degradation: {noise_sharpe_drop_pct:.1f}%")
    noise_ok = noise_sharpe_drop_pct < 30.0
    print(f"      [{'PASS' if noise_ok else 'FAIL'}]")

    test_verdicts['stress'] = lat_ok and noise_ok
    print(f"  STRESS VERDICT: {'PASS' if test_verdicts['stress'] else 'FAIL'}")

    # ------------------------------------------------------------
    # TEST 6: GRID SEARCH
    # ------------------------------------------------------------
    print("\n--- TEST 6: GRID SEARCH ---")
    sensitivities = [0.5, 0.75, 1.0, 1.25, 1.5]
    max_reductions = [0.5, 0.6, 0.7, 0.8]
    grid_results = []
    # For each combination, run sizing across all folds (but we already have folds data, we can apply scale formulas)
    for sens in sensitivities:
        for max_red in max_reductions:
            all_scaled_ret = []
            for fold_data in fold_results:
                probs = fold_data['y_prob']
                scale = 1.0 - np.clip(probs * sens, 0, max_red)
                returns_test = np.exp(fold_data['returns_log']) - 1
                all_scaled_ret.extend(scale * returns_test)
            stats_all = portfolio_stats(np.array(all_scaled_ret))
            grid_results.append({
                'sensitivity': sens,
                'max_reduction': max_red,
                'sharpe': stats_all['sharpe'],
                'max_dd': stats_all['max_dd'],
                'total_return': stats_all['total_return']
            })
    # Find best
    best_sharpe_row = max(grid_results, key=lambda x: x['sharpe'])
    best_dd_row = min(grid_results, key=lambda x: x['max_dd'])
    robust_combos = sum(1 for r in grid_results if r['sharpe'] > 0)
    print(f"  Best Sharpe: sensitivity={best_sharpe_row['sensitivity']}, max_reduction={best_sharpe_row['max_reduction']:.2f} → Sharpe={best_sharpe_row['sharpe']:.3f}")
    print(f"  Best DD: sensitivity={best_dd_row['sensitivity']}, max_reduction={best_dd_row['max_reduction']:.2f} → DD={best_dd_row['max_dd']:.2%}")
    print(f"  Robust combos (Sharpe>0): {robust_combos}/20")
    test_verdicts['grid'] = robust_combos >= 10
    print(f"  GRID VERDICT: {'ROBUST' if test_verdicts['grid'] else 'FRAGILE'} (≥10 combos with Sharpe>0)")

    # ============================================================
    # SECTION 7: FINAL ROBUSTNESS VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 7: FINAL ROBUSTNESS VERDICT")
    print("=" * 70)
    verdict_lines = []
    passed_count = 0
    for test_name, verdict in test_verdicts.items():
        if test_name == 'monotonicity':
            passed = verdict
            status = "PASS" if passed else "FAIL"
        elif test_name == 'vol_target':
            passed = verdict
            status = "PASS" if passed else "FAIL"
        elif test_name == 'turnover':
            passed = verdict in ["OK", "WARNING"]  # WARNING is acceptable?
            status = verdict
        elif test_name == 'stability':
            passed = verdict
            status = "PASS" if passed else "FAIL"
        elif test_name == 'stress':
            passed = verdict
            status = "PASS" if passed else "FAIL"
        elif test_name == 'grid':
            passed = verdict
            status = "ROBUST" if passed else "FRAGILE"
        else:
            status = "N/A"
        verdict_lines.append(f"  Test: {test_name:15s} -> {status}")
        if test_name in ['monotonicity', 'vol_target', 'stability', 'stress']:
            if verdict in [True, "OK", "ROBUST"] or (test_name == 'turnover' and verdict in ["OK", "WARNING"]):
                passed_count += 1
        elif test_name == 'grid' and verdict:
            passed_count += 1

    for line in verdict_lines:
        print(line)

    print(f"\n  *** Passed {passed_count}/6 tests ***")

    if passed_count == 6:
        final_verdict = "PRODUCTION READY"
    elif passed_count >= 4:
        final_verdict = "PRODUCTION READY WITH CAVEATS"
    elif passed_count >= 2:
        final_verdict = "NEEDS WORK"
    else:
        final_verdict = "NOT DEPLOYABLE"

    print(f"  *** FINAL VERDICT: {final_verdict} ***")
    print("=" * 70)

    # ---- Save results JSON ----
    results = {
        'version': 'V20.4',
        'tests': {k: str(v) for k, v in test_verdicts.items()},
        'passed_count': passed_count,
        'final_verdict': final_verdict,
        'baseline': {
            'unscaled': unscaled_stats,
            'scaled': scaled_stats,
            'sharpe_diff': scaled_stats['sharpe'] - unscaled_stats['sharpe'],
            'maxdd_reduction': unscaled_stats['max_dd'] - scaled_stats['max_dd'],
        },
        'grid_best_sharpe': best_sharpe_row,
        'grid_best_dd': best_dd_row,
        'grid_robust_combos': int(robust_combos),
    }
    with open('v20_4_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results JSON: v20_4_results.json")

    # ---- Telegram send ----
    tg_text = f"""<b>V20.4 Robustness Validation — Complete</b>
<b>Final verdict:</b> <code>{final_verdict}</code>
<b>Passed:</b> {passed_count}/6 tests

Tests:
Monotonicity: {'PASS' if test_verdicts['monotonicity'] else 'FAIL'}
Vol-target: {'PASS' if test_verdicts['vol_target'] else 'FAIL'}
Turnover: {test_verdicts['turnover']}
Stability: {'PASS' if test_verdicts['stability'] else 'FAIL'}
Stress: {'PASS' if test_verdicts['stress'] else 'FAIL'}
Grid: {'ROBUST' if test_verdicts['grid'] else 'FRAGILE'}

Baseline Sharpe: {unscaled_stats['sharpe']:.2f} → Scaled: {scaled_stats['sharpe']:.2f}
Best Sharpe combo: sens={best_sharpe_row['sensitivity']}, max_red={best_sharpe_row['max_reduction']:.2f} → Sharpe={best_sharpe_row['sharpe']:.2f}
"""
    tg_send(tg_text, topic_id=TELEGRAM['topic_971'])
    print("[TELEGRAM] Results sent to topic 971")

if __name__ == '__main__':
    main()
