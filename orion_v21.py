#!/home/ubuntu/orion/venv/bin/python3
# -*- coding: utf-8 -*-
"""
ORION V21.0 — Multi-Asset Paper Trading Engine with ETH Edge Detection
V20.9 base + ETH edge layer (LightGBM edge detection model)
ETH: edge-gated (EDGE_ON/OFF). BTC/SOL: V20.9 puro.

Author: Claude Code (Anthropic)
Date: 2026-04-29
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
from scipy.stats import pearsonr
import joblib
import lightgbm as lgb

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# ============================================================
# ASSET CONFIGURATION (V21.0 — MULTI-ASSET + ETH EDGE)
# ============================================================
ASSETS = {
    "ETH/USDT": {
        "mapping": "baseline",
        "regime_guard": False,
        "target_vol": 0.15,
        "vol_window": 168,
        "position_cap": 0.5,
        "virtual_balance": 10000.0,
    },
    "BTC/USDT": {
        "mapping": "baseline",
        "regime_guard": False,
        "target_vol": 0.12,
        "vol_window": 252,
        "position_cap": 0.5,
        "virtual_balance": 10000.0,
    },
    "SOL/USDT": {
        "mapping": "confidence_weighted",
        "regime_guard": True,
        "target_vol": 0.18,
        "vol_window": 168,
        "position_cap": 0.5,
        "virtual_balance": 10000.0,
    },
}

MAX_PORTFOLIO_EXPOSURE = 0.8
DD_FLOOR = -0.50

MODEL_PATH = '/home/ubuntu/orion/model_v20_6_1.pkl'
LOG_PATH = '/home/ubuntu/orion/paper_trading_log_v21.csv'
STATE_PATH = '/home/ubuntu/orion/state_v21.json'
STATE_TEMP_PATH = '/home/ubuntu/orion/state_v21.tmp'

# V21 Edge Detection — ETH only
V21_MODEL_PATH = '/home/ubuntu/orion/lab/v21_lgbm_model.pkl'
V21_EDGE_THRESHOLD = 0.2005  # percentil 15% del backtest v2
V21_ETH_MIN_CANDLES = 700    # necesita rolling(500) para features V21

EXCHANGE = 'okx'
TIMEFRAME = '4H'

SENSITIVITY = 1.0
MAX_REDUCTION = 0.7
VOL_LOOKBACK = 42

FEE_BPS = 5
SLIP_BPS = 5
TOTAL_FRICTION = (FEE_BPS + SLIP_BPS) / 10000.0

CANDLE_SCHEDULE_HOURS = [0, 4, 8, 12, 16, 20]
CANDLE_DELAY_SECONDS = 150

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '-1003505760554')
TOPIC_ALERTS = int(os.getenv('TOPIC_ALERTS', '973'))
TOPIC_RESULTS = int(os.getenv('TOPIC_RESULTS', '971'))

MAX_DD_GUARDRAIL = -0.40
DATA_FEED_MAX_AGE_MINUTES = 30

API_TIMEOUT_MS = 30000
API_MAX_RETRIES = 3

DRIFT_THRESHOLD = 0.5
DRIFT_CONSECUTIVE_BARS = 3

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

def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4h', n_candles=200):
    base_url = "https://www.okx.com"
    endpoint = "/api/v5/market/history-candles"
    params = {'instId': symbol, 'bar': timeframe, 'limit': str(min(100, n_candles))}
    all_candles = []
    remaining = n_candles

    for attempt in range(API_MAX_RETRIES):
        try:
            while remaining > 0:
                params['limit'] = str(min(100, remaining))
                r = requests.get(base_url + endpoint, params=params, timeout=API_TIMEOUT_MS/1000)
                data = r.json()
                if data.get('code') != '0':
                    raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
                candles = data.get('data', [])
                if not candles:
                    break
                all_candles.extend(candles)
                remaining -= len(candles)
                if candles:
                    params['after'] = candles[-1][0]
                time.sleep(0.1)
            break
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                print(f"[OKX] Retry {attempt+1}/{API_MAX_RETRIES} for {symbol}: {e}")
                time.sleep(2)
            else:
                raise RuntimeError(f"Failed after {API_MAX_RETRIES} attempts: {e}")

    if not all_candles:
        raise RuntimeError("No data fetched from OKX")

    df = pd.DataFrame(all_candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote', 'confirm'])
    df['ts'] = pd.to_datetime(df['ts'].astype(np.int64), unit='ms', utc=True)
    df = df.sort_values('ts').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col])

    return df[['ts', 'open', 'high', 'low', 'close', 'vol']].rename(columns={'ts': 'timestamp'})

def build_features(df):
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret
    eps = 1e-8

    df['ret_4h'] = log_ret.rolling(6).sum().shift(1)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + eps)
    rsi = 100 - 100 / (1 + rs)
    df['rsi_norm'] = (rsi / 100).fillna(0.5).shift(1)

    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_pos = (df['close'] - lower) / (upper - lower + eps)
    df['bb_position'] = bb_pos.fillna(0.5).shift(1)

    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9).mean()
    macd_norm = (macd - macd_sig) / (df['close'] * 0.01 + eps)
    df['macd_norm'] = macd_norm.fillna(0).shift(1)

    df['ret_4h_lag1'] = df['ret_4h'].shift(1)
    df['ret_4h_lag2'] = df['ret_4h'].shift(2)

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_norm'] = (atr / df['close']).fillna(0).shift(1)

    rv_6 = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_mean = rv_6.rolling(180).mean().shift(1)
    rv_std = rv_6.rolling(180).std().shift(1)
    df['vol_zscore'] = ((rv_6 - rv_mean) / (rv_std + eps)).fillna(0).shift(1)

    vol_rank = rv_6.rolling(180).rank(pct=True).shift(1)
    df['vol_regime'] = pd.cut(vol_rank, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2]).astype(float).fillna(1).shift(1)

    df['ret_8h'] = log_ret.rolling(12).sum().shift(1)
    df['ret_24h'] = log_ret.rolling(24).sum().shift(1)

    ema_20 = df['close'].ewm(span=20).mean()
    slope = (ema_20 - ema_20.shift(1)) / (ema_20.shift(1) + eps)
    df['ema_slope'] = slope.fillna(0).shift(1)

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

    roll_max = df['close'].rolling(100).max()
    dd = (df['close'] - roll_max) / (roll_max + eps)
    df['drawdown_market'] = dd.fillna(0).shift(1)

    mom_short = df['close'] / df['close'].shift(6) - 1.0
    mom_long = df['close'] / df['close'].shift(24) - 1.0
    df['tf_coherence'] = (np.sign(mom_short) * np.sign(mom_long) *
                          np.minimum(np.abs(mom_short), np.abs(mom_long))).fillna(0)

    ema_200 = df['close'].ewm(span=200).mean()
    df['dist_ema200'] = ((df['close'] - ema_200) / (atr + eps)).fillna(0).shift(1)

    hl_ratio = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt(
        (hl_ratio ** 2).rolling(42).mean().shift(1) / (4 * np.log(2))
    ).fillna(0)

    rv_1d = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    rv_7d = log_ret.rolling(42).std().shift(1) * np.sqrt(6)
    df['realized_vol_1d'] = rv_1d
    df['realized_vol_7d'] = rv_7d
    df['vol_compression'] = rv_1d / (rv_7d + eps)

    gk = 0.5 * (np.log(df['high'] / df['low'])) ** 2 - (2*np.log(2)-1) * (np.log(df['close']/df['open']))**2
    df['garman_klass_vol'] = np.sqrt(gk.rolling(6).mean().shift(1).clip(lower=0))

    vol_rank_pct = rv_7d.rolling(180).rank(pct=True).shift(1)
    df['vol_regime_rank'] = vol_rank_pct.fillna(0.5)

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

def compute_v21_features(df):
    """
    Calcula las 29 features del modelo V21 edge detection.
    Copiada de v21_lgbm_training.py compute_features() + add_lags().
    """
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * df['std20']
    df['bb_lower'] = df['sma20'] - 2 * df['std20']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
    df['bb_width_hist'] = df['bb_width'].rolling(500, min_periods=120).quantile(0.2)
    df['squeeze_duration'] = (df['bb_width'] < df['bb_width_hist']).astype(int)
    df['squeeze_duration'] = df['squeeze_duration'].groupby((df['squeeze_duration'] != df['squeeze_duration'].shift()).cumsum()).cumsum()
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr14'] = df['tr'].rolling(14).mean()
    df['atr50'] = df['tr'].rolling(50).mean()
    df['atr_compression'] = df['atr14'] / df['atr50']
    df['realized_vol_20d'] = df['log_return'].rolling(120).std() * np.sqrt(6)
    df['realized_vol_60d'] = df['log_return'].rolling(360).std() * np.sqrt(6)
    df['vol_regime'] = df['realized_vol_20d'] / df['realized_vol_60d']
    df['volume_sma_120'] = df['volume'].rolling(120).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_120']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    plus_dm_raw = df['high'] - df['high'].shift(1)
    minus_dm_raw = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0)
    df['minus_dm'] = np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr14'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr14'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['trend_strength'] = df['dx'].rolling(14).mean()
    df['price_slope_20'] = df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['rsi_slope_20'] = df['rsi_14'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else 0)
    df['momentum_divergence'] = np.sign(df['price_slope_20']) - np.sign(df['rsi_slope_20'])
    df['vol_of_vol'] = df['atr14'].rolling(20).std()

    # Lag features
    lag_features = ['bb_width', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14']
    lags = [1, 3, 6]
    for feat in lag_features:
        for lag in lags:
            df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

    # ROC features
    for feat in lag_features:
        df[f'{feat}_roc'] = df[feat] / df[feat].shift(1) - 1

    return df

V21_FEATURE_COLS = [
    'bb_width', 'squeeze_duration', 'atr_compression', 'vol_regime',
    'volume_ratio', 'rsi_14', 'trend_strength', 'momentum_divergence',
    'vol_of_vol',
    'bb_width_lag1', 'bb_width_lag3', 'bb_width_lag6',
    'atr_compression_lag1', 'atr_compression_lag3', 'atr_compression_lag6',
    'vol_regime_lag1', 'vol_regime_lag3', 'vol_regime_lag6',
    'volume_ratio_lag1', 'volume_ratio_lag3', 'volume_ratio_lag6',
    'rsi_14_lag1', 'rsi_14_lag3', 'rsi_14_lag6',
    'bb_width_roc', 'atr_compression_roc', 'vol_regime_roc',
    'volume_ratio_roc', 'rsi_14_roc'
]

def calculate_position(proba_high, asset_config, asset_state, log_returns_series):
    realized_vol = log_returns_series.rolling(window=asset_config["vol_window"]).std() * np.sqrt(6 * 365)

    if len(realized_vol) > 0 and pd.notna(realized_vol.iloc[-1]):
        rv_current = float(realized_vol.iloc[-1])
    else:
        rv_current = asset_config["target_vol"]

    vol_ratio = asset_config["target_vol"] / (rv_current + 1e-8)
    vol_ratio = np.clip(vol_ratio, 0.5, 2.0)

    mapping = asset_config.get("mapping", "baseline")
    if mapping == "confidence_weighted":
        raw_position = proba_high * vol_ratio
    else:
        raw_position = (1 - proba_high) * vol_ratio
    position = np.clip(raw_position, 0, asset_config["position_cap"])

    prev_position = asset_state.get("prev_position", 0.0)
    position = 0.7 * position + 0.3 * prev_position

    current_dd = asset_state.get("current_dd", 0.0)
    dd_scalar = np.clip(1.0 + current_dd / abs(DD_FLOOR), 0.1, 1.0)
    position = position * dd_scalar

    return float(position), {
        "realized_vol": rv_current,
        "vol_ratio": vol_ratio,
        "dd_scalar": dd_scalar,
        "position_before_dd": 0.7 * raw_position + 0.3 * prev_position
    }

def apply_portfolio_exposure_cap(results_dict):
    total_exposure = sum(results_dict[asset]["position"] for asset in results_dict)

    if total_exposure > MAX_PORTFOLIO_EXPOSURE:
        scale_factor = MAX_PORTFOLIO_EXPOSURE / total_exposure
        for asset in results_dict:
            results_dict[asset]["position"] *= scale_factor
            results_dict[asset]["exposure_scaled"] = True
        print(f"[PORTFOLIO] Exposure capped: {total_exposure:.2%} → {MAX_PORTFOLIO_EXPOSURE:.0%} (×{scale_factor:.2f})")
    else:
        for asset in results_dict:
            results_dict[asset]["exposure_scaled"] = False

def update_drift_counter(asset, asset_state, features_df, raw_features, proba_high):
    ic_history = asset_state.get("ic_history", [])
    ic_history.append({
        "proba": proba_high,
        "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'
    })

    ic_history = ic_history[-200:]
    asset_state["ic_history"] = ic_history

    drift_counter = asset_state.get("drift_counter", 0)

    if len(ic_history) >= 50:
        recent_probas = [entry["proba"] for entry in ic_history[-50:]]
        proba_std = np.std(recent_probas) if len(recent_probas) > 1 else 0.0
        expected_std = 0.25

        if proba_std < expected_std * DRIFT_THRESHOLD:
            drift_counter += 1
        else:
            drift_counter = 0

    asset_state["drift_counter"] = drift_counter

    if drift_counter >= DRIFT_CONSECUTIVE_BARS:
        return True, drift_counter
    else:
        return False, drift_counter

def load_state():
    if not os.path.exists(STATE_PATH):
        print("[STATE] No state found, starting fresh")
        state = {
            "version": "21.0",
            "last_update": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z',
            "assets": {},
            "portfolio": {"total_balance": 0.0, "portfolio_dd": 0.0, "portfolio_peak": 0.0}
        }
        for symbol in ASSETS:
            state["assets"][symbol] = {
                "virtual_balance": ASSETS[symbol]["virtual_balance"],
                "peak_balance": ASSETS[symbol]["virtual_balance"],
                "current_dd": 0.0,
                "prev_position": 0.0,
                "prev_close": None,
                "drift_counter": 0,
                "ic_history": [],
                "bar_count": 0
            }
        total = sum(ASSETS[s]["virtual_balance"] for s in ASSETS)
        state["portfolio"]["total_balance"] = total
        state["portfolio"]["portfolio_peak"] = total
        return state

    try:
        with open(STATE_PATH, 'r') as f:
            state = json.load(f)

        if "assets" not in state or "portfolio" not in state:
            raise ValueError("Invalid state structure")

        for symbol in ASSETS:
            if symbol not in state["assets"]:
                print(f"[STATE] Adding missing asset: {symbol}")
                state["assets"][symbol] = {
                    "virtual_balance": ASSETS[symbol]["virtual_balance"],
                    "peak_balance": ASSETS[symbol]["virtual_balance"],
                    "current_dd": 0.0,
                    "prev_position": 0.0,
                    "prev_close": None,
                    "drift_counter": 0,
                    "ic_history": [],
                    "bar_count": 0
                }

        with open(STATE_PATH + '.backup', 'w') as f:
            json.dump(state, f, indent=2)

        print(f"[STATE] Loaded V{state.get('version', 'unknown')} — {len(state['assets'])} assets")
        return state
    except Exception as e:
        print(f"[STATE] Failed to load, starting fresh: {e}")
        state_fallback = {
            "version": "21.0",
            "last_update": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z',
            "assets": {},
            "portfolio": {"total_balance": 0.0, "portfolio_dd": 0.0, "portfolio_peak": 0.0}
        }
        for symbol in ASSETS:
            state_fallback["assets"][symbol] = {
                "virtual_balance": ASSETS[symbol]["virtual_balance"],
                "peak_balance": ASSETS[symbol]["virtual_balance"],
                "current_dd": 0.0,
                "prev_position": 0.0,
                "prev_close": None,
                "drift_counter": 0,
                "ic_history": [],
                "bar_count": 0
            }
        total = sum(ASSETS[s]["virtual_balance"] for s in ASSETS)
        state_fallback["portfolio"]["total_balance"] = total
        state_fallback["portfolio"]["portfolio_peak"] = total
        return state_fallback

def save_state(state):
    try:
        with open(STATE_TEMP_PATH, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(STATE_TEMP_PATH, STATE_PATH)

        with open(STATE_PATH + '.backup', 'w') as f:
            json.dump(state, f, indent=2)

        total = state["portfolio"]["total_balance"]
        print(f"[STATE] Saved: portfolio=${total:,.2f}")
    except Exception as e:
        print(f"[STATE] Save failed: {e}")

def init_log():
    if not os.path.exists(LOG_PATH):
        header = ','.join([
            'timestamp', 'asset', 'price_close', 'proba_high',
            'position_size', 'pnl', 'virtual_balance', 'current_dd',
            'exposure_scaled', 'drift_reduced', 'latency_ms',
            'realized_vol', 'vol_ratio', 'dd_scalar', 'features_hash',
            'edge_state', 'edge_proba'
        ]) + '\n'
        with open(LOG_PATH, 'w') as f:
            f.write(header)
        print(f"[LOG] Created: {LOG_PATH}")

def log_bar(asset, data_dict):
    line = ','.join([
        data_dict['timestamp'],
        asset,
        f"{data_dict['price_close']:.6f}",
        f"{data_dict['proba_high']:.6f}",
        f"{data_dict['position_size']:.6f}",
        f"{data_dict['pnl']:.2f}",
        f"{data_dict['virtual_balance']:.2f}",
        f"{data_dict['current_dd']:.6f}",
        str(data_dict.get('exposure_scaled', False)),
        str(data_dict.get('drift_reduced', False)),
        str(data_dict['latency_ms']),
        f"{data_dict.get('realized_vol', 0.0):.6f}",
        f"{data_dict.get('vol_ratio', 0.0):.6f}",
        f"{data_dict.get('dd_scalar', 0.0):.6f}",
        data_dict['features_hash'],
        data_dict.get('edge_state', 'N/A'),
        f"{data_dict.get('edge_proba', 0.0):.6f}"
    ]) + '\n'
    with open(LOG_PATH, 'a') as f:
        f.write(line)

def execution_cycle():
    start_time = time.time()
    state = load_state()

    if state.get('portfolio', {}).get('trading_halted', False):
        print("[HALT] Portfolio trading is halted — skipping execution")
        return

    print(f"\n{'='*70}")
    print(f"ORION V21.0 — Multi-Asset Paper Trading + ETH Edge Detection")
    print(f"UTC: {datetime.now(timezone.utc).replace(tzinfo=None).isoformat()} | Step: {state['assets']['ETH/USDT']['bar_count'] + 1}")
    print(f"Assets: {', '.join(ASSETS.keys())}")
    print(f"{'='*70}")

    print("[MODEL] Loading LightGBM V20.6.1...")
    model = joblib.load(MODEL_PATH)

    # V21 Edge Detection model
    v21_model = None
    try:
        import pickle
        with open(V21_MODEL_PATH, 'rb') as f:
            v21_models = pickle.load(f)
        v21_model = v21_models.get('ETH/USDT')
        if v21_model is not None:
            print(f"[V21] ETH edge model loaded from {V21_MODEL_PATH}")
        else:
            print(f"[V21] WARNING: ETH model not found in pickle — edge detection DISABLED")
    except Exception as e:
        print(f"[V21] WARNING: Could not load V21 model: {e} — edge detection DISABLED")

    results = {}

    for symbol, asset_config in ASSETS.items():
        asset_start = time.time()
        asset_state = state["assets"][symbol]

        try:
            print(f"\n[{symbol}] Starting processing...")

            # V21: ETH necesita más barras para features de edge detection
            n_candles = V21_ETH_MIN_CANDLES if symbol == 'ETH/USDT' else 200
            df = fetch_okx_ohlcv(symbol=symbol.replace("/", "-"), timeframe=TIMEFRAME, n_candles=n_candles)
            df = df.iloc[:-1]
            print(f"[{symbol}] Got {len(df)} candles")

            last_candle_ts = df['timestamp'].iloc[-1]
            now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None), tz='UTC')
            expected_close = last_candle_ts + pd.Timedelta(hours=4)
            age_minutes = (now - expected_close).total_seconds() / 60

            if age_minutes < 0:
                print(f"[{symbol}] Candle not yet closed (opens in {abs(age_minutes):.0f} min) — SKIPPING")
                results[symbol] = {"error": "candle_not_closed", "skipped": True}
                continue
            elif age_minutes > DATA_FEED_MAX_AGE_MINUTES:
                print(f"[{symbol}] Candle too old ({age_minutes:.0f} min) — SKIPPING")
                results[symbol] = {"error": "stale_data", "skipped": True}
                continue

            df, raw_features = build_features(df)
            latest_features = df[raw_features].iloc[-1].values.astype(np.float64)
            features_hash = hashlib.md5(latest_features.tobytes()).hexdigest()[:8]

            proba_high = float(model.predict(latest_features.reshape(1, -1))[0])

            # --- V21 Edge Detection (SOLO ETH) ---
            edge_state = 'N/A'
            edge_proba = 0.0

            if symbol == 'ETH/USDT' and v21_model is not None:
                try:
                    # Calcular features V21
                    df_v21 = compute_v21_features(df.copy())

                    if df_v21 is not None and len(df_v21) > 0:
                        latest_v21 = df_v21[V21_FEATURE_COLS].iloc[-1].values.astype(np.float64)

                        # Verificar que no hay NaN
                        if not np.any(np.isnan(latest_v21)):
                            edge_proba = float(v21_model.predict(latest_v21.reshape(1, -1))[0])

                            if edge_proba >= V21_EDGE_THRESHOLD:
                                edge_state = 'EDGE_ON'
                                # No cambiar nada — V20.9 opera normal
                                print(f'[{symbol}] V21 EDGE_ON (proba={edge_proba:.4f} >= {V21_EDGE_THRESHOLD})')
                            else:
                                edge_state = 'EDGE_OFF'
                                # OVERRIDE: position = 0
                                print(f'[{symbol}] V21 EDGE_OFF (proba={edge_proba:.4f} < {V21_EDGE_THRESHOLD})')
                        else:
                            edge_state = 'ERROR_NAN'
                            print(f'[{symbol}] V21 features contain NaN — falling back to V20.9')
                    else:
                        edge_state = 'ERROR_EMPTY'
                        print(f'[{symbol}] V21 features empty — falling back to V20.9')
                except Exception as e:
                    edge_state = f'ERROR_{type(e).__name__}'
                    print(f'[{symbol}] V21 edge detection failed: {e} — falling back to V20.9')

            log_ret_series = pd.Series(df['log_ret'].values)

            position, sizing_meta = calculate_position(
                proba_high, asset_config, asset_state, log_ret_series
            )

            # --- V21 Edge Gate: OVERRIDE position to 0 if EDGE_OFF ---
            if symbol == 'ETH/USDT' and edge_state == 'EDGE_OFF':
                position = 0.0
                # Mantener prev_position para el smoothing (0.7/0.3)
                # El smoothing natural lo bajará gradualmente

            current_price = float(df['close'].iloc[-1])
            prev_price = asset_state.get("prev_close")
            prev_position = asset_state.get("prev_position", 0.0)
            virtual_balance = asset_state.get("virtual_balance", asset_config["virtual_balance"])

            pnl = 0.0
            if prev_price is not None and prev_price > 0:
                price_change_pct = (current_price - prev_price) / prev_price
                position_usd = virtual_balance * prev_position
                raw_pnl = position_usd * price_change_pct
                delta_position = abs(position - prev_position)
                cycle_fee = virtual_balance * delta_position * TOTAL_FRICTION
                net_pnl = raw_pnl - cycle_fee
                virtual_balance += net_pnl
                pnl = net_pnl

            peak_balance = asset_state.get("peak_balance", virtual_balance)
            virtual_balance = max(virtual_balance, 0.0)
            peak_balance = max(peak_balance, virtual_balance)
            current_dd = (virtual_balance - peak_balance) / peak_balance if peak_balance > 0 else 0.0

            drift_detected, drift_counter = update_drift_counter(
                symbol, asset_state, df, raw_features, proba_high
            )
            if drift_detected:
                position *= 0.5
                print(f"[{symbol}] DRIFT DETECTED — position reduced 50% (counter={drift_counter})")
                tg_send(f"⚠️ DRIFT DETECTED en {symbol} — counter={drift_counter} — posición reducida 50%", topic_id=TOPIC_ALERTS)
            asset_state["virtual_balance"] = virtual_balance
            asset_state["peak_balance"] = peak_balance
            asset_state["current_dd"] = current_dd
            asset_state["prev_close"] = current_price
            asset_state["prev_position"] = position
            asset_state["bar_count"] += 1
            asset_state["drift_counter"] = drift_counter

            if symbol == 'ETH/USDT':
                asset_state['last_edge_state'] = edge_state
                asset_state['last_edge_proba'] = edge_proba

            results[symbol] = {
                "price": current_price,
                "proba": proba_high,
                "position": position,
                "pnl": pnl,
                "virtual_balance": virtual_balance,
                "current_dd": current_dd,
                "drift_counter": drift_counter,
                "drift_reduced": drift_detected,
                "exposure_scaled": False,
                "features_hash": features_hash,
                "realized_vol": sizing_meta["realized_vol"],
                "vol_ratio": sizing_meta["vol_ratio"],
                "dd_scalar": sizing_meta["dd_scalar"],
                "skipped": False,
                "edge_state": edge_state,
                "edge_proba": edge_proba,
            }

            log_bar(symbol, {
                'timestamp': datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z',
                'price_close': current_price,
                'proba_high': proba_high,
                'position_size': position,
                'pnl': pnl,
                'virtual_balance': virtual_balance,
                'current_dd': current_dd,
                'exposure_scaled': False,
                'drift_reduced': drift_detected,
                'latency_ms': int((time.time() - asset_start) * 1000),
                'realized_vol': sizing_meta["realized_vol"],
                'vol_ratio': sizing_meta["vol_ratio"],
                'dd_scalar': sizing_meta["dd_scalar"],
                'features_hash': features_hash,
                'edge_state': edge_state,
                'edge_proba': edge_proba,
            })

            print(f"[{symbol}] P={proba_high:.3f}, Pos={position*100:.0f}%, Vol={sizing_meta['realized_vol']:.3f}, DD={current_dd*100:.1f}%")

        except Exception as e:
            print(f"[{symbol}] ERROR: {e}")
            results[symbol] = {"error": str(e), "skipped": True}
            continue

    active_results = {s: r for s, r in results.items() if not r.get("skipped", False)}
    pre_cap_exposure = sum(active_results[s]["position"] for s in active_results) if len(active_results) > 0 else 0.0
    if len(active_results) > 0:
        apply_portfolio_exposure_cap(active_results)
        # Actualizar prev_position en state con la posición post-cap
        for symbol in active_results:
            state["assets"][symbol]["prev_position"] = active_results[symbol]["position"]
    post_cap_exposure = sum(active_results[s]["position"] for s in active_results) if len(active_results) > 0 else 0.0
    scale_factor = post_cap_exposure / pre_cap_exposure if pre_cap_exposure > 0 else 1.0

    for symbol in active_results:
        asset_state = state["assets"][symbol]
        active_results[symbol]["drift_active"] = (asset_state["drift_counter"] >= DRIFT_CONSECUTIVE_BARS)

    total_balance = sum(active_results[s]["virtual_balance"] for s in active_results)
    total_peak = sum(state["assets"][s]["peak_balance"] for s in active_results)
    portfolio_dd = (total_peak - total_balance) / total_peak if total_peak > 0 else 0.0

    state["portfolio"]["total_balance"] = total_balance
    state["portfolio"]["portfolio_peak"] = total_peak
    state["portfolio"]["portfolio_dd"] = portfolio_dd

    if portfolio_dd < MAX_DD_GUARDRAIL:
        state["portfolio"]["trading_halted"] = True
        halt_msg = f"🚨 PORTFOLIO DD FLOOR: {portfolio_dd:.2%} < {MAX_DD_GUARDRAIL:.2%} — ALL TRADING HALTED"
        print(halt_msg)
        tg_send(halt_msg, topic_id=TOPIC_ALERTS)
        with open('/home/ubuntu/orion/TRADING_HALTED', 'w') as f:
            f.write(datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z\n')

    state["last_update"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'

    save_state(state)

    total_latency = int((time.time() - start_time) * 1000)

    report_lines = []
    report_lines.append(f"✅ ORION V21.0 | {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M')} UTC")

    for symbol in sorted(ASSETS.keys()):
        if symbol in results:
            r = results[symbol]
            if r.get("skipped", False):
                report_lines.append(f"❌ {symbol}: Skipped — {r.get('error', 'unknown')}")
            else:
                pct_change = ((r["virtual_balance"] - ASSETS[symbol]["virtual_balance"]) / ASSETS[symbol]["virtual_balance"] * 100) if ASSETS[symbol]["virtual_balance"] > 0 else 0.0
                line = f"💰 {symbol}: ${r['virtual_balance']:,.0f} ({pct_change:+.1f}%)\n"
                line += f"   Price: ${r['price']:,.2f} | Proba: {r['proba']:.3f} | Pos: {r['position']*100:.0f}%\n"
                line += f"   DD: {r['current_dd']*100:.1f}% | Vol: {r.get('realized_vol', 0):.3f} | Scale: {r.get('dd_scalar', 1):.2f}"
                if symbol == 'ETH/USDT':
                    edge_info = f" | Edge: {r.get('edge_state', 'N/A')} ({r.get('edge_proba', 0):.4f})"
                    line += edge_info
                if r.get("drift_active", False):
                    line += " | Drift ACTIVE"
                report_lines.append(line)
                if r.get("drift_reduced", False):
                    report_lines.append(f"   ⚠️ DRIFT REDUCED (counter={r['drift_counter']})")

    report_lines.append(f"📊 Portfolio: ${total_balance:,.0f} ({portfolio_dd*100:+.1f}%)")
    report_lines.append(f"   Total Exposure: {pre_cap_exposure*100:.0f}% → {post_cap_exposure*100:.0f}% (scale ×{scale_factor:.2f})")
    report_lines.append(f"   Latency: {total_latency}ms")

    tg_text = "\n".join(report_lines)
    tg_send(tg_text, topic_id=TOPIC_RESULTS)

    print(f"[CYCLE] Completed in {total_latency}ms")
    print(f"  Portfolio balance: ${total_balance:,.2f}, DD: {portfolio_dd*100:.1f}%")

def next_candle_time():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    current_hour = now.hour
    next_hour = None
    for h in CANDLE_SCHEDULE_HOURS:
        if h > current_hour:
            next_hour = h
            break
    if next_hour is None:
        next_hour = CANDLE_SCHEDULE_HOURS[0]
        target = datetime(now.year, now.month, now.day, next_hour, 0, 0) + timedelta(days=1)
    else:
        target = datetime(now.year, now.month, now.day, next_hour, 0, 0)

    target += timedelta(seconds=CANDLE_DELAY_SECONDS)
    return target

def wait_and_run():
    next_run = next_candle_time()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    delay = (next_run - now).total_seconds()
    if delay < 0:
        delay = 60
    print(f"[SCHED] Next run at {next_run.isoformat()} (in {delay/60:.0f} min)")
    time.sleep(delay)
    execution_cycle()

def main():
    print("=" * 70)
    print("ORION V21.0 — Multi-Asset Paper Trading + ETH Edge Detection")
    print("=" * 70)
    print(f"Start time: {datetime.now(timezone.utc).replace(tzinfo=None).isoformat()} UTC")
    print(f"Model: {MODEL_PATH}")
    print(f"V21 Edge Model: {V21_MODEL_PATH}")
    print(f"V21 Edge Threshold: {V21_EDGE_THRESHOLD} (percentil 15%)")
    print(f"V21 ETH Gate: EDGE_ON={'>='}{V21_EDGE_THRESHOLD}, EDGE_OFF={'<'}{V21_EDGE_THRESHOLD}")
    print(f"Log: {LOG_PATH}")
    print(f"State: {STATE_PATH}")
    print(f"Assets: {', '.join(ASSETS.keys())} ({TIMEFRAME})")
    print(f"Telegram: Chat {TELEGRAM_CHAT_ID}, Alerts T{TOPIC_ALERTS}, Results T{TOPIC_RESULTS}")
    print(f"Portfolio Exposure Cap: {MAX_PORTFOLIO_EXPOSURE*100:.0f}%")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    init_log()

    tg_send(f"🚀 ORION V21.0 Multi-Asset + ETH Edge Detection STARTED\nUTC: {datetime.now(timezone.utc).replace(tzinfo=None).isoformat()}\nAssets: {', '.join(ASSETS.keys())}\nETH Edge: threshold={V21_EDGE_THRESHOLD} (top 15%)\nBTC/SOL: V20.9 puro", topic_id=TOPIC_ALERTS)

    while True:
        try:
            wait_and_run()
        except KeyboardInterrupt:
            print("\n[MAIN] Keyboard interrupt — exiting")
            tg_send("🛑 ORION V21.0 STOPPED (keyboard)", topic_id=TOPIC_ALERTS)
            break
        except Exception as e:
            print(f"[MAIN] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == '__main__':
    main()
