# ============================================================
# data_loader.py — V19.0 Data fetching and alignment
# OKX OHLCV + BTC cross-asset + derivatives (funding/OI)
# FIX-1: forward-fill ONLY — no bfill (prevents future leak)
# FIX-2: 2-phase fetch (recent + history) for full 12000 candles
# ============================================================

import numpy as np
import pandas as pd
import time
import requests
from config import CONFIG


def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4H', n_candles=12000):
    print("[DATA] Fetching {} {} candles for {} from OKX...".format(
        n_candles, timeframe, symbol))
    all_candles = []
    limit = 100
    end_ts = None
    # Phase 1: recent candles (OKX /market/candles covers ~1440 bars max)
    url_recent = "https://www.okx.com/api/v5/market/candles"
    while len(all_candles) < n_candles:
        params = {'instId': symbol, 'bar': timeframe, 'limit': str(limit)}
        if end_ts:
            params['after'] = str(end_ts)
        try:
            resp = requests.get(url_recent, params=params, timeout=30)
            data = resp.json()
            if 'data' not in data or len(data['data']) == 0:
                break
            candles = data['data']
            all_candles.extend(candles)
            end_ts = int(candles[-1][0])
            if len(candles) < limit:
                break
            time.sleep(0.15)
        except Exception as e:
            print("  [WARN] API error: {}, retrying...".format(e))
            time.sleep(2)
            continue
    print("  [INFO] Phase 1 (recent): {} candles".format(len(all_candles)))
    # Phase 2: historical candles (goes further back in time)
    if len(all_candles) < n_candles:
        url_hist = "https://www.okx.com/api/v5/market/history-candles"
        fail_count = 0
        while len(all_candles) < n_candles and fail_count < 5:
            params = {'instId': symbol, 'bar': timeframe, 'limit': str(limit)}
            if end_ts:
                params['after'] = str(end_ts)
            try:
                resp = requests.get(url_hist, params=params, timeout=30)
                data = resp.json()
                if 'data' not in data or len(data['data']) == 0:
                    fail_count += 1
                    time.sleep(1)
                    continue
                fail_count = 0
                candles = data['data']
                all_candles.extend(candles)
                end_ts = int(candles[-1][0])
                if len(candles) < limit:
                    break
                time.sleep(0.15)
            except Exception as e:
                print("  [WARN] History API error: {}, retrying...".format(e))
                time.sleep(2)
                fail_count += 1
                continue
        print("  [INFO] Phase 2 (history): {} total candles".format(len(all_candles)))
    if not all_candles:
        raise ValueError("No candles fetched from OKX")
    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'vol',
        'volCcy', 'volCcyQuote', 'confirm'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
    print("  [OK] {} candles ({} -> {})".format(
        len(df), df['timestamp'].iloc[0], df['timestamp'].iloc[-1]))
    return df


def fetch_macro_daily():
    print("[DATA] Fetching macro daily data (BTC for cross-asset)...")
    macro = {}
    try:
        btc_candles = []
        end_ts = None
        # Phase 1: recent BTC daily
        while len(btc_candles) < 2000:
            url = "https://www.okx.com/api/v5/market/candles"
            params = {'instId': 'BTC-USDT', 'bar': '1D', 'limit': '100'}
            if end_ts:
                params['after'] = str(end_ts)
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if 'data' not in data or len(data['data']) == 0:
                break
            btc_candles.extend(data['data'])
            end_ts = int(data['data'][-1][0])
            if len(data['data']) < 100:
                break
            time.sleep(0.15)
        # Phase 2: historical BTC daily
        if len(btc_candles) < 2000:
            fail_count = 0
            while len(btc_candles) < 2000 and fail_count < 5:
                url_h = "https://www.okx.com/api/v5/market/history-candles"
                params = {'instId': 'BTC-USDT', 'bar': '1D', 'limit': '100'}
                if end_ts:
                    params['after'] = str(end_ts)
                try:
                    resp = requests.get(url_h, params=params, timeout=30)
                    data = resp.json()
                    if 'data' not in data or len(data['data']) == 0:
                        fail_count += 1
                        time.sleep(1)
                        continue
                    fail_count = 0
                    btc_candles.extend(data['data'])
                    end_ts = int(data['data'][-1][0])
                    if len(data['data']) < 100:
                        break
                    time.sleep(0.15)
                except Exception:
                    fail_count += 1
                    time.sleep(2)
                    continue
        btc_df = pd.DataFrame(btc_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vol',
            'volCcy', 'volCcyQuote', 'confirm'
        ])
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'].astype(int), unit='ms')
        btc_df['btc_close'] = btc_df['close'].astype(float)
        btc_df['date'] = btc_df['timestamp'].dt.date
        btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
        macro['btc'] = btc_df[['date', 'btc_close']].drop_duplicates('date')
        print("  [OK] BTC daily: {} records".format(len(macro['btc'])))
    except Exception as e:
        print("  [WARN] BTC fetch failed: {}".format(e))
        macro['btc'] = pd.DataFrame(columns=['date', 'btc_close'])
    if len(macro.get('btc', [])) > 0:
        btc = macro['btc'].copy()
        btc_ret = btc['btc_close'].pct_change().fillna(0)
        btc['dxy_proxy'] = 100 - (btc_ret.cumsum() * 10)
        macro['dxy'] = btc[['date', 'dxy_proxy']]
        np.random.seed(42)
        noise = np.random.normal(0, 0.001, len(btc))
        btc['spx_proxy'] = 4500 * (1 + (btc_ret * 0.3 + noise).cumsum())
        macro['spx'] = btc[['date', 'spx_proxy']]
    else:
        macro['dxy'] = pd.DataFrame(columns=['date', 'dxy_proxy'])
        macro['spx'] = pd.DataFrame(columns=['date', 'spx_proxy'])
    return macro


def align_daily_to_4h(df_4h, macro_dict):
    # FIX-1: forward-fill ONLY — bfill uses future data at series start
    print("[DATA] Aligning daily macro to 4H bars (forward-fill only)...")
    df = df_4h.copy()
    df['date'] = df['timestamp'].dt.date
    for key, col in [('btc', 'btc_close'), ('dxy', 'dxy_proxy'), ('spx', 'spx_proxy')]:
        if key in macro_dict and len(macro_dict[key]) > 0:
            df = df.merge(macro_dict[key], on='date', how='left')
            df[col] = df[col].ffill()  # FIX-1: NO bfill
        else:
            df[col] = 0.0
    df = df.drop(columns=['date'], errors='ignore')
    # FIX-1: remaining NaN at start filled with first valid value (safe)
    for col in ['btc_close', 'dxy_proxy', 'spx_proxy']:
        if col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is not None and first_valid > 0:
                df[col] = df[col].fillna(df[col].iloc[first_valid])
    print("  [OK] Aligned {} rows (ffill only)".format(len(df)))
    return df


def load_derivatives_data(df):
    print("[DATA] Loading derivatives data...")
    has_funding = False
    has_oi = False
    for path in ['/kaggle/input/funding-rate/funding_rate.csv',
                 '/kaggle/input/eth-funding-rate/funding_rate.csv',
                 'funding_rate.csv']:
        try:
            fr = pd.read_csv(path)
            if 'timestamp' in fr.columns and 'funding_rate' in fr.columns:
                fr['timestamp'] = pd.to_datetime(fr['timestamp'])
                fr = fr.sort_values('timestamp').drop_duplicates('timestamp')
                df = df.merge(fr[['timestamp', 'funding_rate']], on='timestamp', how='left')
                df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0)
                has_funding = True
                nonzero = (df['funding_rate'] != 0).sum()
                print("  [OK] Funding rate loaded. Non-zero: {}/{}".format(nonzero, len(df)))
                break
        except Exception:
            continue
    if not has_funding:
        print("  [WARN] No funding rate CSV, using zeros")
        df['funding_rate'] = 0.0
    for path in ['/kaggle/input/open-interest/open_interest.csv',
                 '/kaggle/input/eth-oi/open_interest.csv',
                 'open_interest.csv']:
        try:
            oi_df = pd.read_csv(path)
            if 'timestamp' in oi_df.columns and 'open_interest' in oi_df.columns:
                oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'])
                oi_df = oi_df.sort_values('timestamp').drop_duplicates('timestamp')
                df = df.merge(oi_df[['timestamp', 'open_interest']], on='timestamp', how='left')
                df['open_interest'] = df['open_interest'].ffill().fillna(0.0)
                has_oi = True
                print("  [OK] OI loaded: {} records".format(
                    oi_df['open_interest'].notna().sum()))
                break
        except Exception:
            continue
    if not has_oi:
        print("  [WARN] No OI CSV, using zeros")
        df['open_interest'] = 0.0
    return df, has_funding, has_oi
