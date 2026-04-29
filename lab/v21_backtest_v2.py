#!/usr/bin/env python3
# Orion V21.0 Backtest v2 - CORREGIDO
# Single-bar returns + compounding real + cross-asset validation

import os
import pickle
import time
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime
import requests

# Telegram config
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = '-1003505760554'
TOPIC_ID = 972

# Assets
ASSETS = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT']

# Feature config
LAG_FEATURES = ['bb_width', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14']
LAGS = [1, 3, 6]

# Test period
TEST_START = '2025-01-01'
TEST_END = '2026-04-29'

# Initial capital
INITIAL_CAPITAL = 10000

# Bars per year for 4H timeframe
BARS_PER_YEAR = 6 * 365  # 6 bars per day * 365 days

# Data directory
DATA_DIR = os.path.expanduser('~/orion/data')


def send_telegram(message):
    if not BOT_TOKEN:
        print("WARNING: No TELEGRAM_BOT_TOKEN found")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown',
            'message_thread_id': TOPIC_ID
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")


def fetch_all_ohlcv(symbol, timeframe='4h', limit=300):
    """
    Descarga datos de OKX. Si falla, intenta leer CSV local.
    """
    try:
        exchange = ccxt.okx()
        all_data = []
        since = exchange.parse8601('2022-01-01T00:00:00Z')

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            except Exception as e:
                print(f'OKX fetch error for {symbol}: {e}')
                raise e

            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break
            time.sleep(0.1)

            if len(all_data) % 3000 == 0:
                print(f'  {symbol}: {len(all_data)} barras descargadas...')

        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

        print(f'Descargado {symbol}: {len(df)} barras')
        return df

    except Exception as e:
        print(f'OKX falló para {symbol}: {e}')
        print('Intentando fallback CSV...')

        data_dir = os.path.expanduser('~/orion/data')
        safe_symbol = symbol.replace('/', '_')
        csv_path = os.path.join(data_dir, f'{safe_symbol}_4h.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f'Cargado desde CSV: {csv_path} ({len(df)} filas)')
            return df
        else:
            raise FileNotFoundError(
                f'No se pudo descargar de OKX ni encontrar CSV en {csv_path}'
            )


def compute_features(df):
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
    for feat in LAG_FEATURES:
        for lag in LAGS:
            df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
    for feat in LAG_FEATURES:
        df[f'{feat}_roc'] = df[feat] / df[feat].shift(1) - 1
    return df


def compute_equity_curve(returns_series, initial_capital=10000):
    """
    Calcula equity curve con compounding real.
    returns_series: log returns de cada barra (sin overlap)
    """
    simple_returns = np.exp(returns_series) - 1

    equity = [initial_capital]
    for r in simple_returns:
        equity.append(equity[-1] * (1 + r))

    equity = pd.Series(equity[1:], index=returns_series.index)
    return equity


def compute_max_drawdown(equity_series):
    """
    Calcula max drawdown como porcentaje (siempre entre -1.0 y 0.0)
    """
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min()
    return max_dd


def calc_metrics(returns_series, label, initial_capital=10000):
    """
    Calcula métricas corregidas con single-bar returns y equity curve real.
    """
    equity = compute_equity_curve(returns_series, initial_capital)

    total_return = (equity.iloc[-1] / initial_capital) - 1

    bars_per_year = 6 * 365
    mean_ret = returns_series.mean()
    std_ret = returns_series.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(bars_per_year) if std_ret > 0 else 0

    max_dd = compute_max_drawdown(equity)
    assert max_dd >= -1.0, f'DD de {max_dd:.2%} es imposible con capital real'

    n_total = len(returns_series)
    n_active = (returns_series != 0).sum()
    exposure = n_active / n_total if n_total > 0 else 0

    years = n_total / bars_per_year
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    active_returns = returns_series[returns_series != 0]
    win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0

    return {
        'label': label,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'exposure': exposure,
        'win_rate': win_rate,
        'n_bars': n_total,
        'n_active': n_active,
    }


def main():
    print("\n" + "="*60)
    print("ORION V21.0 BACKTEST v2 (CORREGIDO)")
    print("="*60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Cargar modelos
    print("\n1. Cargando modelos...")
    model_path = os.path.expanduser('~/orion/lab/v21_lgbm_model.pkl')
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    print('Modelos disponibles:', list(models.keys()))
    eth_model = models['ETH/USDT']
    btc_model = models['BTC/USDT']
    sol_model = models['SOL/USDT']
    print("Modelos ETH, BTC, SOL cargados")

    # 2. Descargar datos frescos
    print("\n2. Descargando datos frescos...")
    all_data = {}
    for asset in ASSETS:
        try:
            df = fetch_all_ohlcv(asset, timeframe='4h', limit=300)
            all_data[asset] = df
        except Exception as e:
            print(f'ERROR: No se pudieron obtener datos para {asset}: {e}')
            return

    # 3. Calcular features para ETH y BTC
    print("\n3. Calculando features...")
    eth_data = compute_features(all_data['ETH/USDT'].copy())
    btc_data = compute_features(all_data['BTC/USDT'].copy())
    sol_data = compute_features(all_data['SOL/USDT'].copy())

    # Feature columns
    feature_cols = ['bb_width', 'squeeze_duration', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14', 'trend_strength', 'momentum_divergence', 'vol_of_vol']
    for feat in LAG_FEATURES:
        for lag in LAGS:
            feature_cols.append(f'{feat}_lag{lag}')
    for feat in LAG_FEATURES:
        feature_cols.append(f'{feat}_roc')

    print(f"Feature columns: {len(feature_cols)}")

    # 4. Calcular single-bar returns para los 3 assets
    print("\n4. Calculando single-bar returns...")
    for asset in ASSETS:
        all_data[asset]['bar_return'] = np.log(all_data[asset]['close'] / all_data[asset]['close'].shift(1))

    # 5. Split temporal y predicciones ETH
    print("\n5. Split temporal y predicciones...")
    eth_test = eth_data[eth_data['timestamp'] >= TEST_START].copy()
    eth_test['pred_proba'] = eth_model.predict(eth_test[feature_cols])

    print(f"ETH test: {len(eth_test)} barras desde {TEST_START}")
    print(f"Predicciones: mean={eth_test['pred_proba'].mean():.4f}, std={eth_test['pred_proba'].std():.4f}")

    # 6. Alinear datos de los 3 assets
    print("\n6. Alineando datos de los 3 assets...")
    asset_data = {}
    for asset in ASSETS:
        asset_df = all_data[asset].copy()
        asset_test = asset_df[asset_df['timestamp'] >= TEST_START].copy()
        merged = pd.merge(eth_test[['timestamp', 'pred_proba']], asset_test[['timestamp', 'bar_return']], on='timestamp', how='inner')
        asset_data[asset] = merged

    print(f"Assets alineados: {len(asset_data['ETH/USDT'])} barras")

    # BLOQUE 1: Selección de percentil óptimo
    print("\n" + "="*60)
    print("BLOQUE 1 — SELECCIÓN DE PERCENTIL ÓPTIMO")
    print("="*60)

    percentiles = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    block1_results = []

    for pct in percentiles:
        threshold = np.percentile(eth_test['pred_proba'], 100 * (1 - pct))
        edge_on_mask = eth_test['pred_proba'] >= threshold

        portfolio_returns_v21 = []
        portfolio_returns_v209 = []

        for asset in ASSETS:
            merged = asset_data[asset]
            asset_mask = merged['pred_proba'] >= threshold

            v21_ret = merged['bar_return'].copy()
            v21_ret[~asset_mask] = 0

            v209_ret = merged['bar_return'].copy()

            portfolio_returns_v21.append(v21_ret)
            portfolio_returns_v209.append(v209_ret)

        portfolio_v21 = pd.concat(portfolio_returns_v21, axis=1).mean(axis=1)
        portfolio_v209 = pd.concat(portfolio_returns_v209, axis=1).mean(axis=1)

        metrics_v21 = calc_metrics(portfolio_v21, f'V21_{int(pct*100)}%', INITIAL_CAPITAL)
        metrics_v209 = calc_metrics(portfolio_v209, f'V209_{int(pct*100)}%', INITIAL_CAPITAL)

        block1_results.append({
            'percentile': pct,
            'threshold': threshold,
            'return_on': metrics_v21['total_return'],
            'return_all': metrics_v209['total_return'],
            'sharpe_on': metrics_v21['sharpe'],
            'sharpe_all': metrics_v209['sharpe'],
            'dd_on': metrics_v21['max_dd'],
            'dd_all': metrics_v209['max_dd'],
            'exposure': metrics_v21['exposure'],
            'calmar_on': metrics_v21['calmar'],
        })

    print("\nMétricas PORTFOLIO (3 assets equally weighted):")
    print("Percentil | Threshold | Return_ON | Return_ALL | Sharpe_ON | Sharpe_ALL | DD_ON    | DD_ALL   | Exposure | Calmar")
    print("-" * 110)
    for r in block1_results:
        print(f"{int(r['percentile']*100):8}% | {r['threshold']:8.4f} | {r['return_on']*100:+8.2f}% | {r['return_all']*100:+8.2f}% | {r['sharpe_on']:8.2f} | {r['sharpe_all']:8.2f} | {r['dd_on']*100:6.2f}% | {r['dd_all']*100:6.2f}% | {r['exposure']*100:7.0f}% | {r['calmar_on']:8.2f}")

    # Seleccionar percentil óptimo (max Calmar con exposure >= 15%)
    valid_results = [r for r in block1_results if r['exposure'] >= 0.15]
    if not valid_results:
        print("ERROR: Ningún percentil cumple exposure >= 15%")
        return

    best_result = max(valid_results, key=lambda x: x['calmar_on'])
    best_pct = best_result['percentile']
    best_threshold = best_result['threshold']

    print(f"\nREGLA: Maximizar Calmar ratio con constraint exposure >= 15%")
    print(f"Percentil óptimo: {int(best_pct*100)}%")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Calmar ratio: {best_result['calmar_on']:.2f}")
    print(f"Exposure: {best_result['exposure']*100:.0f}%")

    # BLOQUE 2: Backtest temporal
    print("\n" + "="*60)
    print(f"BLOQUE 2 — BACKTEST TEMPORAL (percentil óptimo = {int(best_pct*100)}%)")
    print("="*60)

    edge_on_mask = eth_test['pred_proba'] >= best_threshold

    n_total_bars = len(asset_data['ETH/USDT'])
    n_edge_on = edge_on_mask.sum()
    n_edge_off = len(edge_on_mask) - n_edge_on

    print(f"Período: {TEST_START} a {eth_test['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"Barras test: {n_total_bars} | EDGE_ON: {n_edge_on} ({n_edge_on/n_total_bars*100:.0f}%) | EDGE_OFF: {n_edge_off} ({n_edge_off/n_total_bars*100:.0f}%)")

    print("\nPOR ASSET:")
    print("Asset      | Strategy | Total Return | Sharpe | Max DD   | Calmar | Win Rate | Exposure")
    print("-" * 90)

    block2_results = []
    for asset in ASSETS:
        merged = asset_data[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        v21_ret = merged['bar_return'].copy()
        v21_ret[~asset_mask] = 0

        v209_ret = merged['bar_return'].copy()

        metrics_v21 = calc_metrics(v21_ret, f'V21_{asset}', INITIAL_CAPITAL)
        metrics_v209 = calc_metrics(v209_ret, f'V209_{asset}', INITIAL_CAPITAL)

        delta_return = metrics_v21['total_return'] - metrics_v209['total_return']
        delta_sharpe = metrics_v21['sharpe'] - metrics_v209['sharpe']
        delta_dd = metrics_v21['max_dd'] - metrics_v209['max_dd']
        delta_exposure = metrics_v21['exposure'] - metrics_v209['exposure']

        print(f"{asset:10} | V21      | {metrics_v21['total_return']*100:+10.2f}% | {metrics_v21['sharpe']:6.2f} | {metrics_v21['max_dd']*100:7.2f}% | {metrics_v21['calmar']:6.2f} | {metrics_v21['win_rate']*100:6.0f}% | {metrics_v21['exposure']*100:6.0f}%")
        print(f"{asset:10} | V20.9    | {metrics_v209['total_return']*100:+10.2f}% | {metrics_v209['sharpe']:6.2f} | {metrics_v209['max_dd']*100:7.2f}% | {metrics_v209['calmar']:6.2f} | {metrics_v209['win_rate']*100:6.0f}% | {metrics_v209['exposure']*100:6.0f}%")
        print(f"{asset:10} | Delta    | {delta_return*100:+10.2f}% | {delta_sharpe:+6.2f} | {delta_dd*100:+7.2f}% |        |          | {delta_exposure*100:+6.0f}%")
        print("-" * 90)

        block2_results.append({
            'asset': asset,
            'v21_total_return': metrics_v21['total_return'],
            'v21_sharpe': metrics_v21['sharpe'],
            'v21_max_dd': metrics_v21['max_dd'],
            'v21_calmar': metrics_v21['calmar'],
            'v21_win_rate': metrics_v21['win_rate'],
            'v21_exposure': metrics_v21['exposure'],
            'v209_total_return': metrics_v209['total_return'],
            'v209_sharpe': metrics_v209['sharpe'],
            'v209_max_dd': metrics_v209['max_dd'],
            'v209_calmar': metrics_v209['calmar'],
            'v209_win_rate': metrics_v209['win_rate'],
            'v209_exposure': metrics_v209['exposure'],
        })

    # Portfolio equally weighted
    portfolio_returns_v21 = []
    portfolio_returns_v209 = []

    for asset in ASSETS:
        merged = asset_data[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        v21_ret = merged['bar_return'].copy()
        v21_ret[~asset_mask] = 0

        v209_ret = merged['bar_return'].copy()

        portfolio_returns_v21.append(v21_ret)
        portfolio_returns_v209.append(v209_ret)

    portfolio_v21 = pd.concat(portfolio_returns_v21, axis=1).mean(axis=1)
    portfolio_v209 = pd.concat(portfolio_returns_v209, axis=1).mean(axis=1)

    metrics_v21 = calc_metrics(portfolio_v21, 'V21_portfolio', INITIAL_CAPITAL)
    metrics_v209 = calc_metrics(portfolio_v209, 'V209_portfolio', INITIAL_CAPITAL)

    delta_return = metrics_v21['total_return'] - metrics_v209['total_return']
    delta_sharpe = metrics_v21['sharpe'] - metrics_v209['sharpe']
    delta_dd = metrics_v21['max_dd'] - metrics_v209['max_dd']
    delta_exposure = metrics_v21['exposure'] - metrics_v209['exposure']

    print("PORTFOLIO (equally weighted):")
    print("Strategy       | Total Return | Sharpe | Max DD   | Calmar | Win Rate | Exposure")
    print("-" * 90)
    print(f"V21            | {metrics_v21['total_return']*100:+12.2f}% | {metrics_v21['sharpe']:6.2f} | {metrics_v21['max_dd']*100:7.2f}% | {metrics_v21['calmar']:6.2f} | {metrics_v21['win_rate']*100:6.0f}% | {metrics_v21['exposure']*100:6.0f}%")
    print(f"V20.9          | {metrics_v209['total_return']*100:+12.2f}% | {metrics_v209['sharpe']:6.2f} | {metrics_v209['max_dd']*100:7.2f}% | {metrics_v209['calmar']:6.2f} | {metrics_v209['win_rate']*100:6.0f}% | {metrics_v209['exposure']*100:6.0f}%")
    print(f"Delta          | {delta_return*100:+12.2f}% | {delta_sharpe:+6.2f} | {delta_dd*100:+7.2f}% |        |          | {delta_exposure*100:+6.0f}%")

    print("\nSANITY CHECKS:")
    print(f"  assert all DD >= -100%: {all(r['v21_max_dd'] >= -1.0 for r in block2_results)}")
    print(f"  assert all DD >= -100%: {all(r['v209_max_dd'] >= -1.0 for r in block2_results)}")
    print(f"  Total barras test: {n_total_bars}")

    # BLOQUE 3: Análisis EDGE_ON vs EDGE_OFF
    print("\n" + "="*60)
    print("BLOQUE 3 — ANÁLISIS EDGE_ON vs EDGE_OFF")
    print("="*60)

    bars_per_year = 6 * 365
    print("Asset      | Período  | Mean Bar | Sharpe | %Neg  | Total PnL")
    print("-" * 70)

    block3_results = []
    for asset in ASSETS:
        merged = asset_data[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        edge_off_returns = merged['bar_return'][~asset_mask]
        edge_on_returns = merged['bar_return'][asset_mask]

        # EDGE_OFF
        mean_off = edge_off_returns.mean()
        std_off = edge_off_returns.std()
        sharpe_off = mean_off / std_off * np.sqrt(bars_per_year) if std_off > 0 else 0
        pct_negative_off = (edge_off_returns < 0).mean()
        total_pnl_off = edge_off_returns.sum()

        print(f"{asset:10} | EDGE_OFF | {mean_off*100:+8.2f}% | {sharpe_off:6.2f} | {pct_negative_off*100:4.0f}% | {total_pnl_off*100:+8.2f}%")

        # EDGE_ON
        mean_on = edge_on_returns.mean()
        std_on = edge_on_returns.std()
        sharpe_on = mean_on / std_on * np.sqrt(bars_per_year) if std_on > 0 else 0
        pct_negative_on = (edge_on_returns < 0).mean()
        total_pnl_on = edge_on_returns.sum()

        print(f"{asset:10} | EDGE_ON  | {mean_on*100:+8.2f}% | {sharpe_on:6.2f} | {pct_negative_on*100:4.0f}% | {total_pnl_on*100:+8.2f}%")
        print("-" * 70)

        block3_results.append({
            'asset': asset,
            'mean_off': mean_off,
            'sharpe_off': sharpe_off,
            'pct_negative_off': pct_negative_off,
            'total_pnl_off': total_pnl_off,
            'mean_on': mean_on,
            'sharpe_on': sharpe_on,
            'pct_negative_on': pct_negative_on,
            'total_pnl_on': total_pnl_on,
        })

    print("\nINTERPRETACIÓN:")
    all_negative = all(r['mean_off'] < 0 for r in block3_results)
    all_negative_sharpe = all(r['sharpe_off'] < 0 for r in block3_results)

    if all_negative and all_negative_sharpe:
        print("Si Mean_ON > 0 y Mean_OFF < 0 → filtro discrimina CORRECTAMENTE")
    elif any(r['mean_on'] > r['mean_off'] for r in block3_results):
        print("Si Mean_ON > Mean_OFF pero ambos positivos → filtro débil pero válido")
    elif any(r['mean_on'] < r['mean_off'] for r in block3_results):
        print("Si Mean_ON < Mean_OFF → filtro INVERTIDO, problema grave")

    # BLOQUE 4: Walk-forward validation
    print("\n" + "="*60)
    print("BLOQUE 4 — WALK-FORWARD VALIDATION")
    print("="*60)

    period1 = eth_test[eth_test['timestamp'] < '2025-10-01']
    period2 = eth_test[eth_test['timestamp'] >= '2025-10-01']

    print(f"Período 1 (in-sample): 2025-01 a 2025-09 ({len(period1)} barras)")
    print(f"Período 2 (out-of-sample): 2025-10 a 2026-04 ({len(period2)} barras)")

    # Calcular percentil óptimo en period1
    p1_results = []
    for pct in percentiles:
        threshold_p1 = np.percentile(period1['pred_proba'], 100 * (1 - pct))
        edge_on_mask_p1 = period1['pred_proba'] >= threshold_p1

        portfolio_returns_v21_p1 = []
        portfolio_returns_v209_p1 = []

        for asset in ASSETS:
            merged_p1 = asset_data[asset][asset_data[asset]['timestamp'] < '2025-10-01'].copy()
            asset_mask_p1 = merged_p1['pred_proba'] >= threshold_p1

            v21_ret_p1 = merged_p1['bar_return'].copy()
            v21_ret_p1[~asset_mask_p1] = 0

            v209_ret_p1 = merged_p1['bar_return'].copy()

            portfolio_returns_v21_p1.append(v21_ret_p1)
            portfolio_returns_v209_p1.append(v209_ret_p1)

        portfolio_v21_p1 = pd.concat(portfolio_returns_v21_p1, axis=1).mean(axis=1)
        portfolio_v209_p1 = pd.concat(portfolio_returns_v209_p1, axis=1).mean(axis=1)

        metrics_v21_p1 = calc_metrics(portfolio_v21_p1, f'V21_p1_{int(pct*100)}%', INITIAL_CAPITAL)
        metrics_v209_p1 = calc_metrics(portfolio_v209_p1, f'V209_p1_{int(pct*100)}%', INITIAL_CAPITAL)

        p1_results.append({
            'percentile': pct,
            'threshold': threshold_p1,
            'calmar': metrics_v21_p1['calmar'],
            'exposure': metrics_v21_p1['exposure'],
            'sharpe_v21': metrics_v21_p1['sharpe'],
            'sharpe_v209': metrics_v209_p1['sharpe'],
        })

    # Seleccionar mejor percentil en P1
    valid_p1 = [r for r in p1_results if r['exposure'] >= 0.15]
    if not valid_p1:
        print("ERROR: Ningún percentil en P1 cumple exposure >= 15%")
        return

    best_p1 = max(valid_p1, key=lambda x: x['calmar'])
    best_pct_p1 = best_p1['percentile']
    threshold_p1 = best_p1['threshold']

    print(f"\nPercentil seleccionado en P1: {int(best_pct_p1*100)}%")

    delta_p1 = best_p1['sharpe_v21'] - best_p1['sharpe_v209']

    print(f"\nP1 (in-sample): 2025-01 a 2025-09")
    print(f"  Sharpe V21: {best_p1['sharpe_v21']:.2f} | Sharpe V20.9: {best_p1['sharpe_v209']:.2f} | Delta: {delta_p1:+.2f}")

    # Aplicar threshold_p1 a period2
    edge_mask_p2 = period2['pred_proba'] >= threshold_p1

    portfolio_returns_v21_p2 = []
    portfolio_returns_v209_p2 = []

    for asset in ASSETS:
        merged_p2 = asset_data[asset][asset_data[asset]['timestamp'] >= '2025-10-01'].copy()
        asset_mask_p2 = merged_p2['pred_proba'] >= threshold_p1

        v21_ret_p2 = merged_p2['bar_return'].copy()
        v21_ret_p2[~asset_mask_p2] = 0

        v209_ret_p2 = merged_p2['bar_return'].copy()

        portfolio_returns_v21_p2.append(v21_ret_p2)
        portfolio_returns_v209_p2.append(v209_ret_p2)

    portfolio_v21_p2 = pd.concat(portfolio_returns_v21_p2, axis=1).mean(axis=1)
    portfolio_v209_p2 = pd.concat(portfolio_returns_v209_p2, axis=1).mean(axis=1)

    metrics_v21_p2 = calc_metrics(portfolio_v21_p2, 'V21_p2', INITIAL_CAPITAL)
    metrics_v209_p2 = calc_metrics(portfolio_v209_p2, 'V209_p2', INITIAL_CAPITAL)

    delta_p2 = metrics_v21_p2['sharpe'] - metrics_v209_p2['sharpe']

    print(f"\nP2 (out-of-sample): 2025-10 a 2026-04")
    print(f"  Sharpe V21: {metrics_v21_p2['sharpe']:.2f} | Sharpe V20.9: {metrics_v209_p2['sharpe']:.2f} | Delta: {delta_p2:+.2f}")

    walk_forward_pass = delta_p2 > 0

    if walk_forward_pass:
        print(f"\nSi Delta > 0 en P2 → ROBUSTO")
    else:
        print(f"\nSi Delta <= 0 en P2 → OVERFITTEADO")

    # BLOQUE 5: Cross-asset validation
    print("\n" + "="*60)
    print("BLOQUE 5 — CROSS-ASSET VALIDATION")
    print("="*60)
    print('"¿Es ETH un buen proxy del régimen global?"')

    # BTC
    print("\nBTC/USDT:")
    btc_test = btc_data[btc_data['timestamp'] >= TEST_START].copy()
    btc_test['pred_proba_eth'] = eth_model.predict(btc_test[feature_cols])
    btc_test['pred_proba_btc'] = btc_model.predict(btc_test[feature_cols])

    btc_merged = pd.merge(btc_test[['timestamp', 'pred_proba_eth', 'pred_proba_btc']],
                          asset_data['BTC/USDT'][['timestamp', 'bar_return']],
                          on='timestamp', how='inner')

    # Caso A: ETH model → decide para BTC
    btc_mask_a = btc_merged['pred_proba_eth'] >= best_threshold
    btc_ret_a = btc_merged['bar_return'].copy()
    btc_ret_a[~btc_mask_a] = 0
    btc_metrics_a = calc_metrics(btc_ret_a, 'BTC_ETH_model', INITIAL_CAPITAL)

    # Caso B: BTC model → decide para BTC
    btc_mask_b = btc_merged['pred_proba_btc'] >= best_threshold
    btc_ret_b = btc_merged['bar_return'].copy()
    btc_ret_b[~btc_mask_b] = 0
    btc_metrics_b = calc_metrics(btc_ret_b, 'BTC_BTC_model', INITIAL_CAPITAL)

    # Caso C: Baseline
    btc_ret_c = btc_merged['bar_return'].copy()
    btc_metrics_c = calc_metrics(btc_ret_c, 'BTC_baseline', INITIAL_CAPITAL)

    print(f"Caso A: ETH model → decide para BTC")
    print(f"  BTC Sharpe: {btc_metrics_a['sharpe']:.2f} | Return: {btc_metrics_a['total_return']*100:+.2f}% | DD: {btc_metrics_a['max_dd']*100:.2f}%")
    print(f"Caso B: BTC model → decide para BTC")
    print(f"  BTC Sharpe: {btc_metrics_b['sharpe']:.2f} | Return: {btc_metrics_b['total_return']*100:+.2f}% | DD: {btc_metrics_b['max_dd']*100:.2f}%")
    print(f"Caso C: Baseline (always on)")
    print(f"  BTC Sharpe: {btc_metrics_c['sharpe']:.2f} | Return: {btc_metrics_c['total_return']*100:+.2f}% | DD: {btc_metrics_c['max_dd']*100:.2f}%")

    btc_eth_proxy_valid = btc_metrics_a['sharpe'] >= btc_metrics_b['sharpe']

    # SOL
    print("\nSOL/USDT:")
    sol_test = sol_data[sol_data['timestamp'] >= TEST_START].copy()
    sol_test['pred_proba_eth'] = eth_model.predict(sol_test[feature_cols])
    sol_test['pred_proba_sol'] = sol_model.predict(sol_test[feature_cols])

    sol_merged = pd.merge(sol_test[['timestamp', 'pred_proba_eth', 'pred_proba_sol']],
                          asset_data['SOL/USDT'][['timestamp', 'bar_return']],
                          on='timestamp', how='inner')

    # Caso A: ETH model → decide para SOL
    sol_mask_a = sol_merged['pred_proba_eth'] >= best_threshold
    sol_ret_a = sol_merged['bar_return'].copy()
    sol_ret_a[~sol_mask_a] = 0
    sol_metrics_a = calc_metrics(sol_ret_a, 'SOL_ETH_model', INITIAL_CAPITAL)

    # Caso B: SOL model → decide para SOL
    sol_mask_b = sol_merged['pred_proba_sol'] >= best_threshold
    sol_ret_b = sol_merged['bar_return'].copy()
    sol_ret_b[~sol_mask_b] = 0
    sol_metrics_b = calc_metrics(sol_ret_b, 'SOL_SOL_model', INITIAL_CAPITAL)

    # Caso C: Baseline
    sol_ret_c = sol_merged['bar_return'].copy()
    sol_metrics_c = calc_metrics(sol_ret_c, 'SOL_baseline', INITIAL_CAPITAL)

    print(f"Caso A: ETH model → decide para SOL")
    print(f"  SOL Sharpe: {sol_metrics_a['sharpe']:.2f} | Return: {sol_metrics_a['total_return']*100:+.2f}% | DD: {sol_metrics_a['max_dd']*100:.2f}%")
    print(f"Caso B: SOL model → decide para SOL")
    print(f"  SOL Sharpe: {sol_metrics_b['sharpe']:.2f} | Return: {sol_metrics_b['total_return']*100:+.2f}% | DD: {sol_metrics_b['max_dd']*100:.2f}%")
    print(f"Caso C: Baseline (always on)")
    print(f"  SOL Sharpe: {sol_metrics_c['sharpe']:.2f} | Return: {sol_metrics_c['total_return']*100:+.2f}% | DD: {sol_metrics_c['max_dd']*100:.2f}%")

    sol_eth_proxy_valid = sol_metrics_a['sharpe'] >= sol_metrics_b['sharpe']

    print("\nINTERPRETACIÓN:")
    if btc_eth_proxy_valid and sol_eth_proxy_valid:
        print("Si Sharpe_A >= Sharpe_B → ETH proxy VÁLIDO (usar enfoque global)")
    elif not btc_eth_proxy_valid or not sol_eth_proxy_valid:
        print("Si Sharpe_A < Sharpe_B significativamente → necesitas edge por asset")

    cross_asset_valid = btc_eth_proxy_valid and sol_eth_proxy_valid

    # VEREDICTO FINAL
    print("\n" + "="*60)
    print("VEREDICTO FINAL — V21.0 (BACKTEST v2 CORREGIDO)")
    print("="*60)

    criteria = {
        '1_sharpe': metrics_v21['sharpe'] > metrics_v209['sharpe'],
        '2_dd': metrics_v21['max_dd'] >= metrics_v209['max_dd'],
        '3_exposure': metrics_v21['exposure'] < 0.80,
        '4_walk_forward': walk_forward_pass,
        '5_cross_asset': cross_asset_valid,
    }

    n_pass = sum(criteria.values())

    if n_pass == 5:
        verdict = 'V21.0 VIVE — PROCEDER A PAPER TRADING'
    elif n_pass == 4:
        verdict = 'V21.0 BORDERLINE — REVISAR CRITERIO FALLIDO'
    elif n_pass == 3:
        verdict = 'V21.0 DÉBIL — CONSIDERAR AJUSTES'
    else:
        verdict = 'V21.0 MUERE — EDGE DETECTION NO AGREGA VALOR'

    print("\nCRITERIOS DE VIDA O MUERTE:")
    print()

    print(f"1. Sharpe(V21) > Sharpe(V20.9)?")
    print(f"   V21: {metrics_v21['sharpe']:.2f} | V20.9: {metrics_v209['sharpe']:.2f} | Delta: {delta_sharpe:+.2f} → {'PASS' if criteria['1_sharpe'] else 'FAIL'}")
    print()

    print(f"2. MaxDD(V21) <= MaxDD(V20.9)?")
    print(f"   V21: {metrics_v21['max_dd']*100:.2f}% | V20.9: {metrics_v209['max_dd']*100:.2f}% | → {'PASS' if criteria['2_dd'] else 'FAIL'}")
    print()

    print(f"3. Exposure < 80%?")
    print(f"   V21: {metrics_v21['exposure']*100:.0f}% | → {'PASS' if criteria['3_exposure'] else 'FAIL'}")
    print()

    print(f"4. Walk-forward (Delta > 0 en P2)?")
    print(f"   P1 Delta: {delta_p1:+.2f} | P2 Delta: {delta_p2:+.2f} | → {'PASS' if criteria['4_walk_forward'] else 'FAIL'}")
    print()

    print(f"5. Cross-asset (ETH proxy válido)?")
    print(f"   ETH→BTC: {btc_metrics_a['sharpe']:.2f} vs BTC→BTC: {btc_metrics_b['sharpe']:.2f} | → {'PASS' if btc_eth_proxy_valid else 'FAIL'}")
    print(f"   ETH→SOL: {sol_metrics_a['sharpe']:.2f} vs SOL→SOL: {sol_metrics_b['sharpe']:.2f} | → {'PASS' if sol_eth_proxy_valid else 'FAIL'}")
    print()

    print(f"RESULTADO: {n_pass}/5 criterios PASS")
    print()
    print("="*60)
    print(verdict)
    print("="*60)
    print(f"Percentil óptimo: {int(best_pct*100)}%")
    print("="*60)

    # Telegram
    print("\nEnviando a Telegram...")
    msg = f"""ORION V21.0 BACKTEST v2 (CORREGIDO)
Sharpe V21: {metrics_v21['sharpe']:.2f} vs V20.9: {metrics_v209['sharpe']:.2f}
Return V21: {metrics_v21['total_return']*100:+.1f}% vs V20.9: {metrics_v209['total_return']*100:+.1f}%
DD V21: {metrics_v21['max_dd']*100:+.1f}% vs V20.9: {metrics_v209['max_dd']*100:+.1f}%
Exposure: {metrics_v21['exposure']*100:.0f}%
Walk-fwd: {'PASS' if walk_forward_pass else 'FAIL'}
Cross-asset: {'PASS' if cross_asset_valid else 'FAIL'}
Percentil: {int(best_pct*100)}%
VEREDICTO: {verdict}"""

    send_telegram(msg)
    print("Telegram enviado")

    # Guardar resultados
    print("\nGuardando resultados...")

    # CSV detallado barra por barra
    results_path = os.path.expanduser('~/orion/v21_backtest_v2_results.csv')
    results_rows = []

    for asset in ASSETS:
        merged = asset_data[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        v21_ret = merged['bar_return'].copy()
        v21_ret[~asset_mask] = 0

        v209_ret = merged['bar_return'].copy()

        equity_v21 = compute_equity_curve(v21_ret, INITIAL_CAPITAL)
        equity_v209 = compute_equity_curve(v209_ret, INITIAL_CAPITAL)

        for i, row in merged.iterrows():
            results_rows.append({
                'timestamp': row['timestamp'],
                'asset': asset,
                'pred_proba': row['pred_proba'],
                'edge_on': asset_mask.iloc[i] if i < len(asset_mask) else False,
                'bar_return': row['bar_return'],
                'v21_return': v21_ret.iloc[i] if i < len(v21_ret) else 0,
                'v209_return': v209_ret.iloc[i] if i < len(v209_ret) else 0,
                'equity_v21': equity_v21.iloc[i] if i < len(equity_v21) else INITIAL_CAPITAL,
                'equity_v209': equity_v209.iloc[i] if i < len(equity_v209) else INITIAL_CAPITAL,
            })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_path, index=False)
    print(f'Resultados guardados en: {results_path}')

    # CSV resumen
    summary_path = os.path.expanduser('~/orion/lab/v21_backtest_v2_summary.csv')
    summary_rows = []

    for r in block1_results:
        for asset in ASSETS:
            summary_rows.append({
                'percentile': r['percentile'],
                'asset': asset,
                'strategy': 'V21',
                'total_return': r['return_on'],
                'sharpe': r['sharpe_on'],
                'max_dd': r['dd_on'],
                'calmar': r['calmar_on'],
                'exposure': r['exposure'],
            })
            summary_rows.append({
                'percentile': r['percentile'],
                'asset': asset,
                'strategy': 'V20.9',
                'total_return': r['return_all'],
                'sharpe': r['sharpe_all'],
                'max_dd': r['dd_all'],
                'calmar': 0,
                'exposure': 1.0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f'Resumen guardado en: {summary_path}')

    print("\n" + "="*60)
    print("BACKTEST v2 COMPLETADO")
    print("="*60)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
