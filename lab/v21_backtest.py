#!/usr/bin/env python3
# Orion V21.0 Backtest - EDGE_ON vs ALWAYS_ON

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

# Forward return horizon
H = 12


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


def compute_forward_returns(df, h=12):
    df = df.copy()
    df['fwd_return'] = np.log(df['close'].shift(-h) / df['close'])
    return df


def calc_metrics(returns_series, n_total_bars, label):
    total_return = returns_series.sum()
    mean_return = returns_series.mean()
    std_return = returns_series.std()

    sharpe = mean_return / std_return * np.sqrt(6 * 365) if std_return > 0 else 0

    cum_returns = returns_series.cumsum()
    running_max = cum_returns.cummax()
    drawdown = cum_returns - running_max
    max_dd = drawdown.min()

    n_active = (returns_series != 0).sum()
    exposure = n_active / n_total_bars if n_total_bars > 0 else 0

    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    return {
        'label': label,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'exposure': exposure,
        'calmar': calmar,
        'n_trades': int(n_active),
    }


def run_backtest(asset_returns, edge_mask, label):
    v21_returns = asset_returns.copy()
    v21_returns[~edge_mask] = 0

    v209_returns = asset_returns.copy()

    v21_equity = v21_returns.cumsum()
    v209_equity = v209_returns.cumsum()

    return v21_returns, v209_returns, v21_equity, v209_equity


def final_verdict(portfolio_v21, portfolio_v209, walk_forward_pass):
    sharpe_v21 = portfolio_v21['sharpe']
    sharpe_v209 = portfolio_v209['sharpe']
    dd_v21 = portfolio_v21['max_dd']
    dd_v209 = portfolio_v209['max_dd']
    exposure_v21 = portfolio_v21['exposure']

    criteria = {
        'sharpe_better': sharpe_v21 > sharpe_v209,
        'dd_better_or_equal': dd_v21 >= dd_v209,
        'exposure_reduced': exposure_v21 < 0.80,
        'walk_forward': walk_forward_pass,
    }

    n_pass = sum(criteria.values())

    if n_pass == 4:
        verdict = 'V21.0 VIVE — PROCEDER A PAPER TRADING'
    elif n_pass == 3:
        verdict = 'V21.0 BORDERLINE — REVISAR CRITERIO FALLIDO'
    elif n_pass == 2:
        verdict = 'V21.0 DÉBIL — NO RECOMENDADO PARA PRODUCCIÓN'
    else:
        verdict = 'V21.0 MUERE — EDGE DETECTION NO AGREGA VALOR'

    return verdict, criteria, n_pass


def main():
    print("\n" + "="*60)
    print("ORION V21.0 BACKTEST DEFINITIVO")
    print("="*60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Cargar modelo ETH
    print("\n1. Cargando modelo ETH...")
    model_path = os.path.expanduser('~/orion/lab/v21_lgbm_model.pkl')
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    print('Keys del modelo:', type(models))
    if isinstance(models, dict):
        print('Dict keys:', list(models.keys()))

    eth_model = models['ETH/USDT']
    print("Modelo ETH cargado")

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

    # 3. Calcular features ETH
    print("\n3. Calculando features ETH...")
    eth_data = all_data['ETH/USDT'].copy()
    eth_data = compute_features(eth_data)

    # Feature columns (deben coincidir con training)
    feature_cols = ['bb_width', 'squeeze_duration', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14', 'trend_strength', 'momentum_divergence', 'vol_of_vol']
    for feat in LAG_FEATURES:
        for lag in LAGS:
            feature_cols.append(f'{feat}_lag{lag}')
    for feat in LAG_FEATURES:
        feature_cols.append(f'{feat}_roc')

    print(f"Feature columns: {len(feature_cols)}")

    # 4. Calcular forward returns para los 3 assets
    print("\n4. Calculando forward returns...")
    for asset in ASSETS:
        all_data[asset] = compute_forward_returns(all_data[asset], h=H)

    # 5. Split temporal y predicciones
    print("\n5. Split temporal y predicciones...")
    eth_test = eth_data[eth_data['timestamp'] >= TEST_START].copy()
    eth_test['pred_proba'] = eth_model.predict(eth_test[feature_cols])

    print(f"ETH test: {len(eth_test)} barras desde {TEST_START}")
    print(f"Predicciones: mean={eth_test['pred_proba'].mean():.4f}, std={eth_test['pred_proba'].std():.4f}")

    # Alinear forward returns de los 3 assets
    asset_returns = {}
    for asset in ASSETS:
        asset_df = all_data[asset].copy()
        asset_test = asset_df[asset_df['timestamp'] >= TEST_START].copy()
        merged = pd.merge(eth_test[['timestamp', 'pred_proba']], asset_test[['timestamp', 'fwd_return']], on='timestamp', how='inner')
        asset_returns[asset] = merged

    print(f"Assets alineados: {len(asset_returns['ETH/USDT'])} barras")

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
            merged = asset_returns[asset]
            asset_mask = merged['pred_proba'] >= threshold

            v21_ret = merged['fwd_return'].copy()
            v21_ret[~asset_mask] = 0

            v209_ret = merged['fwd_return'].copy()

            portfolio_returns_v21.append(v21_ret)
            portfolio_returns_v209.append(v209_ret)

        portfolio_v21 = pd.concat(portfolio_returns_v21, axis=1).mean(axis=1)
        portfolio_v209 = pd.concat(portfolio_returns_v209, axis=1).mean(axis=1)

        n_total = len(portfolio_v21)

        metrics_v21 = calc_metrics(portfolio_v21, n_total, f'V21_{int(pct*100)}%')
        metrics_v209 = calc_metrics(portfolio_v209, n_total, f'V209_{int(pct*100)}%')

        block1_results.append({
            'percentile': pct,
            'threshold': threshold,
            'sharpe_on': metrics_v21['sharpe'],
            'sharpe_all': metrics_v209['sharpe'],
            'dd_on': metrics_v21['max_dd'],
            'dd_all': metrics_v209['max_dd'],
            'exposure': metrics_v21['exposure'],
            'calmar_on': metrics_v21['calmar'],
        })

    print("\nMétricas GLOBALES (3 assets combinados, equally weighted):")
    print("Percentil | Threshold | Sharpe_ON | Sharpe_ALL | DD_ON   | DD_ALL  | Exposure | Calmar_ON")
    print("-" * 80)
    for r in block1_results:
        print(f"{int(r['percentile']*100):8}% | {r['threshold']:8.4f} | {r['sharpe_on']:8.2f} | {r['sharpe_all']:9.2f} | {r['dd_on']*100:6.2f}% | {r['dd_all']*100:6.2f}% | {r['exposure']*100:7.0f}% | {r['calmar_on']:8.2f}")

    # Seleccionar percentil óptimo (max Calmar con exposure >= 15%)
    valid_results = [r for r in block1_results if r['exposure'] >= 0.15]
    if not valid_results:
        print("ERROR: Ningún percentil cumple exposure >= 15%")
        return

    best_result = max(valid_results, key=lambda x: x['calmar_on'])
    best_pct = best_result['percentile']
    best_threshold = best_result['threshold']

    print(f"\nREGLA DE SELECCIÓN:")
    print(f"Percentil óptimo: {int(best_pct*100)}%")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Calmar ratio: {best_result['calmar_on']:.2f}")
    print(f"Exposure: {best_result['exposure']*100:.0f}%")

    # BLOQUE 2: Backtest temporal
    print("\n" + "="*60)
    print(f"BLOQUE 2 — BACKTEST TEMPORAL (percentil óptimo = {int(best_pct*100)}%)")
    print("="*60)

    edge_on_mask = eth_test['pred_proba'] >= best_threshold

    n_total_bars = len(asset_returns['ETH/USDT'])
    n_edge_on = edge_on_mask.sum()
    n_edge_off = len(edge_on_mask) - n_edge_on

    print(f"Período: {TEST_START} a {eth_test['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"Percentil: {int(best_pct*100)}% | Threshold: {best_threshold:.4f}")
    print(f"Barras test: {n_total_bars} | EDGE_ON: {n_edge_on} ({n_edge_on/n_total_bars*100:.0f}%) | EDGE_OFF: {n_edge_off} ({n_edge_off/n_total_bars*100:.0f}%)")

    print("\nPOR ASSET:")
    print("Asset      | Strategy | Total Return | Sharpe | Max DD   | Calmar | Exposure")
    print("-" * 80)

    block2_results = []
    for asset in ASSETS:
        merged = asset_returns[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        v21_ret = merged['fwd_return'].copy()
        v21_ret[~asset_mask] = 0

        v209_ret = merged['fwd_return'].copy()

        metrics_v21 = calc_metrics(v21_ret, n_total_bars, f'V21_{asset}')
        metrics_v209 = calc_metrics(v209_ret, n_total_bars, f'V209_{asset}')

        delta_return = metrics_v21['total_return'] - metrics_v209['total_return']
        delta_sharpe = metrics_v21['sharpe'] - metrics_v209['sharpe']
        delta_dd = metrics_v21['max_dd'] - metrics_v209['max_dd']
        delta_calmar = metrics_v21['calmar'] - metrics_v209['calmar']
        delta_exposure = metrics_v21['exposure'] - metrics_v209['exposure']

        print(f"{asset:10} | V21      | {metrics_v21['total_return']*100:+10.2f}% | {metrics_v21['sharpe']:6.2f} | {metrics_v21['max_dd']*100:7.2f}% | {metrics_v21['calmar']:6.2f} | {metrics_v21['exposure']*100:6.0f}%")
        print(f"{asset:10} | V20.9    | {metrics_v209['total_return']*100:+10.2f}% | {metrics_v209['sharpe']:6.2f} | {metrics_v209['max_dd']*100:7.2f}% | {metrics_v209['calmar']:6.2f} | {metrics_v209['exposure']*100:6.0f}%")
        print(f"{asset:10} | Δ        | {delta_return*100:+10.2f}% | {delta_sharpe:+6.2f} | {delta_dd*100:+7.2f}% | {delta_calmar:+6.2f} | {delta_exposure*100:+6.0f}%")
        print("-" * 80)

        block2_results.append({
            'asset': asset,
            'v21_total_return': metrics_v21['total_return'],
            'v21_sharpe': metrics_v21['sharpe'],
            'v21_max_dd': metrics_v21['max_dd'],
            'v21_calmar': metrics_v21['calmar'],
            'v21_exposure': metrics_v21['exposure'],
            'v209_total_return': metrics_v209['total_return'],
            'v209_sharpe': metrics_v209['sharpe'],
            'v209_max_dd': metrics_v209['max_dd'],
            'v209_calmar': metrics_v209['calmar'],
            'v209_exposure': metrics_v209['exposure'],
        })

    # Portfolio equally weighted
    portfolio_returns_v21 = []
    portfolio_returns_v209 = []

    for asset in ASSETS:
        merged = asset_returns[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        v21_ret = merged['fwd_return'].copy()
        v21_ret[~asset_mask] = 0

        v209_ret = merged['fwd_return'].copy()

        portfolio_returns_v21.append(v21_ret)
        portfolio_returns_v209.append(v209_ret)

    portfolio_v21 = pd.concat(portfolio_returns_v21, axis=1).mean(axis=1)
    portfolio_v209 = pd.concat(portfolio_returns_v209, axis=1).mean(axis=1)

    metrics_v21 = calc_metrics(portfolio_v21, n_total_bars, 'V21_portfolio')
    metrics_v209 = calc_metrics(portfolio_v209, n_total_bars, 'V209_portfolio')

    delta_return = metrics_v21['total_return'] - metrics_v209['total_return']
    delta_sharpe = metrics_v21['sharpe'] - metrics_v209['sharpe']
    delta_dd = metrics_v21['max_dd'] - metrics_v209['max_dd']
    delta_calmar = metrics_v21['calmar'] - metrics_v209['calmar']
    delta_exposure = metrics_v21['exposure'] - metrics_v209['exposure']

    print("PORTFOLIO (3 assets equally weighted):")
    print("Strategy       | Total Return | Sharpe | Max DD   | Calmar | Exposure")
    print("-" * 80)
    print(f"V21            | {metrics_v21['total_return']*100:+12.2f}% | {metrics_v21['sharpe']:6.2f} | {metrics_v21['max_dd']*100:7.2f}% | {metrics_v21['calmar']:6.2f} | {metrics_v21['exposure']*100:6.0f}%")
    print(f"V20.9          | {metrics_v209['total_return']*100:+12.2f}% | {metrics_v209['sharpe']:6.2f} | {metrics_v209['max_dd']*100:7.2f}% | {metrics_v209['calmar']:6.2f} | {metrics_v209['exposure']*100:6.0f}%")
    print(f"Δ (V21-V20.9)  | {delta_return*100:+12.2f}% | {delta_sharpe:+6.2f} | {delta_dd*100:+7.2f}% | {delta_calmar:+6.2f} | {delta_exposure*100:+6.0f}%")

    # BLOQUE 3: Análisis de períodos EDGE_OFF
    print("\n" + "="*60)
    print("BLOQUE 3 — ANÁLISIS DE PERÍODOS EDGE_OFF")
    print("="*60)
    print('"¿Qué estamos evitando?"')
    print("Asset      | Mean_OFF | Sharpe_OFF | %Neg | Worst Bar | Total PnL OFF")
    print("-" * 80)

    block3_results = []
    for asset in ASSETS:
        merged = asset_returns[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        edge_off_returns = merged['fwd_return'][~asset_mask]

        if len(edge_off_returns) > 0:
            mean_off = edge_off_returns.mean()
            std_off = edge_off_returns.std()
            sharpe_off = mean_off / std_off * np.sqrt(6 * 365) if std_off > 0 else 0
            pct_negative = (edge_off_returns < 0).mean()
            worst_bar = edge_off_returns.min()
            total_pnl_off = edge_off_returns.sum()

            print(f"{asset:10} | {mean_off*100:8.2f}% | {sharpe_off:10.2f} | {pct_negative*100:4.0f}% | {worst_bar*100:8.2f}% | {total_pnl_off*100:10.2f}%")

            block3_results.append({
                'asset': asset,
                'mean_off': mean_off,
                'sharpe_off': sharpe_off,
                'pct_negative': pct_negative,
                'worst_bar': worst_bar,
                'total_pnl_off': total_pnl_off,
            })

    print("\nINTERPRETACIÓN:")
    all_negative = all(r['mean_off'] < 0 for r in block3_results)
    all_negative_sharpe = all(r['sharpe_off'] < 0 for r in block3_results)

    if all_negative and all_negative_sharpe:
        print("✓ Mean_OFF < 0 y Sharpe_OFF < 0 → estamos evitando CORRECTAMENTE períodos malos")
    elif any(r['mean_off'] > 0 for r in block3_results):
        print("✗ Mean_OFF > 0 → estamos dejando dinero en la mesa (el filtro es demasiado agresivo)")

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
            merged_p1 = asset_returns[asset][asset_returns[asset]['timestamp'] < '2025-10-01'].copy()
            asset_mask_p1 = merged_p1['pred_proba'] >= threshold_p1

            v21_ret_p1 = merged_p1['fwd_return'].copy()
            v21_ret_p1[~asset_mask_p1] = 0

            v209_ret_p1 = merged_p1['fwd_return'].copy()

            portfolio_returns_v21_p1.append(v21_ret_p1)
            portfolio_returns_v209_p1.append(v209_ret_p1)

        portfolio_v21_p1 = pd.concat(portfolio_returns_v21_p1, axis=1).mean(axis=1)
        portfolio_v209_p1 = pd.concat(portfolio_returns_v209_p1, axis=1).mean(axis=1)

        n_total_p1 = len(portfolio_v21_p1)

        metrics_v21_p1 = calc_metrics(portfolio_v21_p1, n_total_p1, f'V21_p1_{int(pct*100)}%')
        metrics_v209_p1 = calc_metrics(portfolio_v209_p1, n_total_p1, f'V209_p1_{int(pct*100)}%')

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
    print(f"Threshold de P1: {threshold_p1:.4f}")

    # Aplicar threshold_p1 a period2
    edge_mask_p2 = period2['pred_proba'] >= threshold_p1

    portfolio_returns_v21_p2 = []
    portfolio_returns_v209_p2 = []

    for asset in ASSETS:
        merged_p2 = asset_returns[asset][asset_returns[asset]['timestamp'] >= '2025-10-01'].copy()
        asset_mask_p2 = merged_p2['pred_proba'] >= threshold_p1

        v21_ret_p2 = merged_p2['fwd_return'].copy()
        v21_ret_p2[~asset_mask_p2] = 0

        v209_ret_p2 = merged_p2['fwd_return'].copy()

        portfolio_returns_v21_p2.append(v21_ret_p2)
        portfolio_returns_v209_p2.append(v209_ret_p2)

    portfolio_v21_p2 = pd.concat(portfolio_returns_v21_p2, axis=1).mean(axis=1)
    portfolio_v209_p2 = pd.concat(portfolio_returns_v209_p2, axis=1).mean(axis=1)

    n_total_p2 = len(portfolio_v21_p2)

    metrics_v21_p2 = calc_metrics(portfolio_v21_p2, n_total_p2, 'V21_p2')
    metrics_v209_p2 = calc_metrics(portfolio_v209_p2, n_total_p2, 'V209_p2')

    delta_p1 = best_p1['sharpe_v21'] - best_p1['sharpe_v209']
    delta_p2 = metrics_v21_p2['sharpe'] - metrics_v209_p2['sharpe']

    print(f"\nPeríodo 1 (in-sample): 2025-01 a 2025-09")
    print(f"  Portfolio Sharpe V21: {best_p1['sharpe_v21']:.2f} | Sharpe V20.9: {best_p1['sharpe_v209']:.2f} | Δ: {delta_p1:+.2f}")

    print(f"\nPeríodo 2 (out-of-sample): 2025-10 a 2026-04")
    print(f"  Portfolio Sharpe V21: {metrics_v21_p2['sharpe']:.2f} | Sharpe V20.9: {metrics_v209_p2['sharpe']:.2f} | Δ: {delta_p2:+.2f}")

    walk_forward_pass = delta_p2 > 0

    if walk_forward_pass:
        print(f"\n✓ Δ > 0 en P2 → ROBUSTO")
    else:
        print(f"\n✗ Δ ≤ 0 en P2 → OVERFITTEADO")

    # VEREDICTO FINAL
    print("\n" + "="*60)
    print("VEREDICTO FINAL — V21.0")
    print("="*60)

    verdict, criteria, n_pass = final_verdict(metrics_v21, metrics_v209, walk_forward_pass)

    print("\nCRITERIOS DE VIDA O MUERTE:")
    print()

    print(f"1. Sharpe(V21) > Sharpe(V20.9)?")
    print(f"   V21: {metrics_v21['sharpe']:.2f} | V20.9: {metrics_v209['sharpe']:.2f} | Δ: {delta_sharpe:+.2f} → {'PASS' if criteria['sharpe_better'] else 'FAIL'}")
    print()

    print(f"2. MaxDD(V21) ≤ MaxDD(V20.9)?  (menos negativo = mejor)")
    print(f"   V21: {metrics_v21['max_dd']*100:.2f}% | V20.9: {metrics_v209['max_dd']*100:.2f}% | → {'PASS' if criteria['dd_better_or_equal'] else 'FAIL'}")
    print()

    print(f"3. Exposure significativamente menor que 100%?")
    print(f"   V21: {metrics_v21['exposure']*100:.0f}% | V20.9: 100% | Umbral: < 80% → {'PASS' if criteria['exposure_reduced'] else 'FAIL'}")
    print()

    print(f"4. Walk-forward validation? (Δ > 0 en período out-of-sample)")
    print(f"   P1 Δ: {delta_p1:+.2f} | P2 Δ: {delta_p2:+.2f} | → {'PASS' if criteria['walk_forward'] else 'FAIL'}")
    print()

    print(f"RESULTADO: {n_pass}/4 criterios PASS")
    print()
    print("="*60)
    print(verdict)
    print("="*60)
    print(f"Percentil óptimo: {int(best_pct*100)}%")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Recomendación: {'PROCEDER A PAPER TRADING' if n_pass >= 3 else 'ABORTAR' if n_pass <= 1 else 'REVISAR'}")
    print("="*60)

    # Telegram
    print("\nEnviando a Telegram...")
    msg = (
        'ORION V21.0 BACKTEST DEFINITIVO\n'
        f'Sharpe V21: {metrics_v21["sharpe"]:.2f} vs V20.9: {metrics_v209["sharpe"]:.2f}\n'
        f'DD V21: {metrics_v21["max_dd"]*100:.1f}% vs V20.9: {metrics_v209["max_dd"]*100:.1f}%\n'
        f'Exposure: {metrics_v21["exposure"]*100:.0f}%\n'
        f'Walk-forward: {"PASS" if walk_forward_pass else "FAIL"}\n'
        f'Percentil: {int(best_pct*100)}%\n'
        f'Criterios: {n_pass}/4\n'
        f'VEREDICTO: {verdict}'
    )

    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'message_thread_id': TOPIC_ID,
        'text': msg,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        print(f'Telegram: {resp.status_code}')
    except Exception as e:
        print(f'Telegram error: {e}')

    # Guardar resultados
    print("\nGuardando resultados...")

    # CSV detallado barra por barra
    results_path = os.path.expanduser('~/orion/v21_backtest_results.csv')
    results_rows = []

    for asset in ASSETS:
        merged = asset_returns[asset]
        asset_mask = merged['pred_proba'] >= best_threshold

        v21_ret = merged['fwd_return'].copy()
        v21_ret[~asset_mask] = 0

        v209_ret = merged['fwd_return'].copy()

        cum_v21 = v21_ret.cumsum()
        cum_v209 = v209_ret.cumsum()

        for i, row in merged.iterrows():
            results_rows.append({
                'timestamp': row['timestamp'],
                'asset': asset,
                'pred_proba': row['pred_proba'],
                'edge_on': asset_mask.iloc[i] if i < len(asset_mask) else False,
                'fwd_return': row['fwd_return'],
                'v21_return': v21_ret.iloc[i] if i < len(v21_ret) else 0,
                'v209_return': v209_ret.iloc[i] if i < len(v209_ret) else 0,
                'cumulative_v21': cum_v21.iloc[i] if i < len(cum_v21) else 0,
                'cumulative_v209': cum_v209.iloc[i] if i < len(cum_v209) else 0,
            })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_path, index=False)
    print(f'Resultados guardados en: {results_path}')

    # CSV resumen
    summary_path = os.path.expanduser('~/orion/lab/v21_backtest_summary.csv')
    summary_rows = []

    for r in block1_results:
        for asset in ASSETS:
            summary_rows.append({
                'percentile': r['percentile'],
                'asset': asset,
                'strategy': 'V21',
                'total_return': 0,
                'sharpe': r['sharpe_on'],
                'max_dd': r['dd_on'],
                'calmar': r['calmar_on'],
                'exposure': r['exposure'],
                'n_trades': 0,
            })
            summary_rows.append({
                'percentile': r['percentile'],
                'asset': asset,
                'strategy': 'V20.9',
                'total_return': 0,
                'sharpe': r['sharpe_all'],
                'max_dd': r['dd_all'],
                'calmar': 0,
                'exposure': 1.0,
                'n_trades': 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f'Resumen guardado en: {summary_path}')

    print("\n" + "="*60)
    print("BACKTEST COMPLETADO")
    print("="*60)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
