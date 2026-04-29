#!/usr/bin/env python3
# Orion V21.0 Edge Detection - LightGBM Training Script
# 3 sequential phases: Autocorrelation Gate -> Feature Engineering -> Training

import os
import sys
import pickle
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Telegram config
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = '-1003505760554'
TOPIC_ID = 972

# Target parameters
H = 12  # 48 hours forward (12 * 4H bars)
K = 1.0  # volatility multiplier
D = 0.5  # threshold

# Assets
ASSETS = ['ETH-USDT', 'BTC-USDT', 'SOL-USDT']

# Feature config
LAG_FEATURES = ['bb_width', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14']
LAGS = [1, 3, 6]  # 4H, 12H, 24H

# LightGBM params
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 80,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'verbose': -1
}

# Kill switches
KILL_SWITCHES = {
    'auc_min': 0.55,
    'auc_borderline': 0.58,
    'auc_exploitable': 0.62,
    'precision_top_k': 0.20,
    'sharpe_edge_on': 0.0
}


def send_telegram(message):
    if not BOT_TOKEN:
        print("WARNING: No TELEGRAM_BOT_TOKEN found")
        return
    try:
        import requests
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
    exchange = ccxt.okx()
    all_candles = []
    since = None
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        if len(candles) < limit:
            break
        since = candles[-1][0] + 1
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_target(df, horizon, k, d):
    """
    Construye target de edge detection.

    horizon: barras forward (6 = 1 día en 4H)
    k: multiplicador de vol para threshold de retorno
    d: tolerancia de path adverso (fracción de vol)
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Realized vol 20 días (120 barras de 4H)
    df['realized_vol_20d'] = df['log_return'].rolling(120).std() * np.sqrt(6)

    # Pre-calcular retornos acumulados forward y max drawdown forward (para evaluación)
    log_ret = df['log_return'].values
    n = len(df)

    cum_returns = np.full(n, np.nan)
    max_dd_paths = np.full(n, np.nan)

    for i in range(n - horizon):
        future_returns = log_ret[i+1 : i+1+horizon]
        cum_returns[i] = np.sum(future_returns)
        cum_path = np.cumsum(future_returns)
        max_dd_paths[i] = np.min(cum_path)

    df['cum_return_forward'] = cum_returns
    df['max_dd_path_forward'] = max_dd_paths

    # Calcular target
    targets = []
    vol = df['realized_vol_20d'].values

    for i in range(len(df)):
        if i >= len(df) - horizon:
            targets.append(np.nan)
            continue

        v = vol[i]
        if np.isnan(v) or v < 1e-8:
            targets.append(np.nan)
            continue

        cum_return = cum_returns[i]
        max_dd_path = max_dd_paths[i]

        return_condition = abs(cum_return) > k * v
        path_condition = max_dd_path > -d * v

        edge = 1 if (return_condition and path_condition) else 0
        targets.append(edge)

    df['edge'] = targets
    return df


def analyze_label_autocorr(df, asset):
    edge = df['edge'].dropna().values
    if len(edge) < 100:
        return None
    n_blocks = 0
    block_sizes = []
    current_block = 0
    for val in edge:
        if val == 1:
            current_block += 1
        else:
            if current_block > 0:
                block_sizes.append(current_block)
                n_blocks += 1
                current_block = 0
    if current_block > 0:
        block_sizes.append(current_block)
        n_blocks += 1
    if len(block_sizes) == 0:
        return None
    mean_block_size = np.mean(block_sizes)
    median_block_size = np.median(block_sizes)
    max_block_size = np.max(block_sizes)
    p90_block_size = np.percentile(block_sizes, 90)
    pct_isolated = np.mean([1 for s in block_sizes if s == 1])
    autocorr_lag1 = np.corrcoef(edge[:-1], edge[1:])[0, 1]
    return {
        'asset': asset,
        'n_blocks': n_blocks,
        'mean_block_size': mean_block_size,
        'median_block_size': median_block_size,
        'max_block_size': max_block_size,
        'p90_block_size': p90_block_size,
        'pct_isolated': pct_isolated,
        'autocorr_lag1': autocorr_lag1,
        'edge_rate': np.mean(edge)
    }


def phase0_autocorrelation_gate():
    print("\n" + "="*60)
    print("FASE 0: AUTOCORRELACIÓN DE LABELS (GATE OBLIGATORIO)")
    print("="*60 + "\n")
    results = []
    for asset in ASSETS:
        print(f"Procesando {asset}...")
        df = fetch_all_ohlcv(asset, '4h', limit=300)
        df = compute_target(df, horizon=H, k=K, d=D)

        # SANITY CHECK OBLIGATORIO
        edge_rate = df['edge'].dropna().mean()
        print(f'SANITY CHECK edge_rate: {edge_rate:.1%} (esperado ~18%)')
        assert 0.10 < edge_rate < 0.30, f'edge_rate {edge_rate:.1%} fuera de rango — compute_target está MAL'

        stats = analyze_label_autocorr(df, asset)
        if stats:
            results.append(stats)
    if not results:
        print("ERROR: No se pudo calcular estadísticas de labels")
        return False, None
    df_stats = pd.DataFrame(results)
    print("\nEstadísticas de Labels por Asset:")
    print(df_stats.to_string(index=False))
    print("\n")
    max_autocorr = df_stats['autocorr_lag1'].max()
    max_mean_block = df_stats['mean_block_size'].max()
    if max_autocorr > 0.7:
        print("RED FLAG: autocorr_lag1 > 0.7")
        print(f"Max autocorr_lag1: {max_autocorr:.3f}")
        print("ABORT: No entrenar modelo")
        return False, df_stats
    if max_mean_block > 8:
        print("WARNING: mean_block_size > 8")
        print(f"Max mean_block_size: {max_mean_block:.1f}")
        print("Continuar con cautela")
    print("GATE PASSED: Proceder a Fase 1")
    return True, df_stats


def compute_features(df):
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * df['std20']
    df['bb_lower'] = df['sma20'] - 2 * df['std20']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
    df['bb_width_hist'] = df['bb_width'].rolling(500).quantile(0.2)
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
    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr14'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr14'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['trend_strength'] = df['dx'].rolling(14).mean()
    df['price_slope_20'] = df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['rsi_slope_20'] = df['rsi_14'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['momentum_divergence'] = np.sign(df['price_slope_20']) - np.sign(df['rsi_slope_20'])
    df['vol_of_vol'] = df['atr14'].rolling(20).std()
    for feat in LAG_FEATURES:
        for lag in LAGS:
            df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
    for feat in LAG_FEATURES:
        df[f'{feat}_roc'] = df[feat] / df[feat].shift(1) - 1
    return df


def phase1_feature_engineering():
    print("\n" + "="*60)
    print("FASE 1: FEATURE ENGINEERING (29 features)")
    print("="*60 + "\n")
    all_data = {}
    for asset in ASSETS:
        print(f"Procesando {asset}...")
        df = fetch_all_ohlcv(asset, '4h', limit=300)
        df = compute_target(df, horizon=H, k=K, d=D)
        df = compute_features(df)
        df['asset'] = asset
        all_data[asset] = df
        print(f"  {asset}: {len(df)} barras")
    print("\nFeatures core (9):")
    core_features = ['bb_width', 'squeeze_duration', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14', 'trend_strength', 'momentum_divergence', 'vol_of_vol']
    for feat in core_features:
        print(f"  - {feat}")
    print("\nLag features (15):")
    for feat in LAG_FEATURES:
        for lag in LAGS:
            print(f"  - {feat}_lag{lag}")
    print("\nROC features (5):")
    for feat in LAG_FEATURES:
        print(f"  - {feat}_roc")
    print(f"\nTotal: 29 features")
    return all_data


def prepare_train_test(all_data):
    print("\n" + "="*60)
    print("PREPARACIÓN DE DATOS: Split Temporal")
    print("="*60 + "\n")
    feature_cols = ['bb_width', 'squeeze_duration', 'atr_compression', 'vol_regime', 'volume_ratio', 'rsi_14', 'trend_strength', 'momentum_divergence', 'vol_of_vol']
    for feat in LAG_FEATURES:
        for lag in LAGS:
            feature_cols.append(f'{feat}_lag{lag}')
    for feat in LAG_FEATURES:
        feature_cols.append(f'{feat}_roc')
    train_data = {}
    test_data = {}
    split_date = pd.Timestamp('2024-12-31')
    purge_gap = 12
    for asset in ASSETS:
        df = all_data[asset].copy()
        df = df.dropna(subset=feature_cols + ['edge', 'cum_return_forward'])
        train_mask = df['timestamp'] <= (split_date - pd.Timedelta(hours=purge_gap*4))
        test_mask = df['timestamp'] > split_date
        train_data[asset] = df[train_mask]
        test_data[asset] = df[test_mask]
        print(f"{asset}:")
        print(f"  Train: {len(train_data[asset])} muestras (<= {split_date - pd.Timedelta(hours=purge_gap*4)})")
        print(f"  Test: {len(test_data[asset])} muestras (> {split_date})")
    return train_data, test_data, feature_cols


def train_model(X_train, y_train, X_test, y_test, model_name):
    print(f"\nEntrenando modelo: {model_name}")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return model


def evaluate_model(model, X_test, y_test, returns_test, model_name):
    print(f"\nEvaluando modelo: {model_name}")
    print("-" * 60)
    pred_proba = model.predict(X_test)
    pred = (pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, pred_proba)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(f"Métricas Base:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    edge_rate = y_test.mean()
    print(f"\nPrecision@top_k:")
    for k_pct in [0.10, 0.15, 0.20, 0.30]:
        k = int(len(pred_proba) * k_pct)
        top_k_idx = np.argsort(pred_proba)[-k:]
        precision_k = y_test.iloc[top_k_idx].mean()
        print(f"  top_{int(k_pct*100)}%: {precision_k:.4f} (baseline edge_rate: {edge_rate:.4f})")
    print(f"\nSharpe por decil:")
    deciles = pd.qcut(pred_proba, 10, duplicates='drop', labels=False)
    sharpe_by_decil = []
    for d in range(10):
        mask = deciles == d
        if mask.sum() > 0:
            decil_returns = returns_test[mask]
            mean_ret = decil_returns.mean()
            std_ret = decil_returns.std()
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            sharpe_by_decil.append((d, mean_ret, sharpe))
    for d, mean_ret, sharpe in sharpe_by_decil:
        print(f"  Decil {d}: mean_return={mean_ret:.4f}, sharpe={sharpe:.4f}")
    print(f"\nPnL Attribution:")
    edge_on_mask = pred_proba >= 0.5
    edge_off_mask = pred_proba < 0.5
    if edge_on_mask.sum() > 0:
        edge_on_returns = returns_test[edge_on_mask]
        edge_on_pnl = edge_on_returns.sum()
        edge_on_sharpe = edge_on_returns.mean() / edge_on_returns.std() if edge_on_returns.std() > 0 else 0
        print(f"  EDGE_ON: PnL={edge_on_pnl:.4f}, Sharpe={edge_on_sharpe:.4f}")
    if edge_off_mask.sum() > 0:
        edge_off_returns = returns_test[edge_off_mask]
        edge_off_pnl = edge_off_returns.sum()
        edge_off_sharpe = edge_off_returns.mean() / edge_off_returns.std() if edge_off_returns.std() > 0 else 0
        print(f"  EDGE_OFF: PnL={edge_off_pnl:.4f}, Sharpe={edge_off_sharpe:.4f}")
    total_pnl = returns_test.sum()
    total_sharpe = returns_test.mean() / returns_test.std() if returns_test.std() > 0 else 0
    print(f"  TOTAL: PnL={total_pnl:.4f}, Sharpe={total_sharpe:.4f}")
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print(f"\nTop 15 Feature Importance (gain):")
    for i, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    results = {
        'model': model_name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'edge_rate': edge_rate,
        'precision_top_10': y_test.iloc[np.argsort(pred_proba)[-int(len(pred_proba)*0.10):]].mean(),
        'precision_top_15': y_test.iloc[np.argsort(pred_proba)[-int(len(pred_proba)*0.15):]].mean(),
        'precision_top_20': y_test.iloc[np.argsort(pred_proba)[-int(len(pred_proba)*0.20):]].mean(),
        'precision_top_30': y_test.iloc[np.argsort(pred_proba)[-int(len(pred_proba)*0.30):]].mean(),
        'edge_on_pnl': edge_on_pnl if edge_on_mask.sum() > 0 else 0,
        'edge_on_sharpe': edge_on_sharpe if edge_on_mask.sum() > 0 else 0,
        'edge_off_pnl': edge_off_pnl if edge_off_mask.sum() > 0 else 0,
        'edge_off_sharpe': edge_off_sharpe if edge_off_mask.sum() > 0 else 0,
        'total_pnl': total_pnl,
        'total_sharpe': total_sharpe
    }
    return results, imp_df


def phase2_training(train_data, test_data, feature_cols):
    print("\n" + "="*60)
    print("FASE 2: ENTRENAMIENTO LIGHTGBM")
    print("="*60 + "\n")
    models = {}
    all_results = []
    all_importance = {}
    for asset in ASSETS:
        X_train = train_data[asset][feature_cols]
        y_train = train_data[asset]['edge']
        X_test = test_data[asset][feature_cols]
        y_test = test_data[asset]['edge']
        returns_test = test_data[asset]['cum_return_forward']
        model = train_model(X_train, y_train, X_test, y_test, asset)
        results, imp_df = evaluate_model(model, X_test, y_test, returns_test, asset)
        models[asset] = model
        all_results.append(results)
        all_importance[asset] = imp_df
    print("\n" + "="*60)
    print("CROSS-ASSET MODEL")
    print("="*60 + "\n")
    X_train_all = pd.concat([train_data[asset][feature_cols] for asset in ASSETS])
    y_train_all = pd.concat([train_data[asset]['edge'] for asset in ASSETS])
    X_test_all = pd.concat([test_data[asset][feature_cols] for asset in ASSETS])
    y_test_all = pd.concat([test_data[asset]['edge'] for asset in ASSETS])
    returns_test_all = pd.concat([test_data[asset]['cum_return_forward'] for asset in ASSETS])
    model_cross = train_model(X_train_all, y_train_all, X_test_all, y_test_all, 'CROSS-ASSET')
    results_cross, imp_df_cross = evaluate_model(model_cross, X_test_all, y_test_all, returns_test_all, 'CROSS-ASSET')
    models['CROSS-ASSET'] = model_cross
    all_results.append(results_cross)
    all_importance['CROSS-ASSET'] = imp_df_cross
    return models, all_results, all_importance


def check_kill_switches(results):
    print("\n" + "="*60)
    print("KILL SWITCHES")
    print("="*60 + "\n")
    verdicts = {}
    for res in results:
        model = res['model']
        auc = res['auc']
        precision_top_20 = res['precision_top_20']
        edge_rate = res['edge_rate']
        edge_on_sharpe = res['edge_on_sharpe']
        total_sharpe = res['total_sharpe']
        verdict = "ALIVE"
        if auc < KILL_SWITCHES['auc_min']:
            verdict = "V21.0 MUERE"
        elif auc < KILL_SWITCHES['auc_borderline']:
            verdict = "BORDERLINE"
        elif auc < KILL_SWITCHES['auc_exploitable']:
            verdict = "EXPLOTABLE"
        else:
            verdict = "TERRENO SERIO"
        if precision_top_20 < edge_rate:
            verdict = "KILL (Precision@top_20% < edge_rate)"
        if edge_on_sharpe <= total_sharpe:
            verdict = "KILL (Sharpe EDGE_ON <= Sharpe TOTAL)"
        verdicts[model] = verdict
        print(f"{model}:")
        print(f"  AUC: {auc:.4f} -> {verdict}")
        print(f"  Precision@top_20%: {precision_top_20:.4f} vs edge_rate: {edge_rate:.4f}")
        print(f"  Sharpe EDGE_ON: {edge_on_sharpe:.4f} vs Sharpe TOTAL: {total_sharpe:.4f}")
    return verdicts


def save_results(models, all_results, all_importance, autocorr_stats):
    print("\n" + "="*60)
    print("GUARDANDO RESULTADOS")
    print("="*60 + "\n")
    lab_dir = os.path.expanduser('~/orion/lab')
    os.makedirs(lab_dir, exist_ok=True)
    model_path = os.path.join(lab_dir, 'v21_lgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Modelo guardado: {model_path}")
    results_df = pd.DataFrame(all_results)
    results_path = os.path.expanduser('~/orion/v21_lgbm_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Resultados guardados: {results_path}")
    all_imp_dfs = []
    for model, imp_df in all_importance.items():
        imp_df['model'] = model
        all_imp_dfs.append(imp_df)
    combined_imp = pd.concat(all_imp_dfs)
    imp_path = os.path.join(lab_dir, 'v21_feature_importance.csv')
    combined_imp.to_csv(imp_path, index=False)
    print(f"Feature importance guardado: {imp_path}")
    if autocorr_stats is not None:
        autocorr_path = os.path.join(lab_dir, 'v21_autocorr_stats.csv')
        autocorr_stats.to_csv(autocorr_path, index=False)
        print(f"Autocorr stats guardado: {autocorr_path}")


def send_summary(all_results, verdicts):
    print("\n" + "="*60)
    print("ENVIANDO RESUMEN A TELEGRAM")
    print("="*60 + "\n")
    summary_lines = ["*Orion V21.0 Edge Detection - Training Results*\n"]
    for res in all_results:
        model = res['model']
        verdict = verdicts.get(model, 'UNKNOWN')
        summary_lines.append(f"*{model}*")
        summary_lines.append(f"AUC: {res['auc']:.4f}")
        summary_lines.append(f"Precision: {res['precision']:.4f}")
        summary_lines.append(f"Recall: {res['recall']:.4f}")
        summary_lines.append(f"F1: {res['f1']:.4f}")
        summary_lines.append(f"Precision@top_20%: {res['precision_top_20']:.4f}")
        summary_lines.append(f"EDGE_ON Sharpe: {res['edge_on_sharpe']:.4f}")
        summary_lines.append(f"Veredicto: {verdict}")
        summary_lines.append("")
    message = "\n".join(summary_lines)
    send_telegram(message)
    print("Resumen enviado a Telegram")


def main():
    print("\n" + "="*60)
    print("ORION V21.0 EDGE DETECTION - LIGHTGBM TRAINING")
    print("="*60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {', '.join(ASSETS)}")
    print(f"Target params: h={H}, k={K}, d={D}")
    gate_passed, autocorr_stats = phase0_autocorrelation_gate()
    if not gate_passed:
        print("\nABORT: Fase 0 falló - No entrenar modelo")
        return
    all_data = phase1_feature_engineering()
    train_data, test_data, feature_cols = prepare_train_test(all_data)
    models, all_results, all_importance = phase2_training(train_data, test_data, feature_cols)
    verdicts = check_kill_switches(all_results)
    save_results(models, all_results, all_importance, autocorr_stats)
    send_summary(all_results, verdicts)
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
