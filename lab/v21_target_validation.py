import ccxt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import requests
import datetime
import os
import time

# ======================================================
# CONSTANTS
# ======================================================
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = '-1003505760554'
TOPIC_ID = 972

ASSETS = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT']
HORIZONS = [8, 12, 16, 24]
K_VALUES = [0.6, 0.8, 1.0]
D_VALUES = [0.5, 0.6, 0.7]

DATA_DIR = os.path.expanduser('~/orion/data')
OUTPUT_CSV = os.path.expanduser('~/orion/v21_target_validation.csv')

# ======================================================
# DATA FETCHING
# ======================================================
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
            time.sleep(0.1)  # rate limit

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

        safe_symbol = symbol.replace('/', '_')
        csv_path = os.path.join(DATA_DIR, f'{safe_symbol}_4h.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f'Cargado desde CSV: {csv_path} ({len(df)} filas)')
            return df
        else:
            raise FileNotFoundError(
                f'No se pudo descargar de OKX ni encontrar CSV en {csv_path}'
            )

# ======================================================
# TARGET COMPUTATION
# ======================================================
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

# ======================================================
# BLOQUE 1: Distribución del Target
# ======================================================
def run_block_1(all_data):
    """
    Calcula distribución del target para todo el grid.
    Retorna lista de dicts con resultados.
    """
    results = []

    for asset in ASSETS:
        df = all_data[asset]
        n_total_data = len(df)

        for horizon in HORIZONS:
            for k in K_VALUES:
                for d in D_VALUES:
                    df_computed = compute_target(df.copy(), horizon, k, d)

                    # Contar
                    edge_series = df_computed['edge']
                    valid_mask = ~edge_series.isna()
                    n_total = valid_mask.sum()
                    n_edge1 = (edge_series[valid_mask] == 1).sum()
                    edge_rate = n_edge1 / n_total if n_total > 0 else 0

                    # Status
                    if edge_rate < 0.15:
                        status = "DEMASIADO ESTRICTO"
                    elif edge_rate > 0.70:
                        status = "DEMASIADO LAXO"
                    else:
                        status = "OK"

                    result = {
                        'asset': asset,
                        'h': horizon,
                        'k': k,
                        'd': d,
                        'n_total': n_total,
                        'n_edge1': n_edge1,
                        'edge_rate': edge_rate,
                        'status': status
                    }
                    results.append(result)

    return results

# ======================================================
# BLOQUE 2: Calidad del Edge
# ======================================================
def run_block_2(all_data, block1_results):
    """
    Calcula calidad del edge para configs OK del Bloque 1.
    Retorna lista de dicts con métricas.
    """
    results = []

    for r in block1_results:
        if r['status'] != "OK":
            continue

        asset = r['asset']
        horizon = r['h']
        k = r['k']
        d = r['d']

        df = all_data[asset].copy()
        df_computed = compute_target(df, horizon, k, d)

        # Filtrar filas con edge definido
        valid = ~df_computed['edge'].isna()
        edge_1 = df_computed[valid & (df_computed['edge'] == 1)]
        edge_0 = df_computed[valid & (df_computed['edge'] == 0)]

        if len(edge_1) == 0 or len(edge_0) == 0:
            continue

        # Métricas
        mean_return_edge1 = edge_1['cum_return_forward'].mean()
        mean_return_edge0 = edge_0['cum_return_forward'].mean()
        std_return_edge1 = edge_1['cum_return_forward'].std()
        std_return_edge0 = edge_0['cum_return_forward'].std()

        sharpe_edge1 = mean_return_edge1 / std_return_edge1 if std_return_edge1 > 0 else np.nan
        sharpe_edge0 = mean_return_edge0 / std_return_edge0 if std_return_edge0 > 0 else np.nan

        all_returns = df_computed.loc[valid, 'cum_return_forward']
        sharpe_total = all_returns.mean() / all_returns.std() if all_returns.std() > 0 else np.nan

        mean_dd_path_edge1 = edge_1['max_dd_path_forward'].mean()
        mean_dd_path_edge0 = edge_0['max_dd_path_forward'].mean()

        result = {
            'asset': asset,
            'h': horizon,
            'k': k,
            'd': d,
            'mean_return_edge1': mean_return_edge1,
            'mean_return_edge0': mean_return_edge0,
            'std_return_edge1': std_return_edge1,
            'std_return_edge0': std_return_edge0,
            'sharpe_edge1': sharpe_edge1,
            'sharpe_edge0': sharpe_edge0,
            'sharpe_total': sharpe_total,
            'mean_dd_path_edge1': mean_dd_path_edge1,
            'mean_dd_path_edge0': mean_dd_path_edge0,
            'n_edge1': len(edge_1),
            'n_edge0': len(edge_0),
            'edge_rate': len(edge_1) / (len(edge_1) + len(edge_0)) if (len(edge_1) + len(edge_0)) > 0 else np.nan
        }
        results.append(result)

    return results

# ======================================================
# BLOQUE 3: Separación Estadística
# ======================================================
def run_block_3(all_data, block1_results):
    """
    Calcula separación estadística para configs OK del Bloque 1.
    Retorna lista de dicts con t-stat y p-value.
    """
    results = []

    for r in block1_results:
        if r['status'] != "OK":
            continue

        asset = r['asset']
        horizon = r['h']
        k = r['k']
        d = r['d']

        df = all_data[asset].copy()
        df_computed = compute_target(df, horizon, k, d)

        valid = ~df_computed['edge'].isna()
        edge_1 = df_computed[valid & (df_computed['edge'] == 1)]
        edge_0 = df_computed[valid & (df_computed['edge'] == 0)]

        if len(edge_1) < 2 or len(edge_0) < 2:
            continue

        # Probabilidad de retorno positivo
        p_pos_edge1 = (edge_1['cum_return_forward'] > 0).mean()
        p_pos_edge0 = (edge_0['cum_return_forward'] > 0).mean()
        separation = p_pos_edge1 - p_pos_edge0

        # T-test
        try:
            t_stat, p_value = ttest_ind(
                edge_1['cum_return_forward'].dropna(),
                edge_0['cum_return_forward'].dropna(),
                equal_var=False
            )
        except:
            t_stat = np.nan
            p_value = np.nan

        result = {
            'asset': asset,
            'h': horizon,
            'k': k,
            'd': d,
            'p_pos_edge1': p_pos_edge1,
            'p_pos_edge0': p_pos_edge0,
            'separation': separation,
            't_stat': t_stat,
            'p_value': p_value
        }
        results.append(result)

    return results

# ======================================================
# BLOQUE 4: Robustez Temporal
# ======================================================
def run_block_4(all_data, block1_results):
    """
    Calcula robustez temporal por quartiles.
    Retorna lista de dicts con sharpes por quartile.
    """
    results = []

    for r in block1_results:
        if r['status'] != "OK":
            continue

        asset = r['asset']
        horizon = r['h']
        k = r['k']
        d = r['d']

        df = all_data[asset].copy()
        n = len(df)

        # Crear quartiles
        quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        df['quartile'] = pd.qcut(df.index, 4, labels=quartile_labels, duplicates='drop')

        df_computed = compute_target(df, horizon, k, d)

        sharpes = {}
        valid_count = 0
        concentrations = []
        alphas = []

        for q in quartile_labels:
            subset = df_computed[df_computed['quartile'] == q]
            edge_1_q = subset[subset['edge'] == 1]

            if len(edge_1_q) > 0:
                sharpe_q = edge_1_q['cum_return_forward'].mean() / edge_1_q['cum_return_forward'].std() \
                    if edge_1_q['cum_return_forward'].std() > 0 else np.nan
                alpha_q = edge_1_q['cum_return_forward'].mean()
            else:
                sharpe_q = np.nan
                alpha_q = 0

            sharpes[q] = sharpe_q
            alphas.append(max(alpha_q, 0))

        # Contar cuántos quartiles tienen Sharpe > 0
        for q in quartile_labels:
            if sharpes[q] is not None and sharpes[q] > 0:
                valid_count += 1

        consistent = valid_count
        total_alpha = sum(alphas)
        max_concentration = max(alphas) / total_alpha if total_alpha > 0 else 1.0
        single_period_fail = max_concentration > 0.80

        if valid_count >= 3 and not single_period_fail:
            status = "PASS"
        else:
            status = "FAIL"

        result = {
            'asset': asset,
            'h': horizon,
            'k': k,
            'd': d,
            'Q1_sharpe': sharpes.get('Q1', np.nan),
            'Q2_sharpe': sharpes.get('Q2', np.nan),
            'Q3_sharpe': sharpes.get('Q3', np.nan),
            'Q4_sharpe': sharpes.get('Q4', np.nan),
            'consistent': f"{valid_count}/4",
            'single_period_fail': single_period_fail,
            'status': status
        }
        results.append(result)

    return results

# ======================================================
# BLOQUE 5: Heatmap de Sensibilidad
# ======================================================
def run_block_5(block2_results):
    """
    Identifica mejor configuración por asset.
    Retorna dict {asset: best_config}.
    """
    # Filtrar: edge_rate entre 25% y 55%
    eligible = []
    for r in block2_results:
        edge_rate = r['edge_rate']
        if 0.25 <= edge_rate <= 0.55:
            eligible.append(r)

    best_configs = {}

    for asset in ASSETS:
        asset_configs = [c for c in eligible if c['asset'] == asset]

        if not asset_configs:
            best_configs[asset] = None
            continue

        # Maximizar sharpe_edge1
        best = max(asset_configs, key=lambda x: x['sharpe_edge1'] if not np.isnan(x['sharpe_edge1']) else -999)

        best_configs[asset] = {
            'h': best['h'],
            'k': best['k'],
            'd': best['d'],
            'edge_rate': best['edge_rate'],
            'sharpe_edge1': best['sharpe_edge1']
        }

    return best_configs

# ======================================================
# RED FLAGS EVALUATION
# ======================================================
def evaluate_red_flags(best_configs, block2_results, block3_results):
    """
    Evalúa kill switch en las mejores configuraciones.
    """
    killed = False
    verdict_details = []

    for asset in ASSETS:
        best = best_configs.get(asset)
        if best is None:
            print(f"RED FLAG: {asset} -- No hay configuración elegible")
            killed = True
            continue

        h, k, d = best['h'], best['k'], best['d']

        # Buscar métricas de block2 y block3
        b2 = next((x for x in block2_results
                   if x['asset'] == asset and x['h'] == h and x['k'] == k and x['d'] == d), None)
        b3 = next((x for x in block3_results
                   if x['asset'] == asset and x['h'] == h and x['k'] == k and x['d'] == d), None)

        if b2 is None or b3 is None:
            print(f"RED FLAG: {asset} h={h} -- Falta métrica en bloque 2 o 3")
            killed = True
            continue

        sharpe_edge1 = b2['sharpe_edge1']
        sharpe_total = b2['sharpe_total']
        p_pos_edge1 = b3['p_pos_edge1']
        p_pos_edge0 = b3['p_pos_edge0']
        edge_rate = best['edge_rate']

        # Condición 1: Sharpe edge1 > Sharpe total
        if sharpe_edge1 <= sharpe_total:
            msg = (f"RED FLAG: {asset} h={h} -- "
                   f"Sharpe edge1 ({sharpe_edge1:.3f}) <= "
                   f"Sharpe total ({sharpe_total:.3f})")
            print(msg)
            verdict_details.append(msg)
            killed = True

        # Condición 2: Separación >= 5pp
        separation = p_pos_edge1 - p_pos_edge0
        if separation < 0.05:
            msg = (f"RED FLAG: {asset} h={h} -- "
                   f"Sin separación ({p_pos_edge1:.1%} vs {p_pos_edge0:.1%})")
            print(msg)
            verdict_details.append(msg)
            killed = True

        # Condición 3: edge_rate entre 10% y 80%
        if edge_rate < 0.10 or edge_rate > 0.80:
            msg = (f"RED FLAG: {asset} h={h} -- edge_rate={edge_rate:.1%} fuera de rango")
            print(msg)
            verdict_details.append(msg)
            killed = True

    print()
    print("=" * 60)
    if killed:
        print("VEREDICTO: NO ENTRENAR MODELO. REDEFINIR TARGET.")
    else:
        print("VEREDICTO: TARGET VÁLIDO. PROCEDER A ENTRENAMIENTO V21.0.")
    print("=" * 60)

    return not killed, verdict_details

# ======================================================
# TELEGRAM NOTIFICATION
# ======================================================
def send_telegram(message):
    import requests
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'message_thread_id': TOPIC_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            print('Telegram: mensaje enviado OK')
        else:
            print(f'Telegram: error {r.status_code}: {r.text}')
    except Exception as e:
        print(f'Telegram: excepción {e}')

# ======================================================
# PRINT HELPERS
# ======================================================
def print_table(headers, rows, widths):
    """Imprime tabla alineada."""
    header_line = ' | '.join(f'{h:<{w}}' for h, w in zip(headers, widths))
    print(header_line)
    print('-' * len(header_line))

    for row in rows:
        formatted = []
        for v, w in zip(row, widths):
            if isinstance(v, float):
                if abs(v) >= 1000:
                    formatted.append(f'{v:>{w}.2f}')
                elif abs(v) >= 10:
                    formatted.append(f'{v:>{w}.2f}')
                elif abs(v) < 0.001 and v != 0:
                    formatted.append(f'{v:>{w}.4f}')
                else:
                    formatted.append(f'{v:>{w}.3f}')
            else:
                formatted.append(f'{str(v):<{w}}')
        row_line = ' | '.join(formatted)
        print(row_line)

def format_percentage(x):
    if isinstance(x, float):
        return f'{x:.1%}'
    return str(x)

# ======================================================
# MAIN
# ======================================================
def main():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 60)
    print(" ORION V21.0 -- VALIDACIÓN DE TARGET")
    print(f" Fecha: {timestamp}")
    print(f" Assets: {', '.join(ASSETS)}")
    print(f" Grid: h={HORIZONS} x k={K_VALUES} x d={D_VALUES}")
    print("=" * 60)
    print()

    # ============================================
    # DESCARGAR DATOS
    # ============================================
    print("Descargando datos de OKX...")
    all_data = {}
    for asset in ASSETS:
        try:
            df = fetch_all_ohlcv(asset, timeframe='4h', limit=300)
            all_data[asset] = df
        except Exception as e:
            print(f'ERROR: No se pudieron obtener datos para {asset}: {e}')
            return

    print(f"Datos descargados: {len(all_data)} assets")
    print()

    # ============================================
    # BLOQUE 1
    # ============================================
    print("BLOQUE 1 -- DISTRIBUCIÓN DEL TARGET")
    print("-" * 60)

    block1_results = run_block_1(all_data)

    # Imprimir tabla Bloque 1
    headers = ['Asset', 'h', 'k', 'd', 'N_total', 'N_edge1', 'edge_rate', 'STATUS']
    widths = [12, 4, 6, 6, 9, 9, 10, 20]

    rows = []
    for r in block1_results:
        rows.append([
            r['asset'],
            r['h'],
            r['k'],
            r['d'],
            r['n_total'],
            r['n_edge1'],
            f"{r['edge_rate']:.1%}",
            r['status']
        ])

    print_table(headers, rows, widths)
    print()

    # ============================================
    # BLOQUE 2
    # ============================================
    print("BLOQUE 2 -- CALIDAD DEL EDGE")
    print("-" * 60)

    block2_results = run_block_2(all_data, block1_results)

    headers = ['Asset', 'h', 'k', 'd', 'Ret_e1', 'Ret_e0', 'Sharpe_e1', 'Sharpe_e0', 'DD_e1', 'DD_e0', 'STATUS']
    widths = [12, 4, 6, 6, 10, 10, 11, 11, 10, 10, 12]

    rows = []
    for r in block2_results:
        # Determinar STATUS
        sharpe_total = r['sharpe_total']
        sharpe_edge1 = r['sharpe_edge1']
        status = "PASS" if sharpe_edge1 > sharpe_total and sharpe_edge1 > 0 else "FAIL"

        rows.append([
            r['asset'],
            r['h'],
            r['k'],
            r['d'],
            f"{r['mean_return_edge1']:.3f}",
            f"{r['mean_return_edge0']:.3f}",
            f"{sharpe_edge1:.2f}" if not np.isnan(sharpe_edge1) else "N/A",
            f"{r['sharpe_edge0']:.2f}" if not np.isnan(r['sharpe_edge0']) else "N/A",
            f"{r['mean_dd_path_edge1']:.2%}",
            f"{r['mean_dd_path_edge0']:.2%}",
            status
        ])

    print_table(headers, rows, widths)
    print()

    # ============================================
    # BLOQUE 3
    # ============================================
    print("BLOQUE 3 -- SEPARACIÓN ESTADÍSTICA")
    print("-" * 60)

    block3_results = run_block_3(all_data, block1_results)

    headers = ['Asset', 'h', 'k', 'd', 'P(+|e1)', 'P(+|e0)', 'Separación', 't-stat', 'p-value', 'STATUS']
    widths = [12, 4, 6, 6, 10, 10, 11, 9, 10, 12]

    rows = []
    for r in block3_results:
        separation = r['separation']
        p_value = r['p_value']

        if separation >= 0.05 and (p_value is None or p_value <= 0.05):
            status = "PASS"
        else:
            status = "FAIL"

        rows.append([
            r['asset'],
            r['h'],
            r['k'],
            r['d'],
            f"{r['p_pos_edge1']:.1%}",
            f"{r['p_pos_edge0']:.1%}",
            f"{separation*100:+.1f}pp",
            f"{r['t_stat']:.2f}" if not np.isnan(r['t_stat']) else "N/A",
            f"{p_value:.3f}" if p_value is not None and not np.isnan(p_value) else "N/A",
            status
        ])

    print_table(headers, rows, widths)
    print()

    # ============================================
    # BLOQUE 4
    # ============================================
    print("BLOQUE 4 -- ROBUSTEZ TEMPORAL")
    print("-" * 60)

    block4_results = run_block_4(all_data, block1_results)

    headers = ['Asset', 'h', 'k', 'd', 'Q1_sharpe', 'Q2_sharpe', 'Q3_sharpe', 'Q4_sharpe', 'Consistent', 'STATUS']
    widths = [12, 4, 6, 6, 11, 11, 11, 11, 10, 12]

    rows = []
    for r in block4_results:
        rows.append([
            r['asset'],
            r['h'],
            r['k'],
            r['d'],
            f"{r['Q1_sharpe']:.2f}" if not np.isnan(r['Q1_sharpe']) else "N/A",
            f"{r['Q2_sharpe']:.2f}" if not np.isnan(r['Q2_sharpe']) else "N/A",
            f"{r['Q3_sharpe']:.2f}" if not np.isnan(r['Q3_sharpe']) else "N/A",
            f"{r['Q4_sharpe']:.2f}" if not np.isnan(r['Q4_sharpe']) else "N/A",
            r['consistent'],
            r['status']
        ])

    print_table(headers, rows, widths)
    print()

    # ============================================
    # BLOQUE 5
    # ============================================
    print("BLOQUE 5 -- HEATMAP DE SENSIBILIDAD (mejor config por asset)")
    print("-" * 60)

    best_configs = run_block_5(block2_results)

    headers = ['Asset', 'MEJOR h', 'MEJOR k', 'MEJOR d', 'edge_rate', 'Sharpe_e1', 'Separación']
    widths = [12, 10, 10, 10, 10, 11, 12]

    rows = []
    for asset in ASSETS:
        best = best_configs.get(asset)
        if best:
            # Buscar separación
            b3 = next((x for x in block3_results
                       if x['asset'] == asset and x['h'] == best['h'] and x['k'] == best['k'] and x['d'] == best['d']), None)
            sep_str = f"{b3['separation']*100:+.1f}pp" if b3 else "N/A"

            rows.append([
                asset,
                best['h'],
                best['k'],
                best['d'],
                f"{best['edge_rate']:.1%}",
                f"{best['sharpe_edge1']:.2f}",
                sep_str
            ])
        else:
            rows.append([asset, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    print_table(headers, rows, widths)
    print()

    # ============================================
    # TABLA COMPLETA DEL GRID (para referencia)
    # ============================================
    print("TABLA COMPLETA DEL GRID (ordenada por Sharpe_e1 descendente)")
    print("-" * 60)

    # Preparar datos completos
    full_grid = []
    for b2 in block2_results:
        b3 = next((x for x in block3_results
                   if x['asset'] == b2['asset'] and x['h'] == b2['h'] and x['k'] == b2['k'] and x['d'] == b2['d']), None)
        full_grid.append({
            'asset': b2['asset'],
            'h': b2['h'],
            'k': b2['k'],
            'd': b2['d'],
            'edge_rate': b2['edge_rate'],
            'sharpe_edge1': b2['sharpe_edge1'],
            'separation': b3['separation'] if b3 else np.nan
        })

    # Ordenar por Sharpe descendente
    full_grid.sort(key=lambda x: x['sharpe_edge1'] if not np.isnan(x['sharpe_edge1']) else -999, reverse=True)

    headers = ['Asset', 'h', 'k', 'd', 'edge_rate', 'Sharpe_e1', 'Separación']
    widths = [12, 4, 6, 6, 10, 11, 12]

    rows = []
    for r in full_grid[:30]:  # Top 30
        rows.append([
            r['asset'],
            r['h'],
            r['k'],
            r['d'],
            f"{r['edge_rate']:.1%}",
            f"{r['sharpe_edge1']:.2f}" if not np.isnan(r['sharpe_edge1']) else "N/A",
            f"{r['separation']*100:+.1f}pp" if not np.isnan(r['separation']) else "N/A"
        ])

    print_table(headers, rows, widths)
    print(f"(Mostrando top 30 de {len(full_grid)} configuraciones)")
    print()

    # ============================================
    # VEREDICTO FINAL
    # ============================================
    print("=" * 60)
    print("VEREDICTO FINAL")
    print("=" * 60)

    passed, details = evaluate_red_flags(best_configs, block2_results, block3_results)

    print()
    print("MEJOR CONFIGURACIÓN RECOMENDADA:")
    for asset in ASSETS:
        best = best_configs.get(asset)
        if best:
            print(f"  {asset}: h={best['h']}, k={best['k']}, d={best['d']} "
                  f"(edge_rate={best['edge_rate']:.0%}, Sharpe={best['sharpe_edge1']:.2f})")
        else:
            print(f"  {asset}: NO HAY CONFIGURACIÓN VÁLIDA")
    print()

    # ============================================
    # GUARDAR CSV
    # ============================================
    # Crear DataFrame consolidado
    csv_rows = []
    for r in full_grid:
        csv_rows.append({
            'asset': r['asset'],
            'horizon': r['h'],
            'k': r['k'],
            'd': r['d'],
            'edge_rate': r['edge_rate'],
            'sharpe_edge1': r['sharpe_edge1'],
            'separation_pp': r['separation'],
            'is_best': (best_configs.get(r['asset']) is not None and
                        best_configs[r['asset']]['h'] == r['h'] and
                        best_configs[r['asset']]['k'] == r['k'] and
                        best_configs[r['asset']]['d'] == r['d'])
        })

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(OUTPUT_CSV, index=False)
    print(f"Resultados guardados en: {OUTPUT_CSV}")
    print()

    # ============================================
    # TELEGRAM
    # ============================================
    veredicto_str = "PASS" if passed else "FAIL"

    msg = (
        f"<b>ORION V21.0 TARGET VALIDATION</b>\n"
        f"Fecha: {timestamp}\n\n"
        f"Veredicto: {veredicto_str}\n\n"
        "MEJOR CONFIG:\n"
    )

    for asset in ASSETS:
        best = best_configs.get(asset)
        if best:
            msg += (f"  {asset}: h={best['h']}, k={best['k']}, d={best['d']} "
                    f"(edge={best['edge_rate']:.0%}, Sharpe={best['sharpe_edge1']:.2f})\n")
        else:
            msg += f"  {asset}: NO VÁLIDA\n"

    if details:
        msg += "\nRED FLAGS:\n"
        for d in details[:5]:  # max 5 para no saturar
            msg += f"  • {d}\n"

    send_telegram(msg)

    print("Script completado.")

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == '__main__':
    main()

# ======================================================
# CHECKLIST PRE-ENTREGA (verificar antes de ejecutar)
# ======================================================
# [x] Sin leakage en vol: realized_vol_20d usa rolling(120) solo hacia atrás
# [x] Sin leakage en target: target usa log_return[i+1:i+1+horizon]
# [x] Grid completo: 3 assets × 4 h × 3 k × 3 d = 108 configs
# [x] Red flags evaluadas: evaluate_red_flags() llama en main()
# [x] Telegram funcional: send_telegram() llama al final
# [x] CSV guardado: df.to_csv(OUTPUT_CSV)
# [x] Self-contained: solo librerías públicas (ccxt, pandas, numpy, scipy, requests)
# [x] Formato salida: tablas con separadores y headers exactos
# [x] Performance: loops optimizados, O(n) por asset-per-config
# [x] Track A intacto: CERO modificaciones a orion_crypto.py u otros tracks
#
# NOTA: Si el script tarda > 5 min, optimizar compute_target() con vectorización.
