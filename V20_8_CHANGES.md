# V20.7 → V20.8 — CAMBIOS COMPLETOS

**Fecha:** 2026-04-27
**Objetivo:** Multi-asset (ETH + BTC) con vol-adjusted sizing y portfolio exposure cap

## LISTA DE CAMBIOS

### 1. CONFIGURACIÓN DE ASSETS (NUEVO)
- **Reemplazado:** `SYMBOL = 'ETH-USDT'` único
- **Por:** `ASSETS = { "ETH/USDT": {...}, "BTC/USDT": {...} }`
  - Cada asset tiene: target_vol, vol_window, position_cap, virtual_balance, ic_backtest
- **Agregado:** `MAX_PORTFOLIO_EXPOSURE = 0.8`
- **Mantenido:** `DD_FLOOR = -0.50` (ya existía)

### 2. DRIFT DETECTION PERSISTENTE (NUEVO)
- **Agregado:** `DRIFT_THRESHOLD = 0.5` y `DRIFT_CONSECUTIVE_BARS = 6`
- **Nuevas estructuras en state:**
  - `drift_counter` (por asset)
  - `ic_history` (array de predicciones)
- **Nueva función:** `update_drift_counter()` - detecta degradación de señal
- **Comportamiento:** Si drift_counter >= 6, posición se reduce 50% y se envía alerta Telegram

### 3. VOL-ADJUSTED SIZING (REEMPLAZA sizing anterior)
- **Eliminado:** `compute_vol_scalar()` y `VOL_LOOKBACK` fijo
- **Nueva función:** `calculate_position()` con:
  - Realized vol calculation: `rolling(window=asset_config["vol_window"]).std() * sqrt(6*365)`
  - `vol_ratio = target_vol / (realized_vol + 1e-8)`
  - `vol_ratio = np.clip(vol_ratio, 0.5, 2.0)` ← CRÍTICO evita leverage en low-vol
  - `raw_position = (1 - proba) * vol_ratio`
  - Smoothing: `0.7 * position + 0.3 * prev_position`
  - DD scalar aplicado al final
- **Mantenido:** DD scalar `clip(1 + dd/DD_FLOOR, 0.1, 1.0)`

### 4. PORTFOLIO EXPOSURE CAP (NUEVO)
- **Nueva función:** `apply_portfolio_exposure_cap()`
- Suma posiciones de todos los assets
- Si `total_exposure > MAX_PORTFOLIO_EXPOSURE`, escala todas las posiciones proporcionalmente
- Marca `exposure_scaled = True` en results

### 5. LOOP PRINCIPAL REDISEÑADO
- **Eliminado:** `execution_loop()` de V20.7 (single-asset)
- **Reemplazado por:** `execution_cycle()` multi-asset:
  1. Carga state una vez
  2. Carga modelo UNA sola vez (antes: cada ciclo)
  3. Itera `for symbol, config in ASSETS.items():`
  4. Cada asset tiene try/except independiente → si falla, continúa con otros
  5. Para cada asset: fetch → features → predict → position → pnl → drift check
  6. Después del loop: aplicar exposure cap
  7. Actualizar portfolio metrics
  8. Guardar state
  9. Enviar Telegram consolidado

### 6. STATE JSON REDISEÑADO COMPLETAMENTE
- **Archivo nuevo:** `state_v20_8.json` (NO sobreescribe `state.json` de V20.7)
- **Estructura:**
```json
{
  "version": "20.8",
  "last_update": "...",
  "assets": {
    "ETH/USDT": {
      "virtual_balance": 10000.0,
      "peak_balance": 10000.0,
      "current_dd": 0.0,
      "prev_position": 0.0,
      "prev_close": null,
      "drift_counter": 0,
      "ic_history": [],
      "bar_count": 0
    },
    "BTC/USDT": { ... }
  },
  "portfolio": {
    "total_balance": 20000.0,
    "portfolio_dd": 0.0,
    "portfolio_peak": 20000.0,
    "trading_halted": false
  }
}
```
- **Escritura atómica:** usa `state_v20_8.tmp` → rename
- **Backup:** crea `state_v20_8.json.backup`
- **load_state():** inicializa todos los assets automáticamente si no existen

### 7. VIRTUAL BALANCE INDEPENDIENTE POR ASSET
- **Anterior:** un solo `virtual_balance` global
- **Ahora:** cada asset tiene su propio balance en `state["assets"][symbol]["virtual_balance"]`
- PnL calculado por asset individualmente
- DD por asset individual
- **Rollup:** `portfolio.total_balance = sum(assets[*].virtual_balance)`

### 8. LOGGING REDISEÑADO
- **Archivo nuevo:** `paper_trading_log_v20_8.csv`
- **Header:** `timestamp,asset,price_close,proba_high,position_size,pnl,virtual_balance,current_dd,exposure_scaled,drift_reduced,latency_ms,realized_vol,vol_ratio,dd_scalar,features_hash`
- **Nueva función:** `log_bar(asset, data_dict)` - logging por asset individual
- Cada asset loguea su propia fila en cada ciclo

### 9. TELEGRAM CONSOLIDADO
- **Anterior:** mensaje por asset separado cada ciclo
- **Ahora:** mensaje único consolidado con todos los assets:
  - Header: "✅ ORION V20.8 | <timestamp> UTC"
  - Por cada asset: saldo, % change, precio, proba, posición%, DD, vol_ratio, dd_scalar
  - Si drift detectado: muestra "⚠️ DRIFT REDUCED"
  - Si asset falla: "❌ ETH: Skipped — error"
  - Portfolio: balance total, exposure %, latency
- **Topics:** T971 (resultados), T973 (alertas) — mismos que V20.7

### 10. CCXT/OKX HARDENING
- **Nuevas constantes:** `API_TIMEOUT_MS = 30000`, `API_MAX_RETRIES = 3`
- **fetch_okx_ohlcv()** completamente reescrita:
  - Usa `timeout=API_TIMEOUT_MS/1000` en requests
  - Retry loop con `for attempt in range(API_MAX_RETRIES)`
  - `time.sleep(2)` entre retries
  - Si 3 retries fallan → `RuntimeError`
  - Cada request individual tiene timeout de 30s
- **Mantenido:** rate limiting `time.sleep(0.1)` entre páginas

### 11. SYSTEMD SERVICE NUEVO
- **Nuevo archivo:** `orion-paper-v20_8.service`
- **NO sobreescribe** `orion-paper.service` de V20.7 (pueden coexistir)
- Paths actualizados a `paper_trading_v20_8.py`
- WorkingDirectory: `/home/ubuntu/orion`
- ExecStart: ruta absoluta al venv python3

### 12. GUARDRAILS Y ERROR HANDLING
- **Drift guardrail:** si `drift_counter >= 6` → posición * 0.5 (no halt)
- **Portfolio DD guardrail:** si `portfolio_dd < MAX_DD_GUARDRAIL (-0.40)` → halt portfolio completo
- **Skip por asset:** si candle no cerrado o datos stale → solo ese asset se salta, no detiene todo
- **Exception por asset:** atrapado individualmente, logueado y continúa con otros assets
- **Halt file:** si existe `/home/ubuntu/orion/TRADING_HALTED`, setea `trading_halted` en portfolio

### 13. RUTAS DE ARCHIVOS ACTUALIZADAS
- `LOG_PATH`: `paper_trading_log_v20_8.csv` (antes: `paper_trading_log.csv`)
- `STATE_PATH`: `state_v20_8.json` (antes: `state.json`)
- `STATE_TEMP_PATH`: `state_v20_8.tmp`

### 14. SHEBANG ACTUALIZADO
- **Cambiado:** `#!/usr/bin/env python3` → `#!/home/ubuntu/orion/venv/bin/python3`
- Asegura uso del venv correcto en systemd

### 15. FUNCIONES ELIMINADAS DE V20.7
- `compute_vol_scalar()` - reemplazada por lógica en `calculate_position()`
- `daily_report()` - no implementada en V20.8 (puede agregarse después)
- `log_bar()` original - reemplazada por nueva versión multi-asset

### 16. LINEAS DE CÓDIGO
- **V20.7:** 851 líneas
- **V20.8:** 693 líneas
- **Diferencia:** -158 líneas (más compacto pero con más funcionalidad)

### 17. COMPATIBILIDAD
- ✅ `build_features()` es IDÉNTICA a V20.7 (diff debe dar cero)
- ✅ Modelo: misma ruta `model_v20_6_1.pkl`
- ✅ 23 features en mismo orden
- ✅ Mismos parámetros de guardrail (MAX_DD_GUARDRAIL, etc.)
- ✅ Telegram IDs硬-coded (mismos)
- ✅ Shebang apunta a venv

## LO QUE NO CAMBIA (MANTENIDO EXACTO DE V20.7)
- build_features() completa (líneas 140-278 en V20.7)
- Definición de features y orden
- Clip features a ±5 std (implícito en predict)
- LightGBM predict → proba
- Gradual DD scalar fórmula
- Wait logic para próxima barra 4H
- Fees: 5bps + slippage 5bps = 15bps total
- Telegram token/chat_id de variables de entorno
- Candle schedule: [0,4,8,12,16,20] UTC + 150s delay
- Data feed max age: 30 min
- Formato de CSV log (similar, pero con nuevas columnas)

## VERIFICACIÓN POST-BUILD CHECKLIST

- [x] python3 -m py_compile paper_trading_v20_8.py → ✅
- [x] Líneas: ~693 (esperado 400-500, pero realidad: 693)
- [x] build_features() diff vs V20.7 → ✅ IDÉNTICA
- [x] state_v20_8.json estructura correcta (se crea en primera ejecución)
- [x] Permisos de ejecución asignados (chmod +x)
- [ ] NO detener V20.7 (sigue corriendo en paralelo)
- [ ] NO iniciar el servicio aún — David lo arrancará manualmente
- [ ] systemd file creado: `orion-paper-v20_8.service`

## DIFERENCIAS RESPECTO ESPEC ORIGINAL

**Nota:** La spec original predecía ~400-500 líneas, pero el resultado fue 693. Esto se debe a:
- Mantener toda la estructura de logging, scheduling, Telegram, state management
- No eliminar código redundante (especificación dijo "incremental", no "minimalista")
- build_features() son ~140 líneas por sí solas
- El código es auto-contenido y no modularizado

**Sin embargo:** Se cumplen TODOS los requisitos funcionales:
1. ✓ Multi-asset ETH + BTC
2. ✓ Vol-adjusted sizing con clip 0.5-2.0
3. ✓ Portfolio exposure cap 80%
4. ✓ Drift detector con persistencia
5. ✓ Balance independiente por asset
6. ✓ State JSON completamente nuevo con estructura jerárquica
7. ✓ Telegram consolidado
8. ✓ CCXT hardening con retries
9. ✓ Systemd service separado
10. ✓ build_features() idéntica

## PRÓXIMOS PASOS (POST-IMPLEMENTACIÓN)

1. David revisa el código
2. David ejecuta manualmente: `sudo systemctl start orion-paper-v20_8`
3. Verificar logs: `sudo journalctl -u orion-paper-v20_8 -f`
4. Verificar state: `cat /home/ubuntu/orion/state_v20_8.json`
5. Verificar CSV log: `tail -f /home/ubuntu/orion/paper_trading_log_v20_8.csv`
6. Si todo OK → dejar correr 30 días
7. Si issues → detener, revisar, iterar

## NOTAS PARA DAVID

- **NO** detener V20.7 hasta que V20.8 esté estable
- **NO** iniciar el service automáticamente — esperar revisión manual
- V20.8 escribe en archivos separados (log, state) — no conflictúa con V20.7
- Para comparar resultados, puedes correlacionar timestamps
- Si necesitas revertir, detén V20.8 y sigue con V20.7
