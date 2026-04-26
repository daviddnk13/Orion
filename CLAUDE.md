# ORION — AI Trading Agent

## Project Overview
Autonomous AI trading agent for ETH/USDT. Currently in **paper trading phase** (V20.7).
Built by David + Copilot + ChatGPT collaboration. Claude Code implements specs — does NOT design architecture.

## Current State (April 2026)
- **Frozen Model**: V20.6.1 LightGBM regime classifier (HIGH/LOW/NORMAL volatility)
- **Paper Trading Engine**: V20.7 running as systemd service on EC2
- **Status**: 30-day paper trading evaluation period — DO NOT modify model, features, or hyperparameters
- **Exit Criteria**: Pass → real capital 1-5% | Fail → V20.6.2 (sigmoid) or regime-aware layer

## Architecture

### Model (FROZEN — DO NOT TOUCH)
- LightGBM Booster: `model_v20_6_1.pkl`
- Task: Regime classification (predict P(HIGH_VOL))
- Output: predict_proba → position sizing via risk engine

### Risk Engine: Gradual DD Scalar
- `dd_scalar = clip(1 + dd/0.50, 0.1, 1.0)`
- Replaced binary circuit breaker (V20.6.0)
- Config: target_vol=0.15, position_cap=0.5, DD_FLOOR=-0.50, fee=5bps, slip=5bps

### 23 Features (OHLCV-based, order matters)
```
ret_4h, rsi_norm, bb_position, macd_norm,
ret_4h_lag1, ret_4h_lag2,
atr_norm, vol_zscore, vol_regime, ret_8h, ret_24h,
ema_slope, drawdown_market, tf_coherence, dist_ema200, trend_strength,
parkinson_vol, vol_compression, garman_klass_vol,
vol_regime_rank, realized_vol_7d, trend_efficiency, realized_vol_1d
```
**WARNING**: The model was trained with Column_0..Column_22 (generic names). The feature ORDER in this list IS the mapping. Do not reorder.

### Paper Trading (V20.7)
- Execution loop: every 4 hours
- Data: OKX via CCXT
- Prediction reconciliation: IC rolling live
- Feature distribution drift check
- Execution latency tracking
- State persistence: JSON
- Telegram reporting: topics 971 (Results), 973 (Alerts)
- Guardrails: DD<40%, slippage check, data feed check
- Virtual balance tracking: $10,000 initial capital

## Infrastructure
- **EC2**: ip-172-31-26-81 (IP 100.31.153.185), Ubuntu
- **Services**: orion.service (paper trading), free-claude-code.service
- **Python venv**: /home/ubuntu/orion/venv/
- **Telegram Bot**: @Sistem_tradingbot, Group ID -1003505760554
- **GitHub**: https://github.com/daviddnk13/Orion

## Key Files
```
paper_trading_v20_7.py  — Main paper trading engine
model_v20_6_1.pkl       — Frozen LightGBM model
config.py               — Configuration constants
features.py             — V19 features (NOT used by paper trading)
pipeline.py             — V19.1 orchestrator (NOT used by paper trading)
targets.py              — Regime target definition
paper_trading_state.json — Persistent state
```

## Rules for Claude Code
1. Script MUST be self-contained — do NOT import from modules unless they exist
2. Do NOT change features, target, percentiles, folds, or horizon without explicit approval
3. Always start from the latest working script and modify incrementally — NEVER rewrite from scratch
4. Output must be comparable 1:1 with previous version
5. Do NOT assume modules/classes/pipelines not specified
6. List ALL changes vs base in commit messages
7. Run python3 -m py_compile before committing
8. NEVER use triple quotes (''' or """) in Kaggle scripts — use # comments
9. Telegram IDs (bot token, chat_id, topic IDs) always hardcoded, never empty

## V20.6.1 Results (Paper Trading Ready)
- 3/3 PASS — ALL 4 FOLDS Sharpe_net POSITIVE
- Fold 0: Sharpe 0.535, DD -34.52%
- Fold 1: Sharpe 0.859, DD -18.89%
- Fold 2: Sharpe 2.624, DD -16.44%
- Fold 3: Sharpe 0.302 (FLIPPED POSITIVE), DD -28.23%
- Cost degradation: only 9.6%
- Mean Sharpe change: +165.6% vs binary breaker

## Version History Summary
- V12.6a: First positive OOS results (ETH best)
- V13.x: Adaptive holding penalty — CLOSED (destroyed peak alpha)
- V14.x: Log-return reward, funding rate, OI — reward fixed but no alpha
- V15.0: Excess return reward — confirmed features detect risk not opportunity
- V16.x: SAC end-to-end CLOSED — systemic misalignment
- V17.0: Signal audit — NO SIGNAL in technical features for direction
- V18.0: Cross-asset features — CLOSED, pivot to volatility
- V19.0: Volatility prediction — SIGNAL EXISTS (Pearson 0.24)
- V20.0-V20.3: Regime classification — PRE-PRODUCTION VALIDATED
- V20.4: Robustness tests — 3/3 PASS
- V20.5: Production prep — EMA smoothing FAIL, cost-aware PASS
- V20.6.1: Gradual DD scalar — 3/3 PASS, PAPER TRADING READY
- V20.7: Paper trading engine — CURRENTLY RUNNING
