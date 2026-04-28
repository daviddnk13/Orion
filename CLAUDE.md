# ORION — AI Trading Agent

## Project Overview
Autonomous AI trading agent for crypto markets. Currently in **paper trading phase** (V20.9 Multi-Asset).
Built by David + Copilot + ChatGPT collaboration. Claude Code implements specs — does NOT design architecture.

## Current State (April 2026)
- **Frozen Model**: V20.6.1 LightGBM regime classifier (HIGH/LOW/NORMAL volatility)
- **Paper Trading Engine**: V20.9 multi-asset (ETH + BTC + SOL) running as systemd service on EC2
- **Status**: Paper trading evaluation — DO NOT modify model, features, or hyperparameters
- **Exit Criteria**: Pass -> real capital 1-5% | Fail -> V20.6.2 or regime-aware layer

## Architecture

### Model (FROZEN — DO NOT TOUCH)
- LightGBM Booster: model_v20_6_1.pkl
- Task: Regime classification (predict P(HIGH_VOL))
- Output: predict_proba -> position sizing via per-asset mapping

### Per-Asset Sizing Mappings
- **ETH**: baseline -> position = (1 - proba)
- **BTC**: baseline -> position = (1 - proba)
- **SOL**: confidence_weighted -> position = proba * vol_ratio (inverted)
- All positions: clipped [0, 0.5], smoothed 0.7/0.3

### Risk Engine
- Gradual DD scalar: dd_scalar = clip(1 + dd/0.50, 0.1, 1.0)
- Regime guard SOL: extreme vol -> reduce position 50%
- Portfolio exposure cap: 80% total across all assets
- Config per asset: target_vol (ETH=0.15, BTC=0.12, SOL=0.18), position_cap=0.5
- Friction: fee=5bps, slip=5bps

### 23 Features (OHLCV-based, order matters)
ret_4h, rsi_norm, bb_position, macd_norm, ret_4h_lag1, ret_4h_lag2,
atr_norm, vol_zscore, vol_regime, ret_8h, ret_24h, ema_slope,
drawdown_market, tf_coherence, dist_ema200, trend_strength,
parkinson_vol, vol_compression, garman_klass_vol,
vol_regime_rank, realized_vol_7d, trend_efficiency, realized_vol_1d

WARNING: Model trained with Column_0..Column_22. Feature ORDER IS the mapping.

### Execution Loop (V20.9)
- Every 4 hours aligned to candle close + 2.5 min margin
- Data: OKX via CCXT (ETH, BTC, SOL simultaneously)
- Telegram: topics 971 (Results), 973 (Alerts)
- Virtual balance: USD 10,000 initial per asset
- State: state_v20_9.json

## Infrastructure
- EC2: ip-172-31-26-81, Ubuntu
- Service: orion-v209.service
- Telegram: @Sistem_tradingbot, Group -1003505760554
- GitHub: https://github.com/daviddnk13/Orion

## Key Files
- orion_crypto.py — V20.9 multi-asset engine (MAIN)
- model_v20_6_1.pkl — Frozen LightGBM model
- state_v20_9.json — Persistent state
- paper_trading_log_v20_9.csv — Trade log
- .env — Telegram tokens (NOT in git)

## Rules for Claude Code
1. Script MUST be self-contained
2. Do NOT change features/target/folds/horizon without approval
3. Start from latest working script, modify incrementally
4. Run python3 -m py_compile before committing
5. Telegram tokens from env vars, NEVER hardcoded
6. One variable at a time
7. Test every code path: fresh state, existing state, all mappings

## Roadmap
- V20.9 (CURRENT): Multi-asset paper trading
- V21.0: Edge Detection layer
- V21.1: Directional model
- V22: Shorts + leverage
- V23: Yield Router
- V24+: Multi-market + Asset Intelligence
