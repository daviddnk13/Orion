# ORION — Autonomous Trading AI System

**Repo**: github.com/daviddnk13/Orion
**Platform**: Kaggle (GPU T4) for script execution
**VPS**: EC2 AWS ip-172-31-26-81 (100.31.153.185)
**Asset**: ETH/USDT 4H
**Vision**: Personal wealth generation through autonomous portfolio management (private, non-commercial)

---

## MY ROLE — IMPLEMENTOR ONLY

**I (Claude Code) AM**:
- The code implementor exclusively
- I receive closed, consensus-based specifications
- I write code that EXACTLY follows the spec
- I DO NOT design architecture
- I DO NOT change parameters without explicit authorization
- I DO NOT assume modules/infrastructure not specified
- I LIST ALL changes vs base

**The Design Team (Copilot + ChatGPT + David)**:
- Debate strategy
- Reach consensus
- Design specifications
- Audit my code
- Approve or reject

**Pipeline**:
Copilot+ChatGPT+David debate → consensus → prompt for me → I implement → all 3 audit

---

## MANDATORY RULES — NEVER VIOLATE

### RULE 1 — SELF-CONTAINED
All Kaggle scripts MUST be self-contained. NO imports from repo modules (data_loader, features, targets, config, etc.). Everything inline in a single file.

### RULE 2 — IMMUTABILITY
DO NOT change features, target, percentiles, folds, horizon, hyperparameters without explicit justification + prior team approval.

### RULE 3 — INCREMENTAL
ALWAYS start from the last working script and modify incrementally. NEVER rewrite from scratch. Base script located in user's Google Drive "Trading" folder.

### RULE 4 — COMPARABILITY
Output must be 1:1 comparable with previous version. Same metrics, same evaluation structure.

### RULE 5 — ZERO ASSUMPTIONS
DO NOT assume modules, classes, or pipelines not specified in the prompt.

### RULE 6 — EXPLICIT CHANGES
List ALL changes vs base in a manifest.

### RULE 7 — KAGGLE SCRIPTS
NEVER use triple quotes (''' or """) — always use # comments only. Telegram IDs always hardcoded, never empty.

### RULE 8 — ONE VARIABLE AT A TIME
Maximum ONE conceptual change per version. If spec says "change reward", DO NOT touch features or architecture.

### RULE 9 — DELIVERY
After generating each script:
- Save copy to Google Drive "Trading" folder
- Push to GitHub repository
- Include compilation check (python3 -m py_compile)

---

## CURRENT PROJECT STATE (April 2026)

### Complete Version History

**V12.6a** — BEST HISTORICAL RESULT ETH
- SAC [128,128], 16 features, holding_penalty k=0.0003
- Walk-forward 4 folds seed 42
- Alpha +42pp, Sharpe +0.38

**V13.x** — Adaptive holding penalty by vol_regime
- Improved consistency (3/4 vs 2/4) but destroyed peak alpha
- V13.1 percentiles = TOTAL FAILURE → LINE CLOSED

**V14.x** — Exogenous features (funding rate, OI, F&G)
- V14.1 = invalid CryptoCompare data
- V14.2a = OKX funding rate = garbage (95% zeros)
- V14.3a = log-return reward, solved overfitting but no alpha
- V14.3b = clipped linear, WORSE than everything → CLOSED

**V15.0** — Excess return reward without penalties
- BELOW LEVEL 1, identical pattern: alpha only in crashes

**V16.x** — SAC end-to-end EXHAUSTED
- V16.0 IC safeguard funding+OI → ABORT no signal
- V16.1 SAC [256,256] → ABORTED massive overfitting
- V16.2 Fear & Greed → FAIL, SAC = risk manager not trader
- **CONSENSUS**: SAC end-to-end CLOSED. Problem = formulation, not tuning.

**V17.x** — Signal audit on technical OHLCV features
- V17.0A 4H → NO SIGNAL (0/4 sig)
- V17.0B 1H → NO SIGNAL (worse than 4H)
- V17.0C 1D → NO SIGNAL
- **CONSENSUS**: Technical OHLCV features CLOSED for ETH direction

**V18.x** — Cross-asset + real derivatives
- V18.0B → NO SIGNAL for direction
- **CONSENSUS**: PIVOT to VOLATILITY PREDICTION

**V19.x** — Vol forecasting continuous → CLOSED (did not beat lag1 baseline)
- **CONSENSUS**: Change to REGIME CLASSIFICATION

**V20.x** — Regime Classification (CURRENT)

- **V20.0** — 3-class regime (HIGH/NORMAL/LOW) → PASS 3/4, low HV Recall
- **V20.1** — Binary (HIGH vs REST) → **PASS 4/4** ← **CURRENT BASE**
  - AUC=0.648, beats persist 4/4
  - 23 features (16 base + 7 vol)
  - LightGBM binary, p66.67 threshold
  - Top features: parkinson_vol (0.33), realized_vol_7d (0.25), atr_norm (0.10)

- **V20.2** — Platt calibration → FAIL
  - ECE improved 4/4 but Recall destroyed (0.244→0.079)
  - Cause: temporal non-stationarity
  - **DECISION**: abandon calibration, P(HIGH) = ranking score

- **V20.3** — Risk Engine (NEXT)
  - risk_engine.py APPROVED (module for repo)
  - v20_3_validate.py PENDING (your first attempt was REJECTED)
  - Rejection reasons: non-existent imports, 21 vs 23 features, p75 vs p66.67, not self-contained

---

## FIXED TECHNICAL SPECS

### 23 Features (16 base + 7 vol)

**BASE (16)**:
```
ret_4h, rsi_norm, bb_position, macd_norm,
ret_4h_lag1, ret_4h_lag2,
atr_norm, vol_zscore, vol_regime, ret_8h, ret_24h,
ema_slope, drawdown_market, tf_coherence, dist_ema200, trend_strength
```

**VOL (7)**:
```
parkinson_vol, vol_compression, garman_klass_vol,
vol_regime_rank, realized_vol_7d, trend_efficiency, realized_vol_1d
```

### Walk-Forward Parameters
- 4 folds
- test_size = 1250
- embargo = 180
- horizon = 6 (24h forward)
- seed = 42
- Anchored expanding window

### LightGBM Parameters (V20.x)
```python
{
    "objective": "binary",
    "metric": "binary_logloss",
    "is_unbalance": True,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "seed": 42,
}
```

### Target Definition
```python
future_vol = std(log_returns[i+1 : i+1+HORIZON]) * sqrt(6)
HIGH = future_vol > percentile(67) of TRAINING set
# Top 33%, NEVER p75, NEVER computed on test
```

### Data Source
- OKX ETH-USDT 4H via direct API (Binance blocked from Kaggle + VPS)
- ~12,000 candles

---

## TELEGRAM CONFIGURATION

- **Bot**: @Sistem_tradingbot (ID: `8723893197`)
- **Group**: "alerta de mi ia" (ID: `-1003505760554`, forum mode)
- **Topics**:
  - 971 = 📊 Resultados
  - 972 = 🤖 Modelos
  - 973 = 🔔 Alertas Live
  - 974 = 📝 Changelog
- **User**: @David_nk (ID: `1210633433`)

Every script must include `tg_send()` function and send results to topic 971.

---

## V20.x EVALUATION CRITERIA

### Signal Quality
| Criterion | Threshold |
|-----------|-----------|
| AUC | > 0.60 |
| Recall | ≥ 0.70 |
| F1 | ≥ 0.40 |
| Signal rate | 10%-40% |
| Beats persist | ≥ 3/4 folds |

### Sizing Impact (V20.3+)
| Criterion | Description |
|-----------|-------------|
| Sharpe(scaled) > Sharpe(unscaled) | ≥ 2/4 folds |
| MaxDD(scaled) < MaxDD(unscaled) | ≥ 2/4 folds |
| Return delta | > -10% |

### Overall Verdict Logic
- `Signal ≥3/4 AND Sizing ≥2/3` → "RISK ENGINE OPERATIONAL"
- `Signal ≥3/4 AND Sizing <2/3` → "SIGNAL OK, SIZING NEEDS TUNING"
- `Recall ≥0.70 only` → "RECALL OK, PRECISION NEEDS WORK"
- `Else` → "THRESHOLD OPTIMIZATION FAILED"

---

## GOOGLE DRIVE — SCRIPT ARCHIVE

Location: User's Google Drive "Trading" folder

**Setup rclone (one-time)**:
```bash
sudo apt install rclone -y
rclone config
# New remote → name: gdrive → type: Google Drive → follow instructions
```

**After each script**:
```bash
rclone copy ~/orion/SCRIPT_NAME.py gdrive:"Trading"/
```

If rclone unavailable, at minimum push to GitHub.

**Naming convention**:
```
V20.3_RiskEngine_Validate.py
V20.3_risk_engine.py
V21.0_DESCRIPTION.py
```

---

## GITHUB WORKFLOW

After each implementation:
```bash
cd ~/orion
git add .
git commit -m "V20.3: Risk Engine module + validation script"
git push origin main
```

Commit format: `"VXX.Y: brief description of change"`

---

## LESSONS FROM YOUR FIRST ATTEMPT (V20.3) — CRITICAL

Your first attempt at `v20_3_validate.py` was **REJECTED** because:

1. **Imported non-existent modules**: `from data_loader import ...`, `from features import ...`, `from targets import ...`, `from config import CONFIG` → CRASH. V20.x scripts must be **self-contained**.

2. **21 features instead of 23**: Lost 2 features. This breaks comparability.

3. **Percentile 75 instead of 66.67**: Changed problem definition without authorization. V20.1 uses p66.67.

4. **Assumed modular architecture**: Read the repo and assumed infrastructure that doesn't exist. V20.x scripts run standalone on Kaggle.

**Lesson**: DO NOT interpret the repo. Follow the spec **LITERALLY**. If something is not in the spec, ASK.

---

## IMMEDIATE NEXT STEP: V20.3

**Status**: `risk_engine.py` is APPROVED and already in the repo.

**Pending**: `v20_3_validate.py` needs rewriting following ALL rules above.

**Spec**: A self-contained Kaggle script that:
- Clones the repo, fetches OKX data
- Generates all 23 features inline (NO imports)
- Binary target p66.67
- Walk-forward 4 folds LightGBM
- Fit RiskEngine on train, predict on test
- Sizing simulation (scaled vs unscaled returns)
- Report signal + sizing metrics
- Emit verdict
- Send to Telegram topic 971

**The team will send the specific prompt when ready. DO NOT generate code without explicit prompt.**

---

## DELIVERY PROTOCOL

Each delivery must include:

- `.py` file that compiles without errors (`python3 -m py_compile`)
- **Change manifest** with:
  - List of ALL changes vs base
  - Exact line numbers and before/after code (if editing existing script)
  - List of everything that DID NOT change
- GitHub push with descriptive commit
- Copy to Google Drive (if rclone configured)
- Line count verification (report total)

---

## REPO STRUCTURE

```
Orion/
├── CLAUDE.md              ← This file (master briefing)
├── risk_engine.py         ← APPROVED (V20.3)
├── pipeline.py            ← Exists but scripts DO NOT depend on it
├── quant_gate.py          ← QVG v1.1c
├── signals/               ← V19.1 signal quality
├── regime/                ← Regime classification
├── quality/               ← Quality gates
├── v20_3_validate.py      ← PENDING rewrite
└── ...
```

---

## ESCALATION PROTOCOL

- If a spec is ambiguous → ASK BEFORE implementing
- If a change seems necessary → PROPOSE but DO NOT implement without approval
- If you find a bug in existing code → REPORT, do not fix without permission
- If you need to install dependencies on VPS → ask permission first

---

## ONE-LINE SUMMARY

**I write code. They decide what code to write. Never the reverse.**

---

**Last Updated**: 2026-04-25
**Version**: V20.3 (Risk Engine Validation pending)
**Status**:风险engine module approved, validation script needs rewrite
