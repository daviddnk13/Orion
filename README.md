# V19.0 — Risk State Inference System

## Volatility Prediction for ETH/USDT

Modular walk-forward volatility prediction system using LightGBM with IC safeguard,
PCA temporal blocks, and multi-horizon targets.

### Architecture

```
trading-agent-v19/
├── config.py            # Hyperparameters, seeds, Telegram config
├── telegram_report.py   # Reporting to 4 Telegram topics
├── data_loader.py       # OKX OHLCV + BTC cross-asset + derivatives
├── features.py          # 21 raw features (no lookahead)
├── targets.py           # Multi-horizon vol targets + PCA + baselines + IC
├── evaluation.py        # LightGBM + 10 metrics + walk-forward + verdict
├── main.py              # Orchestrator (entry point)
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── kaggle_notebook.py   # 15-line Kaggle notebook runner
```

### How to Run

#### Option A: Kaggle (recommended)
1. Upload all `.py` files as a Kaggle Dataset named `trading-agent-v19`
2. Create a new Kaggle Notebook
3. Add the dataset and run:
```python
import sys
sys.path.insert(0, '/kaggle/input/trading-agent-v19')
from main import main
results = main()
```

#### Option B: Local
```bash
pip install -r requirements.txt
python main.py
```

### Telegram Setup
Edit `config.py` and set your bot token in `TELEGRAM['bot_token']`.
Topics: 971=Resultados, 972=Modelos, 973=Alertas, 974=Changelog

### Key Design Decisions
- **No triple quotes** — all comments use `#` (Kaggle compatibility)
- **No f-strings in critical paths** — `.format()` for maximum compatibility
- **Forward-fill only** — no bfill to prevent future data leakage (FIX-1)
- **Deterministic seeds** — set once at top-level only (FIX-5)
- **IC safeguard** — features must pass information coefficient threshold
- **10 evaluation metrics** — Pearson, Spearman, QLIKE, cross-horizon, regime

### Version History
- V19.0: Initial modular release (pivot from direction to volatility prediction)
- V18.0B: Final direction prediction attempt (NO SIGNAL — consensus STOP)
