# Orion V20.7 — Paper Trading Engine Dependencies

## Required Python Packages

Install via:
```bash
pip install numpy pandas joblib lightgbm requests schedule scipy
```

## Package Versions (minimum)

- Python: 3.9+
- numpy: >=1.20.0
- pandas: >=1.3.0
- joblib: >=1.1.0
- lightgbm: >=3.3.0
- requests: >=2.25.0
- schedule: >=1.1.0
- scipy: >=1.7.0

## System Requirements

- Linux (Ubuntu 20.04+ recommended)
- Internet connectivity for OKX API and Telegram
- 4GB+ RAM
- Write access to `/home/ubuntu/orion/` for logs and state

## Configuration

All constants are in the script:
- `TELEGRAM_BOT_TOKEN`: Set via environment variable `TELEGRAM_BOT_TOKEN` or edit script
- `TELEGRAM_CHAT_ID`: -1003505760554
- `MODEL_PATH`: Must point to existing `model_v20_6_1.pkl` file

## File Structure Expected

```
/home/ubuntu/orion/
├── paper_trading_v20_7.py     (this script)
├── model_v20_6_1.pkl          (pre-trained LightGBM model)
├── paper_trading_log.csv      (created on first run)
├── state.json                 (created on first run)
└── state_backup.json          (auto-created)
```

## Telegram Bot Setup

Bot token: `8723893197:AAFfIORXd2Y-qQ8TclOq23afEPt_knr7xrU`
Chat ID: `-1003505760554`
Topics:
- Alerts: 973
- Results: 971

Ensure bot is member of the forum chat and can post to topics.
