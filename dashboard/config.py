import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv('/home/ubuntu/orion/.env')

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'orion-dashboard-v209-default-key-change-me')
    DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'orion2026')

    ORION_STATE_PATH = '/home/ubuntu/orion/state_v20_9.json'
    ORION_LOG_PATH = '/home/ubuntu/orion/paper_trading_log_v20_9.csv'
    ORION_MODEL_PATH = '/home/ubuntu/orion/model_v20_6_1.pkl'

    OKX_TICKER_URL = 'https://www.okx.com/api/v5/market/ticker'
    PRICE_CACHE_SECONDS = 30

    SESSION_TIMEOUT = timedelta(hours=8)

    # Cloudflare Turnstile
    TURNSTILE_SITE_KEY = '0x4AAAAAADEpySAMSQbaZejD'
    TURNSTILE_SECRET_KEY = '0x4AAAAAADEpyZG85YMCcs6zgZWBUGjgmsI'
    TURNSTILE_VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'

    # Rate Limiting
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_BLOCK_SECONDS = 900  # 15 min
