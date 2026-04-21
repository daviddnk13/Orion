# ============================================================
# config.py — V19.0 RISK STATE INFERENCE SYSTEM
# All hyperparameters, seeds, paths, and Telegram config
# ============================================================

import numpy as np
import random

# Deterministic seeds (set once at top-level, never repeated)
np.random.seed(42)
random.seed(42)

SAFE_DEBUG = False

# ---- Telegram Configuration ----
# Bot: @Sistem_tradingbot | Group: alerta de mi ia
# Topics: 971=Resultados, 972=Modelos, 973=Alertas, 974=Changelog
TELEGRAM = {
    'bot_token': '8723893197:AAFfIORXd2Y-qQ8TclOq23afEPt_knr7xrU',
    'chat_id': '-1003505760554',
    'topics': {
        'resultados': 971,
        'modelos': 972,
        'alertas': 973,
        'changelog': 974,
    },
    'enabled': True,
}

# ---- Main Configuration ----
CONFIG = {
    'symbol': 'ETH-USDT',
    'timeframe': '4H',
    'n_candles': 6000 if SAFE_DEBUG else 12000,
    'walk_forward_folds': 2 if SAFE_DEBUG else 4,
    'test_size': 1250,
    'embargo': 180,
    'seed': 42,
    'epsilon': 1e-8,
    'pca_min_variance': 0.50,
    'pca_min_ic': 0.02,
    'target_norm_shift': 7,
    'lgbm_params': {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 30,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'n_estimators': 300,
        'random_state': 42,
        'verbose': -1,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
    },
    'early_stopping_rounds': 50,
    'garch_alpha': 0.10,
    'garch_beta': 0.90,
    'feature_clip': 1e6,
    'pred_vol_clip_max': 0.5,
    'isoos_warn_threshold': 2.5,
    'snr_alert_threshold': 2.0,
    'feature_stability_min': 50.0,
    'pred_skew_max': 2.0,
    'stability_score_min': 70,
}

# ---- PCA Temporal Blocks ----
PCA_BLOCKS = {
    'short': {
        'features': ['realized_vol_1d', 'parkinson_vol', 'vol_compression',
                      'trend_efficiency', 'garman_klass_vol'],
        'name': 'latent_short_risk'
    },
    'medium': {
        'features': ['realized_vol_7d', 'vol_regime_state',
                      'funding_rate_zscore', 'funding_regime_shift',
                      'oi_zscore'],
        'name': 'latent_medium_risk'
    },
    'long': {
        'features': ['eth_btc_zscore', 'dxy_corr_rolling',
                      'spx_beta_rolling', 'cross_decorrelation'],
        'name': 'latent_long_risk'
    },
}

if SAFE_DEBUG:
    print("[P1] SAFE_DEBUG=ON: folds=2, candles=6000")
