# ============================================================
# main.py — V19.0 RISK STATE INFERENCE SYSTEM
# Orchestrator: data -> features -> targets -> walk-forward
# Telegram reporting integrated on all 4 topics
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, TELEGRAM, SAFE_DEBUG
from data_loader import (fetch_okx_ohlcv, fetch_macro_daily,
                          align_daily_to_4h, load_derivatives_data)
from features import build_features, validate_features
from targets import build_targets
from evaluation import run_walk_forward
from telegram_report import send_alert, send_changelog


def main():
    print("=" * 70)
    print("V19.0-AUDIT-FINAL + 6 PRODUCTION FIXES")
    print("Risk State Inference System")
    print("SAFE_DEBUG = {}".format(SAFE_DEBUG))
    print("Telegram = {}".format('ENABLED' if TELEGRAM['enabled'] else 'DISABLED'))
    print("=" * 70)

    # Send changelog on start
    if TELEGRAM['enabled']:
        send_changelog(
            "V19.0 MODULAR started\n"
            "- 7 modules (config, telegram, data, features, targets, evaluation, main)\n"
            "- 6 production fixes applied\n"
            "- Telegram reporting on 4 topics",
            TELEGRAM
        )

    # P2: Pre-execution environment check
    print("\n[P2] Pre-execution environment check...")
    try:
        test_arr = np.random.RandomState(42).randn(100)
        test_std = np.std(test_arr)
        assert test_std > 0, "NumPy RNG broken"
        print("  NumPy OK (test_std={:.4f})".format(test_std))
    except Exception as e:
        print("  [ABORT] NumPy check failed: {}".format(e))
        if TELEGRAM['enabled']:
            send_alert("ABORT: NumPy check failed: {}".format(e), TELEGRAM)
        return None

    # ---- Step 1: Fetch data ----
    print("\n[STEP 1] Fetching data...")
    try:
        df = fetch_okx_ohlcv(
            symbol=CONFIG['symbol'],
            timeframe=CONFIG['timeframe'],
            n_candles=CONFIG['n_candles']
        )
    except Exception as e:
        print("  [ABORT] Data fetch failed: {}".format(e))
        if TELEGRAM['enabled']:
            send_alert("ABORT: Data fetch failed: {}".format(e), TELEGRAM)
        return None

    macro = fetch_macro_daily()
    df = align_daily_to_4h(df, macro)
    df, has_funding, has_oi = load_derivatives_data(df)

    # ---- Step 2: Features ----
    print("\n[STEP 2] Building features...")
    df, raw_features = build_features(df, has_funding, has_oi)
    n_issues = validate_features(df, raw_features)
    if n_issues > 5:
        print("  [WARN] {} feature issues — proceeding with caution".format(n_issues))
        if TELEGRAM['enabled']:
            send_alert("{} feature issues detected".format(n_issues), TELEGRAM)

    # ---- Step 3: Targets ----
    print("\n[STEP 3] Building targets...")
    df = build_targets(df)

    # ---- Step 4: Cleanup ----
    warmup = 200
    df = df.iloc[warmup:].reset_index(drop=True)
    print("\n[MAIN] After warmup drop: {} bars".format(len(df)))

    valid_mask = df['target_z_vol24h'].notna() & df['vol_24h_future'].notna()
    df = df[valid_mask].reset_index(drop=True)
    print("[MAIN] After target NaN drop: {} bars".format(len(df)))

    if len(df) < 3000:
        print("[ABORT] Insufficient data after cleaning ({} < 3000)".format(len(df)))
        if TELEGRAM['enabled']:
            send_alert("ABORT: Only {} bars after cleaning".format(len(df)), TELEGRAM)
        return None

    # ---- Step 5: Walk-forward ----
    print("\n[STEP 5] Running walk-forward validation...")
    results = run_walk_forward(df, raw_features, telegram_config=TELEGRAM)

    print("\n[DONE] V19.0-AUDIT-FINAL execution complete.")
    return results


if __name__ == '__main__':
    main()
