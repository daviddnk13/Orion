# ============================================================
# features.py — V19.0 Feature engineering (21 raw features)
# All shift(1) applied — no lookahead
# ============================================================

import numpy as np
import pandas as pd
from config import CONFIG


def build_features(df, has_funding=False, has_oi=False):
    print("[FEATURES] Building 21 raw features...")
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret
    clip = CONFIG['feature_clip']

    # --- Volatility block (short) ---
    df['realized_vol_1d'] = log_ret.rolling(6).std().shift(1) * np.sqrt(6)
    df['realized_vol_7d'] = log_ret.rolling(42).std().shift(1) * np.sqrt(6)
    hl_ratio = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt(
        (hl_ratio ** 2).rolling(42).mean().shift(1) / (4 * np.log(2)))
    df['vol_compression'] = df['realized_vol_1d'] / (df['realized_vol_7d'] + CONFIG['epsilon'])

    # --- Trend efficiency ---
    close_diff_24 = (df['close'] - df['close'].shift(24)).abs()
    sum_abs_diff = df['close'].diff().abs().rolling(24).sum()
    df['trend_efficiency'] = (close_diff_24 / (sum_abs_diff + CONFIG['epsilon'])).shift(1)

    # --- Garman-Klass ---
    gk_comp = 0.5 * (np.log(df['high'] / df['low'])) ** 2 \
              - (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])) ** 2
    df['garman_klass_vol'] = np.sqrt(gk_comp.rolling(6).mean().shift(1).clip(lower=0))

    # --- Vol regime state ---
    vol_7d_unshifted = log_ret.rolling(42).std() * np.sqrt(6)
    vol_rank = vol_7d_unshifted.rolling(180).rank(pct=True).shift(1)
    df['vol_regime_state'] = vol_rank.fillna(0.5)

    # --- Cross-asset (ETH/BTC) ---
    eth_ret = log_ret
    if 'btc_close' in df.columns and df['btc_close'].notna().sum() > 100:
        eth_btc_ratio = df['close'] / (df['btc_close'] + CONFIG['epsilon'])
        eth_btc_log = np.log(eth_btc_ratio / eth_btc_ratio.shift(1))
        df['eth_btc_return_24h'] = eth_btc_log.rolling(6).sum().shift(1)
        df['eth_btc_return_72h'] = eth_btc_log.rolling(18).sum().shift(1)
        ratio_mean = eth_btc_ratio.rolling(180).mean().shift(1)
        ratio_std = eth_btc_ratio.rolling(180).std().shift(1)
        df['eth_btc_zscore'] = ((eth_btc_ratio.shift(1) - ratio_mean) /
                                 (ratio_std + CONFIG['epsilon']))
    else:
        df['eth_btc_return_24h'] = 0.0
        df['eth_btc_return_72h'] = 0.0
        df['eth_btc_zscore'] = 0.0

    # --- DXY correlation ---
    if 'dxy_proxy' in df.columns and df['dxy_proxy'].std() > 0:
        dxy_ret = np.log(df['dxy_proxy'] / df['dxy_proxy'].shift(1))
        df['dxy_corr_rolling'] = eth_ret.rolling(42).corr(dxy_ret).shift(1).fillna(0)
    else:
        dxy_ret = pd.Series(0.0, index=df.index)
        df['dxy_corr_rolling'] = 0.0

    # --- SPX beta ---
    if 'spx_proxy' in df.columns and df['spx_proxy'].std() > 0:
        spx_ret = np.log(df['spx_proxy'] / df['spx_proxy'].shift(1))
        cov_roll = eth_ret.rolling(180).cov(spx_ret).shift(1)
        var_roll = spx_ret.rolling(180).var().shift(1)
        df['spx_beta_rolling'] = (cov_roll / (var_roll + CONFIG['epsilon'])).fillna(0)
    else:
        df['spx_beta_rolling'] = 0.0

    # --- Cross-decorrelation ---
    if 'dxy_proxy' in df.columns and df['dxy_proxy'].std() > 0:
        dxy_ret_local = np.log(df['dxy_proxy'] / df['dxy_proxy'].shift(1))
        corr_short = eth_ret.rolling(6).corr(dxy_ret_local).shift(1).fillna(0)
        corr_long = eth_ret.rolling(42).corr(dxy_ret_local).shift(1).fillna(0)
        df['cross_decorrelation'] = corr_short - corr_long
    else:
        df['cross_decorrelation'] = 0.0

    # --- Funding rate features ---
    df['funding_rate_change'] = df['funding_rate'].diff().shift(1).fillna(0)
    fr_mean = df['funding_rate'].rolling(180).mean().shift(1)
    fr_std = df['funding_rate'].rolling(180).std().shift(1)
    df['funding_rate_zscore'] = ((df['funding_rate'].shift(1) - fr_mean) /
                                  (fr_std + CONFIG['epsilon'])).fillna(0)
    ema_short_fr = df['funding_rate'].ewm(span=6).mean().shift(1)
    ema_long_fr = df['funding_rate'].ewm(span=42).mean().shift(1)
    df['funding_regime_shift'] = (ema_short_fr - ema_long_fr).fillna(0)

    # --- OI features ---
    df['oi_change_pct'] = df['open_interest'].pct_change().shift(1).fillna(0).clip(-0.5, 0.5)
    oi_change = df['open_interest'].pct_change().fillna(0)
    df['oi_acceleration'] = oi_change.diff().shift(1).fillna(0).clip(-0.2, 0.2)
    oi_mean = df['open_interest'].rolling(180).mean().shift(1)
    oi_std = df['open_interest'].rolling(180).std().shift(1)
    df['oi_zscore'] = ((df['open_interest'].shift(1) - oi_mean) /
                        (oi_std + CONFIG['epsilon'])).fillna(0)

    # --- Interaction features ---
    df['funding_oi_divergence'] = (df['funding_rate_zscore'] * df['oi_zscore']).fillna(0)
    price_zscore_raw = log_ret.rolling(42).sum()
    price_z_mean = price_zscore_raw.rolling(180).mean().shift(1)
    price_z_std = price_zscore_raw.rolling(180).std().shift(1)
    price_zscore = ((price_zscore_raw.shift(1) - price_z_mean) /
                     (price_z_std + CONFIG['epsilon'])).fillna(0)
    df['oi_price_divergence'] = (df['oi_zscore'] - price_zscore).fillna(0)

    # --- Feature list ---
    raw_features = [
        'realized_vol_1d', 'parkinson_vol', 'vol_compression',
        'trend_efficiency', 'garman_klass_vol',
        'realized_vol_7d', 'vol_regime_state',
        'eth_btc_return_24h', 'eth_btc_return_72h', 'eth_btc_zscore',
        'dxy_corr_rolling', 'spx_beta_rolling', 'cross_decorrelation',
        'funding_rate_change', 'funding_rate_zscore', 'funding_regime_shift',
        'oi_change_pct', 'oi_acceleration', 'oi_zscore',
        'funding_oi_divergence', 'oi_price_divergence',
    ]
    for f in raw_features:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-clip, clip)
    print("  [OK] {} raw features built".format(len(raw_features)))
    return df, raw_features


def validate_features(df, raw_features):
    print("[VALIDATE] Checking features for silent errors...")
    issues = 0
    for f in raw_features:
        if f not in df.columns:
            print("  [ERROR] Feature '{}' missing!".format(f))
            issues += 1
            continue
        col = df[f]
        nan_pct = col.isna().mean() * 100
        if nan_pct > 5:
            print("  [WARN] {}: {:.1f}% NaN".format(f, nan_pct))
            issues += 1
        if col.std() < 1e-10:
            print("  [WARN] {}: CONSTANT (std={:.2e})".format(f, col.std()))
            issues += 1
        if col.dtype == object:
            print("  [ERROR] {}: dtype=object".format(f))
            issues += 1
    if issues == 0:
        print("  [OK] All features valid")
    else:
        print("  [WARN] {} issues detected".format(issues))
    return issues
