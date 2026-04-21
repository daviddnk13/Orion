# ============================================================
# targets.py — V19.0 Target engineering, PCA, baselines, IC safeguard
# Multi-horizon vol targets + HAR/GARCH baselines
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from config import CONFIG


def build_targets(df):
    print("[TARGETS] Building forward vol targets...")
    log_ret = df['log_ret'].values
    n = len(log_ret)
    tshift = CONFIG['target_norm_shift']

    # vol_4h: 1-bar forward absolute return annualized
    vol_4h = np.full(n, np.nan)
    for t in range(n - 1):
        vol_4h[t] = abs(log_ret[t + 1]) * np.sqrt(6)

    # vol_24h: 6-bar forward realized vol (PRIMARY TARGET)
    vol_24h = np.full(n, np.nan)
    for t in range(n - 6):
        window = log_ret[t + 1: t + 7]
        if len(window) == 6:
            vol_24h[t] = np.std(window, ddof=1) * np.sqrt(6)

    # vol_72h: 18-bar forward realized vol
    vol_72h = np.full(n, np.nan)
    for t in range(n - 18):
        window = log_ret[t + 1: t + 19]
        if len(window) == 18:
            vol_72h[t] = np.std(window, ddof=1) * np.sqrt(6)

    # vol_7d: 42-bar forward realized vol
    vol_7d = np.full(n, np.nan)
    for t in range(n - 42):
        window = log_ret[t + 1: t + 43]
        if len(window) == 42:
            vol_7d[t] = np.std(window, ddof=1) * np.sqrt(6)

    df['vol_4h_future'] = vol_4h
    df['vol_24h_future'] = vol_24h
    df['vol_72h_future'] = vol_72h
    df['vol_7d_future'] = vol_7d

    # Z-score normalization of primary target
    vol_series = df['vol_24h_future']
    rolling_mean = vol_series.rolling(180, min_periods=30).mean().shift(tshift)
    rolling_std = vol_series.rolling(180, min_periods=30).std().shift(tshift)
    df['vol_rolling_mean'] = rolling_mean
    df['vol_rolling_std'] = rolling_std
    df['target_z_vol24h'] = (vol_series - rolling_mean) / (rolling_std + CONFIG['epsilon'])

    valid_z = df['target_z_vol24h'].dropna()
    if len(valid_z) > 100:
        print("  [TARGET] z-score: mean={:.3f} std={:.3f} "
              "skew={:.3f} kurt={:.3f} "
              "outliers(>4s)={:.2f}%".format(
                  valid_z.mean(), valid_z.std(),
                  valid_z.skew(), valid_z.kurtosis(),
                  (valid_z.abs() > 4).mean() * 100))
    print("  [OK] Targets built. shift={}".format(tshift))
    return df


# ============================================================
# PCA TEMPORAL BLOCKS
# ============================================================

def build_pca_risk_factors(X_train, X_test, block_config, block_name,
                           y_train_vol=None):
    available = [f for f in block_config['features']
                 if f in X_train.columns and X_train[f].std() > 1e-10]
    if len(available) < 2:
        print("    [{}] <2 valid features, skipping PCA".format(block_name))
        return None, None, {'block': block_name, 'status': 'SKIP',
                            'explained': 0, 'ic': 0, 'n_features': len(available)}

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train[available].values)
    test_scaled = scaler.transform(X_test[available].values)

    pca = PCA(n_components=1, random_state=CONFIG['seed'])
    pca.fit(train_scaled)
    explained = pca.explained_variance_ratio_[0]

    if explained < CONFIG['pca_min_variance']:
        print("    [{}] PC1 var={:.3f} < 0.50 -> FALLBACK".format(block_name, explained))
        return None, None, {'block': block_name, 'explained': explained,
                            'status': 'FALLBACK_VAR', 'ic': 0, 'n_features': len(available)}

    # Sign convention: max absolute loading positive
    max_abs_idx = np.argmax(np.abs(pca.components_[0]))
    if pca.components_[0][max_abs_idx] < 0:
        pca.components_[0] *= -1

    train_factor = pca.transform(train_scaled)[:, 0]
    test_factor = pca.transform(test_scaled)[:, 0]

    # IC check
    ic = 0.0
    if y_train_vol is not None:
        mask = np.isfinite(train_factor) & np.isfinite(y_train_vol)
        if mask.sum() > 100:
            ic, _ = stats.spearmanr(train_factor[mask], y_train_vol[mask])
            ic = ic if np.isfinite(ic) else 0.0

    if abs(ic) < CONFIG['pca_min_ic']:
        print("    [{}] var={:.3f} OK but IC={:+.4f} < 0.02 -> FALLBACK_IC".format(
            block_name, explained, ic))
        return None, None, {'block': block_name, 'explained': explained,
                            'status': 'FALLBACK_IC', 'ic': ic, 'n_features': len(available)}

    print("    [{}] var={:.3f} IC={:+.4f} OK ({} feat)".format(
        block_name, explained, ic, len(available)))
    return train_factor, test_factor, {
        'block': block_name, 'explained': explained, 'ic': ic,
        'status': 'OK', 'n_features': len(available), 'features_used': available}


# ============================================================
# HAR BASELINE
# ============================================================

def build_har_baseline(df, train_idx, test_idx):
    print("  [HAR] Building HAR baseline (log-space fit)...")
    log_ret = df['log_ret'].values
    rv_d = pd.Series(log_ret).rolling(6).std().shift(1).values * np.sqrt(6)
    rv_w = pd.Series(log_ret).rolling(42).std().shift(1).values * np.sqrt(6)
    rv_m = pd.Series(log_ret).rolling(180).std().shift(1).values * np.sqrt(6)

    target = df['vol_24h_future'].values

    rv_d_tr, rv_w_tr, rv_m_tr = rv_d[train_idx], rv_w[train_idx], rv_m[train_idx]
    rv_d_te, rv_w_te, rv_m_te = rv_d[test_idx], rv_w[test_idx], rv_m[test_idx]
    y_tr_raw = target[train_idx]
    y_te_raw = target[test_idx]

    X_tr = np.column_stack([rv_d_tr, rv_w_tr, rv_m_tr])
    X_te = np.column_stack([rv_d_te, rv_w_te, rv_m_te])

    mask_tr = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr_raw) & (y_tr_raw > 0)
    mask_te = np.all(np.isfinite(X_te), axis=1) & np.isfinite(y_te_raw) & (y_te_raw > 0)

    if mask_tr.sum() < 100 or mask_te.sum() < 50:
        print("  [HAR] Insufficient valid data, returning NaN")
        return np.full(len(test_idx), np.nan), {}

    y_tr = np.log(y_tr_raw[mask_tr] + CONFIG['epsilon'])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr[mask_tr])
    X_te_s = scaler.transform(X_te[mask_te])

    model = LinearRegression().fit(X_tr_s, y_tr)
    pred_log = model.predict(X_te_s)
    pred_vol_masked = np.exp(pred_log)

    pred_full = np.full(len(test_idx), np.nan)
    pred_full[mask_te] = pred_vol_masked
    pred_full = np.clip(np.nan_to_num(pred_full, nan=np.nanmedian(pred_vol_masked)),
                        0, CONFIG['pred_vol_clip_max'])

    info = {
        'coefs': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'r2_train': float(model.score(X_tr_s, y_tr)),
        'n_train': int(mask_tr.sum()),
        'n_test': int(mask_te.sum()),
    }
    print("  [HAR] R2_train={:.4f} n_tr={} n_te={}".format(
        info['r2_train'], info['n_train'], info['n_test']))
    return pred_full, info


# ============================================================
# GARCH(1,1) BASELINE
# ============================================================

def build_garch_baseline(df, train_idx, test_idx):
    print("  [GARCH] Building GARCH(1,1) proxy baseline...")
    log_ret = df['log_ret'].values
    alpha = CONFIG['garch_alpha']
    beta = CONFIG['garch_beta']

    train_returns = log_ret[train_idx]
    valid_tr = train_returns[np.isfinite(train_returns)]
    if len(valid_tr) < 50:
        print("  [GARCH] Insufficient training data")
        return np.full(len(test_idx), np.nan), {}

    sigma2_init = np.var(valid_tr)
    all_idx = np.concatenate([train_idx, test_idx])
    n_all = len(all_idx)
    sigma2 = np.full(n_all, sigma2_init)

    for t in range(1, n_all):
        r_prev = log_ret[all_idx[t - 1]]
        if np.isfinite(r_prev):
            sigma2[t] = alpha * r_prev ** 2 + beta * sigma2[t - 1]
        else:
            sigma2[t] = sigma2[t - 1]

    garch_vol = np.sqrt(np.clip(sigma2, 0, None)) * np.sqrt(6)
    n_train = len(train_idx)
    pred_test = garch_vol[n_train:]
    pred_test = np.clip(pred_test, 0, CONFIG['pred_vol_clip_max'])

    info = {
        'alpha': alpha, 'beta': beta,
        'sigma2_init': float(sigma2_init),
        'mean_vol_test': float(np.nanmean(pred_test)),
    }
    print("  [GARCH] mean_vol_test={:.6f}".format(info['mean_vol_test']))
    return pred_test, info


# ============================================================
# IC SAFEGUARD
# ============================================================

def ic_safeguard(X_train, y_train, feature_names):
    print("  [IC] Running IC safeguard (hardened)...")
    all_ics = {}
    for f in feature_names:
        if f not in X_train.columns:
            continue
        vals = X_train[f].values
        mask = np.isfinite(vals) & np.isfinite(y_train)
        if mask.sum() < 100:
            all_ics[f] = 0.0
            continue
        ic, _ = stats.spearmanr(vals[mask], y_train[mask])
        all_ics[f] = ic if np.isfinite(ic) else 0.0

    ic_values = np.array(list(all_ics.values()))
    abs_ics = np.abs(ic_values[np.isfinite(ic_values)])
    if len(abs_ics) > 3:
        threshold = max(0.02, 0.5 * np.std(abs_ics), np.percentile(abs_ics, 60))
    else:
        threshold = 0.02

    passed = [f for f, ic in all_ics.items() if abs(ic) > threshold]

    if len(passed) > 15:
        passed = sorted(passed, key=lambda f: abs(all_ics[f]), reverse=True)[:15]

    failed = [f for f in all_ics if f not in passed]

    print("  [IC] threshold={:.4f} | passed={} | failed={}".format(
        threshold, len(passed), len(failed)))
    for f in passed:
        print("    [PASS] {}: IC={:+.4f}".format(f, all_ics[f]))
    for f in failed[:5]:
        print("    [FAIL] {}: IC={:+.4f}".format(f, all_ics[f]))

    return passed, all_ics, threshold
