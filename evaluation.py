# ============================================================
# evaluation.py — V19.0 Walk-forward engine, metrics, diagnostics
# LightGBM training + 10 metrics + aggregate verdict
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from config import CONFIG, PCA_BLOCKS
from targets import (build_pca_risk_factors, build_har_baseline,
                     build_garch_baseline, ic_safeguard)
from telegram_report import (send_fold_report, send_aggregate_report,
                              send_alert, send_start_report)


# ============================================================
# QLIKE LOSS
# ============================================================

def compute_qlike(realized, predicted):
    mask = (np.isfinite(realized) & np.isfinite(predicted)
            & (predicted > 0) & (realized > 0))
    if mask.sum() < 10:
        return np.nan
    r = realized[mask]
    p = predicted[mask]
    ratio = r / p
    qlike = np.mean(ratio - np.log(ratio) - 1)
    return float(qlike)


# ============================================================
# 10 EVALUATION METRICS
# ============================================================

def compute_metrics(y_true_vol, pred_vol, har_pred, garch_pred,
                    vol_4h=None, vol_72h=None, vol_7d=None, fold_label=''):
    metrics = {}
    mask = np.isfinite(y_true_vol) & np.isfinite(pred_vol) & (y_true_vol > 0)
    if mask.sum() < 30:
        print("  [METRICS] {} insufficient valid data ({})".format(
            fold_label, mask.sum()))
        return {'M{}'.format(i): np.nan for i in range(1, 11)}
    y = y_true_vol[mask]
    p = pred_vol[mask]

    # M1: Pearson
    r_pearson, _ = stats.pearsonr(y, p)
    metrics['M1_pearson'] = float(r_pearson)

    # M2: Spearman
    r_spearman, _ = stats.spearmanr(y, p)
    metrics['M2_spearman'] = float(r_spearman)

    # M3: Quantile hit rate (terciles)
    try:
        q33, q66 = np.percentile(y, [33.3, 66.6])
        true_q = np.digitize(y, [q33, q66])
        pred_q = np.digitize(p, [q33, q66])
        metrics['M3_quantile_hit'] = float(np.mean(true_q == pred_q))
    except Exception:
        metrics['M3_quantile_hit'] = np.nan

    # M4: Calibration MAE (quintiles)
    try:
        quintiles = np.percentile(y, [20, 40, 60, 80])
        true_bins = np.digitize(y, quintiles)
        cal_errors = []
        for b in range(5):
            mask_b = true_bins == b
            if mask_b.sum() > 5:
                cal_errors.append(abs(np.mean(y[mask_b]) - np.mean(p[mask_b])))
        metrics['M4_calibration_mae'] = float(np.mean(cal_errors)) if cal_errors else np.nan
    except Exception:
        metrics['M4_calibration_mae'] = np.nan

    # M5: Cross-horizon 4h
    if vol_4h is not None:
        v4h = vol_4h[mask] if len(vol_4h) == len(y_true_vol) else vol_4h
        p_sub = p[:len(v4h)] if len(v4h) < len(p) else p
        v4h_sub = v4h[:len(p_sub)]
        v4h_norm = v4h_sub / (np.sqrt(6) + CONFIG['epsilon'])
        p_norm = p_sub / (np.sqrt(6) + CONFIG['epsilon'])
        m5 = np.isfinite(v4h_norm) & np.isfinite(p_norm)
        if m5.sum() > 30:
            metrics['M5_cross_4h'] = float(stats.spearmanr(p_norm[m5], v4h_norm[m5])[0])
        else:
            metrics['M5_cross_4h'] = np.nan
    else:
        metrics['M5_cross_4h'] = np.nan

    # M6: Cross-horizon 72h
    if vol_72h is not None:
        v72 = vol_72h[mask] if len(vol_72h) == len(y_true_vol) else vol_72h
        p_sub = p[:len(v72)] if len(v72) < len(p) else p
        v72_sub = v72[:len(p_sub)]
        v72_norm = v72_sub / (np.sqrt(18) + CONFIG['epsilon'])
        p_norm = p_sub / (np.sqrt(6) + CONFIG['epsilon'])
        m6 = np.isfinite(v72_norm) & np.isfinite(p_norm)
        if m6.sum() > 30:
            metrics['M6_cross_72h'] = float(stats.spearmanr(p_norm[m6], v72_norm[m6])[0])
        else:
            metrics['M6_cross_72h'] = np.nan
    else:
        metrics['M6_cross_72h'] = np.nan

    # M6b: Cross-horizon 7d
    if vol_7d is not None:
        v7d = vol_7d[mask] if len(vol_7d) == len(y_true_vol) else vol_7d
        p_sub = p[:len(v7d)] if len(v7d) < len(p) else p
        v7d_sub = v7d[:len(p_sub)]
        v7d_norm = v7d_sub / (np.sqrt(42) + CONFIG['epsilon'])
        p_norm = p_sub / (np.sqrt(6) + CONFIG['epsilon'])
        m7d = np.isfinite(v7d_norm) & np.isfinite(p_norm)
        if m7d.sum() > 30:
            metrics['M6b_cross_7d'] = float(stats.spearmanr(p_norm[m7d], v7d_norm[m7d])[0])
        else:
            metrics['M6b_cross_7d'] = np.nan
    else:
        metrics['M6b_cross_7d'] = np.nan

    # M7: QLIKE
    metrics['M7_qlike'] = compute_qlike(y, p)

    # M8: QLIKE vs HAR
    har_m = None
    if har_pred is not None and len(har_pred) == len(y_true_vol):
        har_m = har_pred[mask]
    if har_m is not None:
        metrics['M8_qlike_har'] = compute_qlike(y, har_m)
    else:
        metrics['M8_qlike_har'] = np.nan

    # M8b: QLIKE vs GARCH
    garch_m = None
    if garch_pred is not None and len(garch_pred) == len(y_true_vol):
        garch_m = garch_pred[mask]
    if garch_m is not None:
        metrics['M8b_qlike_garch'] = compute_qlike(y, garch_m)
    else:
        metrics['M8b_qlike_garch'] = np.nan

    # M9: Skill vs HAR
    mse_model = mean_squared_error(y, p)
    if har_m is not None:
        mask_h = np.isfinite(har_m)
        if mask_h.sum() > 30:
            mse_har = mean_squared_error(y[mask_h], har_m[mask_h])
            metrics['M9_skill_har'] = float(1 - mse_model / (mse_har + CONFIG['epsilon']))
        else:
            metrics['M9_skill_har'] = np.nan
    else:
        metrics['M9_skill_har'] = np.nan

    # M9b: Skill vs GARCH
    if garch_m is not None:
        mask_g = np.isfinite(garch_m)
        if mask_g.sum() > 30:
            mse_garch = mean_squared_error(y[mask_g], garch_m[mask_g])
            metrics['M9b_skill_garch'] = float(1 - mse_model / (mse_garch + CONFIG['epsilon']))
        else:
            metrics['M9b_skill_garch'] = np.nan
    else:
        metrics['M9b_skill_garch'] = np.nan

    # M10: Regime metrics
    vol_median = np.median(y)
    high_mask = y > vol_median
    low_mask = y <= vol_median
    if high_mask.sum() > 20:
        metrics['M10_regime_corr_high'] = float(stats.spearmanr(y[high_mask], p[high_mask])[0])
    else:
        metrics['M10_regime_corr_high'] = np.nan
    if low_mask.sum() > 20:
        metrics['M10_regime_corr_low'] = float(stats.spearmanr(y[low_mask], p[low_mask])[0])
    else:
        metrics['M10_regime_corr_low'] = np.nan

    metrics['M10_mse'] = float(mse_model)
    metrics['M10_mae'] = float(mean_absolute_error(y, p))

    return metrics


# ============================================================
# PREDICTION DIAGNOSTICS
# ============================================================

def compute_prediction_diagnostics(pred_vol, y_true_vol, fold_label=''):
    diag = {}
    mask = np.isfinite(pred_vol) & np.isfinite(y_true_vol)
    if mask.sum() < 30:
        return {'status': 'INSUFFICIENT_DATA'}

    p = pred_vol[mask]
    y = y_true_vol[mask]

    diag['pred_mean'] = float(np.mean(p))
    diag['pred_std'] = float(np.std(p))
    diag['pred_skew'] = float(stats.skew(p))
    diag['true_mean'] = float(np.mean(y))
    diag['true_std'] = float(np.std(y))

    real_corr = float(stats.spearmanr(y, p)[0]) if len(y) > 10 else 0.0
    diag['real_corr'] = real_corr

    np.random.seed(CONFIG['seed'] + 999)
    y_shuffled = np.random.permutation(y)
    rand_corr = float(stats.spearmanr(y_shuffled, p)[0]) if len(y) > 10 else 0.0
    diag['random_corr'] = rand_corr

    snr = abs(real_corr) / (abs(rand_corr) + CONFIG['epsilon'])
    diag['snr'] = float(snr)

    if snr < CONFIG['snr_alert_threshold']:
        diag['snr_alert'] = "ALERT: SNR={:.2f} < {}".format(snr, CONFIG['snr_alert_threshold'])
        print("  [P5] {} {}".format(fold_label, diag['snr_alert']))
    else:
        diag['snr_alert'] = 'OK'

    diag['status'] = 'OK'
    return diag


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def run_walk_forward(df, raw_features, telegram_config=None):
    n = len(df)
    n_folds = CONFIG['walk_forward_folds']
    test_size = CONFIG['test_size']
    embargo = CONFIG['embargo']

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("  folds={} test_size={} embargo={}".format(n_folds, test_size, embargo))
    print("=" * 70)

    # Send start notification
    if telegram_config:
        send_start_report('19.0', n_folds, n, telegram_config)

    # P2: Pre-execution sanity
    vol_target = df['vol_24h_future'].dropna()
    print("\n[P2] PRE-EXECUTION SANITY:")
    print("  mean(vol_24h_future) = {:.6f}".format(vol_target.mean()))
    print("  std(vol_24h_future)  = {:.6f}".format(vol_target.std()))
    autocorr1 = vol_target.autocorr(lag=1)
    print("  autocorr(lag=1)      = {:.4f}".format(autocorr1))
    nan_ratio = df[raw_features].isna().mean()
    high_nan = nan_ratio[nan_ratio > 0.05]
    if len(high_nan) > 0:
        print("  [WARN] Features with >5% NaN: {}".format(dict(high_nan)))
    if vol_target.mean() != vol_target.mean() or vol_target.std() < 1e-12:
        print("  [ABORT] Target is NaN or constant!")
        return None

    all_fold_results = []
    all_fold_metrics = []
    prev_top_features = []

    for fold in range(n_folds):
        print("\n{}".format("=" * 60))
        print("FOLD {}/{}".format(fold + 1, n_folds))
        print("{}".format("=" * 60))

        test_end = n - (n_folds - fold - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - embargo

        if train_end < 500:
            print("  [SKIP] Fold {}: insufficient training data".format(fold + 1))
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        print("  Train: 0-{} ({} bars)".format(train_end, train_end))
        print("  Test:  {}-{} ({} bars)".format(test_start, test_end, test_size))
        print("  Embargo: {} bars".format(embargo))

        # Ratio check
        tr_mean_vol = df['vol_24h_future'].iloc[train_idx].mean()
        te_mean_vol = df['vol_24h_future'].iloc[test_idx].mean()
        ratio_vol = tr_mean_vol / (te_mean_vol + CONFIG['epsilon'])
        print("  Train/Test mean vol ratio: {:.3f}".format(ratio_vol))

        # Target
        z_train = df['target_z_vol24h'].iloc[train_idx].values
        z_test = df['target_z_vol24h'].iloc[test_idx].values
        y_true_vol = df['vol_24h_future'].iloc[test_idx].values
        vol_rm_test = df['vol_rolling_mean'].iloc[test_idx].values
        vol_rs_test = df['vol_rolling_std'].iloc[test_idx].values

        vol_4h_test = df['vol_4h_future'].iloc[test_idx].values
        vol_72h_test = df['vol_72h_future'].iloc[test_idx].values
        vol_7d_test = df['vol_7d_future'].iloc[test_idx].values

        # Features
        X_train_raw = df[raw_features].iloc[train_idx].copy()
        X_test_raw = df[raw_features].iloc[test_idx].copy()

        # PCA blocks
        pca_reports = []
        pca_train_factors = {}
        pca_test_factors = {}
        y_train_vol_for_ic = df['vol_24h_future'].iloc[train_idx].values

        for bkey, bconf in PCA_BLOCKS.items():
            tr_f, te_f, report = build_pca_risk_factors(
                X_train_raw, X_test_raw, bconf, bconf['name'],
                y_train_vol=y_train_vol_for_ic)
            pca_reports.append(report)
            if tr_f is not None:
                pca_train_factors[bconf['name']] = tr_f
                pca_test_factors[bconf['name']] = te_f

        X_train_full = X_train_raw.copy()
        X_test_full = X_test_raw.copy()
        for fname, fvals in pca_train_factors.items():
            X_train_full[fname] = fvals
        for fname, fvals in pca_test_factors.items():
            X_test_full[fname] = fvals

        all_features = list(X_train_full.columns)

        # IC safeguard
        valid_tr = np.isfinite(z_train)
        passed_features, ic_dict, ic_threshold = ic_safeguard(
            X_train_full[valid_tr], z_train[valid_tr], all_features)

        if len(passed_features) < 3:
            print("  [ABORT] Fold {}: <3 features passed IC".format(fold + 1))
            all_fold_results.append({'fold': fold + 1, 'status': 'ABORT_IC'})
            if telegram_config:
                send_alert("Fold {} ABORTED: <3 features passed IC".format(fold + 1),
                           telegram_config)
            continue

        # LightGBM train/test
        X_tr_lgb = X_train_full[passed_features].values
        X_te_lgb = X_test_full[passed_features].values
        y_tr_lgb = z_train.copy()

        valid_tr_mask = np.isfinite(y_tr_lgb) & np.all(np.isfinite(X_tr_lgb), axis=1)
        valid_te_mask = np.all(np.isfinite(X_te_lgb), axis=1)

        X_tr_lgb = X_tr_lgb[valid_tr_mask]
        y_tr_lgb = y_tr_lgb[valid_tr_mask]
        X_te_lgb_clean = X_te_lgb[valid_te_mask]

        # Split train -> train/val for early stopping
        val_split = max(int(len(X_tr_lgb) * 0.15), 100)
        X_fit = X_tr_lgb[:-val_split]
        y_fit = y_tr_lgb[:-val_split]
        X_val = X_tr_lgb[-val_split:]
        y_val = y_tr_lgb[-val_split:]

        print("  [LGB] Training: fit={} val={} test={}".format(
            len(X_fit), len(X_val), len(X_te_lgb_clean)))

        model = lgb.LGBMRegressor(**CONFIG['lgbm_params'])
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(CONFIG['early_stopping_rounds'], verbose=False),
                       lgb.log_evaluation(period=0)]
        )

        train_mse = mean_squared_error(y_fit, model.predict(X_fit))
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        isoos_ratio = train_mse / (val_mse + CONFIG['epsilon'])
        print("  [LGB] Train MSE={:.6f} Val MSE={:.6f} IS/OOS={:.2f}".format(
            train_mse, val_mse, isoos_ratio))
        if isoos_ratio > CONFIG['isoos_warn_threshold']:
            print("  [P3-D] WARNING: IS/OOS={:.2f} > {}".format(
                isoos_ratio, CONFIG['isoos_warn_threshold']))
            if telegram_config:
                send_alert("Fold {} IS/OOS={:.2f} WARNING".format(
                    fold + 1, isoos_ratio), telegram_config)

        # Predict z-score
        pred_z_clean = model.predict(X_te_lgb_clean)

        # FIX-2 & FIX-3: Inverse transform with strict validation
        pred_vol = np.full(len(test_idx), np.nan)
        valid_te = np.where(valid_te_mask)[0]

        rm_clean = vol_rm_test[valid_te]
        rs_clean = vol_rs_test[valid_te]
        pred_z = pred_z_clean

        # FIX-2: Assert alignment
        assert len(rm_clean) == len(pred_z), \
            "ALIGNMENT ERROR: rm_clean({}) != pred_z({})".format(len(rm_clean), len(pred_z))
        assert len(rs_clean) == len(pred_z), \
            "ALIGNMENT ERROR: rs_clean({}) != pred_z({})".format(len(rs_clean), len(pred_z))

        # FIX-3: NaN safety guard
        nan_mask = np.isfinite(pred_z) & np.isfinite(rm_clean) & np.isfinite(rs_clean)
        n_dropped = (~nan_mask).sum()
        if n_dropped > 0:
            print("  [FIX-3] Dropped {}/{} NaN-mismatched indices".format(
                n_dropped, len(pred_z)))
        pred_z_safe = pred_z[nan_mask]
        rm_safe = rm_clean[nan_mask]
        rs_safe = rs_clean[nan_mask]
        valid_te_safe = valid_te[nan_mask]

        pred_vol_raw = pred_z_safe * (rs_safe + CONFIG['epsilon']) + rm_safe
        pred_vol_clipped = np.clip(pred_vol_raw, 0, CONFIG['pred_vol_clip_max'])
        pred_vol[valid_te_safe] = pred_vol_clipped

        # Fill remaining NaN with median
        med_val = np.nanmedian(pred_vol_clipped) if len(pred_vol_clipped) > 0 else 0.01
        pred_vol = np.where(np.isfinite(pred_vol), pred_vol, med_val)

        # Baselines
        har_pred, har_info = build_har_baseline(df, train_idx, test_idx)
        garch_pred, garch_info = build_garch_baseline(df, train_idx, test_idx)

        # P3: Per-fold diagnostics
        diag = compute_prediction_diagnostics(pred_vol, y_true_vol, "Fold{}".format(fold + 1))
        print("\n  [P3-A] Fold{} DISTRIBUTION:".format(fold + 1))
        print("    pred: mean={:.6f} std={:.6f} skew={:.3f}".format(
            diag.get('pred_mean', 0), diag.get('pred_std', 0), diag.get('pred_skew', 0)))
        print("    true: mean={:.6f} std={:.6f}".format(
            diag.get('true_mean', 0), diag.get('true_std', 0)))

        har_corr = 0.0
        if har_pred is not None:
            m_hc = np.isfinite(pred_vol) & np.isfinite(har_pred)
            if m_hc.sum() > 30:
                har_corr = float(stats.spearmanr(pred_vol[m_hc], har_pred[m_hc])[0])
        garch_corr = 0.0
        if garch_pred is not None:
            m_gc = np.isfinite(pred_vol) & np.isfinite(garch_pred)
            if m_gc.sum() > 30:
                garch_corr = float(stats.spearmanr(pred_vol[m_gc], garch_pred[m_gc])[0])

        print("  [P3-B] corr(pred,true)={:.4f} "
              "corr(pred,HAR)={:.4f} corr(pred,GARCH)={:.4f}".format(
                  diag.get('real_corr', 0), har_corr, garch_corr))

        # P3-C: Feature consistency
        importances = model.feature_importances_
        top5_idx = np.argsort(importances)[::-1][:5]
        top5_features = [passed_features[i] for i in top5_idx]
        print("  [P3-C] Top 5 features: {}".format(top5_features))
        if prev_top_features:
            overlap = len(set(top5_features) & set(prev_top_features))
            overlap_pct = overlap / 5 * 100
            print("         Overlap with prev fold: {:.0f}%".format(overlap_pct))
        prev_top_features = top5_features

        # Metrics
        metrics = compute_metrics(
            y_true_vol, pred_vol, har_pred, garch_pred,
            vol_4h=vol_4h_test, vol_72h=vol_72h_test, vol_7d=vol_7d_test,
            fold_label="Fold{}".format(fold + 1)
        )

        print("\n  [METRICS] Fold{}:".format(fold + 1))
        for k, v in metrics.items():
            if isinstance(v, float):
                if np.isfinite(v):
                    print("    {} = {:.4f}".format(k, v))
                else:
                    print("    {} = NaN".format(k))

        fold_result = {
            'fold': fold + 1,
            'status': 'OK',
            'metrics': metrics,
            'diagnostics': diag,
            'pca_reports': pca_reports,
            'ic_threshold': ic_threshold,
            'passed_features': passed_features,
            'n_features_used': len(passed_features),
            'n_passed': len(passed_features),
            'isoos_ratio': isoos_ratio,
            'har_info': har_info,
            'garch_info': garch_info,
            'top5_features': top5_features,
        }
        all_fold_results.append(fold_result)
        all_fold_metrics.append(metrics)

        # Send fold report to Telegram
        if telegram_config:
            send_fold_report(fold + 1, fold_result, telegram_config)

    # ============================================================
    # AGGREGATE
    # ============================================================
    if not all_fold_metrics:
        print("\n[FINAL] NO FOLDS COMPLETED")
        if telegram_config:
            send_alert("EXPERIMENT FAILED: No folds completed", telegram_config)
        return None

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    agg = {}
    for key in all_fold_metrics[0]:
        vals = [m[key] for m in all_fold_metrics
                if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
        if vals:
            agg["mean_{}".format(key)] = float(np.mean(vals))
            agg["std_{}".format(key)] = float(np.std(vals))

    print("\n[AGG] Mean metrics:")
    for k, v in agg.items():
        if k.startswith('mean_'):
            print("  {} = {:.4f}".format(k, v))

    # P4: EDGE_VALID
    mean_pearson = agg.get('mean_M1_pearson', 0)
    mean_skill_har = agg.get('mean_M9_skill_har', -1)
    cross_pass = 0
    for m in all_fold_metrics:
        m5 = m.get('M5_cross_4h', 0) or 0
        m6 = m.get('M6_cross_72h', 0) or 0
        if m5 > 0.15 or m6 > 0.15:
            cross_pass += 1

    # Feature stability
    if len(all_fold_results) >= 2:
        first_top = set(all_fold_results[0].get('top5_features', []))
        overlaps = []
        for fr in all_fold_results[1:]:
            curr_top = set(fr.get('top5_features', []))
            if first_top and curr_top:
                overlaps.append(len(first_top & curr_top) / 5 * 100)
        feature_stability = float(np.mean(overlaps)) if overlaps else 0.0
    else:
        feature_stability = 0.0

    # Prediction skew
    pred_skews = [fr.get('diagnostics', {}).get('pred_skew', 0)
                  for fr in all_fold_results]
    max_skew = max(abs(s) for s in pred_skews) if pred_skews else 0

    # P5: SNR
    snrs = [fr.get('diagnostics', {}).get('snr', 0) for fr in all_fold_results]
    mean_snr = float(np.mean(snrs)) if snrs else 0

    EDGE_VALID = (
        mean_pearson > 0.20 and
        mean_skill_har > 0 and
        cross_pass >= 2 and
        feature_stability >= CONFIG['feature_stability_min'] and
        max_skew < CONFIG['pred_skew_max']
    )

    print("\n[P4] EDGE_VALID = {}".format(EDGE_VALID))
    print("  mean_pearson={:.4f} (>0.20: {})".format(
        mean_pearson, 'PASS' if mean_pearson > 0.20 else 'FAIL'))
    print("  mean_skill_har={:.4f} (>0: {})".format(
        mean_skill_har, 'PASS' if mean_skill_har > 0 else 'FAIL'))
    print("  cross_pass={} (>=2: {})".format(
        cross_pass, 'PASS' if cross_pass >= 2 else 'FAIL'))
    print("  feature_stability={:.1f}% (>={:.0f}%: {})".format(
        feature_stability, CONFIG['feature_stability_min'],
        'PASS' if feature_stability >= CONFIG['feature_stability_min'] else 'FAIL'))
    print("  max_skew={:.3f} (<{}: {})".format(
        max_skew, CONFIG['pred_skew_max'],
        'PASS' if max_skew < CONFIG['pred_skew_max'] else 'FAIL'))
    print("  mean_SNR={:.2f} (>={}: {})".format(
        mean_snr, CONFIG['snr_alert_threshold'],
        'PASS' if mean_snr >= CONFIG['snr_alert_threshold'] else 'FAIL'))

    # P6: Composite stability score
    score = 0
    if mean_pearson > 0.10:
        score += 15
    if mean_pearson > 0.20:
        score += 10
    if mean_skill_har > 0:
        score += 15
    if agg.get('mean_M9b_skill_garch', -1) > 0:
        score += 10
    if cross_pass >= 2:
        score += 10
    if feature_stability >= 50:
        score += 10
    if max_skew < 2:
        score += 10
    if mean_snr >= 2:
        score += 10
    if all(fr.get('isoos_ratio', 99) < CONFIG['isoos_warn_threshold']
           for fr in all_fold_results if fr.get('status') == 'OK'):
        score += 10

    print("\n[P6] STABILITY SCORE = {}/100".format(score))

    # Verdict
    print("\n{}".format("=" * 60))
    if not EDGE_VALID:
        print("VERDICT: NO EDGE -- SIGNAL INVALID OR UNSTABLE")
        level = 0
    elif score < CONFIG['stability_score_min']:
        print("VERDICT: EDGE DETECTED BUT UNSTABLE (score={} < {})".format(
            score, CONFIG['stability_score_min']))
        level = 1
    elif mean_snr < CONFIG['snr_alert_threshold']:
        print("VERDICT: EDGE DETECTED BUT LOW SNR ({:.2f})".format(mean_snr))
        level = 2
    else:
        print("VERDICT: EDGE CONFIRMED -- SIGNAL VALID AND STABLE")
        level = 3
    print("LEVEL = {}".format(level))
    print("{}".format("=" * 60))

    results = {
        'folds': all_fold_results,
        'aggregate': agg,
        'edge_valid': EDGE_VALID,
        'stability_score': score,
        'level': level,
        'mean_snr': mean_snr,
        'feature_stability': feature_stability,
    }

    # Send aggregate report to Telegram
    if telegram_config:
        send_aggregate_report(results, telegram_config)

    return results
