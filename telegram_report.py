# ============================================================
# telegram_report.py — V19.0 Telegram reporting
# Bot: @Sistem_tradingbot | Group: alerta de mi ia
# Topics: 971=Resultados, 972=Modelos, 973=Alertas, 974=Changelog
# ============================================================

import requests
import numpy as np
from datetime import datetime


def send_telegram(text, topic_id, telegram_config):
    # Send message to specific Telegram topic
    if not telegram_config.get('enabled', False):
        print("  [TG] Telegram disabled, skipping")
        return False
    bot_token = telegram_config['bot_token']
    chat_id = telegram_config['chat_id']
    url = "https://api.telegram.org/bot{}/sendMessage".format(bot_token)
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML',
        'message_thread_id': topic_id,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            print("  [TG] Message sent OK")
            return True
        else:
            print("  [TG] Error {}: {}".format(resp.status_code, resp.text[:100]))
            return False
    except Exception as e:
        print("  [TG] Send failed: {}".format(e))
        return False


def send_start_report(version, n_folds, n_candles, telegram_config):
    # Send experiment start notification to Alertas topic
    topic = telegram_config['topics']['alertas']
    text = (
        "<b>V{} EXPERIMENT STARTED</b>\n"
        "Folds: {}\n"
        "Candles: {}\n"
        "Time: {}"
    ).format(version, n_folds, n_candles, datetime.now().strftime('%Y-%m-%d %H:%M'))
    send_telegram(text, topic, telegram_config)


def send_fold_report(fold_num, fold_result, telegram_config):
    # Send per-fold results to topic 971 (Resultados)
    topic = telegram_config['topics']['resultados']
    m = fold_result.get('metrics', {})
    d = fold_result.get('diagnostics', {})
    status = fold_result.get('status', 'UNKNOWN')

    pearson = m.get('M1_pearson', 0)
    spearman = m.get('M2_spearman', 0)
    mae = m.get('M3_mae', 0)
    rmse = m.get('M4_rmse', 0)
    dir_acc = m.get('M5_directional_acc', 0)
    qhit = m.get('M6_quantile_hit', 0)
    regime_acc = m.get('M7_regime_acc', 0)
    skill_har = m.get('M9_skill_har', 0)
    isoos = fold_result.get('isoos_ratio', 0)
    snr = d.get('snr', 0)
    n_feat = fold_result.get('n_features_used', 0)

    text = (
        "<b>V19.0 FOLD {} -- {}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Pearson:     {:+.4f}\n"
        "Spearman:    {:+.4f}\n"
        "MAE:         {:.6f}\n"
        "RMSE:        {:.6f}\n"
        "Dir Acc:     {:.1f}%\n"
        "Q-Hit:       {:.1f}%\n"
        "Regime Acc:  {:.1f}%\n"
        "Skill/HAR:   {:+.4f}\n"
        "IS/OOS:      {:.2f}x\n"
        "SNR:         {:.2f}\n"
        "Features:    {}\n"
        "Top5:        {}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "{}"
    ).format(
        fold_num, status,
        pearson, spearman, mae, rmse,
        dir_acc * 100, qhit * 100, regime_acc * 100,
        skill_har, isoos, snr, n_feat,
        fold_result.get('top5_features', []),
        datetime.now().strftime('%Y-%m-%d %H:%M')
    )
    send_telegram(text, topic, telegram_config)


def send_aggregate_report(results, telegram_config):
    # Send aggregate + verdict to topic 971 (Resultados)
    topic = telegram_config['topics']['resultados']
    agg = results.get('aggregate', {})
    level = results.get('level', 0)
    score = results.get('stability_score', 0)
    edge = results.get('edge_valid', False)
    mean_snr = results.get('mean_snr', 0)
    fstab = results.get('feature_stability', 0)

    verdicts = {
        0: 'NO EDGE',
        1: 'EDGE UNSTABLE',
        2: 'EDGE LOW SNR',
        3: 'EDGE CONFIRMED',
    }

    text = (
        "<b>V19.0 AGGREGATE RESULTS</b>\n"
        "{'='*30}\n"
        "VERDICT:     {}\n"
        "LEVEL:       {}/3\n"
        "EDGE_VALID:  {}\n"
        "Score:       {}/100\n"
        "Mean SNR:    {:.2f}\n"
        "Feat Stab:   {:.1f}%\n"
        "{'='*30}\n"
        "mean_Pearson:   {:.4f}\n"
        "mean_Spearman:  {:.4f}\n"
        "mean_MAE:       {:.6f}\n"
        "mean_DirAcc:    {:.1f}%\n"
        "mean_Skill/HAR: {:.4f}\n"
        "{'='*30}\n"
        "{}"
    ).format(
        verdicts.get(level, 'UNKNOWN'),
        level, edge, score, mean_snr, fstab,
        agg.get('mean_M1_pearson', 0),
        agg.get('mean_M2_spearman', 0),
        agg.get('mean_M3_mae', 0),
        agg.get('mean_M5_directional_acc', 0) * 100,
        agg.get('mean_M9_skill_har', 0),
        datetime.now().strftime('%Y-%m-%d %H:%M')
    )
    send_telegram(text, topic, telegram_config)

    # Also send to Modelos topic (972)
    model_topic = telegram_config['topics']['modelos']
    model_text = (
        "<b>V19.0 MODEL SUMMARY</b>\n"
        "Level: {}/3\n"
        "Edge: {}\n"
        "Score: {}/100\n"
        "Verdict: {}\n"
        "{}"
    ).format(level, edge, score, verdicts.get(level, 'UNKNOWN'),
             datetime.now().strftime('%Y-%m-%d %H:%M'))
    send_telegram(model_text, model_topic, telegram_config)


def send_alert(message, telegram_config):
    # Send alert to topic 973 (Alertas)
    topic = telegram_config['topics']['alertas']
    text = "<b>V19.0 ALERT</b>\n{}\n{}".format(
        message, datetime.now().strftime('%H:%M'))
    send_telegram(text, topic, telegram_config)


def send_changelog(changes, telegram_config):
    # Send changelog to topic 974 (Changelog)
    topic = telegram_config['topics']['changelog']
    text = "<b>V19.0 CHANGELOG</b>\n{}\n{}".format(
        changes, datetime.now().strftime('%Y-%m-%d %H:%M'))
    send_telegram(text, topic, telegram_config)
