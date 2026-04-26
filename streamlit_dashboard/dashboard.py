#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion V20.7 Streamlit Dashboard
Interactive monitoring interface with login.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
import requests

# =========================
# CONFIGURATION
# =========================
DATA_DIR = os.getenv('DATA_DIR', '/data')
STATE_PATH = os.path.join(DATA_DIR, 'state.json')
LOG_PATH = os.path.join(DATA_DIR, 'paper_trading_log.csv')
USERS_FILE = os.getenv('USERS_FILE', 'users.json')  # Relative to workdir (inside container)

OKX_SYMBOL = 'ETH-USDT'
OKX_TIMEFRAME = '4H'
OKX_N_CANDLES = 200  # para gráficos de mercado
REFRESH_INTERVAL = 30  # segundos entre refrescos de datos (cached)
MARKET_DATA_TTL = 300  # segundos para cache de datos de mercado (5 min)

# =========================
# AUTHENTICATION
# =========================
def load_users():
    users = {}
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
        except Exception as e:
            st.error(f"Error loading users file: {e}")
    return users

def check_credentials(username, password):
    users = load_users()
    if username in users:
        stored_hash = users[username]
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        return stored_hash == pwd_hash
    env_user = os.getenv('DASHBOARD_USERNAME')
    env_pass = os.getenv('DASHBOARD_PASSWORD')
    if env_user and env_pass:
        return username == env_user and password == env_pass
    return False

def login_ui():
    st.title("🔐 Orion Dashboard Login")
    st.write("Please enter your credentials.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.user = username
                st.success("Login successful!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid username or password")

def logout():
    if 'authenticated' in st.session_state:
        del st.session_state.authenticated
    if 'user' in st.session_state:
        del st.session_state.user
    st.rerun()

# =========================
# DATA FUNCTIONS
# =========================
def load_state():
    """Carga state.json del bot."""
    try:
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        st.error("state.json corrupto")
        return {}
    except Exception as e:
        st.error(f"Error leyendo state.json: {e}")
        return {}

def load_log():
    """Carga paper_trading_log.csv como DataFrame."""
    try:
        if not os.path.exists(LOG_PATH):
            return pd.DataFrame(columns=[
                'timestamp','price_close','proba_high','scale_raw','scale_final',
                'position_size','theoretical_return','equity_curve','current_drawdown',
                'volatility_estimate','execution_latency_ms','features_hash','drift_warning'
            ])
        df = pd.read_csv(LOG_PATH)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        numeric_cols = ['price_close','proba_high','scale_raw','scale_final',
                        'position_size','theoretical_return','equity_curve','current_drawdown',
                        'volatility_estimate','execution_latency_ms']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error leyendo log CSV: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calcula métricas de rendimiento desde el log."""
    if df.empty or len(df) < 2:
        return {}
    rets = df['theoretical_return'].dropna()
    if len(rets) < 2:
        return {}
    mean_ret = rets.mean()
    std_ret = rets.std()
    if std_ret > 0:
        sharpe = mean_ret / std_ret * np.sqrt(6 * 365)
    else:
        sharpe = 0.0
    win_rate = (rets > 0).mean()
    if 'equity_curve' in df.columns:
        equity = df['equity_curve'].dropna()
        if len(equity) > 0:
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
            max_dd = drawdown.min()
        else:
            max_dd = np.nan
    else:
        max_dd = np.nan
    if 'equity_curve' in df.columns and len(df) > 0:
        start_eq = df['equity_curve'].iloc[0]
        end_eq = df['equity_curve'].iloc[-1]
        if pd.notna(start_eq) and pd.notna(end_eq) and start_eq != 0:
            total_return = (end_eq / start_eq - 1) * 100
        else:
            total_return = np.nan
    else:
        total_return = np.nan
    if 'position_size' in df.columns:
        pos_changes = (df['position_size'] != df['position_size'].shift()).sum()
        n_trades = pos_changes
    else:
        n_trades = np.nan
    return {
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'total_return': total_return,
        'n_bars': len(df),
        'n_trades': n_trades
    }

def fetch_okx_ohlcv(symbol='ETH-USDT', timeframe='4H', n_candles=12000):
    """Descarga velas de OKX."""
    url_recent = "https://www.okx.com/api/v5/market/candles"
    url_hist = "https://www.okx.com/api/v5/market/history-candles"
    all_candles = []
    limit = 100
    end_ts = None
    while len(all_candles) < n_candles:
        params = {'instId': symbol, 'bar': timeframe, 'limit': str(limit)}
        if end_ts:
            params['after'] = str(end_ts)
        try:
            resp = requests.get(url_recent, params=params, timeout=10)
            data = resp.json()
            if 'data' not in data or len(data['data']) == 0:
                break
            candles = data['data']
            all_candles.extend(candles)
            end_ts = int(candles[-1][0])
            if len(candles) < limit:
                break
            time.sleep(0.1)
        except Exception as e:
            st.warning(f"Error fetching recent candles: {e}")
            break
    if len(all_candles) < n_candles:
        fail_count = 0
        while len(all_candles) < n_candles and fail_count < 3:
            params = {'instId': symbol, 'bar': timeframe, 'limit': str(limit)}
            if end_ts:
                params['after'] = str(end_ts)
            try:
                resp = requests.get(url_hist, params=params, timeout=10)
                data = resp.json()
                if 'data' not in data or len(data['data']) == 0:
                    fail_count += 1
                    time.sleep(1)
                    continue
                candles = data['data']
                all_candles.extend(candles)
                end_ts = int(candles[-1][0])
                if len(candles) < limit:
                    break
                fail_count = 0
                time.sleep(0.1)
            except Exception as e:
                st.warning(f"Error fetching history candles: {e}")
                fail_count += 1
                time.sleep(1)
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'volume_ccy', 'volume_quote', 'confirm'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) > n_candles:
        df = df.tail(n_candles).reset_index(drop=True)
    return df

@st.cache_data(ttl=MARKET_DATA_TTL)
def get_market_data():
    return fetch_okx_ohlcv(OKX_SYMBOL, OKX_TIMEFRAME, OKX_N_CANDLES)

# =========================
# DASHBOARD LAYOUT
# =========================
def dashboard_page():
    with st.sidebar:
        st.header(f"🤖 Orion")
        st.write(f"User: **{st.session_state.user}**")
        if st.button("Logout"):
            logout()
        st.divider()
        state = load_state()
        bot_status = "❓ Unknown"
        if state:
            last_ts_str = state.get('last_timestamp') or state.get('last_bar_ts')
            if last_ts_str:
                try:
                    last_ts = datetime.fromisoformat(last_ts_str.replace('Z', '+00:00'))
                    age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
                    if age_min < 60:
                        bot_status = f"✅ Active ({age_min:.1f} min ago)"
                    else:
                        bot_status = f"⚠️ Stale ({age_min:.1f} min ago)"
                except:
                    bot_status = "⚠️ Invalid timestamp"
        st.write(f"Bot status: {bot_status}")
        st.caption("V20.7 Paper Trading")
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)

    st.title("🔷 Orion V20.7 Dashboard")
    st.markdown("---")

    state = load_state()
    df_log = load_log()
    metrics = calculate_metrics(df_log)

    col1, col2, col3, col4 = st.columns(4)
    equity = state.get('equity', 1.0)
    col1.metric("Equity", f"{equity:.4f}")
    position = state.get('position', state.get('position_size', 0.0))
    col2.metric("Position", f"{position*100:.1f}%")
    dd = state.get('current_drawdown', state.get('drawdown_pct', 0.0))
    if dd > 1:
        dd_display = f"{dd:.2f}%"
        dd_num = dd / 100
    else:
        dd_display = f"{dd*100:.2f}%"
        dd_num = dd
    col3.metric("Drawdown", dd_display)
    prob = state.get('last_proba')
    col4.metric("P(HIGH)", f"{prob*100:.1f}%" if prob is not None else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    sharpe = metrics.get('sharpe', np.nan)
    col5.metric("Sharpe (anual)", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
    max_dd = metrics.get('max_dd', np.nan)
    col6.metric("Max Drawdown", f"{max_dd*100:.1f}%" if not np.isnan(max_dd) else "N/A")
    win_rate = metrics.get('win_rate', np.nan)
    col7.metric("Win Rate", f"{win_rate*100:.1f}%" if not np.isnan(win_rate) else "N/A")
    total_ret = metrics.get('total_return', np.nan)
    col8.metric("Total Return", f"{total_ret:+.2f}%" if not np.isnan(total_ret) else "N/A")

    st.markdown("## 📈 Performance Charts")
    if not df_log.empty and 'equity_curve' in df_log.columns and 'timestamp' in df_log.columns:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=df_log['timestamp'], y=df_log['equity_curve'],
                                    mode='lines', name='Equity', line=dict(color='#00bcd4')))
        fig_eq.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity',
                             template='plotly_dark', height=300)
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No equity curve data available yet.")

    st.markdown("## 📉 Price & Position")
    try:
        market_df = get_market_data()
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        market_df = pd.DataFrame()
    if not market_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3],
                            subplot_titles=('Price (ETH-USDT)', 'Position Size'))
        fig.add_trace(go.Candlestick(
            x=market_df['timestamp'],
            open=market_df['open'],
            high=market_df['high'],
            low=market_df['low'],
            close=market_df['close'],
            name='OHLC'
        ), row=1, col=1)
        if not df_log.empty and 'timestamp' in df_log.columns:
            fig.add_trace(go.Bar(
                x=df_log['timestamp'],
                y=df_log['position_size'] * 100,
                name='Position %',
                marker_color='rgba(255, 100, 100, 0.6)'
            ), row=2, col=1)
        fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No market data available (OKX fetch failed).")

    st.markdown("## 📋 Latest Signals")
    if not df_log.empty:
        display_cols = ['timestamp', 'price_close', 'proba_high', 'position_size',
                        'equity_curve', 'current_drawdown', 'volatility_estimate']
        available_cols = [col for col in display_cols if col in df_log.columns]
        df_display = df_log[available_cols].copy()
        if 'timestamp' in df_display.columns:
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        if 'price_close' in df_display.columns:
            df_display['price_close'] = df_display['price_close'].map('${:.2f}'.format)
        if 'proba_high' in df_display.columns:
            df_display['proba_high'] = df_display['proba_high'].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else 'N/A')
        if 'position_size' in df_display.columns:
            df_display['position_size'] = df_display['position_size'].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else 'N/A')
        if 'equity_curve' in df_display.columns:
            df_display['equity_curve'] = df_display['equity_curve'].map('{:.4f}'.format)
        if 'current_drawdown' in df_display.columns:
            df_display['current_drawdown'] = df_display['current_drawdown'].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')
        if 'volatility_estimate' in df_display.columns:
            df_display['volatility_estimate'] = df_display['volatility_estimate'].map('{:.4f}'.format)
        st.dataframe(df_display.sort_values('timestamp', ascending=False).head(15), use_container_width=True)
    else:
        st.info("No signals logged yet.")

    st.markdown("---")
    st.caption("Dashboard auto-refreshes every 30 seconds if enabled.")
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

def main():
    st.set_page_config(page_title="Orion Dashboard", page_icon="🔷", layout="wide")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if not st.session_state.authenticated:
        login_ui()
    else:
        dashboard_page()

if __name__ == '__main__':
    main()
