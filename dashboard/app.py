#!/home/ubuntu/orion/dashboard/venv/bin/python3
# -*- coding: utf-8 -*-
"""
Orion Dashboard V20.9 — Flask backend for paper trading monitoring
Security: Cloudflare Turnstile + IP Rate Limiting
"""

import os
import json
import csv
import time
import psutil
import subprocess
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

import flask
from flask import Flask, render_template, session, redirect, url_for, request, jsonify, flash
import requests

from config import Config

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)
app.permanent_session_lifetime = Config.SESSION_TIMEOUT

# --- Rate Limiting ---
_login_attempts = defaultdict(list)

def _clean_old_attempts(ip):
    cutoff = time.time() - Config.LOGIN_BLOCK_SECONDS
    _login_attempts[ip] = [t for t in _login_attempts[ip] if t > cutoff]

def _is_blocked(ip):
    _clean_old_attempts(ip)
    return len(_login_attempts[ip]) >= Config.MAX_LOGIN_ATTEMPTS

def _record_attempt(ip):
    _login_attempts[ip].append(time.time())

def _clear_attempts(ip):
    _login_attempts.pop(ip, None)

# --- Turnstile Verification ---
def verify_turnstile(token, ip):
    if not token:
        return False
    try:
        resp = requests.post(Config.TURNSTILE_VERIFY_URL, data={
            'secret': Config.TURNSTILE_SECRET_KEY,
            'response': token,
            'remoteip': ip
        }, timeout=5)
        result = resp.json()
        return result.get('success', False)
    except Exception as e:
        app.logger.error(f"Turnstile verification error: {e}")
        return False

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip and ',' in ip:
        ip = ip.split(',')[0].strip()

    if request.method == 'POST':
        # Check rate limit
        if _is_blocked(ip):
            _clean_old_attempts(ip)
            remaining = int(Config.LOGIN_BLOCK_SECONDS - (time.time() - _login_attempts[ip][0]))
            mins = max(1, remaining // 60)
            flash(f'Too many attempts. Try again in {mins} min.')
            return render_template('index.html', turnstile_site_key=Config.TURNSTILE_SITE_KEY), 429

        # Verify Turnstile
        turnstile_token = request.form.get('cf-turnstile-response', '')
        if not verify_turnstile(turnstile_token, ip):
            flash('Security verification failed. Please try again.')
            return render_template('index.html', turnstile_site_key=Config.TURNSTILE_SITE_KEY), 403

        # Check password
        password = request.form.get('password', '')
        if password == Config.DASHBOARD_PASSWORD:
            session['authenticated'] = True
            session.permanent = True
            _clear_attempts(ip)
            return redirect(url_for('index'))
        else:
            _record_attempt(ip)
            remaining = Config.MAX_LOGIN_ATTEMPTS - len(_login_attempts[ip])
            if remaining > 0:
                flash(f'Invalid password. {remaining} attempts remaining.')
            else:
                flash(f'Too many attempts. Blocked for {Config.LOGIN_BLOCK_SECONDS // 60} min.')
            return render_template('index.html', turnstile_site_key=Config.TURNSTILE_SITE_KEY), 401

    return render_template('index.html', turnstile_site_key=Config.TURNSTILE_SITE_KEY)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

_price_cache = {'data': None, 'timestamp': 0}

def get_okx_prices():
    cache_ttl = Config.PRICE_CACHE_SECONDS
    now = time.time()
    if _price_cache['data'] and (now - _price_cache['timestamp']) < cache_ttl:
        return _price_cache['data']

    prices = {}
    for inst_id in ['ETH-USDT', 'BTC-USDT', 'SOL-USDT']:
        try:
            r = requests.get(Config.OKX_TICKER_URL, params={'instId': inst_id}, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get('code') == '0' and data.get('data'):
                item = data['data'][0]
                last = float(item.get('last', 0))
                open24h = float(item.get('open24h', 0))
                change_pct = ((last - open24h) / open24h * 100) if open24h > 0 else 0.0
                symbol = inst_id.replace('-', '/')
                prices[symbol] = {
                    'price': last,
                    'change_pct': round(change_pct, 2),
                    'high24h': float(item.get('high24h', 0)),
                    'low24h': float(item.get('low24h', 0)),
                    'vol24h': float(item.get('volCcy24h', 0)),
                    'timestamp': now
                }
        except Exception as e:
            app.logger.error(f"OKX API error for {inst_id}: {e}")
    if prices:
        _price_cache['data'] = prices
        _price_cache['timestamp'] = now
    return prices

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat() + 'Z'})

@app.route('/api/status')
@login_required
def api_status():
    state_path = Config.ORION_STATE_PATH
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        return jsonify(state)
    except FileNotFoundError:
        return jsonify({'error': 'State file not found'}), 404
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v21/status')
@login_required
def api_v21_status():
    state_path = Config.ORION_V21_STATE_PATH
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        return jsonify(state)
    except FileNotFoundError:
        return jsonify({'error': 'State file not found'}), 404
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/prices')
@login_required
def api_prices():
    prices = get_okx_prices()
    return jsonify(prices)

@app.route('/api/history')
@login_required
def api_history():
    asset = request.args.get('asset', '').strip()
    limit = request.args.get('limit', 100, type=int)
    limit = max(1, min(limit, 1000))

    log_path = Config.ORION_LOG_PATH
    history = []
    try:
        with open(log_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if asset:
                rows = [r for r in rows if r.get('asset') == asset]
            for row in reversed(rows[:limit]):
                history.append(row)
        return jsonify(history)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        app.logger.error(f"History read error: {e}")
        return jsonify([])

@app.route('/api/v21/history')
@login_required
def api_v21_history():
    asset = request.args.get('asset', '').strip()
    limit = request.args.get('limit', 100, type=int)
    limit = max(1, min(limit, 1000))

    log_path = Config.ORION_V21_LOG_PATH
    history = []
    try:
        with open(log_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if asset:
                rows = [r for r in rows if r.get('asset') == asset]
            for row in reversed(rows[:limit]):
                history.append(row)
        return jsonify(history)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        app.logger.error(f"V21 History read error: {e}")
        return jsonify([])

@app.route('/api/system')
@login_required
def api_system():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        hostname = subprocess.getoutput('hostname').strip()

        # Check both services
        service_status = {}
        for service in ['orion-v209', 'orion-v21']:
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', service],
                    capture_output=True, text=True, timeout=3
                )
                status = result.stdout.strip()
                if result.returncode != 0:
                    status = 'unknown'
                service_status[service] = status
            except Exception:
                service_status[service] = 'error'

        try:
            uptime_sec = time.time() - psutil.boot_time()
            uptime = str(timedelta(seconds=int(uptime_sec)))
        except Exception:
            uptime = 'unknown'

        return jsonify({
            'cpu_percent': cpu_percent,
            'memory': {
                'percent': memory.percent,
                'used_gb': memory.used / (1024**3),
                'total_gb': memory.total / (1024**3)
            },
            'disk': {
                'percent': disk.percent,
                'used_gb': disk.used / (1024**3),
                'total_gb': disk.total / (1024**3)
            },
            'service_status': service_status,
            'hostname': hostname,
            'uptime': uptime
        })
    except Exception as e:
        app.logger.error(f"System info error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
