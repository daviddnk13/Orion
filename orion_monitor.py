#!/usr/bin/env python3
"""
Orion V20.7 — Paper Trading Monitor
Lightweight Flask dashboard for real-time monitoring.
"""

import json
import csv
import os
import subprocess
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

STATE_FILE = '/home/ubuntu/orion/state.json'
LOG_FILE = '/home/ubuntu/orion/paper_trading_log.csv'
SERVICE_NAME = 'orion-paper'

# ─────────────────────────────────────────────
# HTML Template (inline)
# ─────────────────────────────────────────────

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Orion V20.7 Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:16px; }
  .header { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; flex-wrap:wrap; gap:8px; }
  .header h1 { font-size:1.5rem; font-weight:600; }
  .header h1 span { color:var(--accent); }
  .badge { padding:4px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }
  .badge-green { background:rgba(63,185,80,0.2); color:var(--green); }
  .badge-red { background:rgba(248,81,73,0.2); color:var(--red); }
  .badge-yellow { background:rgba(210,153,34,0.2); color:var(--yellow); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:20px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }
  .card-label { font-size:0.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
  .card-value { font-size:1.5rem; font-weight:700; }
  .card-sub { font-size:0.8rem; color:var(--muted); margin-top:2px; }
  .chart-container { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:20px; }
  .chart-container h2 { font-size:1rem; margin-bottom:12px; color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:0.85rem; }
  th { text-align:left; padding:8px 12px; color:var(--muted); border-bottom:1px solid var(--border); font-weight:500; }
  td { padding:8px 12px; border-bottom:1px solid var(--border); }
  tr:hover td { background:rgba(88,166,255,0.04); }
  .pos-green { color:var(--green); } .pos-red { color:var(--red); }
  .refresh-bar { text-align:center; color:var(--muted); font-size:0.75rem; padding:12px; }
  @media(max-width:600px) { .grid { grid-template-columns:1fr 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>&#9678; <span>Orion</span> V20.7</h1>
  <div>
    <span id="service-badge" class="badge badge-green">LOADING</span>
    <span id="refresh-timer" class="badge badge-yellow">--s</span>
  </div>
</div>

<div class="grid" id="metrics"></div>

<div class="chart-container">
  <h2>Equity Curve</h2>
  <canvas id="equityChart" height="80"></canvas>
</div>

<div class="chart-container">
  <h2>Position History</h2>
  <canvas id="positionChart" height="60"></canvas>
</div>

<div class="chart-container">
  <h2>Trade Log (Last 50 bars)</h2>
  <div style="overflow-x:auto;">
    <table id="logTable">
      <thead>
        <tr>
          <th>UTC</th><th>Price</th><th>P(HIGH)</th><th>Position</th>
          <th>Equity</th><th>DD%</th><th>Latency</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="refresh-bar">Auto-refresh every 30s &bull; <span id="last-update">--</span></div>

<script>
let equityChart, positionChart;

async function fetchData() {
  try {
    const [statusRes, historyRes] = await Promise.all([
      fetch('/api/status'), fetch('/api/history')
    ]);
    const status = await statusRes.json();
    const history = await historyRes.json();
    renderMetrics(status);
    renderCharts(history);
    renderTable(history);
    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();

    const badge = document.getElementById('service-badge');
    if (status.service_active) {
      badge.textContent = 'ACTIVE'; badge.className = 'badge badge-green';
    } else {
      badge.textContent = 'DOWN'; badge.className = 'badge badge-red';
    }
  } catch(e) {
    console.error(e);
  }
}

function renderMetrics(s) {
  const eq = s.equity != null ? s.equity : 1;
  const dd = s.drawdown_pct != null ? s.drawdown_pct : 0;
  const pos = s.position != null ? s.position : 0;
  const prob = s.last_proba != null ? s.last_proba : '--';
  const step = s.step != null ? s.step : 0;
  const lastTs = s.last_bar_ts || '--';
  const pnl = ((eq - 1) * 100);
  const pnlClass = pnl >= 0 ? 'pos-green' : 'pos-red';
  const ddClass = dd > 20 ? 'pos-red' : dd > 10 ? 'pos-yellow' : 'pos-green';

  document.getElementById('metrics').innerHTML = `
    <div class="card">
      <div class="card-label">Equity</div>
      <div class="card-value">${eq.toFixed(4)}</div>
      <div class="card-sub ${pnlClass}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%</div>
    </div>
    <div class="card">
      <div class="card-label">Position</div>
      <div class="card-value">${(pos * 100).toFixed(1)}%</div>
      <div class="card-sub">of capital</div>
    </div>
    <div class="card">
      <div class="card-label">P(HIGH VOL)</div>
      <div class="card-value">${typeof prob === 'number' ? (prob * 100).toFixed(1) + '%' : prob}</div>
      <div class="card-sub">regime probability</div>
    </div>
    <div class="card">
      <div class="card-label">Drawdown</div>
      <div class="card-value ${ddClass}">${dd.toFixed(2)}%</div>
      <div class="card-sub">from peak</div>
    </div>
    <div class="card">
      <div class="card-label">Bars Processed</div>
      <div class="card-value">${step}</div>
      <div class="card-sub">4H candles</div>
    </div>
    <div class="card">
      <div class="card-label">Last Bar</div>
      <div class="card-value" style="font-size:1rem">${lastTs}</div>
      <div class="card-sub">UTC timestamp</div>
    </div>
  `;
}

function renderCharts(history) {
  const labels = history.map(r => r.bar_timestamp ? r.bar_timestamp.slice(5,16) : '');
  const eqData = history.map(r => parseFloat(r.equity_after || 1));
  const posData = history.map(r => parseFloat(r.position || 0) * 100);

  const eqCtx = document.getElementById('equityChart').getContext('2d');
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(eqCtx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Equity',
        data: eqData,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      }
    }
  });

  const posCtx = document.getElementById('positionChart').getContext('2d');
  if (positionChart) positionChart.destroy();
  positionChart = new Chart(posCtx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Position %',
        data: posData,
        backgroundColor: posData.map(v => v > 30 ? 'rgba(63,185,80,0.6)' : 'rgba(210,153,34,0.6)'),
        borderRadius: 2
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { display: false } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, max: 60 }
      }
    }
  });
}

function renderTable(history) {
  const tbody = document.querySelector('#logTable tbody');
  const last50 = history.slice(-50).reverse();
  tbody.innerHTML = last50.map(r => {
    const dd = parseFloat(r.drawdown_pct || 0);
    const ddClass = dd > 20 ? 'pos-red' : dd > 10 ? 'pos-yellow' : '';
    const pos = (parseFloat(r.position || 0) * 100).toFixed(1);
    return `<tr>
      <td>${(r.bar_timestamp || '').slice(0,16)}</td>
      <td>$${parseFloat(r.price || 0).toFixed(2)}</td>
      <td>${(parseFloat(r.proba_high || 0) * 100).toFixed(1)}%</td>
      <td>${pos}%</td>
      <td>${parseFloat(r.equity_after || 1).toFixed(4)}</td>
      <td class="${ddClass}">${dd.toFixed(2)}%</td>
      <td>${r.execution_latency_ms || '--'}ms</td>
    </tr>`;
  }).join('');
}

// Refresh timer
let countdown = 30;
setInterval(() => {
  countdown--;
  document.getElementById('refresh-timer').textContent = countdown + 's';
  if (countdown <= 0) { countdown = 30; fetchData(); }
}, 1000);

fetchData();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(TEMPLATE)


@app.route('/api/status')
def api_status():
    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
        except Exception:
            state = {'error': 'Failed to read state'}

    # Check if systemd service is active
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', SERVICE_NAME],
            capture_output=True, text=True, timeout=5
        )
        state['service_active'] = result.stdout.strip() == 'active'
    except Exception:
        state['service_active'] = False

    return jsonify(state)


@app.route('/api/history')
def api_history():
    rows = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            rows = []
    return jsonify(rows)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
