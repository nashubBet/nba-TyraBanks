"""
NBA Betting Model — Flask Web App
Run locally or deploy to Railway/Render
"""

from flask import Flask, render_template_string, jsonify
from datetime import datetime
import json, os, sys

# Import your model (same directory)
sys.path.insert(0, os.path.dirname(__file__))
try:
    from nba_betting_model import BettingAnalyzer, Backtester, ModelConfig
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────
config = ModelConfig(
    bankroll=float(os.environ.get("BANKROLL", 1000)),
    kelly_fraction=0.25,
    min_edge_threshold=0.03,
    min_ev_threshold=0.04,
    confidence_threshold=0.55,
    odds_api_key=os.environ.get("ODDS_API_KEY", "YOUR_ODDS_API_KEY"),
)

# Initialize and train once on startup
analyzer = None
if MODEL_AVAILABLE:
    analyzer = BettingAnalyzer(config)
    analyzer.setup()

# ── HTML Template ───────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="NBA Edge">
<title>NBA Edge</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080c10;
    --surface: #0f1520;
    --surface2: #161e2e;
    --border: #1e2d42;
    --accent: #f97316;
    --accent2: #3b82f6;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --text: #e8edf5;
    --muted: #64748b;
    --safe: env(safe-area-inset-top);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    padding-bottom: 40px;
    background-image:
      radial-gradient(ellipse 60% 40% at 80% -10%, rgba(249,115,22,0.08) 0%, transparent 60%),
      radial-gradient(ellipse 50% 30% at -10% 80%, rgba(59,130,246,0.06) 0%, transparent 60%);
  }

  /* ── Header ── */
  .header {
    padding: calc(var(--safe) + 20px) 20px 16px;
    background: linear-gradient(180deg, rgba(8,12,16,0.98) 0%, transparent 100%);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
  }
  .header-top { display: flex; align-items: center; justify-content: space-between; }
  .logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 2px;
    color: var(--text);
  }
  .logo span { color: var(--accent); }
  .date-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 20px;
  }
  .bankroll-bar {
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .bankroll-item { display: flex; flex-direction: column; gap: 2px; }
  .bankroll-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .bankroll-value { font-family: 'DM Mono', monospace; font-size: 16px; color: var(--green); font-weight: 500; }
  .bankroll-value.neutral { color: var(--text); }

  /* ── Refresh Button ── */
  .refresh-btn {
    width: 36px; height: 36px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
    margin-left: auto;
  }
  .refresh-btn:active { transform: scale(0.92); background: var(--border); }
  .refresh-btn svg { width: 16px; height: 16px; color: var(--muted); }
  .refresh-btn.spinning svg { animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Tabs ── */
  .tabs {
    display: flex;
    padding: 16px 20px 0;
    gap: 6px;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tab {
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
  }
  .tab.active {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  /* ── Content ── */
  .content { padding: 16px 20px; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── Section Label ── */
  .section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 10px;
    margin-top: 4px;
  }

  /* ── Game Card ── */
  .game-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 12px;
    overflow: hidden;
    transition: transform 0.15s;
  }
  .game-card:active { transform: scale(0.99); }
  .game-card.has-bet { border-color: rgba(249,115,22,0.4); }

  .game-header {
    padding: 14px 16px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .game-teams { display: flex; flex-direction: column; gap: 2px; }
  .team-matchup {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 17px;
    letter-spacing: 1px;
    color: var(--text);
    line-height: 1.2;
  }
  .game-time { font-size: 11px; color: var(--muted); }

  .bet-badge {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .bet-badge.hot { background: rgba(249,115,22,0.15); color: var(--accent); border: 1px solid rgba(249,115,22,0.3); }
  .bet-badge.edge { background: rgba(59,130,246,0.12); color: var(--accent2); border: 1px solid rgba(59,130,246,0.25); }
  .bet-badge.skip { background: rgba(100,116,139,0.1); color: var(--muted); border: 1px solid var(--border); }

  /* ── Prob Row ── */
  .prob-row {
    padding: 12px 16px;
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .prob-team { flex: 1; }
  .prob-name { font-size: 12px; color: var(--muted); margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .prob-bar-wrap { background: var(--surface2); border-radius: 4px; height: 6px; overflow: hidden; }
  .prob-bar { height: 100%; border-radius: 4px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }
  .prob-bar.home { background: linear-gradient(90deg, var(--accent2), #60a5fa); }
  .prob-bar.away { background: linear-gradient(90deg, var(--accent), #fb923c); }
  .prob-pct { font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; min-width: 38px; text-align: right; }

  .vs-divider {
    font-size: 10px;
    color: var(--border);
    font-weight: 700;
    padding: 0 2px;
  }

  /* ── Bet Recommendations ── */
  .bet-recs { padding: 0 16px 14px; display: flex; flex-direction: column; gap: 8px; }
  .bet-rec {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .bet-rec.high { border-color: rgba(249,115,22,0.3); background: rgba(249,115,22,0.05); }

  .bet-left { display: flex; flex-direction: column; gap: 3px; }
  .bet-type-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
  }
  .bet-team { font-size: 15px; font-weight: 600; color: var(--text); }
  .bet-meta { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }

  .bet-right { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; }
  .bet-amount {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px;
    letter-spacing: 1px;
    color: var(--green);
    line-height: 1;
  }
  .bet-odds {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }
  .ev-chip {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 10px;
    background: rgba(34,197,94,0.1);
    color: var(--green);
    border: 1px solid rgba(34,197,94,0.2);
  }

  /* ── No bet state ── */
  .no-bet { padding: 10px 16px 14px; }
  .no-bet-inner {
    background: var(--surface2);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .no-bet-text { font-size: 13px; color: var(--muted); }

  /* ── Stats Panel ── */
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
  }
  .stat-card.full { grid-column: 1 / -1; }
  .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 6px; }
  .stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 30px;
    letter-spacing: 1px;
    line-height: 1;
  }
  .stat-value.green { color: var(--green); }
  .stat-value.orange { color: var(--accent); }
  .stat-value.blue { color: var(--accent2); }
  .stat-sub { font-size: 11px; color: var(--muted); margin-top: 4px; }

  /* ── Rules Card ── */
  .rules-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .rules-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 16px;
    letter-spacing: 1px;
    color: var(--accent);
    margin-bottom: 12px;
  }
  .rule-row {
    display: flex;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    align-items: flex-start;
  }
  .rule-row:last-child { border-bottom: none; padding-bottom: 0; }
  .rule-icon { font-size: 14px; margin-top: 1px; flex-shrink: 0; }
  .rule-text { font-size: 13px; color: var(--text); line-height: 1.5; }
  .rule-text span { color: var(--muted); font-size: 12px; display: block; }

  /* ── Loading ── */
  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    gap: 16px;
  }
  .loader {
    width: 36px; height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  .loading-text { font-size: 14px; color: var(--muted); }

  /* ── Empty State ── */
  .empty {
    text-align: center;
    padding: 50px 20px;
  }
  .empty-icon { font-size: 40px; margin-bottom: 12px; }
  .empty-title { font-family: 'Bebas Neue', sans-serif; font-size: 22px; letter-spacing: 1px; color: var(--muted); }
  .empty-sub { font-size: 13px; color: var(--muted); margin-top: 6px; opacity: 0.7; }

  /* ── Toast ── */
  .toast {
    position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%) translateY(80px);
    background: var(--surface2); border: 1px solid var(--border);
    padding: 10px 18px; border-radius: 20px;
    font-size: 13px; color: var(--text);
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1);
    z-index: 999; white-space: nowrap;
  }
  .toast.show { transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-top">
    <div class="logo">NBA <span>EDGE</span></div>
    <div style="display:flex;gap:8px;align-items:center;">
      <div class="date-badge" id="dateLabel">---</div>
      <button class="refresh-btn" id="refreshBtn" onclick="loadPicks()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
          <path d="M3 3v5h5"/>
        </svg>
      </button>
    </div>
  </div>
  <div class="bankroll-bar">
    <div class="bankroll-item">
      <div class="bankroll-label">Bankroll</div>
      <div class="bankroll-value neutral" id="bankrollVal">$1,000</div>
    </div>
    <div class="bankroll-item">
      <div class="bankroll-label">Today's Action</div>
      <div class="bankroll-value" id="todayAction">$0</div>
    </div>
    <div class="bankroll-item">
      <div class="bankroll-label">Bets Flagged</div>
      <div class="bankroll-value" id="betCount">0</div>
    </div>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <button class="tab active" onclick="switchTab('picks', this)">🔥 Today's Picks</button>
  <button class="tab" onclick="switchTab('stats', this)">📊 Performance</button>
  <button class="tab" onclick="switchTab('rules', this)">📋 Rules</button>
</div>

<!-- Content -->
<div class="content">

  <!-- PICKS TAB -->
  <div class="tab-panel active" id="tab-picks">
    <div id="picksContent">
      <div class="loading">
        <div class="loader"></div>
        <div class="loading-text">Loading today's games...</div>
      </div>
    </div>
  </div>

  <!-- STATS TAB -->
  <div class="tab-panel" id="tab-stats">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value green" id="statWinRate">—</div>
        <div class="stat-sub">Target: 53%+</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">ROI</div>
        <div class="stat-value orange" id="statROI">—</div>
        <div class="stat-sub">Return on bets</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Bets</div>
        <div class="stat-value blue" id="statTotalBets">—</div>
        <div class="stat-sub">This session</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Net P&L</div>
        <div class="stat-value green" id="statPnL">—</div>
        <div class="stat-sub">All bets</div>
      </div>
      <div class="stat-card full">
        <div class="stat-label">Model Notes</div>
        <div style="font-size:13px;color:var(--muted);line-height:1.6;margin-top:4px;">
          XGBoost + Neural Net ensemble. Only bets with ≥3% edge and ≥4% EV are shown.
          Quarter Kelly sizing (25%). Break-even = 52.38% win rate.
        </div>
      </div>
    </div>
    <button onclick="runBacktest()" style="width:100%;padding:14px;background:var(--surface);border:1px solid var(--border);border-radius:14px;color:var(--text);font-size:14px;font-weight:600;cursor:pointer;">
      Run Backtest Simulation
    </button>
    <div id="backtestResult" style="margin-top:12px;"></div>
  </div>

  <!-- RULES TAB -->
  <div class="tab-panel" id="tab-rules">
    <div class="rules-card">
      <div class="rules-title">🏀 Betting Rules</div>
      <div class="rule-row">
        <div class="rule-icon">💰</div>
        <div class="rule-text">Never bet more than 5% of bankroll per game
          <span>Quarter Kelly keeps variance low over a full season</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">📉</div>
        <div class="rule-text">Only bet when model edge ≥ 3%
          <span>Below this, you're fighting the vig — not the book</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">🚫</div>
        <div class="rule-text">No parlays
          <span>They compound the book's edge against you</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">📊</div>
        <div class="rule-text">Track closing line value (CLV)
          <span>If your picks beat the closing line, your edge is real</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">😴</div>
        <div class="rule-text">B2B games favor the rested team
          <span>0 rest days = 44% cover rate, under hits 57%</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">🔁</div>
        <div class="rule-text">Recalibrate monthly
          <span>Update team stats at least once a week during season</span>
        </div>
      </div>
    </div>
    <div class="rules-card">
      <div class="rules-title">⚙️ Setup Checklist</div>
      <div class="rule-row">
        <div class="rule-icon">🔑</div>
        <div class="rule-text">Add ODDS_API_KEY to Railway environment
          <span>Free at the-odds-api.com — 500 req/month</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">💵</div>
        <div class="rule-text">Set BANKROLL in Railway environment variables
          <span>Default: $1,000</span>
        </div>
      </div>
      <div class="rule-row">
        <div class="rule-icon">📱</div>
        <div class="rule-text">Add to iPhone home screen
          <span>Safari → Share → Add to Home Screen</span>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
const BASE = '';

// ── Date ────────────────────────────────────────────────────
function setDate() {
  const d = new Date();
  const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('dateLabel').textContent =
    days[d.getDay()] + ' ' + months[d.getMonth()] + ' ' + d.getDate();
}
setDate();

// ── Toast ────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

// ── Tabs ────────────────────────────────────────────────────
function switchTab(id, el) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
}

// ── Confidence Badge ─────────────────────────────────────────
function badgeHTML(recs) {
  if (!recs || recs.length === 0)
    return '<span class="bet-badge skip">Skip</span>';
  const hasHigh = recs.some(r => r.confidence === 'HIGH');
  return hasHigh
    ? '<span class="bet-badge hot">🔥 Hot</span>'
    : '<span class="bet-badge edge">📊 Edge</span>';
}

// ── Render Picks ─────────────────────────────────────────────
function renderPicks(data) {
  const games = data.games || [];
  let totalAction = 0;
  let totalBets = 0;

  if (games.length === 0) {
    document.getElementById('picksContent').innerHTML = `
      <div class="empty">
        <div class="empty-icon">🏀</div>
        <div class="empty-title">No Games Today</div>
        <div class="empty-sub">Check back tomorrow for picks</div>
      </div>`;
    return;
  }

  let html = '<p class="section-label">Games & Picks</p>';
  games.forEach(g => {
    const recs = g.recommendations || [];
    const hasBet = recs.length > 0;
    totalBets += recs.length;
    recs.forEach(r => totalAction += (r.bet_amount || 0));

    const homeProb = Math.round((g.model_prob_home || 0.5) * 100);
    const awayProb = 100 - homeProb;
    const parts = g.game ? g.game.split(' @ ') : ['Away', 'Home'];
    const awayTeam = parts[0] || 'Away';
    const homeTeam = parts[1] || 'Home';

    html += `
    <div class="game-card ${hasBet ? 'has-bet' : ''}">
      <div class="game-header">
        <div class="game-teams">
          <div class="team-matchup">${awayTeam.split(' ').pop()} @ ${homeTeam.split(' ').pop()}</div>
          <div class="game-time">Today • NBA</div>
        </div>
        ${badgeHTML(recs)}
      </div>

      <div class="prob-row">
        <div class="prob-team">
          <div class="prob-name">${homeTeam.split(' ').slice(-2).join(' ')}</div>
          <div class="prob-bar-wrap"><div class="prob-bar home" style="width:${homeProb}%"></div></div>
        </div>
        <div class="vs-divider">VS</div>
        <div class="prob-team">
          <div class="prob-name">${awayTeam.split(' ').slice(-2).join(' ')}</div>
          <div class="prob-bar-wrap"><div class="prob-bar away" style="width:${awayProb}%"></div></div>
        </div>
        <div class="prob-pct" style="color:var(--accent2)">${homeProb}%</div>
      </div>

      ${hasBet ? `
      <div class="bet-recs">
        ${recs.map(r => {
          const isTotal = r.bet_type === 'TOTAL';
          const teamLabel = isTotal
            ? (r.side + ' ' + (g.predicted_total || ''))
            : r.team.split(' ').slice(-2).join(' ');
          const edgeVal = isTotal
            ? (r.edge_pts ? r.edge_pts.toFixed(1) + ' pts' : '—')
            : (r.edge ? (r.edge * 100).toFixed(1) + '%' : '—');
          const evVal = r.ev ? (r.ev * 100).toFixed(1) + '%' : '—';
          return `
          <div class="bet-rec ${r.confidence === 'HIGH' ? 'high' : ''}">
            <div class="bet-left">
              <div class="bet-type-label">${r.bet_type}</div>
              <div class="bet-team">${teamLabel}</div>
              <div class="bet-meta">Edge: ${edgeVal} · EV: ${evVal}</div>
            </div>
            <div class="bet-right">
              <div class="bet-amount">$${(r.bet_amount || 0).toFixed(0)}</div>
              <div class="bet-odds">${r.odds > 0 ? '+' : ''}${r.odds}</div>
              <div class="ev-chip">+EV</div>
            </div>
          </div>`;
        }).join('')}
      </div>` : `
      <div class="no-bet">
        <div class="no-bet-inner">
          <span style="font-size:16px">⛔</span>
          <div class="no-bet-text">No edge detected — skip this game</div>
        </div>
      </div>`}
    </div>`;
  });

  document.getElementById('picksContent').innerHTML = html;
  document.getElementById('todayAction').textContent = '$' + totalAction.toFixed(0);
  document.getElementById('betCount').textContent = totalBets;
  if (totalBets > 0) {
    document.getElementById('todayAction').style.color = 'var(--accent)';
  }
}

// ── Load Picks ───────────────────────────────────────────────
async function loadPicks() {
  const btn = document.getElementById('refreshBtn');
  btn.classList.add('spinning');
  document.getElementById('picksContent').innerHTML = `
    <div class="loading"><div class="loader"></div>
    <div class="loading-text">Analyzing today's games...</div></div>`;

  try {
    const res = await fetch('/api/picks');
    const data = await res.json();
    renderPicks(data);
    document.getElementById('bankrollVal').textContent =
      '$' + (data.bankroll || 1000).toLocaleString();
    showToast('✅ Updated ' + new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}));
  } catch(e) {
    document.getElementById('picksContent').innerHTML = `
      <div class="empty">
        <div class="empty-icon">⚠️</div>
        <div class="empty-title">Connection Error</div>
        <div class="empty-sub">Make sure the server is running</div>
      </div>`;
  } finally {
    btn.classList.remove('spinning');
  }
}

// ── Backtest ─────────────────────────────────────────────────
async function runBacktest() {
  document.getElementById('backtestResult').innerHTML =
    '<div class="loading" style="padding:30px 0"><div class="loader"></div></div>';
  try {
    const res = await fetch('/api/backtest');
    const d = await res.json();
    const roiColor = d.roi >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('statWinRate').textContent = (d.win_rate * 100).toFixed(1) + '%';
    document.getElementById('statROI').textContent = (d.roi >= 0 ? '+' : '') + (d.roi * 100).toFixed(1) + '%';
    document.getElementById('statTotalBets').textContent = d.total_bets;
    document.getElementById('statPnL').textContent =
      (d.total_pnl >= 0 ? '+$' : '-$') + Math.abs(d.total_pnl).toFixed(0);
    document.getElementById('statPnL').style.color = d.total_pnl >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('backtestResult').innerHTML =
      `<p style="text-align:center;font-size:13px;color:var(--green);padding:10px 0">✅ Backtest complete — see stats above</p>`;
    showToast('Backtest done!');
  } catch(e) {
    document.getElementById('backtestResult').innerHTML =
      '<p style="color:var(--muted);font-size:13px;text-align:center">Backtest failed — check server</p>';
  }
}

// ── Init ─────────────────────────────────────────────────────
loadPicks();
</script>
</body>
</html>
"""

# ── Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/picks")
def api_picks():
    if not MODEL_AVAILABLE or analyzer is None:
        # Return demo data if model not loaded
        return jsonify({
            "bankroll": config.bankroll,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "games": [
                {
                    "game": "Golden State Warriors @ Boston Celtics",
                    "model_prob_home": 0.63,
                    "model_prob_away": 0.37,
                    "book_prob_home": 0.58,
                    "book_prob_away": 0.42,
                    "predicted_total": 224.5,
                    "book_total": 218.5,
                    "recommendations": [
                        {
                            "bet_type": "MONEYLINE",
                            "team": "Boston Celtics",
                            "side": "HOME",
                            "odds": -155,
                            "model_prob": 0.63,
                            "book_prob": 0.58,
                            "edge": 0.05,
                            "ev": 0.06,
                            "kelly_pct": 0.032,
                            "bet_amount": 32.0,
                            "confidence": "HIGH",
                        },
                        {
                            "bet_type": "TOTAL",
                            "team": "GSW vs BOS",
                            "side": "OVER",
                            "odds": -110,
                            "model_total": 224.5,
                            "book_total": 218.5,
                            "edge_pts": 6.0,
                            "bet_amount": 15.0,
                            "confidence": "HIGH",
                        }
                    ],
                },
                {
                    "game": "Miami Heat @ Milwaukee Bucks",
                    "model_prob_home": 0.51,
                    "model_prob_away": 0.49,
                    "book_prob_home": 0.55,
                    "book_prob_away": 0.45,
                    "predicted_total": 216.0,
                    "book_total": 218.5,
                    "recommendations": [],
                }
            ],
        })

    results = analyzer.analyze_today()
    return jsonify({
        "bankroll": config.bankroll,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "games": results,
    })

@app.route("/api/backtest")
def api_backtest():
    bt = Backtester(config)
    sim = bt.run_simulation(500)
    bets = sim[sim["bet"] == True]
    if bets.empty:
        return jsonify({"error": "no bets"})
    wins = int(bets["won"].sum())
    total = len(bets)
    pnl = float(bets["pnl"].sum())
    wagered = float(bets["bet_size"].sum())
    return jsonify({
        "win_rate": round(wins / total, 4),
        "roi": round(pnl / wagered, 4),
        "total_bets": total,
        "wins": wins,
        "total_pnl": round(pnl, 2),
        "total_wagered": round(wagered, 2),
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_AVAILABLE})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
