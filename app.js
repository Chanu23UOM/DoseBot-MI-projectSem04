'use strict';
/* =========================================================
   DoseBot — app.js
   Firebase · Chart.js · Scroll animations · All UI logic
   EE2044 Group 29 · University of Moratuwa
   Firebase path: /dosebot/sensors
   ========================================================= */

// ===== FIREBASE CONFIG =====
const firebaseConfig = {
  databaseURL: 'https://dosebot-g29-default-rtdb.asia-southeast1.firebasedatabase.app'
};

firebase.initializeApp(firebaseConfig);
const db = firebase.database();

// ===== CONSTANTS =====
const MAX_POINTS        = 30;
const TEMP_THRESHOLD    = 28;
const CONN_TIMEOUT_MS   = 5000;
const DISPENSE_COOLDOWN = 12000;

// ===== STATE =====
const state = {
  connected:       false,
  latestData:      null,
  tempHistory:     [],
  humHistory:      [],
  dispenses:       loadLS('db2_dispenses', []),
  lastDispenseTime: 0,
  lastDataTime:    0,
  connCheckTimer:  null,
  simInterval:     null,
  tempChart:       null,
  currentRating:   0,
};

// ===== LOCAL STORAGE =====
function loadLS(key, def) {
  try { return JSON.parse(localStorage.getItem(key)) || def; }
  catch { return def; }
}
function saveLS(key, val) {
  try { localStorage.setItem(key, JSON.stringify(val)); } catch (_) {}
}

// ===== TIMESTAMP =====
function fmtTimestamp() {
  const d = new Date();
  return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
    + ' at ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtTime() {
  return new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ===== SMOOTH NUMBER ANIMATION (RAF) =====
function animateNumber(el, target, decimals, duration) {
  if (!el) return;
  duration = duration || 600;
  const start    = parseFloat(el.textContent.replace(/[^\d.-]/g, '')) || 0;
  const diff     = target - start;
  const startTime = performance.now();
  const prev = el._raf;
  if (prev) cancelAnimationFrame(prev);

  function step(now) {
    const p = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);   // easeOutCubic
    el.textContent = (start + diff * eased).toFixed(decimals);
    if (p < 1) el._raf = requestAnimationFrame(step);
  }
  el._raf = requestAnimationFrame(step);
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min(100, Math.max(0, pct)) + '%';
}

// ===== CONNECTION STATUS =====
function setConnectionStatus(status) {
  // status: 'connecting' | 'live' | 'offline'
  state.connected = status === 'live';
  const dot  = document.getElementById('navConnDot');
  const text = document.getElementById('navConnText');
  if (!dot || !text) return;

  dot.className  = 'nav-conn-dot ' + (status === 'live' ? 'live' : status === 'offline' ? 'offline' : '');
  text.textContent = status === 'live' ? 'Live' : status === 'offline' ? 'Offline' : 'Connecting…';
}

function startConnectionWatchdog() {
  if (state.connCheckTimer) clearInterval(state.connCheckTimer);
  state.connCheckTimer = setInterval(() => {
    if (state.lastDataTime > 0 && Date.now() - state.lastDataTime > CONN_TIMEOUT_MS) {
      setConnectionStatus('offline');
    }
  }, 1000);
}

// ===== SYSTEM STATE BADGE =====
const STATES = {
  idle:     { cls: 'idle',     text: '⬤  IDLE — Place bottle to begin' },
  cooling:  { cls: 'cooling',  text: '🌡  COOLING — Temperature high' },
  detected: { cls: 'detected', text: '📦  BOTTLE DETECTED — Checking…' },
  ready:    { cls: 'ready',    text: '✅  READY TO DISPENSE' },
};

function updateSystemState(d) {
  let key;
  if (!d || d.bottle !== 1)              key = 'idle';
  else if (d.temp > TEMP_THRESHOLD)      key = 'cooling';
  else if (d.ready !== 1)               key = 'detected';
  else                                   key = 'ready';

  const s = STATES[key];
  ['heroStateBadge', 'dashStateBadge'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = 'system-state-badge ' + s.cls;
    const textEl = el.querySelector('.state-text');
    if (textEl) textEl.textContent = s.text;
    else el.lastElementChild.textContent = s.text;
  });
  ['heroStateText', 'dashStateText'].forEach(id => setText(id, s.text));
}

// ===== METRIC CARDS =====
function updateMetricCards(d) {
  // Temperature with color coding
  const tempEl = document.getElementById('mTempVal');
  animateNumber(tempEl, d.temp, 1, 600);

  const mCardTemp = document.getElementById('mcard-temp');
  const tempAlert = document.getElementById('tempAlert');
  if (mCardTemp) {
    mCardTemp.classList.remove('temp-safe', 'temp-warn', 'temp-danger');
    if (d.temp < 25)       { mCardTemp.classList.add('temp-safe');   if(tempAlert) tempAlert.hidden = true; }
    else if (d.temp <= 28) { mCardTemp.classList.add('temp-warn');   if(tempAlert) tempAlert.hidden = true; }
    else                   { mCardTemp.classList.add('temp-danger'); if(tempAlert) tempAlert.hidden = false; }
  }

  animateNumber(document.getElementById('mHumVal'),    d.humidity, 0, 600);
  animateNumber(document.getElementById('mWeightVal'), d.weight,   1, 600);
  animateNumber(document.getElementById('mCountVal'),  d.count,    0, 600);

  setBar('mTempBar',   (d.temp / 50) * 100);
  setBar('mHumBar',    d.humidity);
  setBar('mWeightBar', (d.weight / 500) * 100);
  setBar('mCountBar',  (d.count / 50) * 100);

  // Pulse all cards
  document.querySelectorAll('.metric-card').forEach(c => {
    c.classList.remove('data-pulse');
    void c.offsetWidth;           // force reflow
    c.classList.add('data-pulse');
  });
}

// ===== LED INDICATORS =====
function setLed(dotId, valId, state, label) {
  const dot = document.getElementById(dotId);
  const val = document.getElementById(valId);
  if (dot) dot.className = 'led-dot' + (state ? ' ' + state : '');
  if (val) val.textContent = label;
}

function updateLEDs(d) {
  const bottleOk = d.bottle === 1;
  const tempSafe  = d.temp < TEMP_THRESHOLD;
  const ready     = d.ready === 1;
  const fanOn     = d.temp > 26 || d.humidity > 70;

  setLed('led-bottle-dot', 'led-bottle-val', bottleOk ? 'led-on' : '',     bottleOk ? 'Detected ✓' : 'Not detected');
  setLed('led-temp-dot',   'led-temp-val',   tempSafe ? 'led-on' : 'led-err', tempSafe ? `${d.temp.toFixed(1)} °C — Safe` : `${d.temp.toFixed(1)} °C — High!`);
  setLed('led-ready-dot',  'led-ready-val',  ready    ? 'led-on' : '',     ready    ? 'Ready ✓' : 'Not ready');
  setLed('led-fan-dot',    'led-fan-val',    fanOn    ? 'led-on' : '',     fanOn    ? 'Running' : 'Idle');
}

// ===== TEMPERATURE CHART =====
function initTempChart() {
  const ctx = document.getElementById('tempChart');
  if (!ctx) return;

  state.tempChart = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Temperature (°C)',
          data: [],
          borderColor: '#0d9488',
          backgroundColor: 'rgba(13,148,136,0.07)',
          borderWidth: 2.5,
          pointRadius: 3,
          pointBackgroundColor: '#0d9488',
          pointBorderColor: 'white',
          pointBorderWidth: 1.5,
          fill: true,
          tension: 0.42,
        },
        {
          label: '28°C Threshold',
          data: [],
          borderColor: '#ef4444',
          borderWidth: 2,
          borderDash: [6, 5],
          pointRadius: 0,
          fill: false,
          tension: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.85)',
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(1)}`
          }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { maxTicksLimit: 6, font: { size: 11 }, color: '#94a3b8' } },
        y: {
          suggestedMin: 15, suggestedMax: 36,
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { font: { size: 11 }, color: '#94a3b8', callback: v => v + '°C' }
        }
      },
      animation: { duration: 300 }
    }
  });
}

function pushTempChart(temp) {
  if (!state.tempChart) return;
  const chart = state.tempChart;
  chart.data.labels.push(fmtTime());
  chart.data.datasets[0].data.push(temp);
  chart.data.datasets[1].data.push(TEMP_THRESHOLD);
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(ds => ds.data.shift());
  }
  chart.update('quiet');
}

// ===== DISPENSE LOG =====
function checkAutoDispense(d) {
  if (d.ready === 1 && d.bottle === 1 && Date.now() - state.lastDispenseTime > DISPENSE_COOLDOWN) {
    state.lastDispenseTime = Date.now();
    addDispenseEntry(d.count);
  }
}

function addDispenseEntry(count) {
  const entry = { id: Date.now(), time: fmtTimestamp(), count };
  state.dispenses.unshift(entry);
  if (state.dispenses.length > 10) state.dispenses.pop();
  saveLS('db2_dispenses', state.dispenses);
  renderDispenseTable();
}

function renderDispenseTable() {
  const tbody = document.getElementById('dispenseTbody');
  const badge = document.getElementById('dispenseCountBadge');
  if (!tbody) return;
  if (badge) badge.textContent = state.dispenses.length + ' events';
  if (!state.dispenses.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-td">No events yet — waiting for data…</td></tr>';
    return;
  }
  tbody.innerHTML = state.dispenses.map((r, i) => `
    <tr>
      <td style="color:var(--text-muted);font-size:11px">${i + 1}</td>
      <td style="font-size:11.5px">${r.time}</td>
      <td>DoseBot Auto</td>
      <td><span class="badge-pill">${r.count}</span></td>
    </tr>`).join('');
}

// ===== CSV EXPORT =====
function exportCSV() {
  const rows = [['#', 'Timestamp', 'Medicine', 'Pills Left'],
    ...state.dispenses.map((r, i) => [i + 1, r.time, 'DoseBot Auto', r.count])];
  const csv  = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `dosebot-dispense-log-${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ===== MAIN DATA PROCESSOR =====
function processData(d) {
  state.latestData    = d;
  state.lastDataTime  = Date.now();

  state.tempHistory.push(d.temp);
  state.humHistory.push(d.humidity);
  if (state.tempHistory.length > MAX_POINTS) state.tempHistory.shift();
  if (state.humHistory.length  > MAX_POINTS) state.humHistory.shift();

  updateMetricCards(d);
  updateLEDs(d);
  updateSystemState(d);
  pushTempChart(d.temp);
  checkAutoDispense(d);

  setText('lastSync', 'Last synced: ' + fmtTime());
  setConnectionStatus('live');
  hideLoadingSpinner();
}

// ===== FIREBASE LISTENER =====
function initFirebase() {
  // Merge data from multiple Firebase top-level paths
  const merged = { temp: 0, humidity: 0, voltage: 0, bottle: 0, ready: 0, count: 0, weight: 0 };

  // /Sensors → temp, humidity, voltage, bottle
  db.ref('/Sensors').on('value',
    snap => {
      const s = snap.val();
      if (!s || typeof s !== 'object') return;
      merged.temp     = typeof s.temp === 'number' ? s.temp : 0;
      merged.humidity  = typeof s.humidity === 'number' ? s.humidity : 0;
      merged.voltage   = typeof s.voltage === 'number' ? s.voltage : 0;
      merged.bottle    = s.bottle === 1 || s.bottle === true ? 1 : 0;
      processData({ ...merged });
    },
    err => { console.error('[DoseBot] Sensors:', err.message); setConnectionStatus('offline'); }
  );

  // /status → bottle_present, ready, temp_safe
  db.ref('/status').on('value',
    snap => {
      const s = snap.val();
      if (!s || typeof s !== 'object') return;
      const bottlePresent = s.bottle_present === 'true' || s.bottle_present === true;
      merged.ready  = (s.ready === 'true' || s.ready === true) ? 1 : 0;
      if (merged.bottle === 0 && bottlePresent) merged.bottle = 1;
      processData({ ...merged });
    },
    err => { console.error('[DoseBot] Status:', err.message); }
  );

  // /counters → pill_count, last_update
  db.ref('/counters').on('value',
    snap => {
      const c = snap.val();
      if (!c || typeof c !== 'object') return;
      merged.count = typeof c.pill_count === 'number' ? c.pill_count : 0;
      processData({ ...merged });
    },
    err => { console.error('[DoseBot] Counters:', err.message); }
  );

  db.ref('.info/connected').on('value', snap => {
    if (!snap.val()) setConnectionStatus(state.lastDataTime > 0 ? 'offline' : 'connecting');
  });
  startConnectionWatchdog();
}

// ===== SIMULATE =====
function genSim() {
  return {
    temp:     +(21 + Math.random() * 12).toFixed(2),
    humidity: +(45 + Math.random() * 40).toFixed(2),
    weight:   +(80  + Math.random() * 250).toFixed(2),
    voltage:  +(1.8 + Math.random() * 0.5).toFixed(3),
    bottle:   Math.random() > 0.25 ? 1 : 0,
    ready:    Math.random() > 0.35 ? 1 : 0,
    count:    Math.floor(Math.random() * 48),
  };
}

function initSimulate() {
  const btn = document.getElementById('simulateBtn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    if (state.simInterval) {
      clearInterval(state.simInterval);
      state.simInterval = null;
      btn.classList.remove('sim-active');
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Simulate`;
    } else {
      processData(genSim());
      state.simInterval = setInterval(() => processData(genSim()), 1000);
      btn.classList.add('sim-active');
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop Sim`;
    }
  });
}

// ===== LOADING SPINNER =====
function hideLoadingSpinner() {
  const el = document.getElementById('loadingSpinner');
  if (el) el.classList.add('hidden');
}

// Auto-hide after 8s even if Firebase never responds
setTimeout(hideLoadingSpinner, 8000);

// ===== NAVBAR SCROLL BEHAVIOR =====
function initNavScroll() {
  const nav = document.getElementById('mainNav');
  if (!nav) return;
  function update() {
    nav.classList.toggle('scrolled', window.scrollY > 30);
  }
  window.addEventListener('scroll', update, { passive: true });
  update();
}

// ===== BACK TO TOP =====
function initBackToTop() {
  const btn = document.getElementById('backToTop');
  if (!btn) return;
  window.addEventListener('scroll', () => {
    btn.classList.toggle('visible', window.scrollY > 500);
  }, { passive: true });
  btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
}

// ===== HAMBURGER MENU =====
function initHamburger() {
  const btn   = document.getElementById('hamburgerBtn');
  const links = document.getElementById('navLinks');
  if (!btn || !links) return;

  btn.addEventListener('click', () => {
    const open = links.classList.toggle('open');
    btn.classList.toggle('open', open);
    btn.setAttribute('aria-expanded', open);
  });

  // Close on nav-link click
  links.querySelectorAll('.nav-link').forEach(a => {
    a.addEventListener('click', () => {
      links.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
    });
  });
}

// ===== SCROLLSPY (active nav link) =====
function initScrollSpy() {
  const sections = ['home','problem','solution','technology','dashboard','team'];
  const links    = document.querySelectorAll('.nav-link');

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => {
          l.classList.toggle('active', l.getAttribute('href') === '#' + entry.target.id);
        });
      }
    });
  }, { threshold: 0.35 });

  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) obs.observe(el);
  });
}

// ===== SCROLL REVEAL (IntersectionObserver) =====
function initScrollReveal() {
  const els = document.querySelectorAll('.reveal-item, .reveal-left, .reveal-right, .reveal-up');

  // Add stagger classes to grids
  document.querySelectorAll('.stat-grid .stat-card, .feature-grid .feature-card, .tech-grid .tech-item').forEach((el, i) => {
    el.classList.add(`stagger-${Math.min(i + 1, 6)}`);
  });

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  els.forEach(el => obs.observe(el));
}

// ===== ANIMATED STAT COUNTERS =====
function initStatCounters() {
  const cards = document.querySelectorAll('.stat-card[data-target]');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      obs.unobserve(entry.target);
      const card    = entry.target;
      const target  = parseFloat(card.dataset.target);
      const decimals = parseInt(card.dataset.decimal || '0');
      const numEl   = card.querySelector('.stat-num');
      if (!numEl) return;

      const start = performance.now();
      const dur   = 2000;

      function tick(now) {
        const p = Math.min((now - start) / dur, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        numEl.textContent = (target * eased).toFixed(decimals);
        if (p < 1) requestAnimationFrame(tick);
        else numEl.textContent = target.toFixed(decimals);
      }
      requestAnimationFrame(tick);
    });
  }, { threshold: 0.5 });

  cards.forEach(c => obs.observe(c));
}

// ===== MAGNETIC HOVER (Request Demo button) =====
function initMagneticBtn() {
  const btn = document.getElementById('requestDemoBtn');
  if (!btn) return;
  btn.addEventListener('mousemove', e => {
    const r = btn.getBoundingClientRect();
    const x = (e.clientX - r.left - r.width  / 2) * 0.2;
    const y = (e.clientY - r.top  - r.height / 2) * 0.2;
    btn.style.transform = `translate(${x}px, ${y}px) scale(1.02)`;
  });
  btn.addEventListener('mouseleave', () => { btn.style.transform = ''; });
  btn.addEventListener('click', () => {
    const dash = document.getElementById('dashboard');
    if (dash) dash.scrollIntoView({ behavior: 'smooth' });
  });
}

// ===== CSV EXPORT BUTTON =====
function initCsvBtn() {
  const btn = document.getElementById('csvExportBtn');
  if (btn) btn.addEventListener('click', exportCSV);
}

// ===== BOOT =====
document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initBackToTop();
  initHamburger();
  initScrollSpy();
  initScrollReveal();
  initStatCounters();
  initMagneticBtn();
  initTempChart();
  initSimulate();
  initCsvBtn();
  initFirebase();
  renderDispenseTable();

  // Set initial state badge text
  updateSystemState(null);
  setConnectionStatus('connecting');
});
