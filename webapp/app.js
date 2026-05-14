'use strict';
/* =========================================================
   DoseBot Web App — app.js
   Auth guard · Live dashboard · Prescription · Chatbot
   ---------------------------------------------------------
   CHATBOT: set CHATBOT_API_URL to your API endpoint.
   Expected request:  POST { messages: [{role, content}] }
   Expected response: { reply: "..." }
   Leave empty to use the built-in placeholder responses.
   ========================================================= */

const CHATBOT_API_URL = ''; // e.g. 'https://your-api.example.com/chat'

// ===== FIREBASE CONFIG (same as auth.js) =====
const FIREBASE_CONFIG = {
  apiKey:            "AIzaSyAY5EOTQQ-RDYlXqmFEZ3VhIKpXlDv3es8",
  authDomain:        "dosebot-g29.firebaseapp.com",
  databaseURL:       "https://dosebot-g29-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId:         "dosebot-g29",
  storageBucket:     "dosebot-g29.firebasestorage.app",
  messagingSenderId: "214133470498",
  appId:             "1:214133470498:web:caf8d149f9d89e3cb3e17b",
  measurementId:     "G-MJV8KET5QT"
};

firebase.initializeApp(FIREBASE_CONFIG);
const auth = firebase.auth();
const db   = firebase.database();

// ===== CONSTANTS =====
const MAX_POINTS        = 30;
const TEMP_THRESHOLD    = 28;
const CONN_TIMEOUT_MS   = 5000;
const DISPENSE_COOLDOWN = 12000;

// ===== STATE =====
const state = {
  user:             null,
  connected:        false,
  latestData:       null,
  tempHistory:      [],
  dispenses:        loadLS('db3_dispenses', []),
  lastDispenseTime: 0,
  lastDataTime:     0,
  connTimer:        null,
  simInterval:      null,
  tempChart:        null,
  chatHistory:      [],
  chatWaiting:      false,
};

// ===== UTILS =====
function loadLS(k, d) { try { return JSON.parse(localStorage.getItem(k)) || d; } catch { return d; } }
function saveLS(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch(_) {} }

function fmtTime() { return new Date().toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit', second:'2-digit' }); }
function fmtDate(ts) {
  return new Date(ts).toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' })
    + ' ' + new Date(ts).toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit', second:'2-digit' });
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

function animateNumber(el, target, dec, dur = 600) {
  if (!el) return;
  const start = parseFloat(el.textContent.replace(/[^\d.-]/g,'')) || 0;
  const diff  = target - start;
  const t0    = performance.now();
  if (el._raf) cancelAnimationFrame(el._raf);
  function step(now) {
    const p = Math.min((now - t0) / dur, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = (start + diff * e).toFixed(dec);
    if (p < 1) el._raf = requestAnimationFrame(step);
  }
  el._raf = requestAnimationFrame(step);
}
function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min(100, Math.max(0, pct)) + '%';
}

// ===== TOAST =====
function toast(msg, type = 'info') {
  const wrap = document.getElementById('toastWrap');
  if (!wrap) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => { t.classList.add('fade-out'); t.addEventListener('animationend', () => t.remove()); }, 3500);
}

// ===== AUTH GUARD =====
auth.onAuthStateChanged(user => {
  const loader = document.getElementById('appLoader');
  if (!user) {
    window.location.replace('index.html');
    return;
  }
  state.user = user;
  if (loader) loader.classList.add('hidden');
  bootApp();
});

// ===== BOOT =====
function bootApp() {
  initSidebar();
  initTopbar();
  initFirebase();
  initSimulate();
  initTempChart();
  initPrescriptionForm();
  initChatbot();
  initCsvBtns();
  renderDispenseTable();
  loadUserProfile();
  if (!CHATBOT_API_URL) document.getElementById('chatApiNote')?.removeAttribute('hidden');
}

// ===== SIDEBAR NAVIGATION =====
const SECTION_TITLES = {
  'dashboard':    'Dashboard',
  'dispense-log': 'Dispense Log',
  'prescription': 'Prescription',
  'chatbot':      'AI Chatbot',
  'profile':      'Profile',
};

function initSidebar() {
  const navItems = document.querySelectorAll('.nav-item[data-section]');
  navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      navItems.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const sec = btn.dataset.section;
      document.querySelectorAll('.section-view').forEach(v => v.classList.remove('active'));
      document.getElementById(`view-${sec}`)?.classList.add('active');
      setText('topbarTitle', SECTION_TITLES[sec] || 'DoseBot');
      closeSidebar();
    });
  });

  // Logout buttons
  document.getElementById('logoutBtn')?.addEventListener('click',    doLogout);
  document.getElementById('profileLogout')?.addEventListener('click', doLogout);

  // Mobile hamburger
  document.getElementById('menuBtn')?.addEventListener('click', () => {
    document.getElementById('sidebar')?.classList.toggle('open');
    document.getElementById('sidebarOverlay')?.classList.toggle('visible');
  });
  document.getElementById('sidebarOverlay')?.addEventListener('click', closeSidebar);
}

function closeSidebar() {
  document.getElementById('sidebar')?.classList.remove('open');
  document.getElementById('sidebarOverlay')?.classList.remove('visible');
}

function doLogout() {
  auth.signOut().then(() => window.location.replace('index.html'));
}

// ===== TOPBAR USER INFO =====
function initTopbar() {
  const user = state.user;
  const name  = user.displayName || user.email?.split('@')[0] || 'User';
  const initials = name.split(' ').map(w => w[0]).join('').substring(0,2).toUpperCase();
  setText('userName',  name);
  setText('userAvatar', initials);
  document.getElementById('userBtn')?.addEventListener('click', () => {
    document.querySelector('.nav-item[data-section="profile"]')?.click();
  });
}

// ===== CONNECTION STATUS =====
function setConnStatus(status) {
  state.connected = status === 'live';
  const dot  = document.getElementById('connDot');
  const text = document.getElementById('connText');
  if (dot)  { dot.className = 'conn-dot' + (status === 'live' ? ' live' : status === 'offline' ? ' offline' : ''); }
  if (text) { text.textContent = status === 'live' ? 'Live' : status === 'offline' ? 'Offline' : 'Connecting…'; }
}

function startConnWatchdog() {
  if (state.connTimer) clearInterval(state.connTimer);
  state.connTimer = setInterval(() => {
    if (state.lastDataTime > 0 && Date.now() - state.lastDataTime > CONN_TIMEOUT_MS) setConnStatus('offline');
  }, 1000);
}

// ===== SYSTEM STATE BADGE =====
const BADGE_STATES = {
  idle:     { cls:'idle',     text:'⬤  IDLE — Place bottle to begin' },
  cooling:  { cls:'cooling',  text:'🌡  COOLING — Temperature high' },
  detected: { cls:'detected', text:'📦  BOTTLE DETECTED — Checking…' },
  ready:    { cls:'ready',    text:'✅  READY TO DISPENSE' },
};

function updateSystemBadge(d) {
  let key = !d || d.bottle !== 1 ? 'idle'
           : d.temp > TEMP_THRESHOLD ? 'cooling'
           : d.ready !== 1 ? 'detected'
           : 'ready';
  const s = BADGE_STATES[key];
  ['systemBadge','rxSystemBadge'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = `system-badge ${s.cls}`;
    const sp = el.querySelector('span:last-child');
    if (sp) sp.textContent = s.text;
  });
  setText('rxSystemBadgeText', s.text);
  setText('systemBadgeText',   s.text);
}

// ===== METRIC CARDS =====
function updateMetricCards(d) {
  animateNumber(document.getElementById('mTempVal'),   d.temp,     1);
  animateNumber(document.getElementById('mHumVal'),    d.humidity, 0);
  animateNumber(document.getElementById('mWeightVal'), d.weight,   1);
  animateNumber(document.getElementById('mCountVal'),  d.count,    0);

  setBar('mTempBar',   (d.temp / 50) * 100);
  setBar('mHumBar',    d.humidity);
  setBar('mWeightBar', (d.weight / 500) * 100);
  setBar('mCountBar',  (d.count / 50) * 100);

  const card  = document.getElementById('mcard-temp');
  const alert = document.getElementById('tempAlert');
  if (card) {
    card.classList.remove('temp-safe','temp-warn','temp-danger');
    if      (d.temp < 25)  { card.classList.add('temp-safe');   if (alert) alert.hidden = true; }
    else if (d.temp <= 28) { card.classList.add('temp-warn');   if (alert) alert.hidden = true; }
    else                   { card.classList.add('temp-danger'); if (alert) alert.hidden = false; }
  }

  document.querySelectorAll('.metric-card').forEach(c => {
    c.classList.remove('data-pulse'); void c.offsetWidth; c.classList.add('data-pulse');
  });
}

// ===== LED INDICATORS =====
function setLed(dotId, valId, cls, label) {
  const dot = document.getElementById(dotId);
  const val = document.getElementById(valId);
  if (dot) dot.className = 'led-dot' + (cls ? ' ' + cls : '');
  if (val) val.textContent = label;
}

function updateLEDs(d) {
  const bottleOk = d.bottle === 1;
  const tempSafe  = d.temp < TEMP_THRESHOLD;
  const ready     = d.ready === 1;
  const fanOn     = d.temp > 26 || d.humidity > 70;

  setLed('led-bottle-dot','led-bottle-val', bottleOk ? 'on' : '',     bottleOk ? 'Detected ✓' : 'Not detected');
  setLed('led-temp-dot',  'led-temp-val',   tempSafe ? 'on' : 'err',  tempSafe ? `${d.temp.toFixed(1)}°C — Safe` : `${d.temp.toFixed(1)}°C — High!`);
  setLed('led-ready-dot', 'led-ready-val',  ready    ? 'on' : '',     ready    ? 'Ready ✓' : 'Not ready');
  setLed('led-fan-dot',   'led-fan-val',    fanOn    ? 'on' : '',     fanOn    ? 'Running' : 'Idle');

  // Prescription panel mirror
  setText('rxBottleStatus', bottleOk ? '✓ Detected' : '✗ Not detected');
  setText('rxTempStatus',   `${d.temp.toFixed(1)}°C — ${tempSafe ? 'Safe' : 'High!'}`);
  setText('rxReadyStatus',  ready ? '✓ Ready' : 'Not ready');
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
        { label:'Temperature (°C)', data:[], borderColor:'#0d9488', backgroundColor:'rgba(13,148,136,0.07)',
          borderWidth:2.5, pointRadius:3, pointBackgroundColor:'#0d9488', pointBorderColor:'rgba(11,17,32,1)',
          pointBorderWidth:1.5, fill:true, tension:0.42 },
        { label:'28°C Threshold', data:[], borderColor:'#ef4444', borderWidth:2,
          borderDash:[6,5], pointRadius:0, fill:false, tension:0 }
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      interaction:{ mode:'index', intersect:false },
      plugins:{
        legend:{ display:false },
        tooltip:{
          backgroundColor:'rgba(19,29,46,0.95)', padding:10, cornerRadius:8,
          callbacks:{ label: c => ` ${c.dataset.label}: ${Number(c.parsed.y).toFixed(1)}` }
        }
      },
      scales:{
        x:{ grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ maxTicksLimit:6, font:{size:10}, color:'#4e6077' } },
        y:{ suggestedMin:15, suggestedMax:36,
            grid:{ color:'rgba(255,255,255,0.04)' },
            ticks:{ font:{size:10}, color:'#4e6077', callback: v => v+'°C' } }
      },
      animation:{ duration:300 }
    }
  });
}

function pushTempChart(temp) {
  if (!state.tempChart) return;
  const c = state.tempChart;
  c.data.labels.push(fmtTime());
  c.data.datasets[0].data.push(temp);
  c.data.datasets[1].data.push(TEMP_THRESHOLD);
  if (c.data.labels.length > MAX_POINTS) {
    c.data.labels.shift();
    c.data.datasets.forEach(ds => ds.data.shift());
  }
  c.update('quiet');
}

// ===== DISPENSE LOG =====
function checkAutoDispense(d) {
  if (d.ready === 1 && d.bottle === 1 && Date.now() - state.lastDispenseTime > DISPENSE_COOLDOWN) {
    state.lastDispenseTime = Date.now();
    addDispenseEntry({ count: d.count });
  }
}

function addDispenseEntry(rx) {
  const entry = {
    id:       Date.now(),
    time:     fmtDate(Date.now()),
    patient:  rx.patient  || 'Auto',
    medicine: rx.medicine || 'DoseBot Auto',
    doctor:   rx.doctor   || '—',
    count:    rx.count    || 0,
  };
  state.dispenses.unshift(entry);
  if (state.dispenses.length > 50) state.dispenses.pop();
  saveLS('db3_dispenses', state.dispenses);
  renderDispenseTable();
}

function renderDispenseTable() {
  const tbody = document.getElementById('dispenseTbody');
  const badge = document.getElementById('dispenseCountBadge');
  const logTbody = document.getElementById('logTbody');
  const logBadge = document.getElementById('logCountBadge');

  if (badge) badge.textContent = state.dispenses.length + ' events';
  if (logBadge) logBadge.textContent = state.dispenses.length + ' events';

  const empty = '<tr><td colspan="6" class="empty-td">No dispense events yet…</td></tr>';
  if (!state.dispenses.length) {
    if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="empty-td">No events yet…</td></tr>';
    if (logTbody) logTbody.innerHTML = empty;
    return;
  }

  const dashRows = state.dispenses.slice(0,5).map((r, i) => `
    <tr>
      <td style="color:var(--text-muted);font-size:11px">${i+1}</td>
      <td style="font-size:11.5px">${r.time}</td>
      <td>${r.patient}</td>
      <td><span class="badge-pill">${r.count}</span></td>
    </tr>`).join('');
  if (tbody) tbody.innerHTML = dashRows;

  const logRows = state.dispenses.map((r, i) => `
    <tr>
      <td style="color:var(--text-muted)">${i+1}</td>
      <td>${r.time}</td>
      <td>${r.patient}</td>
      <td>${r.medicine}</td>
      <td><span class="badge-pill">${r.count}</span></td>
      <td style="color:var(--text-muted)">${r.doctor}</td>
    </tr>`).join('');
  if (logTbody) logTbody.innerHTML = logRows;
}

// ===== CSV EXPORT =====
function exportCSV() {
  const rows = [['#','Timestamp','Patient','Medicine','Pills','Doctor'],
    ...state.dispenses.map((r,i) => [i+1, r.time, r.patient, r.medicine, r.count, r.doctor])];
  const csv  = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `dosebot-log-${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

function initCsvBtns() {
  document.getElementById('csvBtnDash')?.addEventListener('click', exportCSV);
  document.getElementById('csvBtnLog')?.addEventListener('click',  exportCSV);
}

// ===== MAIN DATA PROCESSOR =====
function processData(d) {
  state.latestData   = d;
  state.lastDataTime = Date.now();
  updateMetricCards(d);
  updateLEDs(d);
  updateSystemBadge(d);
  pushTempChart(d.temp);
  checkAutoDispense(d);
  setText('lastSync', 'Last sync: ' + fmtTime());
  setConnStatus('live');
}

// ===== FIREBASE RTDB LISTENER =====
function initFirebase() {
  db.ref('/dosebot/Sensors').on('value',
    snap => { const d = snap.val(); if (d && typeof d === 'object') processData(d); },
    err  => { console.error('[DoseBot]', err.message); setConnStatus('offline'); }
  );
  db.ref('.info/connected').on('value', snap => {
    if (!snap.val()) setConnStatus(state.lastDataTime > 0 ? 'offline' : 'connecting');
  });

  // Listen to current prescription in Firebase
  db.ref('/dosebot/prescription').on('value', snap => {
    const rx = snap.val();
    if (rx) {
      setText('curRxPatient',  rx.patient  || '—');
      setText('curRxMedicine', rx.medicine || '—');
      setText('curRxCount',    rx.count !== undefined ? `${rx.count} pills` : '—');
      setText('curRxDoctor',   rx.doctor   || '—');
    }
  });

  startConnWatchdog();
}

// ===== SIMULATION =====
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
  const btn = document.getElementById('simBtn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    if (state.simInterval) {
      clearInterval(state.simInterval); state.simInterval = null;
      btn.classList.remove('active');
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Simulate`;
    } else {
      processData(genSim());
      state.simInterval = setInterval(() => processData(genSim()), 1000);
      btn.classList.add('active');
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop`;
    }
  });
}

// ===== PRESCRIPTION FORM =====
function initPrescriptionForm() {
  const countInput = document.getElementById('rxCount');
  document.getElementById('rxCountMinus')?.addEventListener('click', () => {
    if (countInput) countInput.value = Math.max(1, (parseInt(countInput.value)||1) - 1);
  });
  document.getElementById('rxCountPlus')?.addEventListener('click', () => {
    if (countInput) countInput.value = Math.min(50, (parseInt(countInput.value)||1) + 1);
  });

  document.getElementById('rxSubmitBtn')?.addEventListener('click', async () => {
    const patient  = document.getElementById('rxPatient')?.value.trim();
    const doctor   = document.getElementById('rxDoctor')?.value.trim();
    const medicine = document.getElementById('rxMedicine')?.value.trim();
    const count    = parseInt(document.getElementById('rxCount')?.value) || 1;
    const notes    = document.getElementById('rxNotes')?.value.trim();

    if (!patient)  { toast('Patient name is required.',  'error'); return; }
    if (!doctor)   { toast('Doctor name is required.',   'error'); return; }
    if (!medicine) { toast('Medication name is required.','error'); return; }

    const btn = document.getElementById('rxSubmitBtn');
    btn.disabled = true;
    btn.textContent = 'Submitting…';

    try {
      const rxData = {
        patient, doctor, medicine, count, notes,
        requestedAt: Date.now(),
        requestedBy: state.user?.uid || 'unknown',
      };
      await db.ref('/dosebot/prescription').set(rxData);
      addDispenseEntry(rxData);
      toast('Prescription submitted! The kiosk will dispense shortly.', 'success');
      // Clear form
      ['rxPatient','rxDoctor','rxMedicine','rxNotes'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
      });
      if (countInput) countInput.value = '1';
    } catch (err) {
      toast('Failed to submit: ' + err.message, 'error');
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg> Submit Prescription`;
    }
  });
}

// ===== USER PROFILE =====
function loadUserProfile() {
  const user = state.user;
  const name = user.displayName || user.email?.split('@')[0] || 'User';
  const initials = name.split(' ').map(w=>w[0]).join('').substring(0,2).toUpperCase();

  setText('profileName',  name);
  setText('profileEmail', user.email || '');
  setText('profileAvatar', initials);
  setText('infoName',     name);
  setText('infoEmail',    user.email || '—');
  setText('infoProvider', user.providerData[0]?.providerId === 'google.com' ? 'Google' : 'Email / Password');

  // Load extra profile from RTDB
  db.ref(`/dosebot/users/${user.uid}`).once('value').then(snap => {
    const p = snap.val();
    if (p) {
      setText('infoName',  p.name  || name);
      setText('infoPhone', p.phone || '—');
      setText('infoJoined', p.registeredAt ? fmtDate(p.registeredAt) : '—');
      if (p.name) {
        const i = p.name.split(' ').map(w=>w[0]).join('').substring(0,2).toUpperCase();
        setText('profileAvatar', i);
        setText('userAvatar',    i);
        setText('profileName',   p.name);
        setText('userName',      p.name.split(' ')[0]);
      }
    }
  });
}

// ===== AI CHATBOT =====
const PLACEHOLDER_RESPONSES = [
  "I can help you with medication questions once the API is configured. Please set CHATBOT_API_URL in app.js.",
  "Great question! Connect your AI API to get intelligent responses about medications and health.",
  "To enable full AI responses, add your chatbot endpoint to the CHATBOT_API_URL constant in app.js.",
  "I'm the DoseBot assistant in demo mode. With your API connected, I can answer medication queries in real-time.",
];
let placeholderIdx = 0;

function initChatbot() {
  const sendBtn = document.getElementById('chatSendBtn');
  const input   = document.getElementById('chatInput');
  if (!sendBtn || !input) return;

  sendBtn.addEventListener('click', sendChat);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg   = input?.value.trim();
  if (!msg || state.chatWaiting) return;

  input.value = ''; input.style.height = 'auto';
  appendChatMsg('user', msg);
  state.chatHistory.push({ role:'user', content: msg });
  state.chatWaiting = true;
  document.getElementById('chatSendBtn').disabled = true;

  const typingEl = appendTypingIndicator();

  try {
    let reply;
    if (CHATBOT_API_URL) {
      const res = await fetch(CHATBOT_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: state.chatHistory }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      reply = data.reply || data.message || data.content || 'No response from API.';
    } else {
      await new Promise(r => setTimeout(r, 900 + Math.random() * 600));
      reply = PLACEHOLDER_RESPONSES[placeholderIdx++ % PLACEHOLDER_RESPONSES.length];
    }
    state.chatHistory.push({ role:'assistant', content: reply });
    typingEl.remove();
    appendChatMsg('bot', reply);
  } catch (err) {
    typingEl.remove();
    appendChatMsg('bot', `Error: ${err.message}. Check your API configuration.`);
  } finally {
    state.chatWaiting = false;
    document.getElementById('chatSendBtn').disabled = false;
  }
}

function appendChatMsg(role, text) {
  const win = document.getElementById('chatWindow');
  if (!win) return;

  // Remove welcome message on first real message
  const welcome = win.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  const wrap = document.createElement('div');
  wrap.className = `chat-msg ${role}`;

  const avatarEl = document.createElement('div');
  avatarEl.className = 'chat-avatar';
  avatarEl.textContent = role === 'user'
    ? (state.user?.displayName?.split(' ').map(w=>w[0]).join('').substring(0,2).toUpperCase() || 'U')
    : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.textContent = text;

  const timeEl = document.createElement('div');
  timeEl.className = 'chat-time';
  timeEl.textContent = fmtTime();

  const inner = document.createElement('div');
  inner.appendChild(bubble);
  inner.appendChild(timeEl);

  wrap.appendChild(avatarEl);
  wrap.appendChild(inner);
  win.appendChild(wrap);
  win.scrollTop = win.scrollHeight;
  return wrap;
}

function appendTypingIndicator() {
  const win = document.getElementById('chatWindow');
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg bot';
  wrap.innerHTML = `
    <div class="chat-avatar">AI</div>
    <div class="chat-typing">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>`;
  win.appendChild(wrap);
  win.scrollTop = win.scrollHeight;
  return wrap;
}
