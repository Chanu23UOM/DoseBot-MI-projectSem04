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

/* ===== AI CHATBOT via Hugging Face Space =====
   A separate Gradio Space that proxies chat to an open-source medical LLM
   via the HF Inference API. Deploy chatbot-space/ to HF and set the ID below.
   Leave empty to use CHATBOT_API_URL or built-in placeholders. */
const CHATBOT_HF_SPACE = 'https://chanu2003-dosebot-chatbot.hf.space';

/* ===== PRESCRIPTION OCR via Hugging Face Space + ESP32-CAM =====
   The DoseBotV2 medicine OCR/classifier is hosted on the HF Space below.
   The /predict_medicine endpoint takes a named `image` arg and returns
   [labelData, text] (see readPrescription). Space: Chanu2003/DoseBotV2 */
const OCR_HF_SPACE = 'https://chanu2003-dosebotv2.hf.space';
const OCR_ENDPOINT = '/predict_medicine';

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
  espCamIp:         '',
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
  initPrescriptionCam();
  initCsvBtns();
  initDetailsToggle();
  initFab();
  initDispenseBtn();
  renderDispenseTable();
  loadUserProfile();
  if (!CHATBOT_HF_SPACE && !CHATBOT_API_URL) document.getElementById('chatApiNote')?.removeAttribute('hidden');
}

// ===== DISPENSE BUTTON INIT =====
function initDispenseBtn() {
  document.getElementById('dispenseBtn')?.addEventListener('click', triggerDispense);
}

// ===== SIDEBAR NAVIGATION =====
const SECTION_TITLES = {
  'dashboard':    'Dashboard',
  'dispense-log': 'Dispense Log',
  'prescription': 'Prescription',
  'scanner':      'Scanner',
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
      updateFabVisibility(sec);
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

// ===== AT-A-GLANCE HERO STATUS =====
const HERO_STATES = {
  connecting: { cls:'state-connecting', emoji:'○', text:'Connecting to DoseBot…', sub:'Waiting for live sensor data' },
  ok:         { cls:'state-ok',         emoji:'✓', text:'All Systems Nominal',     sub:'Your meds are safe and secure.' },
  busy:       { cls:'state-busy',       emoji:'◌', text:'Dispensing in Progress…', sub:'Please collect your medication at the kiosk.' },
  danger:     { cls:'state-danger',     emoji:'!', text:'Warning: High Temperature', sub:'Cooling engaged to protect your medication.' },
};

function updateHero(d) {
  let key;
  if (!d)                          key = 'connecting';
  else if (d.temp > TEMP_THRESHOLD) key = 'danger';
  else if (d.bottle === 1 && d.ready === 1) key = 'busy';
  else                              key = 'ok';

  const s = HERO_STATES[key];
  const ring = document.getElementById('heroRing');
  if (ring) ring.className = 'hero-ring ' + s.cls;
  setText('heroEmoji', s.emoji);
  setText('heroTemp',  d ? (typeof d.temp === 'number' ? d.temp.toFixed(1) : '--') : '--');
  setText('heroText',  s.text);
  setText('heroSub',   s.sub);
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
        { label:'Temperature (°C)', data:[], borderColor:'#0d9488', backgroundColor:'rgba(13,148,136,0.08)',
          borderWidth:2.5, pointRadius:3, pointBackgroundColor:'#0d9488', pointBorderColor:'#ffffff',
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
          backgroundColor:'rgba(28,37,54,0.95)', padding:10, cornerRadius:8,
          callbacks:{ label: c => ` ${c.dataset.label}: ${Number(c.parsed.y).toFixed(1)}` }
        }
      },
      scales:{
        x:{ grid:{ color:'rgba(20,30,50,0.06)' }, ticks:{ maxTicksLimit:6, font:{size:10}, color:'#8696ad' } },
        y:{ suggestedMin:15, suggestedMax:36,
            grid:{ color:'rgba(20,30,50,0.06)' },
            ticks:{ font:{size:10}, color:'#8696ad', callback: v => v+'°C' } }
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

// ===== PROGRESSIVE DISCLOSURE (View Details) =====
function initDetailsToggle() {
  const btn     = document.getElementById('detailsToggle');
  const details = document.getElementById('dashDetails');
  const label   = document.getElementById('detailsToggleLabel');
  if (!btn || !details) return;
  btn.addEventListener('click', () => {
    const open = details.hasAttribute('hidden');
    if (open) details.removeAttribute('hidden');
    else      details.setAttribute('hidden', '');
    btn.setAttribute('aria-expanded', String(open));
    if (label) label.textContent = open ? 'Hide Details' : 'View Details';
    if (open) details.scrollIntoView({ behavior:'smooth', block:'nearest' });
  });
}

// ===== FLOATING ACTION BUTTON =====
function initFab() {
  document.getElementById('fabNewRx')?.addEventListener('click', () => {
    document.querySelector('.nav-item[data-section="prescription"]')?.click();
    document.getElementById('rxPatient')?.focus();
  });
}

// Hide the FAB while the Prescription view is already open
function updateFabVisibility(sec) {
  document.getElementById('fabNewRx')?.classList.toggle('hidden', sec === 'prescription');
}

// ===== MAIN DATA PROCESSOR =====
function processData(d) {
  state.latestData   = d;
  state.lastDataTime = Date.now();
  updateHero(d);
  updateMetricCards(d);
  updateLEDs(d);
  updateSystemBadge(d);
  pushTempChart(d.temp);
  checkAutoDispense(d);
  setText('lastSync', 'Last sync: ' + fmtTime());
  setConnStatus('live');
}

// ===== DISPENSE TRIGGER =====
function triggerDispense() {
  const d = state.latestData;
  if (!d) { toast('No sensor data yet — cannot dispense.', 'error'); return; }
  if (d.bottle !== 1) { toast('Place a bottle first.', 'error'); return; }
  if (d.temp > TEMP_THRESHOLD) { toast('Temperature too high — wait for cooling.', 'error'); return; }
  if (d.ready !== 1) { toast('System not ready — check device status.', 'error'); return; }

  const btn = document.getElementById('dispenseBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Dispensing…'; }

  db.ref('/commands/dispense_trigger').set(true)
    .then(() => {
      toast('Dispense command sent!', 'success');
      addDispenseEntry({ count: d.count, patient: state.user?.displayName || 'Manual' });
    })
    .catch(err => toast('Dispense failed: ' + err.message, 'error'))
    .finally(() => {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 5v14M5 12h14"/></svg> Dispense`;
      }
    });
}

// ===== FIREBASE RTDB LISTENER =====
function initFirebase() {
  // --- Merge data from multiple Firebase paths into a unified object ---
  const merged = { temp: 0, humidity: 0, voltage: 0, bottle: 0, ready: 0, count: 0, weight: 0, temp_safe: false, bottle_present: false, last_update: '' };

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
    err => { console.error('[DoseBot] Sensors:', err.message); setConnStatus('offline'); }
  );

  // /status → bottle_present, ready, temp_safe
  db.ref('/status').on('value',
    snap => {
      const s = snap.val();
      if (!s || typeof s !== 'object') return;
      merged.bottle_present = s.bottle_present === 'true' || s.bottle_present === true;
      merged.ready          = (s.ready === 'true' || s.ready === true) ? 1 : 0;
      merged.temp_safe      = s.temp_safe === 'true' || s.temp_safe === true;
      // Use bottle_present from status if Sensors.bottle is not set
      if (merged.bottle === 0 && merged.bottle_present) merged.bottle = 1;
      processData({ ...merged });
    },
    err => { console.error('[DoseBot] Status:', err.message); }
  );

  // /counters → pill_count, last_update
  db.ref('/counters').on('value',
    snap => {
      const c = snap.val();
      if (!c || typeof c !== 'object') return;
      merged.count       = typeof c.pill_count === 'number' ? c.pill_count : 0;
      merged.last_update = c.last_update || '';
      processData({ ...merged });
    },
    err => { console.error('[DoseBot] Counters:', err.message); }
  );

  // /commands → dispense_trigger (monitor state)
  db.ref('/commands/dispense_trigger').on('value', snap => {
    const val = snap.val();
    const btn = document.getElementById('dispenseBtn');
    if (btn && val === true) {
      btn.classList.add('dispensing');
    } else if (btn) {
      btn.classList.remove('dispensing');
    }
  });

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

// ===== ESP32-CAM / UPLOAD PRESCRIPTION SCANNER =====
function initPrescriptionCam() {
  const ipInput    = document.getElementById('espIp');
  const connectBtn = document.getElementById('camConnectBtn');
  const captureBtn = document.getElementById('captureReadBtn');
  const stream     = document.getElementById('espStream');
  const placeholder= document.getElementById('camPlaceholder');
  const fileInput  = document.getElementById('prescFile');
  const uploadBtn  = document.getElementById('uploadReadBtn');
  if (!ipInput) return;

  connectBtn?.addEventListener('click', () => {
    const ip = ipInput.value.trim();
    if (!ip) return;
    state.espCamIp = ip;
    stream.src = `http://${ip}:81/stream`;   // CameraWebServer MJPEG stream
    stream.hidden = false;
    if (placeholder) placeholder.hidden = true;
    captureBtn.disabled = false;
  });

  stream?.addEventListener('error', () => {
    appendChatMsg('bot', `Could not reach the camera stream at ${state.espCamIp}. Check the IP and that you are on the same WiFi.`);
  });

  // Read from the ESP32-CAM
  captureBtn?.addEventListener('click', async () => {
    const ip = state.espCamIp;
    if (!ip) return;
    await readPrescription(async () => {
      // The ESP32-CAM shares one frame buffer between the stream (port 81) and
      // /capture (port 80). Stop the stream and let the camera free up, or
      // /capture gets starved and hangs.
      const liveSrc = stream.src;
      stream.removeAttribute('src');
      await new Promise(r => setTimeout(r, 500));

      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 15000); // don't hang forever
      try {
        const res = await fetch(`http://${ip}/capture?_=${Date.now()}`, {
          cache: 'no-store', signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`camera /capture returned ${res.status}`);
        return await res.blob();
      } catch (e) {
        if (e.name === 'AbortError') throw new Error('camera /capture timed out (camera busy — power-cycle the ESP32)');
        throw e;
      } finally {
        clearTimeout(timer);
        stream.src = liveSrc; // resume the live view
      }
    });
  });

  // Read from an uploaded / phone-camera photo (no ESP32 needed)
  uploadBtn?.addEventListener('click', async () => {
    const file = fileInput?.files?.[0];
    if (!file) { toast('Please choose an image first.', 'warning'); return; }
    await readPrescription(async () => file);
  });
}

// Shared OCR pipeline: `getBlob()` supplies the image (ESP32 capture or upload).
async function readPrescription(getBlob) {
  if (state.scanWaiting) return;
  const captureBtn = document.getElementById('captureReadBtn');
  const uploadBtn  = document.getElementById('uploadReadBtn');
  const resultWrap = document.getElementById('scannerResultWrap');
  const resultText = document.getElementById('scannerResultText');

  state.scanWaiting = true;
  if (captureBtn) captureBtn.disabled = true;
  if (uploadBtn)  uploadBtn.disabled  = true;
  
  if (resultWrap) resultWrap.hidden = false;
  if (resultText) resultText.innerHTML = '<span style="color:var(--text-muted)">Analyzing image...</span>';

  try {
    const blob = await getBlob();

    if (!OCR_HF_SPACE) {
      if (resultText) resultText.innerHTML = '<span style="color:var(--danger)">OCR model is not configured yet. Set OCR_HF_SPACE in app.js.</span>';
      return;
    }

    // Send it to the DoseBotV2 model hosted on a Hugging Face Space.
    // Endpoint /predict_medicine takes a named `image` arg and returns
    // [labelData, text], where labelData = { label, confidences: [{label, confidence}] }.
    // (@gradio/client accepts a Blob directly.)
    const { Client } = await import('https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js');
    const client = await Client.connect(OCR_HF_SPACE);
    const result = await client.predict(OCR_ENDPOINT, { image: blob });

    const labelData = result?.data?.[0];
    const textData  = result?.data?.[1];

    const labelName = labelData?.label || '';
    const confObj = labelData?.confidences?.find(c => c.label === labelName);
    const confidenceVal = confObj ? confObj.confidence : null;

    const note = labelName ? `<br><small style="color:var(--text-muted)">(${labelName}${confidenceVal != null ? `, ${Math.round(confidenceVal * 100)}%` : ''})</small>` : '';
    if (resultText) resultText.innerHTML = `<strong>📝 Detected Text:</strong><br>${textData || '(no text detected)'}${note}`;
  } catch (err) {
    const hint = /failed to fetch|networkerror|load failed/i.test(err.message)
      ? '<br><small>Hint: Check ESP32 CORS or network connection.</small>'
      : '';
    if (resultText) resultText.innerHTML = `<span style="color:var(--danger)">Could not read the prescription: ${err.message}${hint}</span>`;
  } finally {
    state.scanWaiting = false;
    if (captureBtn) captureBtn.disabled = false;
    if (uploadBtn)  uploadBtn.disabled  = false;
  }
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

    if (CHATBOT_HF_SPACE) {
      // ---- HF Space via @gradio/client (same pattern as OCR) ----
      const { Client } = await import('https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js');
      const client = await Client.connect(CHATBOT_HF_SPACE);
      const historyJson = JSON.stringify(state.chatHistory.slice(-10));
      const result = await client.predict('/predict', {
        message: msg,
        history: historyJson,
      });
      reply = (Array.isArray(result?.data) ? result.data[0] : result?.data) || 'No response from the AI model.';

    } else if (CHATBOT_API_URL) {
      // ---- Legacy: direct REST API ----
      const res = await fetch(CHATBOT_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: state.chatHistory }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      reply = data.reply || data.message || data.content || 'No response from API.';

    } else {
      // ---- Placeholder (demo mode) ----
      await new Promise(r => setTimeout(r, 900 + Math.random() * 600));
      reply = PLACEHOLDER_RESPONSES[placeholderIdx++ % PLACEHOLDER_RESPONSES.length];
    }

    state.chatHistory.push({ role:'assistant', content: reply });
    typingEl.remove();
    appendChatMsg('bot', reply);
  } catch (err) {
    typingEl.remove();
    const hint = /failed to fetch|networkerror|load failed/i.test(err.message)
      ? ' The chatbot Space may be sleeping — visit it once on Hugging Face to wake it up.'
      : '';
    appendChatMsg('bot', `Error: ${err.message}.${hint}`);
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
  document.getElementById('chatLayout')?.classList.remove('chat-empty');

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
