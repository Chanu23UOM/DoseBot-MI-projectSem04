'use strict';
/* =========================================================
   DoseBot Web App — auth.js
   Firebase Authentication: Google + Email/Password
   ---------------------------------------------------------
   SETUP REQUIRED:
   1. Go to Firebase Console → your project → Project Settings
   2. Copy your web app config and paste below
   3. Enable Authentication → Sign-in methods → Google + Email/Password
   ========================================================= */

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

// ===== INIT =====
firebase.initializeApp(FIREBASE_CONFIG);
const auth = firebase.auth();
const db   = firebase.database();

// ===== TOAST =====
function showToast(msg, type = 'info') {
  const wrap = document.getElementById('toastWrap');
  if (!wrap) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => {
    t.classList.add('fade-out');
    t.addEventListener('animationend', () => t.remove());
  }, 3500);
}

// ===== AUTH STATE OBSERVER =====
auth.onAuthStateChanged(user => {
  const loader = document.getElementById('appLoader');
  if (loader) loader.classList.add('hidden');
  if (user) {
    window.location.replace('app.html');
  }
});

// ===== TABS =====
document.querySelectorAll('.auth-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.auth-tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected','false'); });
    tab.classList.add('active');
    tab.setAttribute('aria-selected','true');
    const which = tab.dataset.tab;
    document.getElementById('loginForm').classList.toggle('hidden',    which !== 'login');
    document.getElementById('registerForm').classList.toggle('hidden', which !== 'register');
    document.getElementById('loginError').textContent    = '';
    document.getElementById('registerError').textContent = '';
  });
});

// ===== GOOGLE SIGN-IN =====
async function googleSignIn() {
  const provider = new firebase.auth.GoogleAuthProvider();
  try {
    const result = await auth.signInWithPopup(provider);
    await ensureUserProfile(result.user, {});
  } catch (err) {
    showToast(friendlyError(err), 'error');
  }
}

document.getElementById('googleSignInBtn')?.addEventListener('click',  googleSignIn);
document.getElementById('googleRegisterBtn')?.addEventListener('click', googleSignIn);

// ===== EMAIL LOGIN =====
document.getElementById('loginBtn')?.addEventListener('click', async () => {
  const email = document.getElementById('loginEmail').value.trim();
  const pass  = document.getElementById('loginPassword').value;
  const errEl = document.getElementById('loginError');
  const btn   = document.getElementById('loginBtn');

  if (!email || !pass) { errEl.textContent = 'Please enter email and password.'; return; }
  errEl.textContent = '';
  btn.classList.add('loading');
  btn.textContent = 'Signing in…';

  try {
    await auth.signInWithEmailAndPassword(email, pass);
  } catch (err) {
    errEl.textContent = friendlyError(err);
    btn.classList.remove('loading');
    btn.textContent = 'Sign In';
  }
});

// Enter key support for login
document.getElementById('loginPassword')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('loginBtn')?.click();
});

// ===== EMAIL REGISTER =====
document.getElementById('registerBtn')?.addEventListener('click', async () => {
  const name  = document.getElementById('regName').value.trim();
  const phone = document.getElementById('regPhone').value.trim();
  const email = document.getElementById('regEmail').value.trim();
  const pass  = document.getElementById('regPassword').value;
  const errEl = document.getElementById('registerError');
  const btn   = document.getElementById('registerBtn');

  if (!name)  { errEl.textContent = 'Full name is required.'; return; }
  if (!email) { errEl.textContent = 'Email is required.'; return; }
  if (pass.length < 6) { errEl.textContent = 'Password must be at least 6 characters.'; return; }
  errEl.textContent = '';
  btn.classList.add('loading');
  btn.textContent = 'Creating account…';

  try {
    const cred = await auth.createUserWithEmailAndPassword(email, pass);
    await cred.user.updateProfile({ displayName: name });
    await ensureUserProfile(cred.user, { name, phone });
  } catch (err) {
    errEl.textContent = friendlyError(err);
    btn.classList.remove('loading');
    btn.textContent = 'Create Account';
  }
});

// ===== SAVE USER PROFILE TO RTDB =====
async function ensureUserProfile(user, extra) {
  const ref = db.ref(`/dosebot/users/${user.uid}`);
  const snap = await ref.once('value');
  if (!snap.exists()) {
    await ref.set({
      name:         extra.name  || user.displayName || '',
      phone:        extra.phone || '',
      email:        user.email  || '',
      registeredAt: Date.now(),
    });
  }
}

// ===== ERROR MESSAGES =====
function friendlyError(err) {
  const map = {
    'auth/user-not-found':        'No account found with this email.',
    'auth/wrong-password':        'Incorrect password.',
    'auth/email-already-in-use':  'An account with this email already exists.',
    'auth/invalid-email':         'Invalid email address.',
    'auth/weak-password':         'Password must be at least 6 characters.',
    'auth/popup-closed-by-user':  'Google sign-in was cancelled.',
    'auth/network-request-failed':'Network error — check your connection.',
    'auth/too-many-requests':     'Too many attempts. Try again later.',
    'auth/configuration-not-found': 'Firebase Auth not configured. Add your Firebase config in auth.js.',
  };
  return map[err.code] || err.message || 'An error occurred. Please try again.';
}
