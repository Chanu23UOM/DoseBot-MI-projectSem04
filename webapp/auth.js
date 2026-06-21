'use strict';
/* =========================================================
   DoseBot Web App — auth.js
   Firebase Authentication: Google Sign-In only
   ---------------------------------------------------------
   SETUP REQUIRED:
   1. Go to Firebase Console → your project → Project Settings
   2. Copy your web app config and paste below
   3. Enable Authentication → Sign-in method → Google
   4. Authentication → Settings → Authorized domains → add every
      domain this app is served from (localhost is included by default)
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

// ===== GOOGLE SIGN-IN =====
async function googleSignIn() {
  const provider = new firebase.auth.GoogleAuthProvider();
  const errEl = document.getElementById('loginError');
  try {
    const result = await auth.signInWithPopup(provider);
    await ensureUserProfile(result.user, {});
  } catch (err) {
    if (errEl) errEl.textContent = friendlyError(err);
    showToast(friendlyError(err), 'error');
  }
}

document.getElementById('googleSignInBtn')?.addEventListener('click', googleSignIn);

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
    'auth/popup-closed-by-user':    'Google sign-in was cancelled.',
    'auth/popup-blocked':           'Your browser blocked the sign-in popup. Allow popups and try again.',
    'auth/network-request-failed':  'Network error — check your connection.',
    'auth/too-many-requests':       'Too many attempts. Try again later.',
    'auth/unauthorized-domain':     'This domain is not authorized for Google sign-in. Add it in Firebase Console → Authentication → Settings → Authorized domains.',
    'auth/configuration-not-found': 'Firebase Auth not configured. Add your Firebase config in auth.js.',
  };
  return map[err.code] || err.message || 'An error occurred. Please try again.';
}
