# DoseBot Dashboard

Smart Pill Dispenser IoT Web Dashboard  
**EE2044 Group 29 · EE Batch 23 · University of Moratuwa**

---

## Quick Start

Open `index.html` directly in a browser — no build tools or server required.

To test without LabVIEW running, click **⚡ Simulate** in the top-right corner of the navbar. This generates random sensor readings every second so every UI feature can be exercised offline.

---

## Firebase Data Path

The dashboard listens on:

```
https://dosebot-g29-default-rtdb.asia-southeast1.firebasedatabase.app/dosebot/Sensors
```

LabVIEW pushes this JSON structure every second:

```json
{
  "weight":   200.00,
  "temp":     24.50,
  "humidity": 62.00,
  "voltage":  2.000,
  "bottle":   1,
  "ready":    1,
  "count":    3
}
```

The database is in **test mode** (unauthenticated reads). No Firebase Auth is required.

---

## Panels

| Panel   | Default | Description |
|---------|---------|-------------|
| **User**   | ✅ | Patient name & appointment entry, medicine prescription dropdown, bottle placement toggle, dispense button, rating tab, live temperature chart, dispense log |
| **Doctor** | —  | Issue prescriptions, pill inventory ring gauges, device status LEDs, pending collections table, activity log |
| **Admin**  | —  | KPI cards, container pill levels, system alert LEDs (temp/humidity/servo/rogue), antitheft lock, community trend charts, users log |

---

## Features

- **Live sensor cards** — Temperature, Humidity, Bottle Weight, Pill Count with animated number transitions  
- **LED status indicators** — Bottle Present, Temp Safe, Ready to Dispense, Fan Active  
- **Temperature chart** — Line chart of last 30 readings with a red dashed threshold at 28 °C  
- **Dispense log** — Last 10 events (newest first) with patient name, medicine, pills left  
- **Doctor prescriptions** — Issue prescriptions; uncollected ones appear in Pending Collections  
- **Admin trends** — Tabbed chart switching between Temperature, Humidity, and Prescriptions/day  
- **Chatbot UI** — DoseBot AI chat panel (API key to be connected)  
- **Rating system** — 5-star ratings with comment; average persisted in localStorage  
- **Simulate button** — Generates random data locally for offline testing  
- **Antitheft lock** — Toggle control in Admin panel  
- **LocalStorage persistence** — Dispense logs, prescriptions, users, ratings survive page refresh  

---

## Connecting the Chatbot API

Open `app.js` and find the `sendMsg()` function inside `initChatbot()`. Replace the placeholder `setTimeout` block with your API call:

```js
// Replace this block:
setTimeout(() => { appendCbMsg('⏳ API connection pending…', 'bot'); }, 700);

// With your API call, e.g.:
const res = await fetch('YOUR_API_ENDPOINT', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer YOUR_KEY' },
  body: JSON.stringify({ message: text })
});
const data = await res.json();
appendCbMsg(data.reply, 'bot');
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI    | Vanilla HTML / CSS / JavaScript (no framework) |
| Fonts | Plus Jakarta Sans (Google Fonts) |
| Charts | Chart.js 4.4 (CDN) |
| Database | Firebase Realtime Database 9.x (compat CDN) |
| Storage | Browser localStorage for session persistence |

---

## File Structure

```
dosebot-dashboard/
├── index.html   — Full dashboard UI (three panels + chatbot + footer)
├── style.css    — Corporate Trust design system (tokens, cards, LEDs, charts)
├── app.js       — Firebase listener, Chart.js, all panel logic
└── README.md    — This file
```

---

*EE2044 — Electronics & Embedded Systems Design Project*  
*University of Moratuwa, Sri Lanka*
