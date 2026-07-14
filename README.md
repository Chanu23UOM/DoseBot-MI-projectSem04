# DoseBot 🤖💊

**Smart medication dispensing kiosk for rural healthcare in Sri Lanka.**

DoseBot reads a doctor's handwritten prescription, checks that the medication is being
stored safely, dispenses the right number of pills, and answers the patient's questions —
combining a CNN, an LLM, a computer-vision pipeline, LabVIEW-driven hardware, and a
real-time web dashboard.

**EE2044 · Group 29 · EE Batch 23 · University of Moratuwa**

---

## Why

Handwritten prescriptions are frequently illegible, and misread prescriptions cause real
medication errors. In rural clinics there often isn't a pharmacist available to
double-check. DoseBot puts a kiosk in that gap: it suggests what the prescription says,
verifies storage conditions before releasing anything, and keeps an auditable log of every
dispense.

---

## System overview

```
┌─────────────────┐     handwritten Rx image      ┌──────────────────────────┐
│                 │ ────────────────────────────► │ DoseBotV2 (HF Space)     │
│                 │ ◄──────────────────────────── │ CNN · top-5 + confidence │
│   Web app       │        top-5 medicines        └──────────────────────────┘
│   (browser)     │
│                 │     medical Q&A               ┌──────────────────────────┐
│  · dashboard    │ ────────────────────────────► │ chatbot-space (HF Space) │
│  · scanner      │ ◄──────────────────────────── │ Llama/Gemma via HF API   │
│  · prescription │                               └──────────────────────────┘
│  · chatbot      │
│                 │     live sensors / commands   ┌──────────────────────────┐
└─────────────────┘ ◄───────────────────────────► │ Firebase Realtime DB     │
                                                  └────────────▲─────────────┘
                                                               │ every 1s
                                    ┌──────────────────────────┴─────────────┐
                                    │ LabVIEW  ·  Arduino  ·  OpenCV         │
                                    │ DHT22 · load cell · servos · ESP32-CAM │
                                    └────────────────────────────────────────┘
```

| Component | Platform | Purpose |
|---|---|---|
| **Prescription OCR** | HF Space (`DoseBotV2`) | Recognise handwritten medicine names |
| **AI Chatbot** | HF Space (`chatbot-space`) | Medical Q&A via an open-source LLM |
| **Web frontend** | Static HTML/CSS/JS | Patient-facing kiosk + dashboard |
| **CV pipeline** | Python + OpenCV | Pill detection and counting |
| **Control** | LabVIEW + Arduino | Sensors, servos, dispensing logic |
| **Data** | Firebase Realtime DB | Live telemetry and prescriptions |

---

## The model

DoseBotV2 is a **custom 4-block CNN** (~1.3M params) that classifies a cropped
handwritten medicine name into one of **78 medicines**. It scores **63.97%** test accuracy
against a 1.28% random baseline — about **50× better than chance** — trained on just 40
images per class.

The decision that mattered most was preprocessing: medicine words are wide and short, so
images are resized to **256×64 with aspect-ratio-preserving white padding** rather than
squashed into a square, which would destroy the letterforms. A close second was learning
that **mild** augmentation (±5°) was essential — aggressive augmentation made the model
underfit so badly it couldn't fit its own training set.

### How we use it

The model runs on a Hugging Face Space; the browser calls it over the Gradio REST API and
gets back the **top-5 candidates with confidence scores**:

```js
// webapp/app.js
const OCR_HF_SPACE = 'https://chanu2003-dosebotv2.hf.space';
const OCR_ENDPOINT = '/predict_medicine';
```

Returning five ranked candidates rather than one answer is the point. At ~64% top-1
accuracy, a single assertion would be wrong about a third of the time — but the correct
medicine is almost always in the top 5, which makes DoseBot a genuinely useful **clinical
decision-support tool** instead of an unreliable oracle.

> **DoseBot never dispenses on the model's prediction alone.** The suggestion is confirmed
> by a human, and dispensing is independently gated by hardware interlocks — bottle
> present, temperature within range, system ready.

The model's known weakness is systematic: it confuses medicine families sharing a prefix
or suffix ("Eso-", "-fex", "-azole"). Full architecture, training details, per-class
metrics and next steps are in **[`ML/README.md`](ML/README.md)**; the complete evaluation
write-up is in `Model Evaluation.pdf`.

---

## Quick start

**Landing page** — open `index.html` in a browser.

**Web app** — open `webapp/index.html` (sign-in) and it hands off to `webapp/app.html`.
No build step, no server: everything is vanilla HTML/CSS/JS loaded from CDNs.

No hardware handy? Open the dashboard and hit **⚡ Simulate** to generate live sensor
readings every second, so the whole UI can be exercised offline.

**CV pipeline**

```bash
pip install opencv-python numpy
python dosebot_cv_pipeline.py --target RED --count 5
```

**ML** — see [`ML/README.md`](ML/README.md). The Kaggle dataset and the 16.3 MB `.h5`
weights are intentionally not committed.

---

## Firebase

The dashboard subscribes to:

```
https://dosebot-g29-default-rtdb.asia-southeast1.firebasedatabase.app/dosebot/Sensors
```

LabVIEW pushes this structure once per second:

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

A representative export is checked in at `LV/dosebot-g29-default-rtdb-export.json`.

---

## Web app

The dashboard leads with an **at-a-glance status ring** rather than a wall of numbers —
one large indicator that answers "are my meds okay?" in plain English:

| State | Colour | Meaning |
|---|---|---|
| Nominal | Teal/green | All systems nominal — meds are safe |
| Dispensing | Amber | Dispensing in progress |
| Warning | Red | High temperature (> 28 °C) — cooling engaged |

Detailed metrics (temperature, humidity, weight, pill count, LED states, trend chart,
dispense log) sit behind a **View Details** toggle, and **New Prescription** is a floating
action button. Sections: Dashboard · Dispense Log · Prescription · Scanner · AI Chatbot ·
Profile.

| Layer | Technology |
|---|---|
| UI | Vanilla HTML / CSS / JavaScript (no framework) |
| Fonts | Plus Jakarta Sans |
| Charts | Chart.js 4.4 (CDN) |
| Auth + data | Firebase Auth + Realtime Database 9.x (compat CDN) |
| ML | Hugging Face Spaces (Gradio REST) |

---

## Repository layout

```
DoseBot-MI-projectSem04/
├── index.html · style.css · app.js · shader.js   — landing page
├── webapp/                                       — the kiosk web app
│   ├── index.html          — sign-in
│   ├── app.html            — dashboard, scanner, prescription, chatbot, profile
│   ├── auth.js · app.js · style.css
│   └── assets/             — mascots, illustrations, artwork
├── ML/                                           — prescription recognition
│   ├── README.md           — model card: architecture, training, results
│   └── model/
│       ├── dosebotv2-space/          — deployed HF Space source (app.py)
│       ├── prescription-recognition/ — training notebooks + script
│       ├── colab V1.ipynb
│       └── DoseBotV2_Viva_Guide.md
├── chatbot-space/                                — HF Space: medical Q&A LLM
├── LV/                                           — LabVIEW VIs + RTDB export
├── firmware/firmware.ino                         — Arduino: DHT22 + servos
├── dosebot_cv_pipeline.py                        — OpenCV pill detection → UDP
├── assets/                                       — shared static assets
├── Model Evaluation.pdf                          — full model evaluation
├── DoseBotV2_Training.ipynb - Colab.pdf          — training run output
├── Project_Proposal_G29.pdf
├── PRESCRIPTION_OCR_SETUP.md                     — OCR + ESP32-CAM setup
└── next phase.md                                 — roadmap: custom TrOCR model
```

---

## Roadmap

Short term: transfer learning from a pretrained backbone (cheapest accuracy win), and
crowdsourcing handwriting samples from real pharmacists. Longer term: a TrOCR
transformer-based OCR and a two-stage detect-then-classify pipeline — see
[`next phase.md`](next%20phase.md).

---

*EE2044 — Electronics & Embedded Systems Design Project*
*University of Moratuwa, Sri Lanka*
