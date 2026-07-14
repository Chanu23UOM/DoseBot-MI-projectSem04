# DoseBotV2 — Prescription Recognition Model

The machine-learning component of DoseBot. It reads a **handwritten medicine name**
from a prescription image and returns the most likely medicine, so the kiosk doesn't
depend on a human correctly deciphering a doctor's handwriting.

**Live Space:** [huggingface.co/spaces/Chanu2003/DoseBotV2](https://huggingface.co/spaces/Chanu2003/DoseBotV2)

---

## At a glance

| Item | Detail |
|---|---|
| Task | Image classification (handwritten medicine name → label) |
| Model | Custom CNN, 4 convolutional blocks, ~1.3M parameters |
| Classes | 78 medicine names |
| Input | 256×64 RGB (width × height), normalised to [0, 1] |
| Test accuracy | **63.97%** (499/780 correct) |
| Test loss | 1.6981 |
| Framework | TensorFlow / Keras 2.20.0 |
| Trained on | Google Colab (T4 GPU) |
| Served by | Hugging Face Spaces (Gradio, free CPU tier) |

---

## Dataset

**Doctor's Handwritten Prescription BD Dataset** (Kaggle) — Bangladeshi medicine brand
names, which overlap heavily with the South Asian pharmacy market.

| Split | Images | Classes | Samples/class |
|---|---|---|---|
| Training | 3,120 | 78 | 40 |
| Validation | 780 | 78 | 10 |
| Testing | 780 | 78 | 10 |
| **Total** | **4,680** | **78** | — |

The dataset is perfectly balanced. It is **not committed to this repo** (~31 MB, 4,683
files) — download it from Kaggle and unpack it to:

```
ML/model/DoseBotV2/Doctor’s Handwritten Prescription BD dataset/
```

The CSVs carry three columns: `IMAGE` (filename), `MEDICINE_NAME` (brand, e.g. `Aceta`)
and `GENERIC_NAME` (e.g. `Paracetamol`).

---

## The one design decision that mattered: 256×64, not 128×128

Handwritten medicine words are **wide and short** ("Azithrocin"). Squashing them into a
square distorts the letterforms and destroys exactly the morphological features the CNN
needs. Instead we scale to fit and pad with white, preserving aspect ratio:

```python
def resize_with_padding(img, target_w=256, target_h=64, pad_color=(255, 255, 255)):
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas
```

Inference applies the **identical** preprocessing — any drift between training and serving
preprocessing would quietly wreck accuracy.

---

## Architecture

```
Input: (64, 256, 3)

Block 1: Conv2D(32)  → BN → Conv2D(32)  → BN → MaxPool(2×2)
Block 2: Conv2D(64)  → BN → Conv2D(64)  → BN → MaxPool(2×2)
Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool(2×2)
Block 4: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool(2×2)

Classifier:
  GlobalAveragePooling2D()
  Dense(512, relu) → BN → Dropout(0.2)
  Dense(78, softmax)
```

| Choice | Why |
|---|---|
| BatchNorm after every Conv2D | Stabilises training, permits higher LR, mild regularisation |
| GlobalAveragePooling2D over Flatten | Flatten would yield a 16,384-element vector → huge dense layer → overfitting. GAP gives 256 outputs |
| Dropout 0.2 (low) | An earlier version used 0.25+0.5+0.5 and **underfit** so badly it couldn't reach 6% train accuracy |
| `padding='same'` | Preserves spatial dims inside each block — important at only 64px height |

---

## Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial LR | 1e-4 |
| Batch size | 32 |
| Epochs | 60 (EarlyStopping, patience=10 on `val_accuracy`) |
| Loss | Categorical cross-entropy |
| Steps/epoch | 97 |

Callbacks: `EarlyStopping` (restores best weights), `ReduceLROnPlateau`
(factor=0.5, patience=5), `ModelCheckpoint` (`save_best_only=True`).

Augmentation is deliberately **mild** — ±5° rotation, 5% zoom/shift, white fill:

> Aggressive augmentation (±15° rotation, brightness, shear) was tried first and made the
> model underfit — it couldn't even fit the training set. Heavy rotation makes handwriting
> unreadable, so there was no signal left to learn.

Progression: val accuracy sat at 1.3% (random) until the LR dropped at epoch 6, jumped to
18.1% by epoch 11, and settled at ~64% val / ~85% train when EarlyStopping restored the
best weights.

---

## Results

**63.97%** test accuracy against a **1.28%** random baseline for 78 classes — roughly
**50× better than chance**, from only 40 training images per class.

**Best performers (F1 ≥ 0.85)** — distinctive letter shapes:

| Medicine | Precision | Recall | F1 |
|---|---|---|---|
| Ketotab | 0.91 | 1.00 | 0.95 |
| Bacaid | 0.91 | 1.00 | 0.95 |
| Candinil | 0.91 | 1.00 | 0.95 |
| Cetisoft | 0.91 | 1.00 | 0.95 |
| Bicozin | 1.00 | 0.80 | 0.89 |

**Worst performers (F1 ≤ 0.40)** — shared prefixes/suffixes:

| Medicine | Precision | Recall | F1 | Likely reason |
|---|---|---|---|---|
| Disopan | 0.11 | 0.40 | 0.17 | Confused with similar-looking words |
| Esoral | 0.50 | 0.20 | 0.29 | "Eso-" collides with Esonix |
| Esonix | 0.67 | 0.20 | 0.31 | "Eso-" collides with Esoral |
| Fexo | 0.33 | 0.30 | 0.32 | Confused with Fexofast, Fenadin |
| Canazole | 0.50 | 0.30 | 0.38 | "-azole" suffix shared with others |

The failure mode is systematic and unsurprising: the model confuses **medicine families
that share a prefix or suffix** ("Eso-", "-fex", "Keto-", "-azole").

### Why 63.97%, honestly

1. **40 training images per class** is very little for deep learning.
2. **High inter-class similarity** — many names look alike handwritten.
3. **High intra-class variation** — every person's handwriting differs.
4. **78 fine-grained classes.**

This is why the kiosk treats the model as **decision support, not an authority** (see below).

### Where we'd take it next

| Approach | Expected impact | Difficulty |
|---|---|---|
| Transfer learning (EfficientNet/ResNet backbone) | High | Low |
| More data (crowdsource from pharmacists) | High | Medium |
| TrOCR (transformer OCR) | Very High | Medium |
| CRNN + CTC (sequence recognition) | Very High | High |
| Top-K verification by a pharmacist | — | Low |

---

## How the kiosk uses it

The model is **not** bundled into the web app. It runs on a Hugging Face Space and the
frontend calls it over the Gradio REST API:

```
webapp (browser)
   │  1. POST image  → {space}/gradio_api/upload
   │  2. POST call   → {space}/gradio_api/call/predict_medicine
   │  3. GET  result → [labelData, summaryText]
   ▼
Hugging Face Space "Chanu2003/DoseBotV2"
   └── app.py → resize_with_padding → model.predict → top-5 + confidences
```

Wired up in [`webapp/app.js`](../webapp/app.js):

```js
const OCR_HF_SPACE = 'https://chanu2003-dosebotv2.hf.space';
const OCR_ENDPOINT = '/predict_medicine';
```

`predict_medicine` returns the **top-5 predictions with confidence scores**, not just a
single answer. That's deliberate: at ~64% top-1, surfacing five ranked candidates for a
human to confirm is far more useful than asserting one possibly-wrong name. The correct
medicine is almost always among the top 5.

> **Safety note.** DoseBot never dispenses on the model's say-so alone. The prediction is
> a suggestion for a pharmacist or patient to confirm, and dispensing is separately gated
> by the kiosk's own sensor interlocks (bottle present, temperature safe, ready).

We call the Gradio REST endpoints with plain `fetch` rather than the `@gradio/client`
library — the library fails in-browser with `TypeError: Failed to fetch` due to
version/protocol skew, while the Spaces themselves send correct CORS headers.

---

## Layout

```
ML/
├── README.md                       — this file
└── model/
    ├── colab V1.ipynb              — first training pass (V1)
    ├── DoseBotV2_Viva_Guide.md     — full written walkthrough
    ├── extract_pdf.py              — helper: pull text out of the evaluation PDF
    ├── dosebotv2-space/            — mirror of the deployed HF Space source
    │   ├── app.py                  — Gradio UI + predict_medicine()
    │   ├── requirements.txt
    │   └── README.md               — Space card
    ├── prescription-recognition/   — training notebooks + script
    │   ├── DoseBotV2_Training.ipynb
    │   ├── prescription-recognition.ipynb
    │   └── prescription_recognition_training.py
    └── DoseBotV2/                  — git-ignored: nested clone of the HF Space + dataset
```

Rendered training output (accuracy/loss curves, confusion matrix, sample predictions)
lives in `DoseBotV2_Training.ipynb - Colab.pdf` at the repo root. The full evaluation
write-up is `Model Evaluation.pdf`.

---

## Reproducing

1. Download the Kaggle dataset and unpack it to the path above.
2. Open `model/prescription-recognition/DoseBotV2_Training.ipynb` in Colab (T4 GPU).
3. Run all cells — trains, evaluates, and emits `prescription_model.h5`,
   `label_map.json` and `model_info.json`.
4. Upload those three files plus `dosebotv2-space/app.py` and `requirements.txt` to the
   Hugging Face Space.
