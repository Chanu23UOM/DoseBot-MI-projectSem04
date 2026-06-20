# NEXT PHASE — Train Your Own Handwriting OCR Model (TrOCR)

This guide takes you from **collecting your own handwriting images** all the way to a
**trained model running inside the DoseBot website**. You will:

1. Collect & label your own data
2. Fine-tune **TrOCR** on **Google Colab** (free GPU)
3. Test it (CER / WER)
4. Upload the model to the **Hugging Face Hub**
5. Deploy it as a **Hugging Face Space**
6. Connect it to the website (one-line change)

> **Why TrOCR?** `microsoft/trocr-base-handwritten` is already pretrained on handwriting, so
> fine-tuning on a small set of *your* images teaches it your doctors' styles without needing
> millions of samples. It reads **one line of text per image**.

> **The design trick:** your new Space will expose the **same function name and outputs** as the
> current one (`classify_image_gradio` → `[text, label, confidence]`). So when you're done,
> connecting it to the website is changing a single line in `webapp/app.js`.

---

## Phase 1 — Collect & label your data

### 1.1 Capture images
Collect images of handwriting — prescriptions, medicine names, dosages. Sources:
- Your **ESP32-CAM**: open `http://<esp-ip>/capture` and save the JPEG.
- A **phone camera** (any photo).

**Crop each image to a single word or single line.** TrOCR transcribes one line at a time, so a
whole multi-line prescription should be cut into several line images.

### 1.2 Organise the folder
```
dataset/
├── images/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
└── labels.csv
```

`labels.csv` — the ground-truth text for every image:
```csv
file_name,text
img_0001.png,Amoxicillin 500mg
img_0002.png,Take twice daily
img_0003.png,Paracetamol
```

> **Tip:** label carefully and consistently — the model can only become as accurate as your
> labels. A spreadsheet (Excel/Google Sheets) exported to CSV is perfectly fine. For bigger sets,
> a free tool like **Label Studio** speeds this up.

### 1.3 How many images?
- **Minimum to see it learn:** ~200–300 labelled lines.
- **Good for a project demo:** ~1,000–3,000.
- More + more handwriting variety = better real-world accuracy.

### 1.4 Upload to Google Drive
Zip the folder and upload `dataset.zip` to your Google Drive (e.g. `MyDrive/dosebot/dataset.zip`).
Colab will read it from there.

---

## Phase 2 — Train on Google Colab (free GPU)

Open <https://colab.research.google.com> → **New notebook**.
**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.**

Run each cell below in order.

### 2.1 Install libraries
```python
!pip install -q transformers datasets evaluate jiwer accelerate
```

### 2.2 Mount Drive and unzip the dataset
```python
from google.colab import drive
drive.mount('/content/drive')

!unzip -q "/content/drive/MyDrive/dosebot/dataset.zip" -d /content/data
# After this you should have /content/data/dataset/images and labels.csv
import os
DATA_DIR = "/content/data/dataset"
print(os.listdir(DATA_DIR))
```

### 2.3 Load & split the labels (80 / 10 / 10)
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"{DATA_DIR}/labels.csv")
train_df, tmp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df,  test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))
print(len(train_df), len(val_df), len(test_df))
```

### 2.4 Dataset class
```python
import torch
from torch.utils.data import Dataset
from PIL import Image

MAX_TARGET_LENGTH = 64  # max characters/tokens per line

class HandwritingDataset(Dataset):
    def __init__(self, df, processor, root):
        self.df = df
        self.processor = processor
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.root}/images/{row['file_name']}").convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            str(row["text"]),
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        ).input_ids
        # Ignore padding tokens in the loss
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
```

### 2.5 Load the model & processor
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

BASE_MODEL = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

# Required token / generation config for fine-tuning
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.config.eos_token_id           = processor.tokenizer.sep_token_id
model.config.max_length             = MAX_TARGET_LENGTH
model.config.num_beams              = 4

train_ds = HandwritingDataset(train_df, processor, DATA_DIR)
val_ds   = HandwritingDataset(val_df,   processor, DATA_DIR)
test_ds  = HandwritingDataset(test_df,  processor, DATA_DIR)
```

### 2.6 Metric (Character Error Rate)
```python
import evaluate
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids   = pred.predictions
    pred_str   = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str  = processor.batch_decode(labels_ids, skip_special_tokens=True)
    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}
```

### 2.7 Train
```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

OUT_DIR = "/content/drive/MyDrive/dosebot/trocr-checkpoints"  # saved to Drive

args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    learning_rate=5e-5,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    save_total_limit=2,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model(OUT_DIR + "/best")
processor.save_pretrained(OUT_DIR + "/best")
```

> Watch the **loss go down** and the **eval `cer`** improve each epoch. If you run out of GPU
> memory, lower `per_device_train_batch_size` to 4 or 2. If accuracy is low, collect more data or
> train more epochs.
>
> **Colab disconnects?** Checkpoints are saved to Drive each epoch, so you can re-mount and resume.

---

## Phase 3 — Test / evaluate

### 3.1 Score on the held-out test set
```python
metrics = trainer.evaluate(test_ds)
print("Test CER:", metrics["eval_cer"])   # lower is better (0 = perfect)
```

### 3.2 Eyeball some predictions
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def read_image(path):
    image = Image.open(path).convert("RGB")
    pv = processor(image, return_tensors="pt").pixel_values.to(device)
    ids = model.generate(pv, max_length=MAX_TARGET_LENGTH)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

for i in range(min(10, len(test_df))):
    row = test_df.iloc[i]
    pred = read_image(f"{DATA_DIR}/images/{row['file_name']}")
    print(f"GT:   {row['text']}\nPRED: {pred}\n---")
```

---

## Phase 4 — Upload the model to Hugging Face Hub

```python
from huggingface_hub import notebook_login
notebook_login()   # paste a token from huggingface.co/settings/tokens (role: write)
```

```python
REPO = "Chanu2003/dosebot-trocr-handwriting"   # change to your username if different
model.push_to_hub(REPO)
processor.push_to_hub(REPO)
print("Pushed to https://huggingface.co/" + REPO)
```

Your trained model now has its own page on Hugging Face, loadable with
`VisionEncoderDecoderModel.from_pretrained("Chanu2003/dosebot-trocr-handwriting")`.

---

## Phase 5 — Deploy a Gradio Space wrapping your model

1. huggingface.co → **New Space** → SDK **Gradio**, hardware **CPU basic (free)**, public.
2. Add these two files to the Space (web editor or `git push`):

**`app.py`**
```python
import gradio as gr
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL = "Chanu2003/dosebot-trocr-handwriting"   # your model from Phase 4
processor = TrOCRProcessor.from_pretrained(MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Keep the SAME function name + 3 outputs as the old Space so the website
# only needs a one-line change to connect.
def classify_image_gradio(image):
    image = image.convert("RGB")
    pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pv, max_length=64)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text, "handwriting", 1.0

demo = gr.Interface(
    fn=classify_image_gradio,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Extracted Text"),
        gr.Textbox(label="Type"),
        gr.Number(label="Confidence"),
    ],
    title="DoseBot Handwriting OCR",
    description="Upload a handwriting image; the trained TrOCR model transcribes it.",
)

if __name__ == "__main__":
    demo.launch()
```

**`requirements.txt`**
```
transformers
torch
sentencepiece
pillow
gradio
```

3. Wait for the Space to build, then **upload a test image** in its UI and confirm it returns text.
4. Open the Space's **"Use via API"** panel and confirm the endpoint is **`/classify_image_gradio`**
   (it matches the function name).

---

## Phase 6 — Connect it to the website

Only **one line** changes. In [`webapp/app.js`](webapp/app.js) (around line 17):

```js
// before
const OCR_HF_SPACE = 'Chanu2003/DoseBotV2';
// after
const OCR_HF_SPACE = 'Chanu2003/dosebot-handwriting-space';   // your new Space id
```

`OCR_ENDPOINT` stays `'/classify_image_gradio'` — no other change needed, because the new Space
keeps the same function name and outputs.

Then test locally:
```bash
cd "DoseBot-MI-projectSem04/webapp"
python -m http.server 8000
```
Open <http://localhost:8000/app.html> → **AI Chatbot** tab →
- **Upload path:** Choose File → Read image → your model's reading appears in chat.
- **Camera path:** Connect → 📸 Capture & Read.

Finally commit and push (same flow as the previous phase):
```bash
git checkout -b use-own-trocr-model
git add webapp/app.js
git commit -m "Point OCR at own trained TrOCR handwriting model"
git checkout main && git merge use-own-trocr-model && git push origin main
```

---

## Checklist
- [ ] `dataset/images/` + `labels.csv` built and zipped to Drive
- [ ] Colab training finished; loss down, eval CER reasonable
- [ ] Test CER printed; sample predictions look sensible
- [ ] Model pushed to `Chanu2003/dosebot-trocr-handwriting`
- [ ] Space live, returns text, endpoint `/classify_image_gradio`
- [ ] `OCR_HF_SPACE` updated in `webapp/app.js`; website shows your model's output

## Troubleshooting
- **CUDA out of memory** → lower `per_device_train_batch_size` (8 → 4 → 2).
- **CER not improving** → more data, more epochs, or check label quality / that crops are single lines.
- **Space build fails** → confirm `requirements.txt` includes `sentencepiece`; check the Space Logs.
- **Website shows nothing** → the result appears in the chat window *below* the camera card; scroll down. Also confirm `OCR_HF_SPACE` matches your Space id exactly.
- **Garbled long text** → TrOCR reads one line per image; split multi-line prescriptions into line crops.
