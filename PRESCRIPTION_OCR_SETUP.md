# Prescription OCR + ESP32-CAM — Setup Guide

This adds a doctor's-handwriting prescription reader to the DoseBot **AI Chat** tab.
You scan a prescription with an ESP32-CAM, the image is sent to a Donut OCR model
hosted on a free Hugging Face Space, and the transcription appears in the chat.

```
[ESP32-CAM] --MJPEG /stream + JPEG /capture--> [Browser: webapp on http]
                                                    | capture blob
                                                    v
                                    [HF Space: Gradio + Donut OCR] --text--> chat window
```

> **Why the model is on Hugging Face, not Replit:** the Donut OCR model is a ~2 GB
> PyTorch transformer and would OOM on Replit's free tier. A free HF CPU Space has
> enough RAM and turns the model into a web API for free.

> **Why the web app runs locally over http:** an HTTPS page cannot fetch
> `http://<esp-ip>` (browser "mixed content" block). Serving the page over plain
> `http://` lets it talk to both the LAN camera (http) and the HF Space (https).

---

## 1. Deploy the OCR model to a Hugging Face Space

1. Sign up at <https://huggingface.co> → **New Space** → SDK **Gradio**,
   hardware **CPU basic (free)**, visibility **Public**.
2. Get the app code:
   ```bash
   git clone https://github.com/JonSnow1807/medical-prescription-ocr.git
   ```
   Copy its `app.py`, any model/inference helper modules, and `requirements.txt`
   into your Space repo. Make sure `requirements.txt` includes:
   `torch`, `transformers`, `pytorch-lightning`, `sentencepiece`, `albumentations`, `gradio`.
3. Confirm `app.py` loads the Donut checkpoint from Hugging Face at startup.
   Commit and push to the Space remote.
4. Wait for the build, then upload a sample prescription image in the Space's UI to
   confirm it returns text.
5. Open the Space's **"Use via API"** link (bottom of the page). Note:
   - the **endpoint name** (e.g. `/predict`)
   - the **input parameter name** (e.g. `image`)

Then set these in [`webapp/app.js`](webapp/app.js) near the top:
```js
const OCR_HF_SPACE   = 'your-username/your-space'; // from step 1
const OCR_ENDPOINT   = '/predict';                 // from step 5
const OCR_PARAM_NAME = 'image';                    // from step 5
```

---

## 2. Flash the ESP32-CAM

1. Arduino IDE → board **AI Thinker ESP32-CAM** → open example
   **ArduinoESP32 → Camera → CameraWebServer**.
2. Set your WiFi SSID and password in the sketch, upload, open Serial Monitor,
   note the printed **LAN IP** (e.g. `192.168.1.50`).
3. Endpoints: live stream `http://<IP>:81/stream`, single JPEG `http://<IP>/capture`.
4. **Add CORS** so the browser can fetch `/capture`. In `app_httpd.cpp`, inside the
   capture handler (and the stream handler), add **before** the body is sent:
   ```c
   httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
   ```
   Re-upload. Without this, `fetch('/capture')` is blocked by CORS.

---

## 3. Run the web app (locally, over http)

```bash
cd "DoseBot-MI-projectSem04/webapp"
python -m http.server 8000
```
Open <http://localhost:8000/app.html> on a device on the **same WiFi** as the camera.

Then in the **AI Chatbot** tab:
1. Enter the ESP32-CAM IP → **Connect** (live view appears).
2. Point the camera at a prescription → **📸 Capture & Read prescription**.
3. The transcription appears as a bot message in the chat.

---

## Troubleshooting / notes

- **CORS error on capture** → the ESP32 firmware is missing the
  `Access-Control-Allow-Origin` header (step 2.4).
- **Mixed-content / blocked request** → you opened the page over `https`. Use the
  local `http://localhost:8000` server instead.
- **No live view** → wrong IP, camera not on the same WiFi, or stream port is 81.
- **First read is slow** → free HF Spaces sleep when idle; the first request wakes it.
- **Accuracy is modest** → the Donut model is trained on synthetic data, not real
  clinical handwriting. For better results, fine-tune it on real prescription
  images (e.g. the Doctor's Handwritten Prescription BD dataset) and push the new
  checkpoint to your Space. Not for clinical use.
```
