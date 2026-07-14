---
title: DoseBotV2
emoji: 💊
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
---

# 🤖💊 DoseBotV2 - Prescription Medicine Recognition

Upload a handwritten prescription image to identify the medicine name.

## Model
- **Architecture:** CNN (4 conv blocks + BatchNorm + GlobalAvgPooling)
- **Input:** 256x64 RGB images (aspect-ratio preserved with padding)
- **Classes:** 78 medicine names
- **Dataset:** Doctor's Handwritten Prescription BD Dataset
