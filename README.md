# ASL Sign Language Detector

## Abstract

This project implements a **real-time American Sign Language (ASL) detection system** using **MediaPipe hand tracking** and a **lightweight neural network**. The application captures live video from a webcam, extracts 21 hand landmarks using Google's MediaPipe framework, and classifies the hand pose into one of 36 ASL signs (A–Z, 0–9) using a trained Dense neural network.

Unlike traditional image-based approaches that rely on heavy CNN models and are sensitive to lighting and background, this system uses **landmark-based classification** — extracting geometric features (x, y, z coordinates of 21 keypoints) that are normalized and scale-invariant. This results in a model that is:

- **Fast** — ~50ms per prediction (vs ~200ms for CNN)
- **Lightweight** — ~50KB model size (vs ~11MB for MobileNetV2)
- **Robust** — works across different lighting conditions and backgrounds

The system also features **sentence building** from detected letters, **multilingual text-to-speech** (English, Hindi, Telugu) with automatic translation, and a modern responsive web interface.

### Key Technologies
- **MediaPipe Hands** — Real-time hand landmark detection (21 keypoints)
- **TensorFlow/Keras** — Dense neural network for landmark classification
- **Flask** — Backend web server with REST API
- **gTTS + deep-translator** — Multilingual translation and speech synthesis
- **HTML/CSS/JavaScript** — Responsive frontend with live camera feed

---

## Project Structure

```
sign-lang/
├── app.py                  # Flask backend (prediction, translation, TTS)
├── train.py                # CNN training script (legacy, optional)
├── train_landmarks.py      # Landmark extraction + Dense NN training
├── requirements.txt        # Python dependencies
├── models/
│   ├── landmark_model.h5   # Trained landmark classifier (~50KB)
│   ├── labels.json         # Class labels (0-9, a-z)
│   └── sign_language_model.h5  # Legacy CNN model (optional)
├── dataset/
│   └── asl_dataset/        # Training images organized by class
│       ├── 0/
│       ├── 1/
│       ├── ...
│       └── z/
├── static/
│   ├── app.css             # Stylesheet
│   └── app.js              # Frontend logic (camera, predictions, UI)
└── templates/
    └── index.html          # Main page template
```

---

## Setup Guide

### Prerequisites

- **Python 3.10 or 3.11** (recommended)
- **pip** (Python package manager)
- **Webcam** (for live detection)
- **Internet connection** (for translation and TTS features)

### Step 1: Clone / Download the Project

```bash
cd "d:\BTECH projects"
# If using git:
# git clone <your-repo-url> sign-lang
cd sign-lang
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

**Activate it:**

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Step 3: Install Dependencies

```bash
pip install flask opencv-python mediapipe tensorflow numpy pillow deep-translator gTTS
```

### Step 4: Prepare the Dat aset

Place your ASL dataset in `dataset/asl_dataset/` with one folder per class:

```
dataset/asl_dataset/
├── 0/    (images of sign "0")
├── 1/    (images of sign "1")
├── ...
├── a/    (images of sign "a")
├── b/    (images of sign "b")
├── ...
└── z/    (images of sign "z")
```

Each folder should contain ~50–100 images of that hand sign (`.jpg`, `.png`).

### Step 5: Train the Model

```bash
python train_landmarks.py
```

This will:
1. Extract hand landmarks from all dataset images using MediaPipe
2. Train a Dense neural network on the landmark features
3. Save the model to `models/landmark_model.h5`
4. Save labels to `models/labels.json`

**Expected output:**
```
Found 36 classes: ['0', '1', ..., 'z']
  [0] 40/70 hands detected
  [1] 65/70 hands detected
  ...
Validation accuracy: 93.96%
Saved landmark model to models/landmark_model.h5
```

### Step 6: Run the Application

```bash
python app.py
```

**Output:**
```
[OK] Loaded landmark model from models/landmark_model.h5
[OK] 36 labels: ['0', '1', ..., 'z']
[OK] Starting server on http://127.0.0.1:5000
```

### Step 7: Use the Application

1. Open **http://127.0.0.1:5000** in your browser (Chrome recommended)
2. Click **"Start Camera"** and allow camera permissions
3. Show ASL hand signs to the camera
4. The app predicts the letter and builds a sentence
5. Select a language (English / Hindi / Telugu) and click **"Speak"** to hear the sentence

---

## Features

| Feature | Description |
|---|---|
| 🖐️ Real-time hand detection | MediaPipe tracks 21 hand landmarks at ~50ms/frame |
| 🔤 ASL letter prediction | Dense NN classifies 36 signs (A–Z, 0–9) |
| 📝 Sentence builder | Stable letters auto-append to build words |
| 🌐 Multilingual TTS | Speak sentences in English, Hindi, or Telugu |
| 🔄 Auto-translation | Translates English text before speaking |
| 🪞 Mirrored camera | Selfie-style view for natural interaction |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Camera not starting | Allow camera permissions in browser, use Chrome |
| No hand detected | Ensure good lighting, show hand clearly in frame |
| Model not loaded | Run `python train_landmarks.py` first |
| Telugu speech not working | Requires internet for Google TTS |
| Warnings in terminal | Harmless TF/MediaPipe logs, can be ignored |
