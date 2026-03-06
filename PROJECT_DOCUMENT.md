# Sign Language Recognition System

## Project Document

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Survey](#3-literature-survey)
4. [System Architecture](#4-system-architecture)
5. [Module Description](#5-module-description)
6. [Implementation Details](#6-implementation-details)
7. [Results and Testing](#7-results-and-testing)
8. [Conclusion and Future Scope](#8-conclusion-and-future-scope)
9. [References](#9-references)

---

## 1. Abstract

This project presents a **Real-Time American Sign Language (ASL) Recognition System** that leverages **MediaPipe Hand Tracking** and a **Dense Neural Network** to detect and classify hand signs into 36 categories (A–Z, 0–9). Unlike traditional CNN-based approaches that process raw images and are sensitive to lighting, background, and computational load, this system uses a **landmark-based classification** approach. It extracts 21 hand keypoints (x, y, z coordinates) using Google's MediaPipe framework, normalizes them, and feeds a lightweight neural network for classification.

The system achieves a **validation accuracy of ~94%**, with a prediction latency of approximately **50ms per frame**, making it suitable for real-time deployment. It also includes features like **sentence building**, **multilingual text-to-speech** (English, Hindi, Telugu) using Google Translate and gTTS, and a **responsive web interface** built with Flask, HTML, CSS, and JavaScript.

**Keywords:** Sign Language Recognition, MediaPipe, Hand Landmarks, Deep Learning, TensorFlow, Flask, Real-Time Detection, Text-to-Speech.

---

## 2. Introduction

### 2.1 Background

According to the World Health Organization, over **430 million people** worldwide require rehabilitation for hearing loss. Sign language is the primary mode of communication for the deaf and hard-of-hearing community. However, a significant communication barrier exists between sign language users and the hearing population who are generally unfamiliar with sign language.

Automated sign language recognition systems can bridge this gap by translating hand gestures into text or speech in real time, enabling seamless communication without the need for a human interpreter.

### 2.2 Problem Statement

To design and implement a real-time web-based system that can:
1. Detect hand gestures from a live webcam feed.
2. Classify the gesture into one of 36 ASL signs (A–Z, 0–9).
3. Build sentences from detected letters.
4. Translate and speak the sentence in multiple languages (English, Hindi, Telugu).

### 2.3 Objectives

- Develop a lightweight, real-time hand gesture recognition system using landmark-based features.
- Achieve high classification accuracy (>90%) with minimal computational overhead.
- Provide a user-friendly web interface accessible through any modern browser.
- Support multilingual text-to-speech for broader accessibility.

### 2.4 Scope

The system focuses on **static ASL fingerspelling** (individual letters and digits). It does not cover dynamic gestures (words or phrases expressed through motion) or two-handed signs. The system is designed for single-hand detection in a webcam-based environment.

---

## 3. Literature Survey

### 3.1 Traditional Approaches

| Approach | Method | Limitations |
|---|---|---|
| Glove-Based Systems | Sensors attached to gloves measure finger positions | Expensive, intrusive, requires specialized hardware |
| Image-Based CNN | Convolutional Neural Networks on raw images | Sensitive to background, lighting; computationally heavy |
| Template Matching | Compares hand shapes to stored templates | Rigid, fails with variations in hand size and orientation |

### 3.2 Modern Approaches

| Approach | Method | Advantages |
|---|---|---|
| MediaPipe + ML | Landmark extraction followed by ML classifier | Fast, lightweight, background-invariant |
| Transfer Learning (MobileNet, ResNet) | Fine-tuning pre-trained models on sign datasets | High accuracy but heavy model size (~11MB+) |
| LSTM/RNN-Based | Sequence models for dynamic gesture recognition | Handles motion-based signs but complex to train |

### 3.3 Why Landmark-Based Classification?

Traditional CNN approaches process the entire image (e.g., 64×64×3 = 12,288 features), making them sensitive to:
- Background clutter
- Lighting variations
- Skin color differences
- Camera angle changes

In contrast, **landmark-based classification** extracts only **63 features** (21 keypoints × 3 coordinates), which are:
- **Scale-invariant** (normalized by hand size)
- **Translation-invariant** (centered on the wrist)
- **Background-independent** (only geometric points are used)

This results in a model that is **~200× smaller** (50KB vs 11MB) and **~4× faster** (50ms vs 200ms) than a comparable CNN approach.

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                          │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────────┐    │
│  │  Camera   │──>│  JavaScript │──>│  Sentence Builder     │    │
│  │  Feed     │   │  (app.js)   │   │  + TTS Controls       │    │
│  └──────────┘   └──────┬──────┘   └──────────────────────┘    │
│                        │ HTTP POST /predict                    │
└────────────────────────┼───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    FLASK SERVER (app.py)                        │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  Image       │  │  MediaPipe     │  │  TensorFlow      │   │
│  │  Decoding    │─>│  Hand Landmark │─>│  Dense NN        │   │
│  │  (Base64)    │  │  Extraction    │  │  Prediction      │   │
│  └──────────────┘  └────────────────┘  └──────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Translation (deep-translator) + TTS (gTTS)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. Browser captures a video frame from the webcam.
2. JavaScript sends the frame as a Base64-encoded JPEG to the Flask backend via HTTP POST.
3. Flask decodes the image and passes it to MediaPipe for hand landmark detection.
4. If a hand is detected, 21 landmarks (63 features) are extracted and normalized.
5. The feature vector is fed to the Dense Neural Network for classification.
6. The predicted label, confidence score, and bounding box are returned to the browser.
7. The browser displays the result and optionally appends stable letters to a sentence.
8. The user can request translation and text-to-speech via the `/speak` endpoint.

### 4.3 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Backend | Flask (Python) | REST API server |
| Hand Detection | MediaPipe Hands | 21-keypoint hand landmark extraction |
| Classification | TensorFlow/Keras | Dense Neural Network model |
| Frontend | HTML, CSS, JavaScript | Responsive web UI with live camera |
| Translation | deep-translator | Google Translate API wrapper |
| Text-to-Speech | gTTS | Google Text-to-Speech synthesis |
| Image Processing | OpenCV, Pillow | Image loading and conversion |

---

## 5. Module Description

### 5.1 Module Overview

The system consists of **5 major modules**:

```
┌─────────────────────────────────────────────────┐
│           Sign Language Recognition System       │
├─────────────┬──────────┬──────────┬─────────────┤
│  Training   │  Backend │ Frontend │ Translation  │
│  Module     │  Module  │ Module   │ & TTS Module │
│             │          │          │              │
│ train_      │ app.py   │ app.js   │ /translate   │
│ landmarks.py│          │ app.css  │ /speak       │
│             │          │ index.   │              │
│             │          │ html     │              │
└─────────────┴──────────┴──────────┴─────────────┘
```

---

### 5.2 Module 1: Training Module (`train_landmarks.py`)

**Purpose:** Extract hand landmarks from dataset images and train a Dense Neural Network classifier.

**Key Functions:**

| Function | Description |
|---|---|
| `normalize_landmarks()` | Translates landmarks so the wrist is at origin, then scales by maximum hand distance to achieve scale and position invariance |
| `extract_dataset()` | Walks the dataset directory, processes each image with MediaPipe, extracts and normalizes landmarks, returns feature matrix X and label vector y |
| `build_model()` | Constructs a Sequential Dense NN with BatchNormalization and Dropout layers |
| `main()` | Orchestrates the full pipeline: extraction → splitting → training → saving |

**Model Architecture:**

```
Input (63 features)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
    ↓
Dense(36, Softmax) → Output (36 classes)
```

**Training Configuration:**
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Sparse Categorical Cross-Entropy
- Batch Size: 64
- Epochs: 100 (with Early Stopping, patience=10)
- Learning Rate Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Train/Validation Split: 80/20

---

### 5.3 Module 2: Backend Module (`app.py`)

**Purpose:** Flask web server that handles prediction requests, translation, and TTS.

**API Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the main HTML page |
| `/predict` | POST | Accepts a Base64 image, returns predicted label, confidence, bounding box, and top-3 predictions |
| `/translate` | POST | Translates English text to Hindi or Telugu |
| `/speak` | POST | Translates text and generates MP3 audio using gTTS, returns Base64-encoded audio |

**Key Design Decisions:**
- **Persistent MediaPipe instance:** The `Hands` detector is created once at startup and reused across all requests, avoiding the overhead of re-initialization (~200ms saved per request).
- **Landmark normalization:** The same `normalize_landmarks()` function used during training is reused in production to ensure feature consistency.
- **Minimum detection confidence:** Set to 0.5 (50%) to balance between false positives and missed detections.

---

### 5.4 Module 3: Frontend Module (`app.js`, `app.css`, `index.html`)

**Purpose:** Responsive web interface with live camera feed, real-time predictions, and sentence building.

**Key Features:**

| Feature | Implementation |
|---|---|
| Live Camera Feed | `navigator.mediaDevices.getUserMedia()` with 640×480 resolution |
| Frame Capture | Off-screen `<canvas>` captures frames every 150ms |
| Prediction Display | Shows predicted letter, confidence %, top-3 alternatives |
| Bounding Box Overlay | Orange rectangle drawn on a transparent `<canvas>` overlay |
| Stability Buffer | Requires 3 out of 5 consecutive frames to agree before adding a letter |
| Mirrored Video | CSS `transform: scaleX(-1)` for natural selfie-style interaction |

**Stability Algorithm:**
```
Buffer Size = 5 frames
Stability Threshold = 3 (majority vote)

1. Each prediction is pushed into a circular buffer.
2. When the buffer is full, count occurrences of each label.
3. If any label appears ≥ 3 times AND is different from the last added letter,
   append it to the sentence.
4. Clear the buffer after appending.
```

---

### 5.5 Module 4: Translation & TTS Module

**Purpose:** Multilingual support for translating and speaking the built sentence.

**Supported Languages:**

| Language | Code | Translation | TTS |
|---|---|---|---|
| English | en-US | No translation needed | ✅ |
| Hindi | hi-IN | Google Translate | ✅ |
| Telugu | te-IN | Google Translate | ✅ |

**Flow:**
1. User builds a sentence (e.g., "HELLO").
2. User selects a language (e.g., Hindi) and clicks "Speak".
3. Backend translates "HELLO" → "नमस्ते" using `deep-translator`.
4. `gTTS` generates an MP3 audio file of "नमस्ते" in Hindi.
5. Audio is Base64-encoded and sent to the browser for playback.

---

## 6. Implementation Details

### 6.1 Dataset

- **Source:** ASL Alphabet Dataset (cropped hand images)
- **Total Images:** ~2,515 images
- **Classes:** 36 (A–Z letters + 0–9 digits)
- **Images per Class:** ~70 images
- **Image Format:** JPEG (cropped hand regions)
- **Directory Structure:**
  ```
  dataset/asl_dataset/
  ├── 0/    (70 images)
  ├── 1/    (70 images)
  ├── ...
  ├── a/    (70 images)
  ├── ...
  └── z/    (70 images)
  ```

### 6.2 Feature Extraction Pipeline

```
Raw Image (JPEG)
    ↓ cv2.imread()
BGR Image
    ↓ cv2.cvtColor(BGR → RGB)
RGB Image
    ↓ MediaPipe Hands.process()
21 Hand Landmarks (x, y, z each)
    ↓ normalize_landmarks()
63-Dimensional Feature Vector (float32)
```

**Normalization Steps:**
1. **Translation:** Subtract wrist coordinates (landmark 0) from all 21 landmarks, centering the hand at the origin.
2. **Scaling:** Divide all coordinates by the maximum Euclidean distance from the wrist, normalizing the hand to unit scale.
3. **Flattening:** Reshape the 21×3 array into a 63-element vector.

### 6.3 Neural Network Training

**Hardware:** Standard laptop CPU (no GPU required)

**Training Output Example:**
```
Found 36 classes: ['0', '1', ..., 'z']
  [0] 40/70 hands detected
  [1] 65/70 hands detected
  ...
Train: 1624, Val: 407
Epoch 1/100 - accuracy: 0.45 - val_accuracy: 0.72
Epoch 10/100 - accuracy: 0.89 - val_accuracy: 0.91
Epoch 25/100 - accuracy: 0.96 - val_accuracy: 0.94
Validation accuracy: 93.96%
Saved landmark model to models/landmark_model.h5
```

### 6.4 Project Structure

```
sign-lang/
├── app.py                  # Flask backend server
├── train.py                # CNN training script (legacy)
├── train_landmarks.py      # Landmark-based training script
├── requirements.txt        # Pinned Python dependencies
├── setup.bat               # One-click setup script (Windows)
├── run_app.bat              # One-click run script (Windows)
├── .gitignore              # Git ignore rules
├── README.md               # Project README
├── models/
│   ├── landmark_model.h5   # Trained landmark classifier (~50KB)
│   ├── labels.json         # 36 class labels
│   └── sign_language_model.h5  # Legacy CNN model
├── dataset/
│   └── asl_dataset/        # Training images (36 classes)
├── static/
│   ├── app.css             # Stylesheet (responsive design)
│   └── app.js              # Frontend logic (camera, predictions)
└── templates/
    └── index.html          # Main HTML page (Jinja2 template)
```

### 6.5 Software Requirements

| Software | Version | Purpose |
|---|---|---|
| Python | 3.10 / 3.11 | Programming language |
| Flask | 3.0.2 | Web framework |
| TensorFlow | 2.15.0 | Deep learning framework |
| MediaPipe | 0.10.11 | Hand landmark detection |
| OpenCV | 4.9.0.80 | Image processing |
| NumPy | 1.26.4 | Numerical computation |
| Pillow | 10.2.0 | Image loading |
| deep-translator | 1.11.4 | Language translation |
| gTTS | 2.5.1 | Text-to-speech |

### 6.6 Hardware Requirements

| Component | Minimum Requirement |
|---|---|
| Processor | Intel i3 / AMD Ryzen 3 or higher |
| RAM | 4 GB (8 GB recommended) |
| Storage | 500 MB free space |
| Camera | Built-in webcam or USB webcam |
| Internet | Required for translation and TTS features |

---

## 7. Results and Testing

### 7.1 Model Performance

| Metric | Value |
|---|---|
| Training Accuracy | ~96% |
| Validation Accuracy | ~94% |
| Model Size | ~50 KB |
| Prediction Latency | ~50 ms per frame |
| Real-Time FPS | ~6–7 frames/second |

### 7.2 Comparison with CNN Approach

| Metric | Landmark Model (Ours) | CNN Model (MobileNetV2) |
|---|---|---|
| Input Features | 63 (landmarks) | 12,288 (64×64×3 pixels) |
| Model Size | ~50 KB | ~11 MB |
| Prediction Time | ~50 ms | ~200 ms |
| Background Sensitivity | Low | High |
| Lighting Sensitivity | Low | High |
| Accuracy | ~94% | ~95% |

### 7.3 Testing Scenarios

| Test Case | Expected Result | Actual Result | Status |
|---|---|---|---|
| Show letter "A" to camera | Predicts "A" with >80% confidence | Predicts "A" at 87% | ✅ Pass |
| Show digit "5" to camera | Predicts "5" with >80% confidence | Predicts "5" at 92% | ✅ Pass |
| No hand in frame | Shows "No hand detected" | Shows "-" with no prediction | ✅ Pass |
| Build sentence "HELLO" | Letters auto-append to sentence | Sentence shows "HELLO" | ✅ Pass |
| Speak in Hindi | Translates and plays Hindi audio | Audio plays correctly | ✅ Pass |
| Speak in Telugu | Translates and plays Telugu audio | Audio plays correctly | ✅ Pass |
| Poor lighting | Lower confidence but still detects | Confidence drops to ~60% | ✅ Pass |
| Multiple hands in frame | Detects only one hand | Uses first detected hand | ✅ Pass |

### 7.4 Screenshots

The web application displays:
- A **live camera feed** with an orange bounding box around the detected hand.
- A **predicted letter** display with confidence percentage.
- **Top-3 predictions** shown as chips for transparency.
- A **sentence builder** with Add Space, Backspace, Clear, and Speak controls.
- A **language selector** for English, Hindi, and Telugu.

---

## 8. Conclusion and Future Scope

### 8.1 Conclusion

This project successfully demonstrates a **real-time ASL sign language recognition system** that is:

1. **Lightweight:** The model is only ~50KB, making it deployable on low-end hardware without a GPU.
2. **Fast:** Predictions take ~50ms, enabling smooth real-time interaction.
3. **Robust:** Landmark-based features are invariant to background, lighting, and skin color.
4. **Accessible:** The web-based interface works on any device with a browser and webcam, with multilingual TTS support for broader reach.
5. **Easy to Deploy:** Setup scripts (`setup.bat`, `run_app.bat`) allow one-click installation and execution on Windows.

The system achieves a validation accuracy of **~94%** on 36 ASL classes, demonstrating that landmark-based approaches can match CNN performance at a fraction of the computational cost.

### 8.2 Future Scope

| Enhancement | Description |
|---|---|
| **Dynamic Gesture Recognition** | Use LSTM/RNN models to recognize motion-based signs (e.g., "thank you", "please") |
| **Two-Handed Signs** | Extend MediaPipe to detect both hands and classify two-handed ASL signs |
| **Word-Level Prediction** | Use NLP models to predict entire words from letter sequences (autocomplete) |
| **Mobile App** | Port the system to a mobile app using TensorFlow Lite for on-device inference |
| **ISL Support** | Extend the dataset to include Indian Sign Language (ISL) gestures |
| **Larger Dataset** | Train on 5,000+ images per class for higher accuracy and generalization |
| **WebSocket Streaming** | Replace HTTP polling with WebSocket for lower-latency video streaming |
| **User Authentication** | Add login system to save personalized sentence history |

---

## 9. References

1. F. Zhang, V. Bazarevsky, A. Vakunov, et al., "MediaPipe Hands: On-device Real-time Hand Tracking," *arXiv preprint arXiv:2006.10214*, 2020.

2. M. Abadi, P. Barham, J. Chen, et al., "TensorFlow: A System for Large-Scale Machine Learning," *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, pp. 265–283, 2016.

3. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4510–4520, 2018.

4. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," *International Conference on Learning Representations (ICLR)*, 2015.

5. S. Amin, "American Sign Language Recognition using Deep Learning and Computer Vision," *International Journal of Advanced Computer Science and Applications*, vol. 11, no. 8, 2020.

6. Google, "MediaPipe Solutions Documentation," Available: https://developers.google.com/mediapipe

7. Pallets Projects, "Flask Documentation," Available: https://flask.palletsprojects.com/

8. World Health Organization, "Deafness and Hearing Loss Fact Sheet," Available: https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss

---

> **Project Repository:** https://github.com/sudheshna28/Sign_Language_Recognition_System
