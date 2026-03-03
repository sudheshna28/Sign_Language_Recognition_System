import os
# Suppress noisy TensorFlow and MediaPipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import base64
import io
import json
import logging

import mediapipe as mp
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

# Suppress werkzeug request-level noise
logging.getLogger("werkzeug").setLevel(logging.WARNING)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(APP_DIR, "dataset", "asl_dataset")
MODEL_PATH = os.path.join(APP_DIR, "models", "landmark_model.h5")
LABELS_PATH = os.path.join(APP_DIR, "models", "labels.json")

NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data

    if not os.path.isdir(DATASET_DIR):
        return []

    labels = []
    for name in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, name)
        if os.path.isdir(path) and len(name) == 1 and name.isalnum():
            labels.append(name)

    return sorted(labels)


LABELS = load_labels()

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Loaded landmark model from {MODEL_PATH}")
    print(f"[OK] {len(LABELS)} labels: {LABELS}")
else:
    print(f"[!] Model not found at {MODEL_PATH}")
    print("    Run 'python train_landmarks.py' to train the model first.")

# Persistent MediaPipe Hands instance — much faster than creating per request
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


def decode_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def normalize_landmarks(landmarks):
    """Normalize landmarks: translate so wrist is origin, scale by hand size."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    coords = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords.flatten()


def detect_hand(image):
    """Detect hand and return normalized landmarks + bounding box."""
    height, width, _ = image.shape
    results = hands_detector.process(image)

    if not results.multi_hand_landmarks:
        return None, None

    hand_lms = results.multi_hand_landmarks[0]
    landmarks = hand_lms.landmark

    # Normalized feature vector
    features = normalize_landmarks(landmarks)

    # Bounding box for overlay
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = max(int(min(xs) * width) - 20, 0)
    y_min = max(int(min(ys) * height) - 20, 0)
    x_max = min(int(max(xs) * width) + 20, width - 1)
    y_max = min(int(max(ys) * height) + 20, height - 1)
    bbox = (x_min, y_min, x_max, y_max)

    return features, bbox


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", model_loaded=model is not None, labels=LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            jsonify(
                {
                    "error": "Model not loaded. Run 'python train_landmarks.py' first.",
                }
            ),
            400,
        )

    data = request.get_json(silent=True) or {}
    img_data = data.get("image")
    if not img_data:
        return jsonify({"error": "Missing image"}), 400

    try:
        image = decode_image(img_data)
        features, bbox = detect_hand(image)

        if features is None:
            return jsonify({"hand_detected": False})

        # Predict using landmark features
        input_data = np.expand_dims(features, axis=0)
        probs = model.predict(input_data, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        confidence = float(probs[idx])

        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [
            {
                "label": LABELS[i] if i < len(LABELS) else str(i),
                "confidence": float(probs[i]),
            }
            for i in top3_idx
        ]

        return jsonify(
            {
                "hand_detected": True,
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "top3": top3,
            }
        )
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "hand_detected": False}), 500

# Language code mapping: browser speech lang → Google Translate lang
LANG_MAP = {
    "en-US": "en",
    "hi-IN": "hi",
    "te-IN": "te",
}


@app.route("/translate", methods=["POST"])
def translate_text():
    """Translate text from English to the target language."""
    from deep_translator import GoogleTranslator

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    target_lang = data.get("lang", "en-US")

    if not text:
        return jsonify({"translated": ""})

    lang_code = LANG_MAP.get(target_lang, "en")

    if lang_code == "en":
        return jsonify({"translated": text})

    try:
        translated = GoogleTranslator(source="en", target=lang_code).translate(text)
        return jsonify({"translated": translated or text})
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return jsonify({"translated": text})


@app.route("/speak", methods=["POST"])
def speak_text():
    """Translate + generate TTS audio server-side using gTTS."""
    from deep_translator import GoogleTranslator
    from gtts import gTTS

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    target_lang = data.get("lang", "en-US")

    if not text:
        return jsonify({"error": "No text"}), 400

    lang_code = LANG_MAP.get(target_lang, "en")

    # Translate if not English
    speak_text = text
    if lang_code != "en":
        try:
            speak_text = GoogleTranslator(source="en", target=lang_code).translate(text) or text
        except Exception as e:
            print(f"[WARN] Translation failed, speaking English: {e}")

    # Generate audio with gTTS
    try:
        tts = gTTS(text=speak_text, lang=lang_code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return jsonify({
            "audio": f"data:audio/mp3;base64,{audio_b64}",
            "translated": speak_text,
        })
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[OK] Starting server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
