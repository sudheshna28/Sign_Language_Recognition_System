"""Extract MediaPipe hand landmarks from dataset images and train a Dense classifier."""

import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(APP_DIR, "dataset", "asl_dataset")
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "landmark_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3  # x, y, z per landmark = 63


def normalize_landmarks(landmarks):
    """Normalize landmarks: translate so wrist is origin, scale by hand size."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    coords = coords - wrist  # center on wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist  # scale to unit
    return coords.flatten()  # shape (63,)


def extract_dataset():
    """Walk the dataset and extract landmark features + labels."""
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"Dataset not found at {DATASET_DIR}")

    class_names = sorted(
        n for n in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, n))
        and len(n) == 1 and n.isalnum()
    )
    print(f"Found {len(class_names)} classes: {class_names}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

    X, y = [], []
    total = 0
    detected = 0

    for label_idx, label in enumerate(class_names):
        label_dir = os.path.join(DATASET_DIR, label)
        files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        label_detected = 0

        for fname in files:
            fpath = os.path.join(label_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            total += 1
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                features = normalize_landmarks(lm)
                X.append(features)
                y.append(label_idx)
                detected += 1
                label_detected += 1

        print(f"  [{label}] {label_detected}/{len(files)} hands detected")

    hands.close()
    print(f"\nTotal: {detected}/{total} images with hands detected ({detected/total*100:.1f}%)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), class_names


def build_model(num_classes):
    """Build a small Dense classifier for 63 landmark features."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURES,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    import tensorflow as tf

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Extracting landmarks from dataset...")
    X, y, class_names = extract_dataset()

    if len(X) == 0:
        raise SystemExit("No landmarks extracted! Check that your dataset has visible hand images.")

    # Save labels
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved {len(class_names)} labels to {LABELS_PATH}")

    # Shuffle and split
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Build and train
    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
    )

    # Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {acc:.2%}")

    model.save(MODEL_PATH)
    print(f"Saved landmark model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
