import json
import os

import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(APP_DIR, "dataset", "asl_dataset")
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42


def main():
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"Dataset not found at {DATASET_DIR}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    with open(LABELS_PATH, "w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)

    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=6)

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
