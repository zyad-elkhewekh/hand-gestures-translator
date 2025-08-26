from pathlib import Path
import os, json, random, shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path().resolve()
DATA_DIR = BASE_DIR / "data"
NUMBERS_DIR = DATA_DIR / "numbers"
NUMBERS_SPLIT = DATA_DIR / "numbers_split"
LETTERS_TRAIN = DATA_DIR / "letters" / "asl_alphabet_train"
LETTERS_TEST = DATA_DIR / "letters" / "asl_alphabet_test"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42

LETTERS_EPOCHS = 100
NUMBERS_EPOCHS = 60
PATIENCE_LETTERS = 15
PATIENCE_NUMBERS = 10


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(SEED)

print("Python:", os.sys.version.splitlines()[0])
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


def split_numbers(src=NUMBERS_DIR, out=NUMBERS_SPLIT, test_ratio=0.2, seed=SEED):
    src = Path(src)
    out = Path(out)
    if not src.exists():
        raise FileNotFoundError(f"Numbers folder not found: {src}")
    if (out / "train").exists() and (out / "test").exists():
        return out / "train", out / "test"
    random.seed(seed)
    train_dir = out / "train"
    test_dir = out / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        imgs = [p for p in cls_dir.iterdir() if p.is_file()]
        random.shuffle(imgs)
        split_at = int(len(imgs) * (1 - test_ratio))
        (train_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        (test_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        for p in imgs[:split_at]:
            shutil.copy(p, train_dir / cls_dir.name / p.name)
        for p in imgs[split_at:]:
            shutil.copy(p, test_dir / cls_dir.name / p.name)
    return train_dir, test_dir


def prepare_datasets(folder, image_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2, shuffle=True):
    folder = str(folder)
    ds_train = keras.utils.image_dataset_from_directory(
        folder,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=val_split,
        subset="training",
        seed=SEED,
        shuffle=shuffle,
    )
    ds_val = keras.utils.image_dataset_from_directory(
        folder,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        shuffle=False,
    )
    class_names = ds_train.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)
    return ds_train, ds_val, class_names


def get_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ], name="data_augment")

def build_basic_cnn(num_classes, input_shape=(*IMG_SIZE, 3), dropout=0.4):
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = get_augmentation()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"Basic_CNN_{num_classes}c")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train_letters(epochs=LETTERS_EPOCHS, val_split=0.2, patience=PATIENCE_LETTERS):
    if not LETTERS_TRAIN.exists():
        raise FileNotFoundError(f"Letters train folder not found: {LETTERS_TRAIN}")
    ds_train, ds_val, class_names = prepare_datasets(LETTERS_TRAIN, val_split=val_split)
    print("Letters classes:", class_names)
    counts = {c: len(list((Path(LETTERS_TRAIN) / c).glob('*'))) for c in class_names}
    print("Class counts (letters):", counts)
    model = build_basic_cnn(len(class_names))
    stamp = timestamp()
    best_path = MODELS_DIR / f"Basic_CNN_letters_best_{stamp}.keras"
    out_path = MODELS_DIR / f"Basic_CNN_letters_final_{stamp}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks)
    model.save(out_path)
    save_json(MODELS_DIR / f"letters_class_names_{stamp}.json", class_names)
    save_json(OUTPUTS_DIR / f"letters_run_meta_{stamp}.json", {
        "seed": SEED, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs_requested": epochs,
        "epochs_ran": len(history.history["loss"]), "patience": patience
    })
    print("Saved letters model to", out_path)
    if LETTERS_TEST.exists():
        print("Sanity-check predictions on asl_alphabet_test:")
        for img in sorted(LETTERS_TEST.iterdir()):
            if not img.is_file(): continue
            x = keras.preprocessing.image.load_img(img, target_size=IMG_SIZE)
            x = keras.preprocessing.image.img_to_array(x) / 255.0
            x = np.expand_dims(x, 0)
            pred = model.predict(x)
            print(img.name, "->", class_names[int(pred.argmax())])
    return history


def train_numbers(epochs=NUMBERS_EPOCHS, val_split=0.2, test_ratio=0.2, patience=PATIENCE_NUMBERS):
    if not NUMBERS_DIR.exists():
        raise FileNotFoundError(f"Numbers folder not found: {NUMBERS_DIR}")
    train_dir, test_dir = split_numbers(src=NUMBERS_DIR, out=NUMBERS_SPLIT, test_ratio=test_ratio, seed=SEED)
    ds_train, ds_val, class_names = prepare_datasets(train_dir, val_split=val_split)
    print("Numbers classes:", class_names)
    counts = {c: len(list((Path(train_dir) / c).glob('*'))) for c in class_names}
    print("Class counts (numbers):", counts)
    total = sum(counts.values())
    class_weight = None
    if min(counts.values()) > 0 and max(counts.values()) / min(counts.values()) > 1.5:
        class_weight = {i: total / (len(counts) * v) for i, v in enumerate(counts.values())}
        print("Using class_weight:", class_weight)
    model = build_basic_cnn(len(class_names))
    stamp = timestamp()
    best_path = MODELS_DIR / f"Basic_CNN_numbers_best_{stamp}.keras"
    out_path = MODELS_DIR / f"Basic_CNN_numbers_final_{stamp}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, class_weight=class_weight)
    model.save(out_path)
    save_json(MODELS_DIR / f"numbers_class_names_{stamp}.json", class_names)
    save_json(OUTPUTS_DIR / f"numbers_run_meta_{stamp}.json", {
        "seed": SEED, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs_requested": epochs,
        "epochs_ran": len(history.history["loss"]), "patience": patience
    })
    print("Saved numbers model to", out_path)
    correct = 0
    total_imgs = 0
    for cls in sorted(test_dir.iterdir()):
        if not cls.is_dir(): continue
        for img in cls.iterdir():
            if not img.is_file(): continue
            x = keras.preprocessing.image.load_img(img, target_size=IMG_SIZE)
            x = keras.preprocessing.image.img_to_array(x) / 255.0
            x = np.expand_dims(x, 0)
            pred = model.predict(x)
            pred_label = class_names[int(pred.argmax())]
            if pred_label == cls.name:
                correct += 1
            total_imgs += 1
    acc = correct / total_imgs if total_imgs else 0
    print(f"Numbers split test accuracy: {acc:.4f} ({correct}/{total_imgs})")
    return history


set_seeds(SEED)
print("Starting training sequence...")

hist_letters = train_letters(epochs=LETTERS_EPOCHS, val_split=0.2, patience=PATIENCE_LETTERS)
hist_numbers = train_numbers(epochs=NUMBERS_EPOCHS, val_split=0.2, test_ratio=0.2, patience=PATIENCE_NUMBERS)

print("Both trainings complete.")
