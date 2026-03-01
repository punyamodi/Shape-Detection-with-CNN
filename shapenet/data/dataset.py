import os
import zipfile
import numpy as np
import cv2
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from shapenet.config import Config


def extract_dataset(zip_path: str, extract_dir: str) -> str:
    if not os.path.exists(extract_dir):
        parent = os.path.dirname(extract_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(parent if parent else ".")
    return extract_dir


def load_images(
    base_dir: str,
    classes: List[str],
    image_size: Tuple[int, int],
    channels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    h, w = image_size
    read_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath, read_flag)
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            if channels == 1:
                img = img.reshape(h, w, 1)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            labels.append(idx)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)


def _resolve_base_dir(cfg: Config) -> str:
    extract_dataset(cfg.data.zip_path, cfg.data.extract_dir)
    base_dir = cfg.data.extract_dir
    if not any(
        os.path.isdir(os.path.join(base_dir, c)) for c in cfg.data.classes
    ):
        subdirs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        if subdirs:
            base_dir = os.path.join(base_dir, subdirs[0])
    return base_dir


def build_splits(
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_dir = _resolve_base_dir(cfg)
    X, y = load_images(base_dir, cfg.data.classes, cfg.data.image_size, cfg.data.channels)
    X = X / 255.0
    y_cat = to_categorical(y, num_classes=len(cfg.data.classes))

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y_cat,
        test_size=cfg.data.test_size,
        stratify=y,
        random_state=cfg.data.random_state,
    )

    y_train_full_raw = np.argmax(y_train_full, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=cfg.data.val_size,
        stratify=y_train_full_raw,
        random_state=cfg.data.random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
