import os
import tempfile

import cv2
import numpy as np
import pytest

from shapenet.data.dataset import load_images


def _create_dummy_dataset(base_dir, classes, size=(200, 200), n=5):
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n):
            img = np.random.randint(0, 255, size, dtype=np.uint8)
            cv2.imwrite(os.path.join(cls_dir, f"{i}.png"), img)


def test_load_images_shape():
    classes = ["circle", "square"]
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_dummy_dataset(tmpdir, classes, size=(200, 200), n=10)
        X, y = load_images(tmpdir, classes, (200, 200), 1)
        assert X.shape == (20, 200, 200, 1)
        assert y.shape == (20,)


def test_load_images_labels():
    classes = ["circle", "square"]
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_dummy_dataset(tmpdir, classes, size=(200, 200), n=5)
        X, y = load_images(tmpdir, classes, (200, 200), 1)
        assert set(y.tolist()) == {0, 1}


def test_pixel_range():
    classes = ["triangle"]
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_dummy_dataset(tmpdir, classes, size=(200, 200), n=5)
        X, _ = load_images(tmpdir, classes, (200, 200), 1)
        assert float(X.max()) <= 255.0
        assert float(X.min()) >= 0.0


def test_missing_class_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_images(tmpdir, ["nonexistent"], (200, 200), 1)
