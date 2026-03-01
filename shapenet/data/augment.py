from typing import Tuple

import numpy as np
import cv2
from tensorflow.keras.utils import Sequence


class AugmentedSequence(Sequence):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        augment: bool = False,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx]

        if self.augment:
            X_batch = self._augment_batch(X_batch)

        return X_batch, y_batch

    def _augment_batch(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for img in batch:
            img = self._random_flip(img)
            img = self._random_rotate(img)
            img = self._random_shift(img)
            out.append(img)
        return np.array(out)

    def _random_flip(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]
        if np.random.rand() > 0.5:
            img = img[::-1, :, :]
        return img

    def _random_rotate(self, img: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-180, 180)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img.squeeze(), M, (w, h))
        return rotated.reshape(img.shape)

    def _random_shift(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        dx = int(np.random.uniform(-0.15 * w, 0.15 * w))
        dy = int(np.random.uniform(-0.15 * h, 0.15 * h))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img.squeeze(), M, (w, h))
        return shifted.reshape(img.shape)
