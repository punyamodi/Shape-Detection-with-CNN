import os
from typing import Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from shapenet.config import Config


class Predictor:
    def __init__(self, cfg: Config, model=None):
        self.cfg = cfg
        self.classes = cfg.data.classes
        self.image_size = cfg.data.image_size
        self.channels = cfg.data.channels

        if model is not None:
            self.model = model
        else:
            model_path = os.path.join(cfg.output.model_dir, cfg.output.model_name)
            self.model = load_model(model_path)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        resized = cv2.resize(img, (w, h))
        if self.channels == 1:
            resized = resized.reshape(h, w, 1)
        arr = resized.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict_image(self, image_path: str) -> Tuple[str, float]:
        read_flag = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(image_path, read_flag)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        processed = self._preprocess(img)
        proba = self.model.predict(processed, verbose=0)[0]
        idx = int(np.argmax(proba))
        return self.classes[idx], float(proba[idx])

    def predict_frame(self, frame: np.ndarray) -> Tuple[str, float, np.ndarray]:
        if self.channels == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        processed = self._preprocess(gray if self.channels == 1 else frame)
        proba = self.model.predict(processed, verbose=0)[0]
        idx = int(np.argmax(proba))
        label = self.classes[idx]
        confidence = float(proba[idx])

        h, w = frame.shape[:2]
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (0, h - 30), (bar_width, h), (0, 200, 0), -1)
        cv2.putText(
            frame,
            f"{label}  {confidence:.1%}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        return label, confidence, frame

    def run_webcam(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")
        print("Press q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            label, confidence, annotated = self.predict_frame(frame.copy())
            cv2.imshow("ShapeNet  q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
