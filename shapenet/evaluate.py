import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

from shapenet.config import Config
from shapenet.data.dataset import build_splits


def evaluate(cfg: Config, model=None):
    cfg.ensure_dirs()

    _, _, X_test, _, _, y_test = build_splits(cfg)

    if model is None:
        model_path = os.path.join(cfg.output.model_dir, cfg.output.model_name)
        model = load_model(model_path)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=cfg.data.classes))

    _plot_confusion_matrix(y_true, y_pred, cfg.data.classes, cfg.output.plots_dir)

    return acc, y_true, y_pred


def _plot_confusion_matrix(y_true, y_pred, classes, plots_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {path}")
