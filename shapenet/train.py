import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam

from shapenet.config import Config
from shapenet.data.augment import AugmentedSequence
from shapenet.data.dataset import build_splits
from shapenet.model.cnn import build_model


def get_callbacks(cfg: Config) -> list:
    model_path = os.path.join(cfg.output.model_dir, cfg.output.model_name)
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.training.reduce_lr_factor,
            patience=cfg.training.reduce_lr_patience,
            min_lr=cfg.training.min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=cfg.output.logs_dir),
    ]


def plot_history(history, plots_dir: str) -> str:
    metrics_df = pd.DataFrame(history.history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics_df["loss"], label="train")
    axes[0].plot(metrics_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(metrics_df["accuracy"], label="train")
    axes[1].plot(metrics_df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def train(cfg: Config):
    cfg.ensure_dirs()

    print("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = build_splits(cfg)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    input_shape = X_train.shape[1:]
    num_classes = len(cfg.data.classes)

    model = build_model(input_shape, num_classes, cfg.model)
    model.summary()

    optimizer = Adam(learning_rate=cfg.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_seq = AugmentedSequence(X_train, y_train, cfg.data.batch_size, augment=True)
    val_seq = AugmentedSequence(X_val, y_val, cfg.data.batch_size, augment=False)

    history = model.fit(
        train_seq,
        epochs=cfg.training.epochs,
        validation_data=val_seq,
        callbacks=get_callbacks(cfg),
    )

    plot_path = plot_history(history, cfg.output.plots_dir)
    print(f"Training curves saved to {plot_path}")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%  |  Test loss: {loss:.4f}")

    return model, history
