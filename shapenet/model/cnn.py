from typing import List, Tuple

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from shapenet.config import ModelConfig


def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    cfg: ModelConfig,
) -> Sequential:
    model = Sequential(name="ShapeNet")

    for i, filters in enumerate(cfg.filters):
        kwargs = dict(
            filters=filters,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=l2(1e-4),
        )
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(Conv2D(**kwargs))
        if cfg.use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())

    for units in cfg.dense_units:
        model.add(Dense(units, activation="relu", kernel_regularizer=l2(1e-4)))
        if cfg.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(cfg.dropout_rate))

    model.add(Dense(num_classes, activation="softmax"))

    return model
