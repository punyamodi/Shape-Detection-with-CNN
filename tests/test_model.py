import numpy as np
import pytest

from shapenet.config import ModelConfig
from shapenet.model.cnn import build_model


def test_output_shape():
    cfg = ModelConfig(filters=[16, 32], dense_units=[64], dropout_rate=0.3, use_batch_norm=False)
    model = build_model((200, 200, 1), 4, cfg)
    x = np.random.rand(2, 200, 200, 1).astype(np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (2, 4)


def test_probabilities_sum_to_one():
    cfg = ModelConfig(filters=[16], dense_units=[32], dropout_rate=0.0, use_batch_norm=False)
    model = build_model((200, 200, 1), 4, cfg)
    x = np.random.rand(3, 200, 200, 1).astype(np.float32)
    out = model.predict(x, verbose=0)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3), atol=1e-5)


def test_output_classes():
    cfg = ModelConfig(filters=[16], dense_units=[32], dropout_rate=0.0, use_batch_norm=False)
    model = build_model((200, 200, 1), 4, cfg)
    assert model.output_shape == (None, 4)


def test_custom_num_classes():
    cfg = ModelConfig(filters=[16], dense_units=[32], dropout_rate=0.0, use_batch_norm=False)
    model = build_model((200, 200, 1), 6, cfg)
    assert model.output_shape == (None, 6)
