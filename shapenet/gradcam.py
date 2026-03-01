import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


def compute_gradcam(model, img_array: np.ndarray, class_idx: int) -> np.ndarray:
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No Conv2D layers found in model")
    last_conv = conv_layers[-1]

    grad_model = Model(inputs=model.input, outputs=[last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if len(original_img.shape) == 2 or original_img.shape[-1] == 1:
        base = cv2.cvtColor(
            original_img.reshape(h, w),
            cv2.COLOR_GRAY2BGR,
        )
    else:
        base = original_img.copy()

    overlay = cv2.addWeighted(base, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
