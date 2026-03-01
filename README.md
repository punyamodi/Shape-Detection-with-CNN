# ShapeNet — Real-time Shape Recognition with CNN

> Detects **circles**, **squares**, **stars**, and **triangles** from images and live webcam feeds using a deep convolutional neural network trained to 99%+ accuracy.

---

## Overview

ShapeNet is a full-fledged computer vision pipeline for geometric shape recognition. It covers data loading and augmentation, model training with callbacks, evaluation with detailed metrics, single-image inference, real-time webcam detection, and Grad-CAM visualizations — all driven by a single YAML config.

### Model Architecture

```
Input (200×200×1)
       │
 ┌─────▼─────┐
 │  Conv2D   │  32 filters  → BN → MaxPool
 │  Conv2D   │  64 filters  → BN → MaxPool
 │  Conv2D   │  128 filters → BN → MaxPool
 │  Conv2D   │  256 filters → BN → MaxPool
 └─────┬─────┘
       │
 GlobalAveragePooling2D
       │
  Dense(256) → BN → Dropout(0.5)
  Dense(128) → BN → Dropout(0.5)
       │
  Dense(4, softmax)
       │
  {circle | square | star | triangle}
```

---

## Project Structure

```
Shape-Detection-with-CNN/
├── shapenet/
│   ├── config.py          # Dataclass-based config with YAML loading
│   ├── train.py           # Training loop with callbacks
│   ├── evaluate.py        # Metrics, confusion matrix, classification report
│   ├── inference.py       # Single-image and real-time webcam prediction
│   ├── gradcam.py         # Grad-CAM saliency maps
│   ├── data/
│   │   ├── dataset.py     # Dataset loading, extraction, and splits
│   │   └── augment.py     # Custom augmentation sequence
│   └── model/
│       └── cnn.py         # CNN architecture builder
├── scripts/
│   ├── train.py           # CLI entry point for training
│   ├── evaluate.py        # CLI entry point for evaluation
│   └── predict.py         # CLI entry point for inference
├── configs/
│   └── default.yaml       # Default hyperparameters and paths
├── tests/
│   ├── test_data.py       # Dataset loading tests
│   └── test_model.py      # Model architecture tests
├── shapes.zip             # Dataset: ~14,970 grayscale 200×200 images
├── requirements.txt
└── setup.py
```

---

## Dataset

The dataset contains **~14,970 grayscale images** (200×200 px) spread evenly across four classes:

| Class    | Count |
|----------|-------|
| Circle   | 3,720 |
| Square   | 3,765 |
| Star     | 3,765 |
| Triangle | 3,720 |

`shapes.zip` is extracted automatically on first run. No manual setup required.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python scripts/train.py --config configs/default.yaml
```

Override hyperparameters without editing the config:

```bash
python scripts/train.py --epochs 20 --batch-size 64 --lr 0.0005
```

### 3. Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml
```

Outputs:
- Test accuracy and loss
- Per-class precision, recall, F1
- Confusion matrix saved to `plots/confusion_matrix.png`

### 4. Predict — single image

```bash
python scripts/predict.py --image path/to/shape.png
```

With Grad-CAM overlay:

```bash
python scripts/predict.py --image path/to/shape.png --gradcam
```

### 5. Predict — live webcam

```bash
python scripts/predict.py --webcam
# or specify a camera index
python scripts/predict.py --webcam --camera 1
```

Press `q` to exit the webcam window.

---

## Configuration

All training and inference parameters live in `configs/default.yaml`:

```yaml
data:
  zip_path: shapes.zip
  image_size: [200, 200]
  channels: 1
  test_size: 0.20
  val_size: 0.15
  batch_size: 32
  classes: [circle, square, star, triangle]

model:
  filters: [32, 64, 128, 256]
  dense_units: [256, 128]
  dropout_rate: 0.5
  use_batch_norm: true

training:
  epochs: 30
  learning_rate: 0.001
  early_stopping_patience: 7
  reduce_lr_patience: 3
  reduce_lr_factor: 0.5

output:
  model_dir: models
  plots_dir: plots
  model_name: shapenet.h5
```

---

## Features

- **Config-driven** — all parameters in a single YAML file, overridable via CLI flags
- **Data augmentation** — random rotation (±180°), horizontal/vertical flips, translation
- **Training callbacks** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
- **Detailed evaluation** — classification report, confusion matrix heatmap
- **Grad-CAM** — visualize which regions the model focuses on for any prediction
- **Real-time webcam** — live shape detection with confidence bar overlay
- **Unit tests** — pytest suite covering data loading and model architecture

---

## Training Outputs

After training, the following artifacts are saved:

| Path | Description |
|------|-------------|
| `models/shapenet.h5` | Best checkpoint (highest val accuracy) |
| `plots/training_curves.png` | Loss and accuracy curves |
| `plots/confusion_matrix.png` | Confusion matrix heatmap |
| `logs/` | TensorBoard event files |

View TensorBoard logs:

```bash
tensorboard --logdir logs
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

- **TensorFlow / Keras** — model building and training
- **OpenCV** — image I/O, webcam capture, Grad-CAM overlay
- **scikit-learn** — train/val/test splits, evaluation metrics
- **seaborn / matplotlib** — confusion matrix and training curves
- **PyYAML** — config management

---

## License

MIT
