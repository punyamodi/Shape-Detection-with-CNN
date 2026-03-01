from dataclasses import dataclass, field
from typing import List, Tuple
import yaml
import os


@dataclass
class DataConfig:
    zip_path: str = "shapes.zip"
    extract_dir: str = "data/shapes"
    image_size: Tuple[int, int] = (200, 200)
    channels: int = 1
    test_size: float = 0.20
    val_size: float = 0.15
    random_state: int = 42
    batch_size: int = 32
    classes: List[str] = field(default_factory=lambda: ["circle", "square", "star", "triangle"])


@dataclass
class ModelConfig:
    filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.5
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.001
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class OutputConfig:
    model_dir: str = "models"
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    model_name: str = "shapenet.h5"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "data" in raw:
            d = raw["data"]
            if "image_size" in d:
                d["image_size"] = tuple(d["image_size"])
            cfg.data = DataConfig(**d)
        if "model" in raw:
            cfg.model = ModelConfig(**raw["model"])
        if "training" in raw:
            cfg.training = TrainingConfig(**raw["training"])
        if "output" in raw:
            cfg.output = OutputConfig(**raw["output"])
        return cfg

    def ensure_dirs(self):
        os.makedirs(self.output.model_dir, exist_ok=True)
        os.makedirs(self.output.logs_dir, exist_ok=True)
        os.makedirs(self.output.plots_dir, exist_ok=True)
