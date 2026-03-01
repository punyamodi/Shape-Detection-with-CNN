import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapenet.config import Config
from shapenet.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShapeNet CNN model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--model", default=None, help="Override path to saved model")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.model:
        cfg.output.model_dir = os.path.dirname(os.path.abspath(args.model))
        cfg.output.model_name = os.path.basename(args.model)

    evaluate(cfg)


if __name__ == "__main__":
    main()
