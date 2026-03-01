import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapenet.config import Config
from shapenet.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description="Run ShapeNet inference")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--image", default=None, help="Path to image for single prediction")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run real-time webcam inference",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Save Grad-CAM overlay alongside prediction",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    predictor = Predictor(cfg)

    if args.webcam:
        predictor.run_webcam(camera_index=args.camera)

    elif args.image:
        if args.gradcam:
            import cv2
            import numpy as np
            from shapenet.gradcam import compute_gradcam, overlay_gradcam

            read_flag = cv2.IMREAD_GRAYSCALE if cfg.data.channels == 1 else cv2.IMREAD_COLOR
            img = cv2.imread(args.image, read_flag)
            if img is None:
                print(f"Cannot read image: {args.image}")
                sys.exit(1)

            h, w = cfg.data.image_size
            img_resized = cv2.resize(img, (w, h))
            arr = img_resized.reshape(h, w, cfg.data.channels).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, 0)

            label, confidence = predictor.predict_image(args.image)
            class_idx = cfg.data.classes.index(label)

            heatmap = compute_gradcam(predictor.model, arr, class_idx)
            overlay = overlay_gradcam(img_resized.reshape(h, w, cfg.data.channels), heatmap)

            os.makedirs(cfg.output.plots_dir, exist_ok=True)
            out_path = os.path.join(cfg.output.plots_dir, "gradcam_overlay.png")
            cv2.imwrite(out_path, overlay)
            print(f"Prediction: {label}  ({confidence:.1%} confidence)")
            print(f"Grad-CAM overlay saved to {out_path}")
        else:
            label, confidence = predictor.predict_image(args.image)
            print(f"Prediction: {label}  ({confidence:.1%} confidence)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
