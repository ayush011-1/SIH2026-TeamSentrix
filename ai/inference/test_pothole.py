import argparse
import sys
from pathlib import Path

import cv2

try:
    from rfdetr import RFDETRSmall
except ImportError:
    print(
        "Error: 'rfdetr' package is not installed.\n"
        "Install required dependencies with: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_MODEL_PATH = Path("output/pothole_rfdetr_s/checkpoint_best_total.pth")
DEFAULT_IMAGE_PATH = Path("test_custom/pothole.png")
DEFAULT_OUTPUT_PATH = Path("outputs/pothole_result.jpg")


def run_inference(
    model_path: Path,
    image_path: Path,
    output_path: Path,
    threshold: float = 0.65,
) -> None:
    if not model_path.exists():
        print(
            f"\nError: Model checkpoint not found at: {model_path}\n\n"
            "To fix this, you can:\n"
            "  1. Train the model using the included dataset:\n"
            "         python ai/train.py --epochs 50\n"
            "  2. Or download 'checkpoint_best_total.pth' from GitHub Releases and place it in:\n"
            f"         {model_path}\n"
            "  3. Or specify a custom checkpoint:\n"
            "         python ai/inference/test_pothole.py --model path/to/your_checkpoint.pth\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trained RF-DETR-S from {model_path}...")
    model = RFDETRSmall(
        pretrain_weights=str(model_path),
        num_classes=1,
    )

    print(f"Running pothole detection on {image_path}...")
    detections = model.predict(
        str(image_path),
        threshold=threshold,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    for box, confidence in zip(
        detections.xyxy,
        detections.confidence,
    ):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"pothole {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(output_path), image)

    print(f"Detections found : {len(detections.xyxy)}")
    print(f"Annotated image  : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RF-DETR Pothole Detection on an Image")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained checkpoint (default: output/pothole_rfdetr_s/checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to input image (default: test_custom/pothole.png)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save output image (default: outputs/pothole_result.jpg)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Detection confidence threshold (default: 0.65)",
    )

    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()