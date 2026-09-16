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
DEFAULT_VIDEO_PATH = Path("test_videos/road.mp4")
DEFAULT_OUTPUT_PATH = Path("outputs/video/road_detected.mp4")


def run_video_inference(
    model_path: Path,
    video_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.65,
    frame_skip: int = 2,
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
            "         python ai/inference/video_pothole.py --model path/to/your_checkpoint.pth\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trained RF-DETR-S from {model_path}...")
    model = RFDETRSmall(
        pretrain_weights=str(model_path),
        num_classes=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0
    processed_frames = 0
    temp_dir = output_path.parent / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "_current_frame.jpg"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            if frame_number % frame_skip != 0:
                writer.write(frame)
                continue

            cv2.imwrite(str(temp_path), frame)

            detections = model.predict(
                str(temp_path),
                threshold=confidence_threshold,
            )

            for box, confidence in zip(
                detections.xyxy,
                detections.confidence,
            ):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = f"pothole {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            writer.write(frame)
            processed_frames += 1

            if processed_frames % 10 == 0 or frame_number == total_frames:
                print(f"Progress: frame {frame_number}/{total_frames} (processed: {processed_frames})")

    finally:
        cap.release()
        writer.release()
        if temp_path.exists():
            temp_path.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    print(f"\nDone! Processed {processed_frames} frames.")
    print(f"Saved video to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETR Pothole Detection on Video")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained checkpoint (default: output/pothole_rfdetr_s/checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to input video (default: test_videos/road.mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output annotated video (default: outputs/video/road_detected.mp4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Detection confidence threshold (default: 0.65)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process detection every N frames for speed (default: 2)",
    )

    args = parser.parse_args()
    run_video_inference(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        confidence_threshold=args.threshold,
        frame_skip=args.frame_skip,
    )


if __name__ == "__main__":
    main()