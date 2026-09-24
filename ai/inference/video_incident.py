"""
Incident / Accident Detection & Tracking on Video
Team Sentrix - SIH 2026
"""

import argparse
import sys
from pathlib import Path
import cv2

try:
    from rfdetr import RFDETRSmall
except ImportError:
    print("Error: 'rfdetr' package is not installed.", file=sys.stderr)
    sys.exit(1)

from ai.inference.tracker import IoUTracker

DEFAULT_MODEL_PATH = Path("output/incident_rfdetr_s/checkpoint_best_total.pth")
DEFAULT_VIDEO_PATH = Path("test_videos/road.mp4")
DEFAULT_OUTPUT_PATH = Path("outputs/video/incident_detected.mp4")


def run_incident_video(
    model_path: Path,
    video_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.50,
    frame_skip: int = 2,
) -> None:
    if not model_path.exists():
        print(f"Error: Incident model checkpoint not found at: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Incident RF-DETR-S model from {model_path}...")
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker = IoUTracker(iou_threshold=0.25)
    temp_dir = output_path.parent / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "_incident_frame.jpg"

    frame_number = 0
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
            detections = model.predict(str(temp_path), threshold=confidence_threshold)

            raw_dets = []
            for box, conf in zip(detections.xyxy, detections.confidence):
                x1, y1, x2, y2 = map(int, box)
                raw_dets.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_name": "accident",
                })

            tracked = tracker.update(raw_dets)
            for item in tracked:
                x1, y1, x2, y2 = item["box"]
                track_id = item.get("track_id", 1)
                color = (50, 50, 240)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                history = item.get("history", [])
                for h_idx in range(1, len(history)):
                    cv2.line(frame, history[h_idx - 1], history[h_idx], (0, 255, 255), 2)

                label = f"Incident #{track_id} {item['confidence']:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        if temp_path.exists():
            temp_path.unlink()

    print(f"Incident video tracking complete: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Incident Detection & Tracking on Video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--frame-skip", type=int, default=2)

    args = parser.parse_args()
    run_incident_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        confidence_threshold=args.threshold,
        frame_skip=args.frame_skip,
    )


if __name__ == "__main__":
    main()
