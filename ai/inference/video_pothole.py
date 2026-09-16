from pathlib import Path

import cv2
from rfdetr import RFDETRSmall


MODEL_PATH = Path("output/pothole_rfdetr_s/checkpoint_best_total.pth")
VIDEO_PATH = Path("test_videos/road.mp4")
OUTPUT_PATH = Path("outputs/video/road_detected.mp4")

CONFIDENCE_THRESHOLD = 0.65
FRAME_SKIP = 2


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading trained RF-DETR-S...")
    model = RFDETRSmall(
        pretrain_weights=str(MODEL_PATH),
        num_classes=1,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number % FRAME_SKIP != 0:
            writer.write(frame)
            continue

        temp_path = Path("outputs/video/_current_frame.jpg")
        cv2.imwrite(str(temp_path), frame)

        detections = model.predict(
            str(temp_path),
            threshold=CONFIDENCE_THRESHOLD,
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

        if processed_frames % 20 == 0:
            print(
                f"Processed {processed_frames} frames "
                f"(source frame {frame_number}/{total_frames})"
            )

    cap.release()
    writer.release()

    if temp_path.exists():
        temp_path.unlink()

    print(f"\nSaved video: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()