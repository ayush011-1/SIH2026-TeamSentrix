from pathlib import Path

import cv2
from rfdetr import RFDETRSmall


MODEL_PATH = Path("output/pothole_rfdetr_s/checkpoint_best_total.pth")
IMAGE_PATH = Path("test_custom/pothole.png")
OUTPUT_PATH = Path("outputs/pothole_result.jpg")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading trained RF-DETR-S...")

    model = RFDETRSmall(
        pretrain_weights=str(MODEL_PATH),
        num_classes=1,
    )

    print("Running pothole detection...")

    detections = model.predict(
        str(IMAGE_PATH),
        threshold=0.65,
    )

    image = cv2.imread(str(IMAGE_PATH))

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

    cv2.imwrite(str(OUTPUT_PATH), image)

    print(f"Detections: {len(detections.xyxy)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()