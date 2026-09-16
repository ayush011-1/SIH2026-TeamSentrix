from pathlib import Path

import cv2
from rfdetr import RFDETRSmall


def main() -> None:
    image_path = Path("test_images/bus.jpeg")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Loading RF-DETR-S...")
    model = RFDETRSmall()

    print(f"Running detection on: {image_path}")
    detections = model.predict(
        str(image_path),
        threshold=0.5,
    )

    image = cv2.imread(str(image_path))

    print("\nDetection Results")
    print("-----------------")

    for i, (box, confidence, class_id) in enumerate(
        zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        ),
        start=1,
    ):
        class_name = detections.data["class_name"][i - 1]

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        print(
            f"{i}. {class_name} | "
            f"confidence={confidence:.3f}"
        )

    output_path = output_dir / "detection_result.jpg"

    cv2.imwrite(
        str(output_path),
        image,
    )

    print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()