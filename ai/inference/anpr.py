"""
TransitEye ANPR Pipeline
RF-DETR-S license plate detection + PaddleOCR recognition

Pipeline:
Image
  -> RF-DETR-S
  -> license_plate bounding box
  -> plate crop
  -> PaddleOCR
  -> registration number
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
from paddleocr import PaddleOCR
from rfdetr import RFDETRSmall


# ---------------------------------------------------------
# DEFAULT PATHS
# ---------------------------------------------------------

DEFAULT_MODEL = Path("output/anpr_rfdetr_s/checkpoint_best_total.pth")
DEFAULT_IMAGE = Path("test_images/bus.jpeg")
DEFAULT_OUTPUT = Path("outputs/anpr_result.jpg")
DEFAULT_CROP_DIR = Path("outputs/anpr_crops")


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_plate_text(text: str) -> str:
    """
    Keep only letters and numbers and convert to uppercase.
    Spaces are removed from the final registration number.
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# ---------------------------------------------------------
# OCR RESULT EXTRACTION
# ---------------------------------------------------------

def extract_ocr_text(results) -> list[str]:
    """
    Extract recognized text strings from PaddleOCR 3.x results.
    """
    texts: list[str] = []

    for result in results:
        # PaddleOCR 3.x returns a result object/dict-like structure.
        try:
            data = result.json
            if callable(data):
                data = data()
        except Exception:
            data = None

        if isinstance(data, dict):
            res = data.get("res", data)

            rec_texts = res.get("rec_texts", [])
            if rec_texts:
                texts.extend(str(t) for t in rec_texts)

        # Fallback for object-style access
        try:
            rec_texts = result.get("rec_texts", [])
            if rec_texts:
                texts.extend(str(t) for t in rec_texts)
        except Exception:
            pass

    return texts


# ---------------------------------------------------------
# ANPR PIPELINE
# ---------------------------------------------------------

def run_anpr(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    crop_dir: Path,
    detection_threshold: float = 0.20,
) -> None:

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"ANPR model checkpoint not found: {model_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRANSITEYE ANPR")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. LOAD RF-DETR MODEL
    # -----------------------------------------------------

    print("\n[1/4] Loading RF-DETR-S ANPR model...")

    detector = RFDETRSmall(
        pretrain_weights=str(model_path),
        num_classes=1,
    )

    print("RF-DETR-S loaded successfully.")

    # -----------------------------------------------------
    # 2. DETECT LICENSE PLATES
    # -----------------------------------------------------

    print("\n[2/4] Detecting license plates...")

    detections = detector.predict(
        str(image_path),
        threshold=detection_threshold,
    )

    detection_count = len(detections.xyxy)

    print(f"License plates detected: {detection_count}")

    # Load original image
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not open image: {image_path}")

    image_height, image_width = image.shape[:2]

    # -----------------------------------------------------
    # NO DETECTIONS
    # -----------------------------------------------------

    if detection_count == 0:
        print("No license plate detected.")
        cv2.imwrite(str(output_path), image)
        print(f"Saved: {output_path}")
        return

    # -----------------------------------------------------
    # 3. PROCESS EVERY DETECTED PLATE
    # -----------------------------------------------------

    print("\n[3/4] Cropping plates and running OCR...")

    ocr = PaddleOCR(lang="en")

    recognized_plates: list[str] = []

    for index, (box, confidence) in enumerate(
        zip(detections.xyxy, detections.confidence),
        start=1,
    ):

        x1, y1, x2, y2 = map(int, box)

        # Add a small padding around the plate.
        pad_x = max(5, int((x2 - x1) * 0.05))
        pad_y = max(5, int((y2 - y1) * 0.10))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y)

        # Crop plate
        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            print(f"Plate {index}: invalid crop, skipping.")
            continue

        crop_path = crop_dir / f"plate_{index}.jpg"
        cv2.imwrite(str(crop_path), plate_crop)

        print(
            f"Plate {index}: "
            f"confidence={float(confidence):.3f}, "
            f"crop={crop_path}"
        )

        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

        results = ocr.predict(str(crop_path))
        raw_texts = extract_ocr_text(results)

        if raw_texts:
            # Use first recognized text for this plate.
            raw_text = raw_texts[0]
            plate_text = clean_plate_text(raw_text)
        else:
            raw_text = ""
            plate_text = ""

        recognized_plates.append(plate_text)

        if plate_text:
            print(
                f"Plate {index}: OCR = {plate_text} "
                f"(raw: {raw_text})"
            )
        else:
            print(f"Plate {index}: OCR could not read text.")

        # -------------------------------------------------
        # DRAW DETECTION RESULT
        # -------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        label = (
            f"{plate_text} "
            f"{float(confidence):.2f}"
            if plate_text
            else f"license_plate {float(confidence):.2f}"
        )

        label_y = max(25, y1 - 10)

        cv2.putText(
            image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # 4. SAVE RESULT
    # -----------------------------------------------------

    cv2.imwrite(str(output_path), image)

    print("\n[4/4] ANPR complete.")
    print(f"Saved annotated image: {output_path}")

    print("\nRecognized plates:")
    for plate in recognized_plates:
        if plate:
            print(f"  {plate}")

    print("=" * 60)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="TransitEye RF-DETR-S + PaddleOCR ANPR"
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Input image",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="RF-DETR ANPR checkpoint",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Annotated output image",
    )

    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=DEFAULT_CROP_DIR,
        help="Directory for cropped license plates",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="RF-DETR detection confidence threshold",
    )

    args = parser.parse_args()

    run_anpr(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        crop_dir=args.crop_dir,
        detection_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()