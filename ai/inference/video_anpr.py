"""
TransitEye Video ANPR
RF-DETR-S license plate detection + IoU tracking + PaddleOCR

Pipeline:

Video
  ↓
RF-DETR-S
  ↓
Confidence filtering
  ↓
Plate-shape filtering
  ↓
IoU tracking
  ↓
Plate crop
  ↓
PaddleOCR
  ↓
Registration number
  ↓
Annotated output video
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import cv2
from paddleocr import PaddleOCR
from rfdetr import RFDETRSmall

from ai.inference.tracker import IoUTracker


# =========================================================
# DEFAULT PATHS
# =========================================================

DEFAULT_MODEL = Path(
    "output/anpr_rfdetr_s/checkpoint_best_total.pth"
)

DEFAULT_VIDEO = Path(
    "test_videos/road.mp4"
)

DEFAULT_OUTPUT = Path(
    "outputs/video/anpr_detected.mp4"
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_plate_text(text: str) -> str:
    """
    Convert OCR result into a clean alphanumeric
    registration number.
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# =========================================================
# OCR RESULT EXTRACTION
# =========================================================

def extract_ocr_text(results: Any) -> list[str]:
    """
    Extract recognized text from PaddleOCR 3.x results.
    """

    texts: list[str] = []

    for result in results:

        data = None

        # PaddleOCR 3.x JSON-style result
        try:
            data = result.json

            if callable(data):
                data = data()

        except Exception:
            data = None

        if isinstance(data, dict):

            res = data.get(
                "res",
                data
            )

            rec_texts = res.get(
                "rec_texts",
                []
            )

            if rec_texts:
                texts.extend(
                    str(t)
                    for t in rec_texts
                )

        # Fallback
        try:

            rec_texts = result.get(
                "rec_texts",
                []
            )

            if rec_texts:
                texts.extend(
                    str(t)
                    for t in rec_texts
                )

        except Exception:
            pass

    return texts


# =========================================================
# PLATE GEOMETRY FILTER
# =========================================================

def is_valid_plate_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    """
    Reject detections that do not resemble a
    horizontal vehicle license plate.
    """

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    # Ignore tiny detections.
    if width < 40 or height < 12:
        return False

    # Ignore enormous regions.
    if width > frame_width * 0.60:
        return False

    if height > frame_height * 0.30:
        return False

    aspect_ratio = width / height

    # Normal plate-like boxes should be wider
    # than they are tall.
    #
    # Perspective can reduce this ratio, so
    # we intentionally don't make it too strict.
    if aspect_ratio < 1.5:
        return False

    # Reject extremely thin horizontal regions.
    if aspect_ratio > 8.0:
        return False

    return True


# =========================================================
# MAIN VIDEO ANPR
# =========================================================

def run_video_anpr(
    model_path: Path,
    video_path: Path,
    output_path: Path,
    detection_threshold: float = 0.20,
    frame_skip: int = 3,
    ocr_interval: int = 10,
) -> None:

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"ANPR model not found:\n{model_path}"
        )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("TRANSITEYE VIDEO ANPR")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD RF-DETR-S
    # -----------------------------------------------------

    print("\n[1/5] Loading RF-DETR-S ANPR model...")

    detector = RFDETRSmall(
        pretrain_weights=str(model_path),
        num_classes=1,
    )

    print("RF-DETR-S loaded successfully.")

    # -----------------------------------------------------
    # LOAD OCR
    # -----------------------------------------------------

    print("\n[2/5] Loading PaddleOCR...")

    ocr = PaddleOCR(
        lang="en"
    )

    print("PaddleOCR loaded successfully.")

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    print("\n[3/5] Opening video...")

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video: "
        f"{frame_width}x{frame_height} "
        f"@ {fps:.2f} FPS"
    )

    print(
        f"Total frames: {total_frames}"
    )

    # -----------------------------------------------------
    # CREATE OUTPUT VIDEO
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    # -----------------------------------------------------
    # TRACKER
    # -----------------------------------------------------

    print("\n[4/5] Starting tracker...")

    tracker = IoUTracker(
        iou_threshold=0.25,
        max_lost=15,
    )

    print("Tracker ready.")

    # -----------------------------------------------------
    # OCR CACHE
    # -----------------------------------------------------

    # Track ID -> recognized registration
    plate_text_by_track: dict[int, str] = {}

    # Track ID -> last frame OCR was attempted
    plate_last_ocr_frame: dict[int, int] = {}

    # -----------------------------------------------------
    # TEMPORARY DETECTION IMAGE
    # -----------------------------------------------------

    temp_frame_path = (
        output_path.parent
        / "_anpr_frame.jpg"
    )

    temp_plate_path = (
        output_path.parent
        / "_anpr_plate.jpg"
    )

    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    frame_number = 0
    processed_frames = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # -------------------------------------------------
            # FRAME SKIPPING
            # -------------------------------------------------

            if frame_number % frame_skip != 0:

                writer.write(frame)
                continue

            processed_frames += 1

            # -------------------------------------------------
            # SAVE CURRENT FRAME FOR RF-DETR
            # -------------------------------------------------

            cv2.imwrite(
                str(temp_frame_path),
                frame,
            )

            # -------------------------------------------------
            # RF-DETR DETECTION
            # -------------------------------------------------

            detections = detector.predict(
                str(temp_frame_path),
                threshold=detection_threshold,
            )

            # -------------------------------------------------
            # CONVERT TO TRACKER FORMAT
            # -------------------------------------------------

            raw_detections = []

            for box, confidence in zip(
                detections.xyxy,
                detections.confidence,
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box,
                )

                confidence = float(
                    confidence
                )

                # -------------------------------------------------
                # CONFIDENCE FILTER
                # -------------------------------------------------

                if confidence < detection_threshold:
                    continue

                # -------------------------------------------------
                # CLAMP BOX
                # -------------------------------------------------

                x1 = max(
                    0,
                    min(
                        x1,
                        frame_width - 1,
                    ),
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        frame_height - 1,
                    ),
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        frame_width,
                    ),
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        frame_height,
                    ),
                )

                # -------------------------------------------------
                # PLATE GEOMETRY FILTER
                # -------------------------------------------------

                if not is_valid_plate_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    frame_width,
                    frame_height,
                ):
                    continue

                raw_detections.append(
                    {
                        "box": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        "confidence": confidence,
                        "class_name": "license_plate",
                    }
                )

            # -------------------------------------------------
            # TRACK DETECTIONS
            # -------------------------------------------------

            tracked = tracker.update(
                raw_detections
            )

            # -------------------------------------------------
            # PROCESS TRACKED PLATES
            # -------------------------------------------------

            for detection in tracked:

                x1, y1, x2, y2 = detection[
                    "box"
                ]

                confidence = float(
                    detection[
                        "confidence"
                    ]
                )

                track_id = int(
                    detection[
                        "track_id"
                    ]
                )

                # -------------------------------------------------
                # CLAMP AGAIN
                # -------------------------------------------------

                x1 = max(
                    0,
                    min(
                        int(x1),
                        frame_width - 1,
                    ),
                )

                y1 = max(
                    0,
                    min(
                        int(y1),
                        frame_height - 1,
                    ),
                )

                x2 = max(
                    0,
                    min(
                        int(x2),
                        frame_width,
                    ),
                )

                y2 = max(
                    0,
                    min(
                        int(y2),
                        frame_height,
                    ),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # -------------------------------------------------
                # CHECK PLATE SHAPE AGAIN
                # -------------------------------------------------

                if not is_valid_plate_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    frame_width,
                    frame_height,
                ):
                    continue

                # -------------------------------------------------
                # OCR INTERVAL
                # -------------------------------------------------

                last_ocr_frame = plate_last_ocr_frame.get(
                    track_id,
                    -100000,
                )

                should_run_ocr = (
                    track_id not in plate_text_by_track
                    or (
                        frame_number
                        - last_ocr_frame
                        >= (
                            ocr_interval
                            * frame_skip
                        )
                    )
                )

                # -------------------------------------------------
                # RUN OCR
                # -------------------------------------------------

                if should_run_ocr:

                    plate_crop = frame[
                        y1:y2,
                        x1:x2,
                    ]

                    if plate_crop.size > 0:

                        # -----------------------------------------
                        # UPSCALE SMALL PLATES
                        # -----------------------------------------

                        crop_height, crop_width = (
                            plate_crop.shape[:2]
                        )

                        if crop_width < 200:

                            scale = 2.5

                            plate_crop = cv2.resize(
                                plate_crop,
                                None,
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_CUBIC,
                            )

                        # -----------------------------------------
                        # LIGHT IMAGE ENHANCEMENT
                        # -----------------------------------------

                        gray = cv2.cvtColor(
                            plate_crop,
                            cv2.COLOR_BGR2GRAY,
                        )

                        gray = cv2.bilateralFilter(
                            gray,
                            7,
                            50,
                            50,
                        )

                        enhanced = cv2.cvtColor(
                            gray,
                            cv2.COLOR_GRAY2BGR,
                        )

                        cv2.imwrite(
                            str(temp_plate_path),
                            enhanced,
                        )

                        try:

                            results = ocr.predict(
                                str(temp_plate_path)
                            )

                            raw_texts = (
                                extract_ocr_text(
                                    results
                                )
                            )

                            # -------------------------------------
                            # CHOOSE BEST OCR RESULT
                            # -------------------------------------

                            best_text = ""

                            best_length = 0

                            for raw_text in raw_texts:

                                cleaned = (
                                    clean_plate_text(
                                        raw_text
                                    )
                                )

                                # Ignore very short garbage.
                                if len(cleaned) < 4:
                                    continue

                                # Prefer longer readings.
                                if len(cleaned) > best_length:

                                    best_text = cleaned
                                    best_length = len(
                                        cleaned
                                    )

                            if best_text:

                                plate_text_by_track[
                                    track_id
                                ] = best_text

                                print(
                                    f"Frame {frame_number} | "
                                    f"Track {track_id} | "
                                    f"Plate: {best_text} | "
                                    f"Detection confidence: "
                                    f"{confidence:.3f}"
                                )

                            plate_last_ocr_frame[
                                track_id
                            ] = frame_number

                        except Exception as exc:

                            print(
                                f"OCR error "
                                f"(track {track_id}): "
                                f"{exc}",
                                file=sys.stderr,
                            )

                # -------------------------------------------------
                # DISPLAY LABEL
                # -------------------------------------------------

                plate_text = (
                    plate_text_by_track.get(
                        track_id
                    )
                )

                if plate_text:

                    label = (
                        f"{plate_text} "
                        f"ID:{track_id} "
                        f"{confidence:.2f}"
                    )

                else:

                    label = (
                        f"PLATE "
                        f"ID:{track_id} "
                        f"{confidence:.2f}"
                    )

                # -------------------------------------------------
                # DRAW BOX
                # -------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3,
                )

                # -------------------------------------------------
                # DRAW LABEL
                # -------------------------------------------------

                label_y = max(
                    30,
                    y1 - 10,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # -------------------------------------------------
            # STATUS OVERLAY
            # -------------------------------------------------

            status = (
                f"ANPR | "
                f"Frame {frame_number}/{total_frames}"
            )

            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # WRITE FRAME
            # -------------------------------------------------

            writer.write(frame)

            # -------------------------------------------------
            # PROGRESS
            # -------------------------------------------------

            if processed_frames % 20 == 0:

                print(
                    f"Processed "
                    f"{processed_frames} frames "
                    f"(source frame "
                    f"{frame_number}/{total_frames})"
                )

    finally:

        cap.release()
        writer.release()

        # Remove temporary files
        try:
            if temp_frame_path.exists():
                temp_frame_path.unlink()

            if temp_plate_path.exists():
                temp_plate_path.unlink()

        except Exception:
            pass

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print("\n" + "=" * 70)
    print("VIDEO ANPR COMPLETE")
    print("=" * 70)

    print(
        f"Output: {output_path}"
    )

    print(
        f"Frames processed: {processed_frames}"
    )

    if plate_text_by_track:

        print("\nRecognized plates:")

        for track_id, plate in sorted(
            plate_text_by_track.items()
        ):

            print(
                f"  Track {track_id}: {plate}"
            )

    else:

        print(
            "\nNo readable license plates were found."
        )

    print("=" * 70)


# =========================================================
# COMMAND LINE
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "TransitEye RF-DETR-S + "
            "PaddleOCR Video ANPR"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to ANPR RF-DETR checkpoint",
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input traffic video",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output annotated video",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="RF-DETR confidence threshold",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Process every Nth frame",
    )

    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=10,
        help="OCR interval in processed frames",
    )

    args = parser.parse_args()

    run_video_anpr(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        detection_threshold=args.threshold,
        frame_skip=args.frame_skip,
        ocr_interval=args.ocr_interval,
    )


if __name__ == "__main__":
    main()