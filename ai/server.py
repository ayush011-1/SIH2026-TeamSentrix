"""
FastAPI Server for RF-DETR Model Testing Frontend.
Team Sentrix - SIH 2026
"""

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Try importing RF-DETR
try:
    from rfdetr import RFDETRSmall
except ImportError:
    RFDETRSmall = None

BASE_DIR = Path(__file__).resolve().parent.parent
POTHOLE_MODEL_PATH = BASE_DIR / "output" / "pothole_rfdetr_s" / "checkpoint_best_total.pth"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Sentrix AI Model Testing Platform",
    description="Interactive Web Testing Interface for RF-DETR Models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache to avoid reloading weights on every single inference request
_MODEL_CACHE: Dict[str, Any] = {}


def get_device() -> str:
    """Detect available hardware acceleration."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str):
    """Retrieve or load model into singleton cache."""
    if RFDETRSmall is None:
        raise HTTPException(
            status_code=500,
            detail="rfdetr package is not installed in the current environment.",
        )

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    device = get_device()

    if model_name == "pothole":
        if not POTHOLE_MODEL_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Pothole model weights not found at {POTHOLE_MODEL_PATH}. "
                    "Train the model using 'python ai/train.py' or download 'checkpoint_best_total.pth' "
                    "into 'output/pothole_rfdetr_s/'."
                ),
            )
        print(f"[Sentrix API] Loading trained pothole model from {POTHOLE_MODEL_PATH} ({device})...")
        model = RFDETRSmall(
            pretrain_weights=str(POTHOLE_MODEL_PATH),
            num_classes=1,
            device=device,
        )
        _MODEL_CACHE["pothole"] = model
        return model

    elif model_name == "coco":
        print(f"[Sentrix API] Loading base COCO RF-DETR model ({device})...")
        model = RFDETRSmall(device=device)
        _MODEL_CACHE["coco"] = model
        return model

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_name '{model_name}'. Choose 'pothole' or 'coco'.",
        )


@app.get("/api/status")
async def get_system_status():
    """System health check, available hardware, and model readiness."""
    device = get_device()
    pothole_ready = POTHOLE_MODEL_PATH.exists()
    pothole_size_mb = (
        round(POTHOLE_MODEL_PATH.stat().st_size / (1024 * 1024), 2)
        if pothole_ready
        else 0
    )

    return {
        "status": "online",
        "device": device,
        "device_name": (
            "Apple Silicon Metal Performance Shaders (MPS)"
            if device == "mps"
            else ("NVIDIA CUDA GPU" if device == "cuda" else "CPU")
        ),
        "models": {
            "pothole": {
                "available": pothole_ready,
                "path": str(POTHOLE_MODEL_PATH.relative_to(BASE_DIR)),
                "size_mb": pothole_size_mb,
                "cached_in_memory": "pothole" in _MODEL_CACHE,
                "classes": ["pothole"],
            },
            "coco": {
                "available": True,
                "description": "Base RF-DETR (80 COCO classes: car, bus, truck, person, etc.)",
                "cached_in_memory": "coco" in _MODEL_CACHE,
            },
        },
    }


@app.get("/api/samples")
async def get_sample_files():
    """List available sample images and videos for rapid 1-click testing."""
    samples = {
        "images": [],
        "videos": [],
    }

    # Sample images
    for dir_name in ["test_custom", "test_images"]:
        p = BASE_DIR / dir_name
        if p.exists():
            for f in p.glob("*.*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    samples["images"].append({
                        "name": f.name,
                        "path": str(f.relative_to(BASE_DIR)),
                        "folder": dir_name,
                        "url": f"/samples/{dir_name}/{f.name}",
                    })

    # Sample videos
    video_dir = BASE_DIR / "test_videos"
    if video_dir.exists():
        for f in video_dir.glob("*.*"):
            if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                samples["videos"].append({
                    "name": f.name,
                    "path": str(f.relative_to(BASE_DIR)),
                    "url": f"/samples/test_videos/{f.name}",
                })

    return samples


@app.post("/api/detect/image")
async def detect_image(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None),
    model_name: str = Form("pothole"),
    threshold: float = Form(0.50),
):
    """Run object detection on an image (uploaded file or predefined sample)."""
    image_bytes = None

    if file is not None and file.filename:
        image_bytes = await file.read()
    elif sample_path:
        sample_file = BASE_DIR / sample_path
        if not sample_file.exists():
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_path}")
        image_bytes = sample_file.read_bytes()
    else:
        raise HTTPException(status_code=400, detail="Provide an image upload or sample_path.")

    # Decode image with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    height, width, _ = image.shape

    # Save to a temporary file for RF-DETR model.predict(path)
    temp_dir = OUTPUTS_DIR / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_input = temp_dir / f"input_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(temp_input), image)

    model = load_model(model_name)

    start_time = time.perf_counter()
    try:
        detections = model.predict(str(temp_input), threshold=threshold)
    finally:
        if temp_input.exists():
            temp_input.unlink()

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    annotated = image.copy()
    detection_items = []

    # Map detections
    for i, (box, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
        x1, y1, x2, y2 = map(int, box)

        if model_name == "coco" and hasattr(detections, "data") and "class_name" in detections.data:
            class_name = detections.data["class_name"][i]
        else:
            class_name = "pothole"

        detection_items.append({
            "id": i + 1,
            "class_name": class_name,
            "confidence": round(float(confidence), 3),
            "confidence_percent": round(float(confidence) * 100, 1),
            "box": [x1, y1, x2, y2],
            "width": x2 - x1,
            "height": y2 - y1,
        })

        # Draw bounding box & label
        color = (0, 230, 118) if class_name == "pothole" else (59, 130, 246)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{class_name} {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Label background badge
        cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + w + 10, max(25, y1)), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

    # Encode annotated image to base64 JPEG
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "model_used": model_name,
        "latency_ms": latency_ms,
        "fps_estimate": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
        "image_width": width,
        "image_height": height,
        "total_detections": len(detection_items),
        "detections": detection_items,
        "image_base64": f"data:image/jpeg;base64,{base64_image}",
    }


@app.post("/api/detect/video")
async def detect_video(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None),
    model_name: str = Form("pothole"),
    threshold: float = Form(0.65),
    frame_skip: int = Form(2),
    max_seconds: int = Form(10),
):
    """Run pothole / hazard detection on video."""
    video_input_path = None
    cleanup_input = False

    if file is not None and file.filename:
        temp_dir = OUTPUTS_DIR / "_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_input_path = temp_dir / f"upload_{int(time.time())}_{file.filename}"
        with open(video_input_path, "wb") as f:
            f.write(await file.read())
        cleanup_input = True
    elif sample_path:
        video_input_path = BASE_DIR / sample_path
        if not video_input_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample video not found: {sample_path}")
    else:
        raise HTTPException(status_code=400, detail="Provide video upload or sample_path.")

    model = load_model(model_name)

    cap = cv2.VideoCapture(str(video_input_path))
    if not cap.isOpened():
        if cleanup_input and video_input_path.exists():
            video_input_path.unlink()
        raise HTTPException(status_code=400, detail="Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames_to_process = min(total_frames, int(fps * max_seconds))

    timestamp = int(time.time())
    output_filename = f"detected_video_{timestamp}.mp4"
    output_filepath = OUTPUTS_DIR / "video" / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Try avc1 first (for native HTML5 browser playback), fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_filepath), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_filepath), fourcc, fps, (width, height))

    temp_frame_path = OUTPUTS_DIR / "_temp" / f"frame_{timestamp}.jpg"
    temp_frame_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    frame_number = 0
    processed_count = 0
    total_detections_in_video = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_number >= max_frames_to_process:
                break

            frame_number += 1

            if frame_number % frame_skip != 0:
                writer.write(frame)
                continue

            cv2.imwrite(str(temp_frame_path), frame)
            detections = model.predict(str(temp_frame_path), threshold=threshold)

            total_detections_in_video += len(detections.xyxy)

            for box, confidence in zip(detections.xyxy, detections.confidence):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 118), 2)
                label = f"pothole {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 230, 118),
                    2,
                )

            writer.write(frame)
            processed_count += 1

    finally:
        cap.release()
        writer.release()
        if temp_frame_path.exists():
            temp_frame_path.unlink()
        if cleanup_input and video_input_path and video_input_path.exists():
            video_input_path.unlink()

    total_time = round(time.perf_counter() - start_time, 2)
    avg_fps = round(processed_count / total_time, 1) if total_time > 0 else 0

    return {
        "success": True,
        "video_url": f"/outputs/video/{output_filename}",
        "frames_total": frame_number,
        "frames_detected": processed_count,
        "total_detections": total_detections_in_video,
        "processing_time_sec": total_time,
        "avg_fps": avg_fps,
        "width": width,
        "height": height,
    }


# Static file mounts
if (BASE_DIR / "test_custom").exists():
    app.mount("/samples/test_custom", StaticFiles(directory=str(BASE_DIR / "test_custom")), name="samples_custom")
if (BASE_DIR / "test_images").exists():
    app.mount("/samples/test_images", StaticFiles(directory=str(BASE_DIR / "test_images")), name="samples_images")
if (BASE_DIR / "test_videos").exists():
    app.mount("/samples/test_videos", StaticFiles(directory=str(BASE_DIR / "test_videos")), name="samples_videos")
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_static")


@app.get("/")
async def serve_index():
    """Serve the testing frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend index.html not yet initialized"}


def run_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"\n🚀 Sentrix AI Model Testing Platform running at: http://{host}:{port}")
    uvicorn.run("ai.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
