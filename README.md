# Sentrix: AI-Powered Road Hazard & Pothole Detection
> Smart India Hackathon (SIH 2026) | Team Sentrix

Real-time, transformer-based road hazard and pothole detection powered by **RF-DETR (Real-time Vision Transformer Object Detector)** with a built-in interactive web testing console.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Interactive Web Testing Dashboard](#interactive-web-testing-dashboard)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Multi-Hazard Extension (Accidents, Waterlogging, ANPR)](#multi-hazard-extension-accidents-waterlogging-anpr)
- [Model Weights & Checkpoints](#model-weights--checkpoints)
- [Running CLI Inference](#running-cli-inference)
  - [Image Detection](#image-detection)
  - [Video Detection](#video-detection)
  - [General Object Detection](#general-object-detection)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Overview
Sentrix utilizes RF-DETR-Small (DINOv2 backbone) fine-tuned on the Transiteye pothole dataset to identify road surface anomalies and potholes in real-time video feeds and static imagery.

Key highlights:
- **State-of-the-Art Architecture**: End-to-end DETR transformer without post-processing NMS bottlenecks.
- **Hardware Acceleration**: Out-of-the-box support for Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), and CPU.
- **Interactive Web Console**: Sleek dark-mode testing interface with drag-and-drop, sample selectors, live telemetry (latency & FPS), and detections inspection.
- **Multi-Hazard Capable**: Readily extensible to Accidents, Waterlogging, and License Plate Recognition (ANPR).

---

## Project Structure
```text
SIH2026-TeamSentrix/
├── app.py                           # Root launcher for Web Testing Console
├── ai/
│   ├── server.py                    # FastAPI backend with model caching
│   ├── train.py                     # Training entrypoint for RF-DETR
│   └── inference/
│       ├── test_pothole.py          # CLI image inference with pothole model
│       ├── video_pothole.py         # CLI video inference with frame sampling
│       └── detect.py                # Pre-trained baseline inference
├── frontend/                        # Web Testing Dashboard UI
│   ├── index.html                   # Modern glassmorphic console
│   ├── styles.css                   # Custom dark design system
│   └── app.js                       # Asynchronous dashboard logic
├── datasets/
│   └── transiteye/                  # Roboflow pothole dataset (train, valid, test)
├── output/
│   └── pothole_rfdetr_s/            # Checkpoints, logs, and metrics
├── outputs/                         # Annotated outputs (images & videos)
├── test_custom/                     # Sample test images
├── test_videos/                     # Sample test videos
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/ayush011-1/SIH2026-TeamSentrix.git
cd SIH2026-TeamSentrix
```

### 2. Set Up Python Environment
We recommend Python 3.10 to 3.12 using Conda or venv:

#### Using Conda (Recommended)
```bash
conda create -n sentrix-env python=3.11 -y
conda activate sentrix-env
pip install -r requirements.txt
```

#### Using venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Interactive Web Testing Dashboard

Launch the built-in testing interface to run visual inference on images or videos with confidence sliders, telemetry badges, and detection tables:

```bash
python app.py
```

Then open your browser at:
👉 **`http://localhost:8000`**

### Features:
- **Image Testing**: Drag and drop any road image or click 1-click sample chips (`pothole.png`, `bus.jpeg`).
- **Video Testing**: Process road videos (`road.mp4`) with configurable frame skips and preview annotated video directly in the browser.
- **Model Selector**: Switch between your fine-tuned Pothole model and the base 80-class COCO detector.
- **Live Telemetry**: Real-time inference latency (ms), FPS, and bounding box inspection table.
- **Multi-Hazard Hub**: Architecture guide for training on accidents, waterlogging, and ANPR.

---

## Dataset
The repository includes the **Transiteye** pothole dataset inside [datasets/transiteye/](datasets/transiteye/).
It is pre-formatted in Roboflow format with:
- `train/` (training images and annotations)
- `valid/` (validation images and annotations)
- `test/` (evaluation images)
- `classes.txt` (`pothole`)

---

## Model Training

To train RF-DETR-Small on the pothole dataset:

```bash
python ai/train.py --epochs 50 --batch-size 4
```

### Training Options
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset-dir` | `datasets/transiteye` | Path to dataset directory |
| `--output-dir` | `output/pothole_rfdetr_s` | Output directory for checkpoints & logs |
| `--epochs` | `50` | Total training epochs |
| `--batch-size` | `4` | Batch size per step |
| `--device` | `auto` | Acceleration device: `auto`, `mps` (Mac), `cuda` (NVIDIA GPU), or `cpu` |
| `--resolution` | `512` | Image input resolution |
| `--lr` | `0.0001` | Learning rate |
| `--resume` | `None` | Path to existing checkpoint (`.pth` or `.ckpt`) to resume training |

#### Examples

**Train on Apple Silicon (M1/M2/M3/M4):**
```bash
python ai/train.py --epochs 50 --batch-size 4 --device mps
```

**Train on an NVIDIA GPU:**
```bash
python ai/train.py --epochs 50 --batch-size 8 --device cuda
```

**Resume from a previous checkpoint:**
```bash
python ai/train.py --epochs 50 --resume output/pothole_rfdetr_s/checkpoint_29.ckpt
```

Training automatically saves the best weights to:
`output/pothole_rfdetr_s/checkpoint_best_total.pth`

---

## Multi-Hazard Extension (Accidents, Waterlogging, ANPR)

Can this model be trained on **Accidents**, **Waterlogging**, and **ANPR**? **Yes, absolutely.**

### 1. Vehicle Accidents & Collisions
- **Supported natively.** Collisions, overturned vehicles, and roadside debris are standard bounding-box object detection tasks.
- RF-DETR's transformer attention captures global road context (angled cars, skid marks) better than traditional CNNs.
- Label classes: `accident`, `overturned_vehicle`, `debris`.

### 2. Waterlogging & Flooded Roads
- **Bounding Boxes**: Detect puddles and flood zones with standard RF-DETR.
- **Instance Segmentation**: The same package provides `RFDETRSegSmall`, which predicts exact polygon masks for water pool boundaries and surface area estimation.

### 3. ANPR (Automatic Number Plate Recognition)
- **Stage 1 (RF-DETR)**: Detects the `license_plate` bounding box in dense traffic at ~30 FPS.
- **Stage 2 (OCR)**: Crop the detected plate and pass it to a lightweight OCR model (e.g. `PaddleOCR` or `EasyOCR`) to read the registration number (e.g. `MH 12 AB 1234`).

### 4. Unified Multi-Hazard Training
You can combine all classes into **one single model**:
```text
# classes.txt
pothole
accident
waterlogging
license_plate
```
Train with `python ai/train.py --dataset-dir datasets/unified_hazards` and the model will detect all road hazards and vehicle plates simultaneously in real time.

---

## Model Weights & Checkpoints

> [!NOTE]
> GitHub has a strict **100 MB** file limit. Trained model weights (`checkpoint_best_total.pth` ~122 MB, `.ckpt` ~486 MB) are git-ignored to prevent repository bloat and push errors.

### How weights are managed:
1. **Base Pretrained Weights**: RF-DETR downloads the official base model automatically when you run `RFDETRSmall()` for the first time.
2. **Sharing Fine-Tuned Weights**:
   - **GitHub Releases**: Download `checkpoint_best_total.pth` from the [GitHub Releases](https://github.com/ayush011-1/SIH2026-TeamSentrix/releases) tab and save it in `output/pothole_rfdetr_s/`.
   - Or run training locally with `python ai/train.py` to produce your own weights.

---

## Running CLI Inference

### Image Detection
```bash
python ai/inference/test_pothole.py --image test_custom/pothole.png --output outputs/pothole_result.jpg
```

### Video Detection
```bash
python ai/inference/video_pothole.py --video test_videos/road.mp4 --output outputs/video/road_detected.mp4
```

### General Object Detection
```bash
python ai/inference/detect.py
```

---

## Troubleshooting & FAQs

- **`ModuleNotFoundError: No module named 'rfdetr'`**:
  Make sure you have installed dependencies in your active Python environment:
  ```bash
  pip install -r requirements.txt
  ```
- **`OutOfMemoryError` during training**:
  Reduce the batch size or input resolution:
  ```bash
  python ai/train.py --batch-size 2 --resolution 512
  ```
- **`Model checkpoint not found`**:
  Train the model once using `python ai/train.py` or download `checkpoint_best_total.pth` into `output/pothole_rfdetr_s/`.