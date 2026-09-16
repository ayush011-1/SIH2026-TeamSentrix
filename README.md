# Sentrix: AI-Powered Road Hazard & Pothole Detection
> Smart India Hackathon (SIH 2026) | Team Sentrix

Real-time, transformer-based road hazard and pothole detection powered by **RF-DETR (Real-time Vision Transformer Object Detector)**.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Weights & Checkpoints](#model-weights--checkpoints)
- [Running Inference](#running-inference)
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
- **Configurable Pipeline**: Flexible scripts for training, single-image inference, and streaming video analysis.

---

## Project Structure
```text
SIH2026-TeamSentrix/
├── ai/
│   ├── train.py                     # Training entrypoint for RF-DETR
│   └── inference/
│       ├── test_pothole.py          # Image inference with pothole model
│       ├── video_pothole.py         # Video inference with frame sampling
│       └── detect.py                # Pre-trained baseline inference
├── datasets/
│   └── transiteye/                  # Roboflow pothole dataset (train, valid, test)
│       ├── classes.txt
│       ├── train/
│       ├── valid/
│       └── test/
├── output/
│   └── pothole_rfdetr_s/            # Checkpoints, logs, and metrics
│       ├── checkpoint_best_total.pth (ignored in git due to file size)
│       └── metrics.csv
├── outputs/                         # Annotated outputs (images & videos)
├── test_custom/                     # Sample test images
├── test_videos/                     # Sample test video
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

## Dataset
The repository includes the **Transiteye** pothole dataset inside [datasets/transiteye/](datasets/transiteye/).
It is pre-formatted in Roboflow format with:
- `train/` (training images and annotations)
- `valid/` (validation images and annotations)
- `test/` (evaluation images)
- `classes.txt` (`pothole`)

---

## Model Training

To train RF-DETR-Small on the pothole dataset from scratch:

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

## Model Weights & Checkpoints

> [!NOTE]
> GitHub has a strict **100 MB** file limit. Trained model weights (`checkpoint_best_total.pth` ~122 MB, `.ckpt` ~486 MB) are git-ignored to prevent repository bloat and push errors.

### How weights are managed:
1. **Base Pretrained Weights**: RF-DETR downloads the official base model automatically when you run `RFDETRSmall()` for the first time.
2. **Sharing Fine-Tuned Weights**:
   - **GitHub Releases**: Download the latest `checkpoint_best_total.pth` from the [GitHub Releases](https://github.com/ayush011-1/SIH2026-TeamSentrix/releases) tab and save it in `output/pothole_rfdetr_s/`.
   - Or run training locally with `python ai/train.py` to produce your own weights.

---

## Running Inference

### Image Detection
Detect potholes in a single image:
```bash
python ai/inference/test_pothole.py --image test_custom/pothole.png --output outputs/pothole_result.jpg
```
**Options:**
- `--model`: Path to model weights (default: `output/pothole_rfdetr_s/checkpoint_best_total.pth`)
- `--image`: Input image path
- `--output`: Output image save path
- `--threshold`: Confidence threshold (default: `0.65`)

---

### Video Detection
Process a video and produce a bounding-box annotated video:
```bash
python ai/inference/video_pothole.py --video test_videos/road.mp4 --output outputs/video/road_detected.mp4
```
**Options:**
- `--video`: Input video file path
- `--output`: Output annotated video path
- `--threshold`: Confidence score threshold (default: `0.65`)
- `--frame-skip`: Skip every N frames for faster processing (default: `2`)

---

### General Object Detection
Run the base pretrained RF-DETR model (detecting common COCO objects like cars, buses, pedestrians):
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