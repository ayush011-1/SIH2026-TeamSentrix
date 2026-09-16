#!/usr/bin/env python3
"""
Universal Training Script for RF-DETR.
Supports: Potholes, Accidents, ANPR (License Plates), Waterlogging, and Custom Multi-Hazard Datasets.
Team Sentrix - SIH 2026
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from rfdetr import RFDETRSmall
except ImportError:
    print(
        "Error: 'rfdetr' package is not installed.\n"
        "Install all required dependencies with:\n"
        "    pip install -r requirements.txt\n"
        "or:\n"
        "    pip install 'rfdetr[train,loggers]'",
        file=sys.stderr,
    )
    sys.exit(1)


def inspect_dataset_classes(dataset_path: Path) -> Tuple[int, List[str]]:
    """Auto-detect number of classes and class names from dataset directory."""
    # 1. Check classes.txt
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return len(names), names

    # 2. Check Roboflow COCO train/_annotations.coco.json
    for json_name in ["train/_annotations.coco.json", "_annotations.coco.json"]:
        json_path = dataset_path / json_name
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cats = data.get("categories", [])
                if cats:
                    names = [c["name"] for c in sorted(cats, key=lambda x: x.get("id", 0))]
                    return len(names), names
            except Exception:
                pass

    # Default fallback
    return 1, ["object"]


def train(
    dataset_dir: str = "datasets/transiteye",
    output_dir: str = "output/pothole_rfdetr_s",
    epochs: int = 50,
    batch_size: int = 4,
    device: str = "auto",
    resolution: int = 512,
    lr: float = 1e-4,
    num_classes: Optional[int] = None,
    dataset_file: str = "roboflow",
    resume: Optional[str] = None,
) -> None:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    if not dataset_path.exists():
        print(
            f"Error: Dataset directory '{dataset_path}' not found.\n"
            "Please ensure the dataset exists at the specified path.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Detect dataset classes
    detected_count, class_names = inspect_dataset_classes(dataset_path)
    target_num_classes = num_classes if num_classes is not None else detected_count

    print("=" * 65)
    print("Team Sentrix: RF-DETR Universal Model Training")
    print("=" * 65)
    print(f"Dataset path   : {dataset_path.resolve()}")
    print(f"Dataset format : {dataset_file}")
    print(f"Classes found  : {target_num_classes} -> {class_names}")
    print(f"Output path    : {output_path.resolve()}")
    print(f"Epochs         : {epochs}")
    print(f"Batch size     : {batch_size}")
    print(f"Resolution     : {resolution}")
    print(f"Device         : {device}")
    print(f"Learning rate  : {lr}")
    if resume:
        print(f"Resume from    : {resume}")
    print("=" * 65)

    print("\nInitializing RF-DETR-Small...")
    if resume and Path(resume).exists():
        print(f"Resuming / fine-tuning from checkpoint: {resume}")
        model = RFDETRSmall(pretrain_weights=resume, num_classes=target_num_classes)
    else:
        # Downloads base pre-trained backbone if not cached locally
        model = RFDETRSmall(num_classes=target_num_classes)

    print("\nStarting training loop...")
    train_kwargs = {
        "dataset_dir": str(dataset_path.resolve()),
        "dataset_file": dataset_file,
        "epochs": epochs,
        "batch_size": batch_size,
        "output_dir": str(output_path.resolve()),
        "resolution": resolution,
        "lr": lr,
    }

    if device and device.lower() != "auto":
        train_kwargs["device"] = device

    if resume:
        train_kwargs["resume"] = resume

    model.train(**train_kwargs)

    print("\n" + "=" * 65)
    print("Training finished successfully!")
    best_ckpt = output_path / "checkpoint_best_total.pth"
    if best_ckpt.exists():
        print(f"Best checkpoint saved to: {best_ckpt.resolve()}")
    else:
        print(f"Checkpoints saved to: {output_path.resolve()}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on any dataset: Potholes, Accidents, ANPR, Waterlogging, etc."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/transiteye",
        help="Path to dataset directory (default: datasets/transiteye)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/pothole_rfdetr_s",
        help="Directory to save checkpoints & metrics (default: output/pothole_rfdetr_s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: 'auto', 'mps' (Mac), 'cuda' (Nvidia GPU), or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image input resolution (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Explicit number of classes (default: auto-detected from dataset)",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="roboflow",
        choices=["roboflow", "coco", "yolo"],
        help="Dataset annotation format: 'roboflow', 'coco', or 'yolo' (default: roboflow)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        resolution=args.resolution,
        lr=args.lr,
        num_classes=args.num_classes,
        dataset_file=args.dataset_format,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
