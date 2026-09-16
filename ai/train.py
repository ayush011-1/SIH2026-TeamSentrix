#!/usr/bin/env python3
"""
Training script for RF-DETR on the Pothole / Road Hazard dataset.
Team Sentrix - SIH 2026
"""

import argparse
import sys
from pathlib import Path

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


def train(
    dataset_dir: str = "datasets/transiteye",
    output_dir: str = "output/pothole_rfdetr_s",
    epochs: int = 50,
    batch_size: int = 4,
    device: str = "auto",
    resolution: int = 512,
    lr: float = 1e-4,
    resume: str | None = None,
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

    print("=" * 60)
    print("Team Sentrix: RF-DETR Pothole Detection Training")
    print("=" * 60)
    print(f"Dataset path : {dataset_path.resolve()}")
    print(f"Output path  : {output_path.resolve()}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch_size}")
    print(f"Resolution   : {resolution}")
    print(f"Device       : {device}")
    print(f"Learning rate: {lr}")
    if resume:
        print(f"Resume from  : {resume}")
    print("=" * 60)

    print("\nInitializing RF-DETR-Small...")
    if resume and Path(resume).exists():
        print(f"Loading weights from checkpoint: {resume}")
        model = RFDETRSmall(pretrain_weights=resume, num_classes=1)
    else:
        # Downloads base pre-trained backbone if not cached locally
        model = RFDETRSmall(num_classes=1)

    print("\nStarting training loop...")
    train_kwargs = {
        "dataset_dir": str(dataset_path.resolve()),
        "dataset_file": "roboflow",
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

    print("\n" + "=" * 60)
    print("Training finished successfully!")
    best_ckpt = output_path / "checkpoint_best_total.pth"
    if best_ckpt.exists():
        print(f"Best checkpoint saved to: {best_ckpt.resolve()}")
    else:
        print(f"Checkpoints saved to: {output_path.resolve()}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR-Small on pothole detection dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/transiteye",
        help="Path to Roboflow dataset directory (default: datasets/transiteye)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/pothole_rfdetr_s",
        help="Directory to save training outputs and checkpoints (default: output/pothole_rfdetr_s)",
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
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
