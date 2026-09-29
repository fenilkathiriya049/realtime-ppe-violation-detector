# src/training/train.py
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_training(config_path: str, epochs: int = 50, batch_size: int = 16, img_size: int = 640):
    config = Path(config_path)
    if not config.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load small baseline model
    model = YOLO("yolo11s.pt")  # or 'yolov8s.pt'

    # Train
    model.train(
        data=str(config),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=0,               # Set to 'cpu' if no dedicated GPU
        workers=4,
        save=True,
        project="runs/train",
        name="ppe_violation_exp"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Safety Violation Detector")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    run_training(args.config, args.epochs, args.batch, args.imgsz)