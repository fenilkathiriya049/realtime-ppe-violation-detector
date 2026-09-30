# src/training/export_onnx.py
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

def export_and_store(
    weights_path: str,
    output_dir: str = "models",
    imgsz: int = 640,
    half: bool = True
):
    source_path = Path(weights_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {source_path}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading PyTorch checkpoint: {source_path}")
    model = YOLO(str(source_path))

    print(f"[*] Exporting to ONNX (half={half}, dynamic=True, simplify=True)...")
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=True,
        simplify=True
    )

    # Ultralytics exports next to the source checkpoint; move it into models/
    source_onnx = Path(exported_file)
    target_onnx = target_dir / "best.onnx"
    shutil.copy2(source_onnx, target_onnx)

    print(f"[*] Successfully saved deployable model to: {target_onnx}")
    return target_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch weights to top-level models/ directory")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/runs/train/ppe_violation_exp/weights/best.pt",
        help="Path to best.pt"
    )
    parser.add_argument("--output_dir", type=str, default="models", help="Destination folder for exported weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", default=True, help="Export in FP16 precision")
    args = parser.parse_args()

    export_and_store(args.weights, args.output_dir, args.imgsz, args.half)