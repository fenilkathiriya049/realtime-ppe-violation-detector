# tests/benchmark.py
import time
import numpy as np
import cv2
from ultralytics import YOLO
from src.inference.engine import PPEInferenceEngine

# 1. Warmup and Benchmark PyTorch (.pt)
print("[*] Benchmarking PyTorch (.pt)...")
pt_model = YOLO("runs/detect/runs/train/ppe_violation_exp-2/weights/best.pt")
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(10):
    _ = pt_model(dummy_img, verbose=False)

pt_times = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = pt_model(dummy_img, verbose=False)
    pt_times.append((time.perf_counter() - t0) * 1000)

# 2. Warmup and Benchmark ONNX Runtime (.onnx)
print("[*] Benchmarking ONNX Runtime FP16 (.onnx)...")
engine = PPEInferenceEngine(model_path="models/best.onnx")

for _ in range(10):
    _ = engine.predict(dummy_img)

onnx_times = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = engine.predict(dummy_img)
    onnx_times.append((time.perf_counter() - t0) * 1000)

print("\n--- Phase 2 Latency Results (50 iterations) ---")
print(f"PyTorch (.pt)       : Avg Latency = {np.mean(pt_times):.2f} ms | FPS = {1000 / np.mean(pt_times):.1f}")
print(f"ONNX Engine (.onnx) : Avg Latency = {np.mean(onnx_times):.2f} ms | FPS = {1000 / np.mean(onnx_times):.1f}")