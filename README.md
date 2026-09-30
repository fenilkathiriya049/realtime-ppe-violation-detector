# Real-Time PPE Compliance & Violation Monitoring Engine

An edge-optimized, production-ready computer vision pipeline engineered to monitor Personal Protective Equipment (PPE) compliance in industrial environments in real time. 

Built with **Ultralytics YOLO11**, serialized to **FP16 ONNX Runtime** using hardware-accelerated **CUDAExecutionProvider**, wrapped with spatial multi-object tracking logic, and served via **FastAPI** REST and live streaming endpoints.

---

## Key Features

- **High-Speed Inference**: Sub-15 ms latency (~68+ FPS) on consumer RTX GPUs via ONNX Runtime FP16 execution.
- **Robust Spatial Association**: Automatically couples helmets and reflective vests to individual workers, with fallback anchoring heuristics when full-body bounding boxes are partially occluded.
- **Real-Time Violation Auditing**: Instant compliance status tagging (`COMPLIANT` vs `VIOLATION: NO_HELMET / NO_VEST`) with dynamic visual HUD overlays.
- **Production REST & Streaming API**: FastAPI microservice supporting single-image JSON analysis, annotated JPEG returns, and continuous MJPEG live streams (webcam / CCTV RTSP).
- **Containerized Deployment**: Multi-stage, GPU-enabled Docker configuration powered by `nvidia/cuda:12.1.1-runtime`.

---

## Hardware Benchmarks

Evaluated on an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** running CUDA 12.1 and TensorRT/ONNX Runtime:

| Framework / Format | Precision | Avg Latency (ms) | Throughput (FPS) | mAP@0.5 |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch (`.pt`)** | FP32 | 15.46 ms | 64.7 FPS | 0.814 |
| **ONNX Runtime Engine** | **FP16** | **14.66 ms** | **68.2 FPS** | **0.814** |
| ONNX Runtime (CPU Fallback) | FP32 | 70.36 ms | 14.2 FPS | 0.814 |

---

## System Architecture

```text
Incoming Stream / Image Payload
              │
              ▼
    [ FastAPI Endpoint ]  (src/api/app.py)
              │
              ▼
 [ Preprocessing & Letterbox ] (NumPy / OpenCV, [0, 1] normalized)
              │
              ▼
 [ ONNX Runtime GPU Session ]  (models/best.onnx via CUDAExecutionProvider)
              │
              ▼
  [ Vectorized NMS Filtering ] (cv2.dnn.NMSBoxes)
              │
              ▼
 [ Spatial Violation Tracker ] (src/inference/tracker.py)
   ├── Person / PPE IoU Association
   ├── Worker Temporal Tracking
   └── Violation Rule Tagging
              │
              ▼
 [ Visual Analytics / JSON Payload ] (draw_ppe_analytics)