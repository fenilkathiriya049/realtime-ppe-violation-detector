# src/api/app.py
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from src.inference.engine import PPEInferenceEngine
from src.inference.tracker import PPEViolationTracker
from src.utils.visualizer import draw_ppe_analytics

MODEL_PATH = "models/best.onnx"
engine: PPEInferenceEngine = None
tracker: PPEViolationTracker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tracker
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    engine = PPEInferenceEngine(model_path=MODEL_PATH, conf_thres=0.25)
    tracker = PPEViolationTracker()
    print("[*] API Lifespan: ONNX CUDA Engine & Tracker loaded successfully.")
    yield
    print("[*] API Lifespan: Shutting down resources.")


app = FastAPI(
    title="PPE Safety Compliance & Violation Monitoring API",
    version="1.0.0",
    description="Real-time ONNX Runtime GPU inference and spatial safety violation tracking.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(file_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    return image


@app.get("/health")
def health():
    provider = engine.session.get_providers()[0] if engine else "not_loaded"
    return {
        "status": "healthy",
        "active_provider": provider,
        "model_path": MODEL_PATH,
    }


@app.post("/api/v1/audit/json")
async def audit_ppe_json(file: UploadFile = File(...)) -> Dict[str, Any]:
    contents = await file.read()
    frame = decode_image(contents)

    detections = engine.predict(frame)
    workers = tracker.evaluate_and_track(detections)

    total_workers = len(workers)
    violations = [w for w in workers if w["status"] == "VIOLATION"]

    return {
        "total_workers": total_workers,
        "compliant_count": total_workers - len(violations),
        "violation_count": len(violations),
        "workers": workers,
        "raw_detections": detections,
    }


@app.post("/api/v1/audit/image")
async def audit_ppe_image(file: UploadFile = File(...)):
    contents = await file.read()
    frame = decode_image(contents)

    detections = engine.predict(frame)
    workers = tracker.evaluate_and_track(detections)
    annotated = draw_ppe_analytics(frame, workers)

    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode response image.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


def generate_video_stream(video_source: Any = 0):
    # Use cv2.CAP_DSHOW for fast, reliable hardware capture on Windows
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[!] Warning: Unable to open video source: {video_source}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detections = engine.predict(frame)
        workers = tracker.evaluate_and_track(detections)
        annotated_frame = draw_ppe_analytics(frame, workers)

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()

@app.get("/api/v1/stream/webcam")
def stream_demo():
    # Path to any sample MP4 video
    return StreamingResponse(
        generate_video_stream(0),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )