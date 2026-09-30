# src/api/app.py
from pathlib import Path
from typing import Any, Dict
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.inference.engine import PPEInferenceEngine
from src.inference.tracker import PPEViolationTracker
from src.utils.visualizer import draw_ppe_analytics

app = FastAPI(
    title="PPE Safety Compliance & Violation Monitoring API",
    version="1.0.0",
    description="Real-time ONNX Runtime GPU inference and spatial safety violation tracking.",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best.onnx"
engine: PPEInferenceEngine = None
tracker: PPEViolationTracker = None


@app.on_event("startup")
def startup_event():
    global engine, tracker
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    # Initialize engine and tracker once on startup
    engine = PPEInferenceEngine(model_path=MODEL_PATH, conf_thres=0.25)
    tracker = PPEViolationTracker()
    print("[*] FastAPI Service Ready: ONNX CUDA Engine & Violation Tracker Loaded.")


def decode_image(file_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    return image


@app.get("/health")
def health():
    """Health check endpoint verifying GPU provider availability."""
    provider = engine.session.get_providers()[0] if engine else "not_loaded"
    return {
        "status": "healthy",
        "active_provider": provider,
        "model_path": MODEL_PATH,
    }


@app.post("/api/v1/audit/json")
async def audit_ppe_json(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an image to get structured compliance metrics and worker violation lists."""
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
    """Upload an image and receive the annotated frame (with HUD and colored status boxes) as JPEG."""
    contents = await file.read()
    frame = decode_image(contents)

    detections = engine.predict(frame)
    workers = tracker.evaluate_and_track(detections)
    annotated = draw_ppe_analytics(frame, workers)

    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode response image.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

from fastapi.responses import StreamingResponse

def generate_video_stream(video_source: Any = 0):
    """
    Generator that captures frames, performs detection & tracking,
    and yields multipart MJPEG image bytes.
    video_source: 0 for default webcam, or path/URL to video/RTSP stream.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run Phase 2 & 3 pipeline
        detections = engine.predict(frame)
        workers = tracker.evaluate_and_track(detections)
        annotated_frame = draw_ppe_analytics(frame, workers)

        # Encode to JPEG
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.get("/api/v1/stream/webcam")
def stream_webcam():
    """Live video stream from webcam or default camera device."""
    return StreamingResponse(
        generate_video_stream(0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )