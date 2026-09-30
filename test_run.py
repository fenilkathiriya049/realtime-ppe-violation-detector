# test_run.py
from pathlib import Path
import cv2
from src.inference.engine import PPEInferenceEngine
from src.inference.tracker import PPEViolationTracker
from src.utils.visualizer import draw_ppe_analytics

# Lower conf_thres to 0.15 to see all candidates
engine = PPEInferenceEngine(model_path="models/best.onnx", conf_thres=0.15)
tracker = PPEViolationTracker()

image_path = Path("data/processed/valid/images") / "frame000060_person01_conf0-71_jpg.rf.6e74bf93e85825aa74065edf149f385a.jpg"
if not image_path.exists():
    image_path = next(Path("data/processed/valid/images").glob("*.jpg"))

frame = cv2.imread(str(image_path))

# 1. Raw Detections
detections = engine.predict(frame)
print(f"\n[*] Total raw detections found: {len(detections)}")
for d in detections:
    print(f"    - Class: {d['class_name']:<12} | Conf: {d['confidence']:.2f} | BBox: {d['bbox']}")

# 2. Track & Audit
evaluated_workers = tracker.evaluate_and_track(detections)
output_frame = draw_ppe_analytics(frame, evaluated_workers)
cv2.imwrite("output_safety_audit.jpg", output_frame)
print(f"\n[✓] Evaluated {len(evaluated_workers)} workers.")