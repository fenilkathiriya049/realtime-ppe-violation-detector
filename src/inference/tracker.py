# src/inference/tracker.py
from typing import Any, Dict, List, Tuple
import numpy as np


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    if xB < xA or yB < yA:
        return 0.0

    inter = (xB - xA) * (yB - yA)
    union = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter)
    return inter / union if union > 0 else 0.0


class PPEViolationTracker:
    def __init__(self, iou_match_threshold: float = 0.30, max_missing_frames: int = 15):
        self.iou_match_threshold = iou_match_threshold
        self.max_missing_frames = max_missing_frames
        self.tracked_workers: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1

    def evaluate_and_track(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        person_detections = [d for d in detections if d["class_name"] == "person"]
        vest_detections = [d for d in detections if d["class_name"] == "vest"]
        helmet_detections = [d for d in detections if d["class_name"] == "helmet"]
        no_helmet_detections = [d for d in detections if d["class_name"] == "no_helmet"]

        # If no explicit 'person' class is detected, anchor workers using 'vest' or head gear
        candidate_workers = []
        if person_detections:
            for p in person_detections:
                candidate_workers.append({"bbox": p["bbox"], "confidence": p["confidence"]})
        elif vest_detections:
            # Expand vest box slightly upward and downward to approximate worker body
            for v in vest_detections:
                vx1, vy1, vx2, vy2 = v["bbox"]
                vh = vy2 - vy1
                candidate_workers.append({
                    "bbox": [vx1, max(0, int(vy1 - 0.4 * vh)), vx2, vy2 + int(0.2 * vh)],
                    "confidence": v["confidence"]
                })
        elif helmet_detections or no_helmet_detections:
            for h in (helmet_detections + no_helmet_detections):
                hx1, hy1, hx2, hy2 = h["bbox"]
                hh = hy2 - hy1
                candidate_workers.append({
                    "bbox": [hx1 - 20, hy1, hx2 + 20, hy2 + int(3.5 * hh)],
                    "confidence": h["confidence"]
                })

        current_frame_workers = []

        for p_idx, worker in enumerate(candidate_workers):
            p_box = worker["bbox"]
            track_id = self.next_track_id
            self.next_track_id += 1

            has_helmet = any(compute_iou(p_box, h["bbox"]) > 0.05 or (h["bbox"][1] <= p_box[1] + (p_box[3] - p_box[1]) * 0.5) for h in helmet_detections)
            has_no_helmet = any(compute_iou(p_box, nh["bbox"]) > 0.05 for nh in no_helmet_detections)
            has_vest = any(compute_iou(p_box, v["bbox"]) > 0.1 for v in vest_detections)

            violations = []
            if has_no_helmet or not has_helmet:
                violations.append("NO_HELMET")
            if not has_vest:
                violations.append("NO_VEST")

            status = "COMPLIANT" if len(violations) == 0 else "VIOLATION"

            current_frame_workers.append({
                "track_id": track_id,
                "bbox": p_box,
                "confidence": worker["confidence"],
                "status": status,
                "violations": violations
            })

        return current_frame_workers