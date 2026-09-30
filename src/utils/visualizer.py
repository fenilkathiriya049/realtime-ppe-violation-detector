# src/utils/visualizer.py
from typing import Any, Dict, List
import cv2
import numpy as np


def draw_ppe_analytics(image: np.ndarray, workers: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()
    h, w = annotated.shape[:2]

    total_workers = len(workers)
    violations = sum(1 for worker in workers if worker["status"] == "VIOLATION")
    compliant = total_workers - violations

    # Draw Worker Boxes
    for worker in workers:
        x1, y1, x2, y2 = worker["bbox"]
        is_compliant = worker["status"] == "COMPLIANT"

        # Compliant = Green (0, 200, 0) | Violation = Red (0, 0, 255)
        color = (0, 200, 0) if is_compliant else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        tag = f"Worker #{worker['track_id']} [COMPLIANT]" if is_compliant else f"Worker #{worker['track_id']} [{', '.join(worker['violations'])}]"
        (text_w, text_h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - text_h - 6)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            annotated,
            tag,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Top HUD Bar
    cv2.rectangle(annotated, (0, 0), (w, 40), (20, 20, 20), -1)
    hud_msg = f"Tracked Workers: {total_workers}  |  Compliant: {compliant}  |  Violations: {violations}"
    cv2.putText(annotated, hud_msg, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated