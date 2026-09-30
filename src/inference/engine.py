# src/inference/engine.py
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch  # Crucial on Windows: preloads CUDA/cuDNN DLLs into the process environment
import onnxruntime as ort


class PPEInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/best.onnx",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = {
            0: "helmet",
            1: "no_helmet",
            2: "none",
            3: "person",
            4: "vest",
        }

        # Hardware provider configuration
        available_providers = ort.get_available_providers()
        print(f"[*] Available ONNX Providers: {available_providers}")

        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"[*] Active Session Provider: {self.session.get_providers()[0]}")

        # Metadata extraction
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.input_type = self.input_meta.type
        self.output_name = self.session.get_outputs()[0].name

        # Resolve expected dimensions [batch, channels, height, width]
        shape = self.input_meta.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize and pad image while maintaining aspect ratio."""
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img, r, (dw, dh)

    def preprocess(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Normalize, letterbox, and convert frame to CHW tensor."""
        padded_img, ratio, dwdh = self.letterbox(img, (self.input_h, self.input_w))
        blob = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        # Dynamic cast to float16 or float32 based on model definition
        target_dtype = np.float16 if "float16" in self.input_type else np.float32
        blob = np.ascontiguousarray(blob, dtype=target_dtype) / 255.0
        return np.expand_dims(blob, axis=0), ratio, dwdh

    def postprocess(
        self,
        raw_preds: np.ndarray,
        ratio: float,
        dwdh: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Parse raw model outputs, apply confidence thresholding, and run NMS."""
        # Transpose shape: [1, 9, 8400] -> [8400, 9] (4 coords + 5 class probabilities)
        predictions = np.squeeze(raw_preds, axis=0).T.astype(np.float32)

        boxes = predictions[:, :4]           # [cx, cy, w, h]
        class_scores = predictions[:, 4:]     # 5 class scores

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Confidence cutoff
        mask = confidences >= self.conf_thres
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert [cx, cy, w, h] -> [x1, y1, w, h] for cv2.dnn.NMSBoxes
        dw, dh = dwdh
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / ratio
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / ratio
        w = boxes[:, 2] / ratio
        h = boxes[:, 3] / ratio

        nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(), self.conf_thres, self.iou_thres
        )

        detections = []
        if len(indices) > 0:
            orig_h, orig_w = orig_shape
            for idx in indices.flatten():
                bx, by, bw, bh = nms_boxes[idx]
                cls_id = int(class_ids[idx])
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.classes.get(cls_id, "unknown"),
                    "confidence": float(round(confidences[idx], 4)),
                    "bbox": [
                        max(0, min(orig_w, int(bx))),
                        max(0, min(orig_h, int(by))),
                        max(0, min(orig_w, int(bx + bw))),
                        max(0, min(orig_h, int(by + bh))),
                    ],
                })

        return detections

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run full pipeline: preprocessing -> inference -> postprocessing."""
        tensor, ratio, dwdh = self.preprocess(frame)
        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self.postprocess(raw_outputs, ratio, dwdh, frame.shape[:2])