# src/inference/engine.py
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any

class ONNXDetector:
    def __init__(self, model_path: str):
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [B, C, H, W]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Resize to model input (typically 640x640)
        h, w = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # HWC to CHW and normalize to [0, 1]
        input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(input_tensor, axis=0)

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List[Dict[str, Any]]:
        tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})
        # outputs[0] contains bounding boxes and class logits
        # Add your NMS / filtering logic here
        return [{"status": "model_ready", "raw_output_shape": outputs[0].shape}]