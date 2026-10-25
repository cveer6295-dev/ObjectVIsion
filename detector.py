"""YOLOv8 detection module. Loads the model once and returns clean results."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
SUPPORTED_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]  # nano, small, medium


class DetectorError(Exception):
    """Raised when the model cannot be loaded or detection fails."""


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    class_id: int


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", models_dir: str | Path = MODELS_DIR):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from exc
        self.models_dir.mkdir(parents=True, exist_ok=True)
        weights = self.models_dir / self.model_name
        try:
            # If the file is missing, Ultralytics downloads it (internet needed once).
            model = YOLO(str(weights))
            logger.info("Loaded model %s", weights)
            return model
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(
                f"Could not load '{self.model_name}'. If this is the first run, check your internet "
                f"connection (the weights download once). Or download the file manually into "
                f"'{self.models_dir}'. Details: {exc}"
            ) from exc

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self.model.names)

    def detect(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45,
               imgsz: int = 640, classes: list[int] | None = None) -> list[Detection]:
        """Run inference on one BGR frame. NMS (controlled by `iou`) is applied inside YOLO."""
        if frame is None or frame.size == 0:
            raise DetectorError("Empty frame received.")
        try:
            result = self.model.predict(frame, conf=conf, iou=iou, imgsz=imgsz,
                                        classes=classes, device="cpu", verbose=False)[0]
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"Detection failed: {exc}") from exc
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cid = int(box.cls[0])
            detections.append(Detection(names[cid], float(box.conf[0]), (x1, y1, x2, y2), cid))
        return detections


def _color_for(class_id: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(class_id * 7 + 3)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_detections(frame: np.ndarray, detections: list[Detection],
                    show_labels: bool = True, show_conf: bool = True) -> np.ndarray:
    """Draw boxes (and optional labels/confidence) on a copy of the frame."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = _color_for(d.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        parts = []
        if show_labels:
            parts.append(d.label)
        if show_conf:
            parts.append(f"{d.confidence:.2f}")
        if parts:
            text = " ".join(parts)
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1 - th - base - 4, 0)
            cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + base + 4), color, -1)
            cv2.putText(out, text, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
