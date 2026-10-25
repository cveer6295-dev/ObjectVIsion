"""Step 2: load the pretrained YOLOv8 nano model. Needs internet only the first time."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import ObjectDetector, DetectorError

try:
    det = ObjectDetector("yolov8n.pt")
    print("Model loaded. It knows", len(det.class_names), "classes, e.g.", list(det.class_names.values())[:5])
except DetectorError as e:
    print("ERROR:", e)
