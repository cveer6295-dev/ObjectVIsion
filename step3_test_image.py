"""Step 3: detect objects in one image.
Usage: python scripts/step3_test_image.py path/to/image.jpg
Saves the annotated image to outputs/image_result.jpg
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from detector import ObjectDetector, DetectorError, draw_detections

if len(sys.argv) < 2:
    sys.exit("Please give an image path, e.g. python scripts/step3_test_image.py sample.jpg")
image = cv2.imread(sys.argv[1])           # read the image as a NumPy array (BGR colours)
if image is None:
    sys.exit("Could not read that image. Check the path and file type.")
try:
    detector = ObjectDetector("yolov8n.pt")
    detections = detector.detect(image, conf=0.25)
except DetectorError as e:
    sys.exit(f"ERROR: {e}")
print(f"Found {len(detections)} objects")
for d in detections:
    print(f"  {d.label:12s} confidence={d.confidence:.2f} box={d.box}")
Path("outputs").mkdir(exist_ok=True)
cv2.imwrite("outputs/image_result.jpg", draw_detections(image, detections))
print("Saved outputs/image_result.jpg")
