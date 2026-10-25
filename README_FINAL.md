# ObjectVision AI — Vision Command Center

A presentation-ready web interface for the ObjectVision AI BCA project.

## What this final version does

- YOLOv8 object detection on uploaded MP4, AVI and MOV videos
- Frame-by-frame processing with OpenCV
- Bounding boxes, labels and confidence scores
- Detection statistics and charts
- Annotated video preview and download
- Webcam snapshot detection
- Session-only run history for comparing scans
- Technical architecture / system-intel page
- Presentation-friendly dark "AI command center" interface
- CPU-ready workflow using the existing project modules

## Run

Recommended Python: 3.10–3.12 (64-bit).

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can also double-click:

`run_objectvision.bat`

## Important

The model weights may download on first use if the selected YOLOv8 `.pt` file is not already present in `models/`.

The core project performs inference with pretrained YOLOv8 weights. It does not train a new model.

Counts are detections accumulated across frames, not unique physical objects. Persistent tracking, line-crossing counts and real-time continuous webcam detection are not part of this core implementation.

## Suggested final-demo flow

1. Open Mission Control.
2. Show the pipeline and project objective.
3. Open Video Analyzer.
4. Upload a short 10–20 second sample.
5. Show clip intelligence and the active detection profile.
6. Start the Vision Scan.
7. Show annotated output, object chart and metrics.
8. Download the processed video.
9. Open System Intel and explain the architecture and limitations.

## Project identity

**Project title:** ObjectVision AI: Object Recognition and Detection in Videos Using Deep Learning

**UI identity:** ObjectVision AI — Vision Command Center

Core stack: Python, YOLOv8/Ultralytics, PyTorch, OpenCV, NumPy, Streamlit and optional imageio-ffmpeg.
