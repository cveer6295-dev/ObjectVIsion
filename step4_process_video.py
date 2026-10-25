"""Step 4: process a video from the command line (no Streamlit).
Usage: python scripts/step4_process_video.py sample_videos/input.mp4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import ObjectDetector, DetectorError
from utils import make_output_path, format_stats
from video_processor import process_video, VideoError

if len(sys.argv) < 2:
    sys.exit("Usage: python scripts/step4_process_video.py <video>")
src = Path(sys.argv[1])
try:
    res = process_video(src, make_output_path("outputs", src.stem), ObjectDetector(),
                        progress_callback=lambda d, t: print(f"\rframe {d}/{t}", end=""))
except (VideoError, DetectorError) as e:
    sys.exit(f"\nERROR: {e}")
print(f"\nSaved: {res.output_path}\nFrames: {res.frames_processed}  Detections: {res.total_detections}  "
      f"Speed: {res.avg_fps:.1f} FPS")
for row in format_stats(res.category_counts):
    print(row)
