"""Frame-by-frame video processing with OpenCV + the detector.

NOTE: this module counts DETECTIONS ACROSS FRAMES. A car visible in 100 frames
adds 100 detections. It does NOT count unique physical objects (that would need
a tracker such as ByteTrack/BoT-SORT - see the advanced features guide).
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from detector import ObjectDetector, DetectorError, draw_detections

logger = logging.getLogger(__name__)


class VideoError(Exception):
    """Raised for unreadable or invalid videos."""


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass
class ProcessingResult:
    output_path: Path
    frames_processed: int
    total_detections: int
    category_counts: dict = field(default_factory=dict)
    max_objects_in_frame: int = 0
    elapsed_seconds: float = 0.0
    avg_fps: float = 0.0
    stopped_early: bool = False
    info: VideoInfo | None = None


def get_video_info(path: str | Path) -> VideoInfo:
    """Open a video, read its properties and make sure at least one frame can be read."""
    path = Path(path)
    if not path.exists():
        raise VideoError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV could not open this video. It may be corrupted or use an unsupported codec.")
        ok, _ = cap.read()
        if not ok:
            raise VideoError("The video opened but no frames could be read (file may be corrupted).")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0 or fps > 240:
            fps = 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = max(count, 0)
        return VideoInfo(width, height, fps, count, count / fps if count else 0.0)
    finally:
        cap.release()


def _open_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):  # try H.264 first, fall back to mp4v
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            logger.info("Using codec %s", codec)
            return writer
        writer.release()
    raise VideoError(f"Could not create output video at {output_path}. Check the folder and permissions.")


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    detector: ObjectDetector,
    conf: float = 0.25,
    show_labels: bool = True,
    show_conf: bool = True,
    imgsz: int = 640,
    progress_callback: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ProcessingResult:
    """Read -> detect -> draw -> write, one frame at a time (low memory use)."""
    input_path, output_path = Path(input_path), Path(output_path)
    info = get_video_info(input_path)

    cap = cv2.VideoCapture(str(input_path))
    writer = None
    counts: Counter = Counter()
    frames = total = max_in_frame = 0
    stopped = False
    start = time.perf_counter()
    try:
        writer = _open_writer(output_path, info.fps, (info.width, info.height))
        while True:
            if should_stop and should_stop():
                stopped = True
                break
            ok, frame = cap.read()
            if not ok:  # end of video
                break
            detections = detector.detect(frame, conf=conf, imgsz=imgsz)
            counts.update(d.label for d in detections)
            total += len(detections)
            max_in_frame = max(max_in_frame, len(detections))
            writer.write(draw_detections(frame, detections, show_labels, show_conf))
            frames += 1
            if progress_callback:
                progress_callback(frames, info.frame_count)
    except DetectorError:
        raise
    except MemoryError as exc:
        raise VideoError("Ran out of memory. Try a smaller video, a smaller model or a smaller image size.") from exc
    finally:  # always free resources, even after an error
        cap.release()
        if writer is not None:
            writer.release()

    if frames == 0:
        raise VideoError("No frames were processed.")
    elapsed = time.perf_counter() - start
    return ProcessingResult(output_path, frames, total, dict(counts), max_in_frame,
                            elapsed, frames / elapsed if elapsed > 0 else 0.0, stopped, info)
