"""Reusable helper functions for ObjectVision AI."""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_MB = 200
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging once (console + outputs/app.log)."""
    root = logging.getLogger()
    if root.handlers:
        return
    OUTPUTS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUTS_DIR / "app.log", encoding="utf-8")],
    )


def validate_video_file(filename: str, size_bytes: int, max_mb: int = MAX_UPLOAD_MB) -> tuple[bool, str]:
    """Check extension and size. Returns (is_valid, message)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Please upload MP4, AVI or MOV."
    if size_bytes <= 0:
        return False, "The file is empty."
    if size_bytes > max_mb * 1024 * 1024:
        return False, f"File is larger than {max_mb} MB. Please upload a smaller video."
    return True, "File looks valid."


def make_output_path(outputs_dir: str | Path, stem: str) -> Path:
    """Create a unique output filename like outputs/myvideo_detected_20260101_101500.mp4."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) or "video"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outputs_dir / f"{safe}_detected_{stamp}.mp4"


def format_stats(category_counts: dict[str, int]) -> list[dict]:
    """Turn {'person': 10, 'car': 5} into table rows sorted by count, with percentages."""
    total = sum(category_counts.values())
    rows = []
    for name, count in sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = (count / total * 100) if total else 0.0
        rows.append({"Category": name, "Detections": count, "Share (%)": round(pct, 1)})
    return rows


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"


def reencode_h264(src: str | Path) -> bool:
    """Try to convert a video to browser-friendly H.264 using imageio-ffmpeg.

    Returns True on success. If imageio-ffmpeg is missing or fails, the original
    file is left untouched and False is returned (the download still works).
    """
    src = Path(src)
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = src.with_name(src.stem + "_h264.mp4")
        cmd = [ffmpeg, "-y", "-i", str(src), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-an", str(tmp)]
        result = subprocess.run(cmd, capture_output=True, timeout=1800)
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(src)
            return True
        tmp.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - optional step, never crash the app
        logging.getLogger(__name__).warning("H.264 re-encode skipped: %s", exc)
    return False
