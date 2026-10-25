"""ObjectVision AI — Vision Command Center
Presentation-ready Streamlit interface for the existing YOLOv8/OpenCV project.
Run with: streamlit run app.py
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import ObjectDetector, DetectorError, SUPPORTED_MODELS, draw_detections
from utils import (
    OUTPUTS_DIR,
    MAX_UPLOAD_MB,
    format_duration,
    format_stats,
    make_output_path,
    reencode_h264,
    setup_logging,
    validate_video_file,
)
from video_processor import VideoError, get_video_info, process_video

setup_logging()
logger = logging.getLogger("objectvision.app")

st.set_page_config(
    page_title="ObjectVision AI | Vision Command Center",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ov-bg: #07111f;
  --ov-panel: #0c1828;
  --ov-panel2: #101f33;
  --ov-text: #edf5ff;
  --ov-muted: #8fa4bd;
  --ov-line: rgba(148, 174, 205, .16);
  --ov-cyan: #56d8ff;
  --ov-violet: #9b8cff;
  --ov-green: #62e6a6;
  --ov-orange: #ffb86b;
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 80% 0%, rgba(86,216,255,.10), transparent 28rem),
    radial-gradient(circle at 10% 25%, rgba(155,140,255,.08), transparent 25rem),
    var(--ov-bg);
  color: var(--ov-text);
}
.block-container { padding-top: 2rem; max-width: 1450px; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.03em; }
.hero {
  padding: 2.4rem 2.6rem;
  border: 1px solid var(--ov-line);
  border-radius: 28px;
  background:
    linear-gradient(135deg, rgba(86,216,255,.10), rgba(155,140,255,.07) 50%, rgba(255,255,255,.02));
  box-shadow: 0 24px 70px rgba(0,0,0,.24);
  position: relative; overflow: hidden;
}
.hero:after {
  content: ""; position:absolute; width:260px; height:260px; right:-80px; top:-100px;
  border:1px solid rgba(86,216,255,.18); border-radius:50%;
  box-shadow:0 0 0 30px rgba(86,216,255,.025),0 0 0 60px rgba(86,216,255,.018);
}
.eyebrow {
  color: var(--ov-cyan); font-size:.78rem; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; margin-bottom:.6rem;
}
.hero-title { font-size: clamp(2.4rem, 5vw, 4.6rem); line-height:.95; margin:0; }
.hero-copy { color:#b9c9da; max-width:760px; font-size:1.05rem; line-height:1.65; margin-top:1rem; }
.pill {
  display:inline-block; border:1px solid rgba(86,216,255,.24); background:rgba(86,216,255,.07);
  color:#bfefff; padding:.38rem .7rem; border-radius:999px; font-size:.75rem; margin:.25rem .3rem .25rem 0;
}
.card {
  background: linear-gradient(145deg, rgba(16,31,51,.92), rgba(9,21,35,.92));
  border:1px solid var(--ov-line); border-radius:20px; padding:1.25rem 1.35rem;
  height:100%; box-shadow:0 12px 35px rgba(0,0,0,.14);
}
.card-title { font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem; margin-bottom:.45rem; }
.muted { color:var(--ov-muted); line-height:1.6; }
.metric-card {
  background:rgba(12,24,40,.86); border:1px solid var(--ov-line); border-radius:18px;
  padding:1rem 1.1rem; min-height:105px;
}
.metric-label { color:var(--ov-muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.09em; }
.metric-value { font-family:'Space Grotesk'; font-size:1.75rem; font-weight:700; margin-top:.25rem; }
.section-label {
  color:#c6d5e5; font-size:.75rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.13em; margin:1.5rem 0 .65rem;
}
.pipeline {
  display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:1rem 0;
}
.step {
  border:1px solid var(--ov-line); background:rgba(255,255,255,.025); border-radius:12px;
  padding:.65rem .8rem; font-size:.82rem;
}
.arrow { color:var(--ov-cyan); }
.status {
  display:inline-flex; align-items:center; gap:.45rem; color:#baf6d7; font-size:.8rem;
}
.dot { width:8px; height:8px; border-radius:50%; background:var(--ov-green); box-shadow:0 0 14px var(--ov-green); }
.notice {
  border-left:3px solid var(--ov-orange); background:rgba(255,184,107,.07);
  padding:.8rem 1rem; border-radius:0 12px 12px 0; color:#d9e3ef;
}
.sidebar-brand { padding:.5rem .2rem 1rem; }
.sidebar-brand .name { font-family:'Space Grotesk'; font-size:1.2rem; font-weight:700; }
.sidebar-brand .tag { color:var(--ov-muted); font-size:.75rem; margin-top:.2rem; }
[data-testid="stSidebar"] { background:#071321; border-right:1px solid var(--ov-line); }
[data-testid="stFileUploader"] {
  border:1px dashed rgba(86,216,255,.30); border-radius:18px; background:rgba(86,216,255,.025);
}
button[kind="primary"] { border-radius:12px; }
.small { font-size:.78rem; color:var(--ov-muted); }
hr { border-color:var(--ov-line) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = None
if "input_path" not in st.session_state:
    st.session_state.input_path = None
if "h264" not in st.session_state:
    st.session_state.h264 = False


@st.cache_resource(show_spinner=False)
def load_detector(model_name: str) -> ObjectDetector:
    return ObjectDetector(model_name)


def metric_card(label: str, value: str, hint: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="small">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar():
    st.sidebar.markdown(
        '<div class="sidebar-brand"><div class="name">◉ OBJECTVISION AI</div>'
        '<div class="tag">Vision Command Center · YOLOv8</div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "WORKSPACE",
        ["Mission Control", "Video Analyzer", "Snapshot Lab", "Run History", "System Intel"],
        label_visibility="visible",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**DETECTION PROFILE**")
    model = st.sidebar.selectbox(
        "Model",
        SUPPORTED_MODELS,
        index=0,
        format_func=lambda x: x.replace("yolov8", "YOLOv8 ").replace(".pt", "").upper(),
        help="Nano is fastest. Larger models can be more accurate but are slower on CPU.",
    )
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.sidebar.selectbox("Inference size", [320, 480, 640], index=2)
    show_labels = st.sidebar.checkbox("Object labels", True)
    show_conf = st.sidebar.checkbox("Confidence scores", True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="small"><b>LOCAL-FIRST</b><br>Processing runs through your Python environment. '
        'No project-specific cloud service is required.</div>',
        unsafe_allow_html=True,
    )
    return page, {
        "model": model,
        "conf": conf,
        "imgsz": imgsz,
        "show_labels": show_labels,
        "show_conf": show_conf,
    }


def page_home():
    st.markdown(
        """
<div class="hero">
  <div class="eyebrow">COMPUTER VISION / DEEP LEARNING</div>
  <div class="hero-title">See what the video<br>is actually showing.</div>
  <div class="hero-copy">
    ObjectVision AI turns prerecorded video into structured visual intelligence using a
    pretrained YOLOv8 detector and OpenCV. Upload a clip, run inference frame-by-frame,
    inspect the detections, and export the annotated result.
  </div>
  <div style="margin-top:1rem">
    <span class="pill">YOLOv8</span><span class="pill">OpenCV</span><span class="pill">PyTorch</span>
    <span class="pill">CPU-ready</span><span class="pill">80 COCO classes</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">SYSTEM SNAPSHOT</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Detection engine", "YOLOv8", "pretrained inference")
    with c2: metric_card("Video pipeline", "OpenCV", "frame-by-frame")
    with c3: metric_card("Deployment", "Local", "standard CPU supported")
    with c4: metric_card("Classes", "80", "COCO object categories")

    st.markdown('<div class="section-label">THE IDEA</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="card"><div class="card-title">01 · Ingest</div><div class="muted">Validate MP4, AVI or MOV input, inspect resolution, frame rate, duration and frame count before inference.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">02 · Understand</div><div class="muted">YOLOv8 evaluates each frame and returns object labels, bounding boxes and confidence scores.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><div class="card-title">03 · Explain</div><div class="muted">Annotated video plus category statistics make the model output understandable instead of opaque.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline"><span class="step">Video input</span><span class="arrow">→</span>'
        '<span class="step">Validation</span><span class="arrow">→</span><span class="step">Frame extraction</span>'
        '<span class="arrow">→</span><span class="step">YOLOv8 inference</span><span class="arrow">→</span>'
        '<span class="step">NMS + confidence</span><span class="arrow">→</span><span class="step">Bounding boxes</span>'
        '<span class="arrow">→</span><span class="step">Statistics</span><span class="arrow">→</span>'
        '<span class="step">Export</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><b>Important:</b> counts are detections accumulated across frames, not unique physical objects. '
        'Persistent tracking is intentionally outside the current core implementation.</div>',
        unsafe_allow_html=True,
    )


def page_video(settings):
    st.title("Video Analyzer")
    st.caption("Turn a prerecorded clip into an annotated detection report.")

    uploaded = st.file_uploader(
        f"Drop a video here · MP4 / AVI / MOV · maximum {MAX_UPLOAD_MB} MB",
        type=["mp4", "avi", "mov"],
    )
    if uploaded is None:
        st.markdown(
            '<div class="card"><div class="card-title">Ready for a scan</div>'
            '<div class="muted">Upload a short test clip to begin. For a presentation demo, '
            '10–20 seconds is usually enough to show the complete workflow.</div></div>',
            unsafe_allow_html=True,
        )
        return

    ok, msg = validate_video_file(uploaded.name, uploaded.size)
    if not ok:
        st.error(msg)
        return

    key = (uploaded.name, uploaded.size)
    if st.session_state.upload_key != key:
        suffix = Path(uploaded.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.close()
        st.session_state.upload_key = key
        st.session_state.input_path = tmp.name
        st.session_state.result = None
        st.session_state.h264 = False

    input_path = st.session_state.input_path
    try:
        info = get_video_info(input_path)
    except VideoError as exc:
        st.error(f"Invalid video: {exc}")
        return

    left, right = st.columns([1.35, .65])
    with left:
        st.markdown("#### Source preview")
        st.video(uploaded.getvalue())
    with right:
        st.markdown("#### Clip intelligence")
        metric_card("Resolution", f"{info.width} × {info.height}", "source dimensions")
        st.write("")
        metric_card("Frame rate", f"{info.fps:.1f} FPS", "source rate")
        st.write("")
        metric_card("Duration", format_duration(info.duration), f"{info.frame_count or 'Unknown'} frames")

    st.markdown('<div class="section-label">ACTIVE PROFILE</div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    with p1: metric_card("Model", settings["model"].replace(".pt","").upper(), "pretrained")
    with p2: metric_card("Threshold", f'{settings["conf"]:.2f}', "confidence cutoff")
    with p3: metric_card("Input size", str(settings["imgsz"]), "inference pixels")
    with p4: metric_card("Labels", "ON" if settings["show_labels"] else "OFF", "overlay mode")

    if info.frame_count > 3000 or info.width >= 1920:
        st.warning("Large clip detected. CPU processing may take longer; YOLOv8n with 320/480 input is the fastest profile.")

    if st.button("▶  START VISION SCAN", type="primary", use_container_width=True):
        try:
            detector = load_detector(settings["model"])
            out_path = make_output_path(OUTPUTS_DIR, Path(uploaded.name).stem)
            bar = st.progress(0.0)
            status = st.empty()

            def on_progress(done: int, total: int):
                pct = min(done / total, 1.0) if total else 0.0
                bar.progress(pct)
                status.markdown(f"**Processing frame {done}**" + (f" / {total}" if total else ""))

            with st.spinner("Running YOLOv8 inference…"):
                result = process_video(
                    input_path, out_path, detector,
                    conf=settings["conf"],
                    show_labels=settings["show_labels"],
                    show_conf=settings["show_conf"],
                    imgsz=settings["imgsz"],
                    progress_callback=on_progress,
                )
                status.markdown("Optimising output for browser playback…")
                result_ok = reencode_h264(result.output_path)

            st.session_state.result = result
            st.session_state.h264 = result_ok
            st.session_state.history.insert(0, {
                "video": uploaded.name,
                "model": settings["model"],
                "confidence": settings["conf"],
                "frames": result.frames_processed,
                "detections": result.total_detections,
                "categories": len(result.category_counts),
                "fps": round(result.avg_fps, 2),
                "duration": format_duration(result.elapsed_seconds),
            })
            st.success("Vision scan complete.")
        except (VideoError, DetectorError) as exc:
            st.error(f"Processing failed: {exc}")
            logger.exception("Processing failed")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            logger.exception("Unexpected error")

    if st.session_state.result is not None:
        show_results(st.session_state.result)


def show_results(result):
    st.markdown('<div class="section-label">ANALYSIS REPORT</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    with m1: metric_card("Frames processed", f"{result.frames_processed:,}", "frames analysed")
    with m2: metric_card("Total detections", f"{result.total_detections:,}", "across all frames")
    with m3: metric_card("Categories", str(len(result.category_counts)), "classes detected")
    with m4: metric_card("Avg speed", f"{result.avg_fps:.1f}", "frames / second")

    st.caption(f"Processing time: {format_duration(result.elapsed_seconds)} · Peak objects in one frame: {result.max_objects_in_frame}")

    v,t = st.columns([1.35,.65])
    with v:
        st.markdown("#### Annotated output")
        if result.output_path.exists():
            data = result.output_path.read_bytes()
            st.video(data)
            if not st.session_state.h264:
                st.caption("Browser preview may depend on your installed codec. The download remains available.")
            st.download_button(
                "⬇  DOWNLOAD ANNOTATED VIDEO",
                data=data,
                file_name=result.output_path.name,
                mime="video/mp4",
                use_container_width=True,
            )
    with t:
        st.markdown("#### Object mix")
        rows = format_stats(result.category_counts)
        if rows:
            chart_data = {r["Category"]: r["Detections"] for r in rows[:12]}
            st.bar_chart(chart_data, height=300)
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No objects detected. Lower the confidence threshold and scan again.")
        st.caption("The category counts represent detections across frames, not unique physical objects.")


def page_snapshot(settings):
    st.title("Snapshot Lab")
    st.caption("A compact visual test bench for single-frame object detection.")
    st.markdown(
        '<div class="card"><div class="card-title">Camera / image workflow</div>'
        '<div class="muted">Capture a snapshot with your webcam, send the frame through the same YOLOv8 detector, '
        'then inspect the returned labels and confidence scores.</div></div>',
        unsafe_allow_html=True,
    )
    try:
        shot = st.camera_input("Capture a frame")
    except Exception as exc:
        st.error(f"Camera unavailable: {exc}")
        return
    if shot is None:
        return
    try:
        frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        detector = load_detector(settings["model"])
        t0 = time.perf_counter()
        dets = detector.detect(frame, conf=settings["conf"], imgsz=settings["imgsz"])
        annotated = draw_detections(frame, dets, settings["show_labels"], settings["show_conf"])
        elapsed = time.perf_counter() - t0
        left,right = st.columns([1.3,.7])
        with left:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{len(dets)} detections · {elapsed:.2f}s", use_container_width=True)
        with right:
            metric_card("Objects found", str(len(dets)), "this frame")
            st.write("")
            metric_card("Inference time", f"{elapsed:.2f}s", "local processing")
            if dets:
                st.markdown("#### Detections")
                st.dataframe(
                    [{"Object": d.label, "Confidence": round(d.confidence, 2)} for d in dets],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No objects crossed the selected confidence threshold.")
    except DetectorError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Could not process the picture: {exc}")


def page_history():
    st.title("Run History")
    st.caption("Session-only comparison of scans completed during this browser session.")
    history = st.session_state.history
    if not history:
        st.markdown(
            '<div class="card"><div class="card-title">No scans yet</div>'
            '<div class="muted">Complete a video scan and its key metrics will appear here for comparison.</div></div>',
            unsafe_allow_html=True,
        )
        return
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-label">QUICK COMPARISON</div>', unsafe_allow_html=True)
    left,right = st.columns(2)
    with left:
        st.markdown("#### Detections per run")
        st.bar_chart({h["video"][:22]: h["detections"] for h in history[:10]}, height=300)
    with right:
        st.markdown("#### Processing speed")
        st.bar_chart({h["video"][:22]: h["fps"] for h in history[:10]}, height=300)


def page_system():
    st.title("System Intel")
    st.caption("A transparent view of what the project does—and what it deliberately does not claim.")
    a,b = st.columns(2)
    with a:
        st.markdown(
            '<div class="card"><div class="card-title">Architecture</div>'
            '<div class="pipeline"><span class="step">User</span><span class="arrow">→</span>'
            '<span class="step">Streamlit</span><span class="arrow">→</span>'
            '<span class="step">Video Processor</span><span class="arrow">→</span>'
            '<span class="step">Detector</span><span class="arrow">→</span>'
            '<span class="step">YOLOv8 / PyTorch</span></div>'
            '<div class="muted">OpenCV handles frame input/output and annotation. Utility functions validate files, '
            'format statistics, create output names and optionally re-encode results for browser playback.</div></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="card"><div class="card-title">Core technologies</div>'
            '<div class="muted"><b>Python</b> — application logic<br>'
            '<b>YOLOv8 / Ultralytics</b> — pretrained object detector<br>'
            '<b>PyTorch</b> — deep-learning runtime<br>'
            '<b>OpenCV</b> — video frames and bounding boxes<br>'
            '<b>NumPy</b> — image arrays<br>'
            '<b>Streamlit</b> — web interface<br>'
            '<b>imageio-ffmpeg</b> — optional browser-friendly encoding</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">INFERENCE LOGIC</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Validate", "Check extension, size and whether OpenCV can read the video."),
        ("02", "Extract", "Read one frame at a time to keep memory usage controlled."),
        ("03", "Infer", "Run the pretrained YOLOv8 model on each frame."),
        ("04", "Filter", "Apply confidence threshold and YOLO non-maximum suppression."),
        ("05", "Annotate", "Draw bounding boxes, labels and optional confidence scores."),
        ("06", "Aggregate", "Accumulate per-category detections and performance metrics."),
        ("07", "Export", "Write the processed frames and provide a downloadable video."),
    ]
    for no,title,desc in steps:
        st.markdown(
            f'<div class="card" style="margin:.5rem 0;display:flex;gap:1rem;align-items:center;">'
            f'<div style="font-family:Space Grotesk;color:#56d8ff;font-weight:700;min-width:32px">{no}</div>'
            f'<div><b>{title}</b><div class="muted">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">LIMITATIONS & RESPONSIBLE USE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice"><b>Pretrained model:</b> this project performs inference; it does not train YOLOv8 from scratch. '
        'The model covers common COCO classes and can miss small, dark, blurred, occluded or out-of-class objects. '
        'Performance varies by hardware and video resolution. Counts are per-frame detections, not unique-object tracking. '
        'Do not use the demo as a safety-critical decision system without proper validation.</div>',
        unsafe_allow_html=True,
    )


def main():
    page, settings = sidebar()
    if page == "Mission Control":
        page_home()
    elif page == "Video Analyzer":
        page_video(settings)
    elif page == "Snapshot Lab":
        page_snapshot(settings)
    elif page == "Run History":
        page_history()
    else:
        page_system()


if __name__ == "__main__":
    main()
