"""
ui.py — Streamlit interface for the People Counting System.

Layout
------
  Sidebar : all configuration controls
  Main    : video upload → process → live frame display + count metrics

Run with:
    streamlit run ui.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import streamlit as st

from main import run_pipeline
from utils import frame_to_rgb

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="People Counter",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — dark card styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .metric-card {
            background: #1e1e2e;
            border-radius: 12px;
            padding: 20px 30px;
            text-align: center;
        }
        .metric-label { color: #a0a0b0; font-size: 0.9rem; letter-spacing: 0.08em; }
        .metric-value { font-size: 2.8rem; font-weight: 700; line-height: 1.1; }
        .metric-in    { color: #6ee7b7; }   /* green */
        .metric-out   { color: #93c5fd; }   /* blue  */
        .metric-total { color: #fbbf24; }   /* amber */
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

def sidebar_controls() -> dict:
    """Render all sidebar widgets and return a config dict."""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.subheader("Model")
    model_choice = st.sidebar.selectbox(
        "YOLOv8 variant",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=0,
        help="Nano is fastest; Large is most accurate.",
    )

    st.sidebar.subheader("Detection")
    conf = st.sidebar.slider(
        "Confidence threshold", 0.10, 0.90, 0.35, 0.05,
        help="Lower = more detections (more false positives).",
    )
    iou = st.sidebar.slider(
        "IoU / NMS threshold", 0.20, 0.90, 0.50, 0.05,
        help="Suppress overlapping boxes above this IoU.",
    )

    st.sidebar.subheader("Counting Line")
    line_axis = st.sidebar.radio(
        "Line orientation", ["horizontal", "vertical"], index=1,
        help="Use 'vertical' for side-view street/corridor footage. "
             "Use 'horizontal' for top-down entrance cameras."
    )
    line_ratio = st.sidebar.slider(
        "Line position (fraction of frame)",
        0.10, 0.90, 0.50, 0.05,
        help="0.5 = centre of the frame.",
    )
    in_direction = st.sidebar.radio(
        "Direction counted as IN",
        options=[1, -1],
        format_func=lambda x: "Top → Bottom / Left → Right" if x == 1 else "Bottom → Top / Right → Left",
        index=0,
    )

    st.sidebar.subheader("Performance")
    max_dim = st.sidebar.select_slider(
        "Max frame dimension (px)",
        options=[480, 640, 720, 1080, 1280],
        value=720,
        help="Smaller = faster inference, lower resolution display.",
    )
    show_conf = st.sidebar.checkbox("Show confidence scores", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with YOLOv8 + ByteTrack · Streamlit")

    return {
        "model_path":   model_choice,
        "conf":         conf,
        "iou":          iou,
        "line_axis":    line_axis,
        "line_ratio":   line_ratio,
        "in_direction": in_direction,
        "max_dim":      max_dim,
        "show_conf":    show_conf,
    }


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

def render_metrics(count_in: int, count_out: int) -> None:
    """Render three metric cards: IN, OUT, NET."""
    net = count_in - count_out
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ENTERED (IN)</div>
                <div class="metric-value metric-in">{count_in}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EXITED (OUT)</div>
                <div class="metric-value metric-out">{count_out}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">NET OCCUPANCY</div>
                <div class="metric-value metric-total">{net:+d}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("👥 Real-time People Counting System")
    st.caption("YOLOv8 detection · ByteTrack tracking · Virtual line crossing")

    config = sidebar_controls()

    # ----------------------------------------------------------------- upload
    st.subheader("1 · Upload Video")
    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Short clips (< 5 min) work best for demo purposes.",
    )

    if uploaded is None:
        st.info("Upload a video above to get started.", icon="ℹ️")
        st.stop()

    # Save to a temp file (Streamlit gives us a BytesIO, OpenCV needs a path)
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp_path = tmp.name

    # ---------------------------------------------------------------- file info
    file_size_mb = Path(tmp_path).stat().st_size / 1_048_576
    cap_probe = cv2.VideoCapture(tmp_path)
    probe_fps   = cap_probe.get(cv2.CAP_PROP_FPS) or 0
    probe_w     = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    probe_h     = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    with st.expander("Video metadata", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resolution", f"{probe_w}×{probe_h}")
        c2.metric("FPS",        f"{probe_fps:.1f}")
        c3.metric("Frames",     probe_total)
        c4.metric("Size",       f"{file_size_mb:.1f} MB")

    # --------------------------------------------------------------- run button
    st.subheader("2 · Run Detection")
    run_btn = st.button("▶ Start Processing", type="primary", use_container_width=True)

    if not run_btn:
        st.stop()

    # --------------------------------------------------------------- processing
    st.subheader("3 · Live Output")

    # Placeholders that will be updated in-place every frame
    frame_placeholder  = st.empty()
    metric_placeholder = st.empty()
    progress_bar       = st.progress(0, text="Processing frames…")
    status_text        = st.empty()

    final_in = final_out = 0
    start_t = time.perf_counter()

    try:
        for rgb_frame, count_in, count_out, frame_idx in _pipeline_rgb(
            tmp_path, config
        ):
            # --- Live frame display ---
            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True,
                caption=f"Frame {frame_idx}",
            )

            # --- Live metric cards ---
            with metric_placeholder.container():
                render_metrics(count_in, count_out)

            # --- Progress bar ---
            if probe_total > 0:
                progress = min(frame_idx / probe_total, 1.0)
                elapsed  = time.perf_counter() - start_t
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                progress_bar.progress(
                    progress,
                    text=f"Frame {frame_idx}/{probe_total} · {proc_fps:.1f} fps",
                )

            final_in, final_out = count_in, count_out

    except Exception as exc:
        st.error(f"Processing error: {exc}", icon="🚨")
        raise

    # --------------------------------------------------------------- summary
    progress_bar.empty()
    elapsed_total = time.perf_counter() - start_t
    status_text.success(
        f"Processing complete — {frame_idx} frames in {elapsed_total:.1f}s "
        f"({frame_idx / elapsed_total:.1f} fps avg)",
        icon="✅",
    )

    st.subheader("4 · Final Counts")
    render_metrics(final_in, final_out)


# ---------------------------------------------------------------------------
# Helper — wraps run_pipeline to convert BGR → RGB before yielding
# ---------------------------------------------------------------------------

def _pipeline_rgb(video_path: str, cfg: dict):
    """
    Thin wrapper around run_pipeline that converts each frame BGR→RGB
    so Streamlit can display it without a colour swap artifact.
    """
    for frame, count_in, count_out, idx in run_pipeline(
        source=video_path,
        model_path=cfg["model_path"],
        line_ratio=cfg["line_ratio"],
        line_axis=cfg["line_axis"],
        in_direction=cfg["in_direction"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        max_dim=cfg["max_dim"],
        show_conf=cfg["show_conf"],
    ):
        yield frame_to_rgb(frame), count_in, count_out, idx


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
