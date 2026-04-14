"""
main.py — Core video-processing pipeline.

Can be used in two ways:
  1. CLI: `python main.py --source video.mp4`
  2. Library: `import main; for frame, counts in main.run_pipeline(...): ...`

The pipeline is implemented as a *generator* so that the Streamlit UI
(ui.py) can consume processed frames one at a time without loading the
entire video into memory.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from counter import LineCrossingCounter
from tracker import PeopleTracker
from utils import (
    draw_bbox,
    draw_hud,
    frame_to_rgb,
    get_video_info,
    make_video_writer,
    resize_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline generator
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str | int,
    model_path: str = "yolov8n.pt",
    line_ratio: float = 0.50,
    line_axis: str = "horizontal",
    in_direction: int = 1,
    conf: float = 0.35,
    iou: float = 0.50,
    output_path: Optional[str] = None,
    max_dim: int = 1280,
    device: Optional[str] = None,
    show_conf: bool = True,
) -> Generator[Tuple[np.ndarray, int, int, int], None, None]:
    """
    Process every frame of a video and yield annotated frames + live counts.

    This is the heart of the application.  All other modules (main CLI,
    Streamlit UI) call this function.

    Args:
        source:       Path to a video file, or 0 / device index for webcam.
        model_path:   YOLOv8 model weight file (downloaded automatically).
        line_ratio:   Where to place the counting line as a fraction of the
                      frame dimension (0.0 = top/left, 1.0 = bottom/right).
        line_axis:    "horizontal" → line spans full width at `line_ratio`
                      of the height.  "vertical" → line spans full height
                      at `line_ratio` of the width.
        in_direction: +1 or -1 — which crossing direction is "IN".
                      For a horizontal line: +1 = top-to-bottom is IN.
        conf:         YOLO detection confidence threshold.
        iou:          NMS IoU threshold.
        output_path:  If given, annotated video is saved here (.mp4).
        max_dim:      Resize frames so the longest side ≤ this value.
                      Reduces memory and speeds up inference.
        device:       Torch device ("cpu", "cuda", "mps", or None = auto).
        show_conf:    Show confidence score in bounding-box labels.

    Yields:
        Tuple of:
          - annotated BGR frame (np.ndarray)
          - current count_in  (int)
          - current count_out (int)
          - frame index       (int)
    """
    # ------------------------------------------------------------------ open
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source!r}")

    info = get_video_info(cap)
    fps      = info["fps"]
    orig_w   = info["width"]
    orig_h   = info["height"]
    total    = info["total_frames"]
    logger.info(
        "Video: %dx%d @ %.1f fps, %d frames total",
        orig_w, orig_h, fps, total,
    )

    # Read first frame to determine actual display dimensions after resize
    ret, probe = cap.read()
    if not ret:
        raise IOError("Could not read first frame from video.")
    probe = resize_frame(probe, max_dim)
    frame_h, frame_w = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # ------------------------------------------------------ counting line
    if line_axis == "horizontal":
        line_y = int(frame_h * line_ratio)
        line_start = (0, line_y)
        line_end   = (frame_w, line_y)
    else:  # vertical
        line_x = int(frame_w * line_ratio)
        line_start = (line_x, 0)
        line_end   = (line_x, frame_h)

    # -------------------------------------------------------- init modules
    tracker = PeopleTracker(model_path=model_path, conf=conf, iou=iou, device=device)
    counter = LineCrossingCounter(
        line_start=line_start,
        line_end=line_end,
        in_direction=in_direction,
    )

    # ---------------------------------------------------- optional writer
    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        writer = make_video_writer(output_path, fps, frame_w, frame_h)
        logger.info("Saving output to: %s", output_path)

    # -------------------------------------------------------- main loop
    frame_idx = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Resize for consistent processing speed
            frame = resize_frame(frame, max_dim)

            # --- Detection + tracking ---
            persons = tracker.track_frame(frame)

            # --- Line-crossing logic ---
            count_in, count_out = counter.update(persons)

            # --- Draw annotations ---
            # 1. Counting line
            counter.draw(frame)

            # 2. Bounding boxes + IDs
            for person in persons:
                draw_bbox(frame, person, show_conf=show_conf)

            # 3. HUD overlay
            # Compute rolling FPS every 15 frames
            frame_idx += 1
            if frame_idx % 15 == 0:
                elapsed = time.perf_counter() - fps_timer
                display_fps = 15.0 / elapsed
                fps_timer = time.perf_counter()

            draw_hud(frame, count_in, count_out, fps=display_fps)

            # --- Write to disk (optional) ---
            if writer:
                writer.write(frame)

            yield frame, count_in, count_out, frame_idx

    finally:
        cap.release()
        if writer:
            writer.release()
        logger.info(
            "Done — %d frames processed.  IN=%d  OUT=%d",
            frame_idx, counter.count_in, counter.count_out,
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time People Counter with YOLOv8 + ByteTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",      default="0",          help="Video file path or webcam index")
    p.add_argument("--model",       default="yolov8n.pt", help="YOLOv8 model weights")
    p.add_argument("--output",      default=None,         help="Save annotated video to this path")
    p.add_argument("--line-ratio",  default=0.50, type=float, help="Line position (fraction of frame)")
    p.add_argument("--line-axis",   default="horizontal", choices=["horizontal", "vertical"])
    p.add_argument("--in-dir",      default=1, type=int,  help="+1 or -1 for IN direction")
    p.add_argument("--conf",        default=0.35, type=float)
    p.add_argument("--iou",         default=0.50, type=float)
    p.add_argument("--max-dim",     default=1280, type=int, help="Resize longest edge to this")
    p.add_argument("--no-display",  action="store_true",  help="Skip cv2.imshow (headless)")
    p.add_argument("--device",      default=None,         help="cuda / cpu / mps")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Allow integer webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    window = "People Counter  [Q to quit]"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for frame, count_in, count_out, idx in run_pipeline(
        source=source,
        model_path=args.model,
        line_ratio=args.line_ratio,
        line_axis=args.line_axis,
        in_direction=args.in_dir,
        conf=args.conf,
        iou=args.iou,
        output_path=args.output,
        max_dim=args.max_dim,
        device=args.device,
    ):
        if not args.no_display:
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                logger.info("User requested exit.")
                break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
