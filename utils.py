"""
utils.py — Drawing helpers, video utilities, and shared constants.

All pure-helper functions live here so the other modules stay focused
on their single responsibility.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackedPerson:
    """Represents one detected & tracked person in a single frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    centroid: Tuple[int, int]          # (cx, cy)


# ---------------------------------------------------------------------------
# Colour palette — unique colour per track ID (deterministic)
# ---------------------------------------------------------------------------

def id_to_color(track_id: int) -> Tuple[int, int, int]:
    """
    Return a deterministic BGR colour for a given track ID.
    Uses a golden-ratio-based hue spread so nearby IDs look distinct.
    """
    hue = int((track_id * 37) % 180)          # spread hues 0-179
    hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ---------------------------------------------------------------------------
# Bounding-box & label drawing
# ---------------------------------------------------------------------------

def draw_bbox(
    frame: np.ndarray,
    person: TrackedPerson,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Draw a rounded bounding box and ID label for one tracked person.

    Args:
        frame:      BGR image (modified in-place and returned).
        person:     TrackedPerson dataclass instance.
        show_conf:  Whether to append the confidence score to the label.

    Returns:
        Annotated frame (same array, modified in-place).
    """
    x1, y1, x2, y2 = person.bbox
    color = id_to_color(person.track_id)
    thickness = 2

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background + text
    label = f"ID:{person.track_id}"
    if show_conf:
        label += f"  {person.confidence:.0%}"

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = max(y1 - 4, th + 4)

    cv2.rectangle(
        frame,
        (x1, label_y - th - baseline - 2),
        (x1 + tw + 4, label_y + baseline - 2),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Centroid dot
    cv2.circle(frame, person.centroid, 4, color, -1)
    return frame


def draw_counting_line(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the virtual counting line and endpoint markers."""
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, pt1, 5, color, -1)
    cv2.circle(frame, pt2, 5, color, -1)
    return frame


def draw_hud(
    frame: np.ndarray,
    count_in: int,
    count_out: int,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Render a semi-transparent HUD panel in the top-left corner showing
    IN / OUT counts and processing FPS.

    Args:
        frame:     BGR image (modified in-place).
        count_in:  Cumulative people counted as entering.
        count_out: Cumulative people counted as exiting.
        fps:       Current processing frame rate (0 to hide).

    Returns:
        Annotated frame.
    """
    panel_w, panel_h = 210, 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"IN  : {count_in:>4}", (18, 36),  font, 0.7, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT : {count_out:>4}", (18, 66),  font, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
    if fps > 0:
        cv2.putText(frame, f"FPS : {fps:>5.1f}", (18, 96), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_video_info(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """Return a dict of metadata from an open VideoCapture."""
    return {
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def make_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an MP4 VideoWriter for the given path and parameters."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def resize_frame(frame: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """
    Downscale a frame so its longest dimension ≤ max_dim,
    preserving aspect ratio.  Returns the original frame if already small.
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def compute_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Return the centroid (cx, cy) of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB (for Streamlit / PIL display)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
