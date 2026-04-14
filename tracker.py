"""
tracker.py — YOLOv8 + ByteTrack people detector / tracker.

Responsibilities:
  - Load a YOLOv8 model once at startup.
  - Run inference + tracking on every frame via `track_frame()`.
  - Return a clean list of TrackedPerson objects (no YOLO internals leak out).

ByteTrack is bundled with the Ultralytics package (bytetrack.yaml) so no
extra installation is needed beyond `pip install ultralytics`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from utils import TrackedPerson, compute_centroid

logger = logging.getLogger(__name__)


# COCO class index for "person"
_PERSON_CLASS_ID = 0


class PeopleTracker:
    """
    Wraps YOLOv8 with ByteTrack to produce per-frame lists of tracked people.

    Usage
    -----
        tracker = PeopleTracker(model_path="yolov8n.pt")
        persons = tracker.track_frame(bgr_frame)

    Parameters
    ----------
    model_path : str
        Path to a local .pt file or an Ultralytics model name
        (e.g. "yolov8n.pt", "yolov8s.pt").  The nano model is the
        fastest; use "yolov8m.pt" for better accuracy at the cost of
        more GPU memory.
    conf : float
        Minimum detection confidence threshold (0–1).
    iou : float
        IoU threshold for NMS inside the tracker.
    device : str | None
        Torch device string: "cpu", "cuda", "mps", or None to auto-select.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device  # None → auto (GPU if available, else CPU)

        # Warm up the model so the first real frame isn't slow
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_frame(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Run detection + ByteTrack on one BGR frame.

        ByteTrack requires `persist=True` so that the tracker keeps its
        internal state (Kalman filter, track buffers) between calls.
        Without it, IDs would reset every frame.

        Args:
            frame: A BGR image as a NumPy uint8 array (H×W×3).

        Returns:
            List of TrackedPerson objects.  May be empty if nobody is
            detected or if a frame is lost.
        """
        # `model.track()` runs detection then ByteTrack in one call.
        results = self.model.track(
            source=frame,
            persist=True,           # ← CRITICAL: maintain tracker state
            conf=self.conf,
            iou=self.iou,
            classes=[_PERSON_CLASS_ID],
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,          # suppress per-frame console spam
            stream=False,
        )

        return self._parse_results(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(results) -> List[TrackedPerson]:
        """
        Convert raw Ultralytics Results into TrackedPerson dataclasses.

        If tracking has not yet assigned IDs (e.g. first frame warm-up),
        the boxes.id tensor may be None — we skip those frames gracefully.
        """
        persons: List[TrackedPerson] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                # Tracker hasn't initialised tracks yet
                continue

            # Extract tensors → NumPy in one shot for speed
            ids         = boxes.id.int().cpu().numpy()        # (N,)
            xyxy        = boxes.xyxy.int().cpu().numpy()      # (N, 4)
            confs       = boxes.conf.float().cpu().numpy()    # (N,)

            for track_id, box, conf in zip(ids, xyxy, confs):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bbox = (x1, y1, x2, y2)
                persons.append(
                    TrackedPerson(
                        track_id=int(track_id),
                        bbox=bbox,
                        confidence=float(conf),
                        centroid=compute_centroid(bbox),
                    )
                )

        return persons
