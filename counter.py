"""
counter.py — Virtual line-crossing counter with double-count prevention.

Algorithm overview
------------------
For every tracked person we maintain which "side" of the virtual line
their centroid was on in the *previous* frame.  When the sign of the
cross-product changes we record a crossing:

    cross = (line_end - line_start) × (centroid - line_start)

    cross > 0  →  side = +1  ("left" or "above" depending on orientation)
    cross < 0  →  side = -1  ("right" or "below")

Crossing direction:
    +1 → -1  means the object moved from side A to side B  → counted as IN
    -1 → +1  means the object moved from side B to side A  → counted as OUT

Double-count prevention
-----------------------
Each track_id can trigger at most ONE crossing event per entry into the
`cooldown` window.  After a crossing we put the ID into a cooldown set for
`cooldown_frames` frames.  This prevents a centroid that oscillates around
the line boundary from generating spurious counts.

Additionally, IDs that leave the scene (not seen for > `max_absent_frames`)
are cleaned up from the state dictionaries to prevent memory leaks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from utils import TrackedPerson, draw_counting_line

logger = logging.getLogger(__name__)


class LineCrossingCounter:
    """
    Counts IN / OUT crossings over a user-defined virtual line segment.

    Parameters
    ----------
    line_start : (int, int)
        Pixel coordinate of the first endpoint of the counting line.
    line_end : (int, int)
        Pixel coordinate of the second endpoint.
    in_direction : int
        Which crossing direction maps to "IN".
        +1  → objects going from side +1 to side -1 are IN
        -1  → objects going from side -1 to side +1 are IN
        Default: +1 (top-to-bottom for a horizontal line).
    cooldown_frames : int
        Frames to ignore repeated crossings for the same track ID.
    max_absent_frames : int
        Frames before a track with no detections is removed from state.
    """

    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        in_direction: int = 1,
        cooldown_frames: int = 10,
        max_absent_frames: int = 30,
    ) -> None:
        self.line_start = line_start
        self.line_end = line_end
        self.in_direction = in_direction
        self.cooldown_frames = cooldown_frames
        self.max_absent_frames = max_absent_frames

        # Cumulative counters
        self.count_in: int = 0
        self.count_out: int = 0

        # Per-ID state: track_id → last known side (+1 or -1)
        self._prev_side: Dict[int, int] = {}

        # Per-ID cooldown: track_id → frames remaining in cooldown
        self._cooldown: Dict[int, int] = {}

        # Per-ID absence counter: track_id → frames not seen
        self._absent: Dict[int, int] = {}

        # Which IDs were seen in the most recent frame
        self._seen_this_frame: Set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, persons: List[TrackedPerson]) -> Tuple[int, int]:
        """
        Process one frame's worth of tracked persons.

        Call this once per frame in order.  Returns the *current cumulative*
        (count_in, count_out) tuple.

        Args:
            persons: List of TrackedPerson objects from the tracker.

        Returns:
            (count_in, count_out) — cumulative totals.
        """
        self._seen_this_frame = {p.track_id for p in persons}

        for person in persons:
            self._process_person(person)

        # Tick absence counters and evict stale IDs
        self._cleanup_stale_ids()

        return self.count_in, self.count_out

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the counting line on the given frame.

        The line is yellow by default; you can change the colour in
        utils.draw_counting_line if needed.
        """
        return draw_counting_line(frame, self.line_start, self.line_end)

    def reset(self) -> None:
        """Reset all counts and internal state (e.g. for a new video)."""
        self.count_in = 0
        self.count_out = 0
        self._prev_side.clear()
        self._cooldown.clear()
        self._absent.clear()
        self._seen_this_frame.clear()
        logger.debug("Counter reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _side_of_line(self, point: Tuple[int, int]) -> int:
        """
        Return +1 or -1 indicating which side of the line 'point' is on.
        Uses the z-component of the 2-D cross product.
        Returns 0 if the point sits exactly on the line (very rare).
        """
        ax = self.line_end[0] - self.line_start[0]
        ay = self.line_end[1] - self.line_start[1]
        bx = point[0] - self.line_start[0]
        by = point[1] - self.line_start[1]
        cross = ax * by - ay * bx
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0  # exactly on the line — treat as no crossing

    def _process_person(self, person: TrackedPerson) -> None:
        """Evaluate line crossing for a single person in the current frame."""
        tid = person.track_id
        current_side = self._side_of_line(person.centroid)

        if current_side == 0:
            # Centroid exactly on the line; don't record a crossing
            return

        # --- Cooldown tick -------------------------------------------------
        if tid in self._cooldown:
            self._cooldown[tid] -= 1
            if self._cooldown[tid] <= 0:
                del self._cooldown[tid]
            # While in cooldown, update side but don't count
            self._prev_side[tid] = current_side
            self._absent[tid] = 0
            return

        # --- First time we see this ID -------------------------------------
        if tid not in self._prev_side:
            self._prev_side[tid] = current_side
            self._absent[tid] = 0
            return

        prev_side = self._prev_side[tid]

        # --- Check for crossing --------------------------------------------
        if prev_side != current_side:
            # Crossing detected!
            if prev_side == self.in_direction:
                # Moved from the "IN" side to the other → counted as IN
                self.count_in += 1
                logger.debug("ID %d crossed IN  (total IN=%d)", tid, self.count_in)
            else:
                # Moved from the other side → counted as OUT
                self.count_out += 1
                logger.debug("ID %d crossed OUT (total OUT=%d)", tid, self.count_out)

            # Start cooldown to block double-counts
            self._cooldown[tid] = self.cooldown_frames

        # Always update the stored side and reset absence
        self._prev_side[tid] = current_side
        self._absent[tid] = 0

    def _cleanup_stale_ids(self) -> None:
        """
        Increment absence counters for IDs not seen this frame and remove
        any that have been gone for too long.  Prevents unbounded memory use.
        """
        all_known = set(self._prev_side.keys())
        absent_ids = all_known - self._seen_this_frame

        for tid in absent_ids:
            self._absent[tid] = self._absent.get(tid, 0) + 1
            if self._absent[tid] > self.max_absent_frames:
                # Track has disappeared — clean up all its state
                self._prev_side.pop(tid, None)
                self._cooldown.pop(tid, None)
                self._absent.pop(tid, None)
                logger.debug("Evicted stale track ID %d", tid)
