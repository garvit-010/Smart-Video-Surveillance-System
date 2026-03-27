"""
tracking.py
-----------
Stage 5 of the pipeline.

Implements centroid-based multi-object tracking (no external library required).

Algorithm:
  - For every frame, compute the centroid of each bounding box.
  - Match new centroids to existing tracks using minimum Euclidean distance.
  - Assign a persistent integer ID to each track.
  - Retire tracks that have not been matched for MAX_DISAPPEARED frames.
"""

import numpy as np
from collections import OrderedDict
from ..config import MAX_DISAPPEARED, MAX_DISTANCE


class CentroidTracker:
    """Tracks multiple objects across video frames using centroid matching.

    Each object receives a unique integer ID that persists until the object
    has not been detected for MAX_DISAPPEARED consecutive frames.

    Attributes:
        next_id       : Counter for assigning new IDs.
        objects       : {id: centroid_array} for currently tracked objects.
        disappeared   : {id: frame_count_since_last_detection}.
    """

    def __init__(self) -> None:
        """Initialise tracker with empty state."""
        self.next_id    : int                 = 0
        self.objects    : OrderedDict         = OrderedDict()   # id → centroid
        self.disappeared: OrderedDict         = OrderedDict()   # id → count

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_centroid(bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Compute the centroid [cx, cy] of an (x, y, w, h) bounding box."""
        x, y, w, h = bbox
        return np.array([x + w // 2, y + h // 2], dtype=int)

    def _register(self, centroid: np.ndarray) -> None:
        """Register a new object with the next available ID."""
        self.objects[self.next_id]    = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id: int) -> None:
        """Remove a track that has disappeared for too long."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, bboxes: list[tuple[int, int, int, int]]) -> OrderedDict:
        """Update tracker with bounding boxes from the current frame.

        Args:
            bboxes: List of (x, y, w, h) detections.  May be empty.

        Returns:
            OrderedDict mapping each active object ID to its current centroid.
        """
        # Case 1: no detections → increment disappeared counters
        if not bboxes:
            stale = [oid for oid, cnt in self.disappeared.items()
                     if cnt + 1 > MAX_DISAPPEARED]
            for oid in stale:
                self._deregister(oid)
            for oid in self.disappeared:
                self.disappeared[oid] += 1
            return self.objects

        # Compute input centroids
        input_centroids = np.array(
            [self._bbox_to_centroid(b) for b in bboxes], dtype=int
        )

        # Case 2: no existing tracks → register all as new
        if not self.objects:
            for c in input_centroids:
                self._register(c)
            return self.objects

        # Case 3: match existing tracks to new centroids via distance matrix
        obj_ids       = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()), dtype=int)

        # Euclidean distance matrix: shape (n_objects, n_inputs)
        D = np.linalg.norm(
            obj_centroids[:, np.newaxis] - input_centroids[np.newaxis], axis=2
        )

        # Greedy assignment: match closest pairs first
        rows = D.min(axis=1).argsort()               # objects ranked by min dist
        cols = D.argmin(axis=1)[rows]                # best input for each object

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > MAX_DISTANCE:
                continue
            oid = obj_ids[row]
            self.objects[oid]     = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing tracks → increment disappeared
        for row in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > MAX_DISAPPEARED:
                self._deregister(oid)

        # Unmatched new centroids → register as new objects
        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col])

        return self.objects
