"""Centroid-based multi-object tracking across video frames."""

import numpy as np
from collections import OrderedDict
from ..config import MAX_DISAPPEARED, MAX_DISTANCE


class CentroidTracker:
    """Tracks multiple objects using centroid matching."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.next_id    : int                 = 0
        self.objects    : OrderedDict         = OrderedDict()   # id → centroid
        self.disappeared: OrderedDict         = OrderedDict()   # id → count

    # ── internal helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_centroid(bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Compute centroid from axis-aligned bounding box."""
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
        """Update tracker with detections and return current tracked objects."""
        if not bboxes:
            stale = [oid for oid, cnt in self.disappeared.items()
                     if cnt + 1 > MAX_DISAPPEARED]
            for oid in stale:
                self._deregister(oid)
            for oid in self.disappeared:
                self.disappeared[oid] += 1
            return self.objects

        input_centroids = np.array(
            [self._bbox_to_centroid(b) for b in bboxes], dtype=int
        )

        if not self.objects:
            for c in input_centroids:
                self._register(c)
            return self.objects

        obj_ids       = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()), dtype=int)

        D = np.linalg.norm(
            obj_centroids[:, np.newaxis] - input_centroids[np.newaxis], axis=2
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

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

        for row in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > MAX_DISAPPEARED:
                self._deregister(oid)

        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col])

        return self.objects
