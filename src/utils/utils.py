"""Utility functions for logging, video I/O, and frame annotations."""

import cv2
import os
import time
import numpy as np
from ..config import (
    OUTPUT_DIR,
    FOURCC,
    OVERLAY_FONT_SCALE,
    OVERLAY_FONT_COLOR,
    OVERLAY_FONT_THICKNESS,
)


# ── Logging ───────────────────────────────────────────────────────────────────

class Logger:
    """Simple console logger."""

    @staticmethod
    def info(msg: str) -> None:
        print(f"[INFO]  {msg}")

    @staticmethod
    def warn(msg: str) -> None:
        print(f"[WARN]  {msg}")

    @staticmethod
    def error(msg: str) -> None:
        print(f"[ERROR] {msg}")




log = Logger()

def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file for reading."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: '{path}'")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: '{path}'")

    return cap


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Extract video metadata from an open VideoCapture."""
    return {
        "width"        : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height"       : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps"          : cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "total_frames" : int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def create_video_writer(output_path: str, props: dict) -> cv2.VideoWriter:
    """Create a video writer for the output file."""
    os.makedirs(os.path.dirname(output_path) or OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    return cv2.VideoWriter(
        output_path, fourcc, props["fps"], (props["width"], props["height"])
    )


# ── Overlay helpers ────────────────────────────────────────────────────────────────

def put_overlay_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
) -> np.ndarray:
    """Render text overlay on a frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        OVERLAY_FONT_COLOR,
        OVERLAY_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_tracking_ids(frame: np.ndarray, tracked_objects: dict) -> np.ndarray:
    """Draw tracking IDs and centroids on a frame."""
    for obj_id, centroid in tracked_objects.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ID {obj_id}",
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed seconds."""
        return time.perf_counter() - self._start

    def elapsed_str(self) -> str:
        """Return formatted elapsed-time string."""
        secs = self.elapsed()
        return f"{secs:.2f}s"


# ── Summary Printing ──────────────────────────────────────────────────────────

def print_summary(
    total_frames: int,
    total_detections: int,
    elapsed: float,
    output_path: str,
) -> None:
    """Print a processing summary to stdout."""
    sep = "─" * 50
    print(f"\n{sep}")
    print("  PROCESSING SUMMARY")
    print(sep)
    print(f"  Frames processed  : {total_frames}")
    print(f"  Total detections  : {total_detections}")
    print(f"  Processing time   : {elapsed:.2f}s")
    print(f"  Avg FPS (pipeline): {total_frames / max(elapsed, 1e-6):.1f}")
    print(f"  Output saved to   : {output_path}")
    print(f"{sep}\n")
