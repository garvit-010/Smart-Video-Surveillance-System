"""Object detection using contour analysis on foreground masks."""

import cv2
import numpy as np
from ..config import MIN_CONTOUR_AREA, BBOX_COLOR, BBOX_THICKNESS


def find_contours(mask: np.ndarray) -> list:
    """Extract contours from a binary foreground mask."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def filter_contours(contours: list) -> list:
    """Filter out small contours below the minimum area threshold."""
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


def contours_to_bboxes(contours: list) -> list[tuple[int, int, int, int]]:
    """Convert contours to axis-aligned bounding boxes."""
    return [cv2.boundingRect(c) for c in contours]


def draw_detections(frame: np.ndarray, bboxes: list) -> np.ndarray:
    """Draw bounding boxes on a frame."""
    annotated = frame.copy()
    for x, y, w, h in bboxes:
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            BBOX_COLOR,
            BBOX_THICKNESS,
        )
    return annotated


def detect_objects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect objects as bounding boxes from a foreground mask."""
    contours = find_contours(mask)
    filtered  = filter_contours(contours)
    bboxes    = contours_to_bboxes(filtered)
    return bboxes
