"""
detection.py
------------
Stage 4 of the pipeline.

Performs contour-based object detection on the foreground mask produced by
segmentation.py.  Each sufficiently large connected region is wrapped in a
bounding box and classified as a detected object.
"""

import cv2
import numpy as np
from ..config import MIN_CONTOUR_AREA, BBOX_COLOR, BBOX_THICKNESS


def find_contours(mask: np.ndarray) -> list:
    """Extract external contours from a binary foreground mask.

    Args:
        mask: Binary foreground mask (output of segmentation stage).

    Returns:
        List of contours (each contour is an (N, 1, 2) numpy array).
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def filter_contours(contours: list) -> list:
    """Discard contours whose area is below the minimum threshold.

    Args:
        contours: Raw list of contours from find_contours().

    Returns:
        Filtered list containing only contours with area ≥ MIN_CONTOUR_AREA.
    """
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


def contours_to_bboxes(contours: list) -> list[tuple[int, int, int, int]]:
    """Convert a list of contours to axis-aligned bounding boxes.

    Args:
        contours: Filtered list of contours.

    Returns:
        List of (x, y, w, h) tuples — top-left corner + width/height.
    """
    return [cv2.boundingRect(c) for c in contours]


def draw_detections(frame: np.ndarray, bboxes: list) -> np.ndarray:
    """Draw bounding boxes on a copy of the frame.

    Args:
        frame:  BGR frame to annotate (modified in-place on a copy).
        bboxes: List of (x, y, w, h) bounding boxes.

    Returns:
        Annotated BGR frame.
    """
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
    """Full detection pipeline: mask → bounding boxes.

    Args:
        mask: Binary foreground mask.

    Returns:
        List of (x, y, w, h) bounding boxes for detected objects.
    """
    contours = find_contours(mask)
    filtered  = filter_contours(contours)
    bboxes    = contours_to_bboxes(filtered)
    return bboxes
