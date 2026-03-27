"""
segmentation.py
---------------
Stage 3 of the pipeline.

Separates moving foreground objects from the static background using:
  1. MOG2 (Mixture of Gaussians v2) background subtractor
  2. Binary thresholding + morphological cleanup to remove noise
"""

import cv2
import numpy as np
from ..config import (
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    MOG2_DETECT_SHADOWS,
    MORPH_KERNEL_SIZE,
)


class BackgroundSubtractor:
    """Wraps OpenCV's MOG2 subtractor with morphological post-processing.

    Usage:
        subtractor = BackgroundSubtractor()
        for frame in frames:
            mask = subtractor.apply(frame)
    """

    def __init__(self) -> None:
        """Initialise the MOG2 model and morphological kernel."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Compute the foreground mask for a single frame.

        Steps:
          1. MOG2 subtraction → raw binary mask
          2. Morphological opening  (removes small blobs / salt noise)
          3. Morphological closing  (fills small gaps inside objects)

        Args:
            frame: BGR video frame (the subtractor keeps the colour frame
                   internally; grayscale conversion is handled by MOG2).

        Returns:
            Cleaned binary foreground mask (H×W, dtype uint8, values 0/255).
        """
        # Step 1: raw foreground mask
        raw_mask = self._mog2.apply(frame)

        # Step 2: threshold to strictly binary (discards shadows if any)
        _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

        # Step 3: morphological cleanup
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self._kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel)

        return closed


def segment_frame(subtractor: BackgroundSubtractor, frame: np.ndarray) -> np.ndarray:
    """Apply background subtraction and return the foreground mask.

    Convenience wrapper so pipeline.py can call a plain function.

    Args:
        subtractor: Stateful BackgroundSubtractor instance.
        frame:      Current BGR frame.

    Returns:
        Binary foreground mask.
    """
    return subtractor.apply(frame)
