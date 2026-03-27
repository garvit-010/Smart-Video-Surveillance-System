"""Background subtraction using MOG2 with morphological post-processing."""

import cv2
import numpy as np
from ..config import (
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    MOG2_DETECT_SHADOWS,
    MORPH_KERNEL_SIZE,
)


class BackgroundSubtractor:
    """MOG2-based background subtractor with morphological cleanup."""

    def __init__(self) -> None:
        """Initialize MOG2 model and morphological kernel."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Compute the foreground mask for a frame using background subtraction."""
        raw_mask = self._mog2.apply(frame)
        _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self._kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel)
        return closed


def segment_frame(subtractor: BackgroundSubtractor, frame: np.ndarray) -> np.ndarray:
    """Apply background subtraction to a frame."""
    return subtractor.apply(frame)
