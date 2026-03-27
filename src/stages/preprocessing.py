"""Frame preprocessing with grayscale conversion, blur, and histogram equalization."""

import cv2
import numpy as np
from ..config import (
    GAUSSIAN_BLUR_KERNEL,
    GAUSSIAN_BLUR_SIGMA,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)


def apply_histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply grayscale, blur, and CLAHE preprocessing to a frame."""
    gray      = to_grayscale(frame)
    blurred   = apply_gaussian_blur(gray)
    enhanced  = apply_histogram_equalization(blurred)
    return enhanced, frame
