"""
preprocessing.py
----------------
Stage 1 of the pipeline.

Applies classical image-quality enhancements to every raw frame before any
feature extraction or segmentation is attempted:
  1. Grayscale conversion
  2. Gaussian blur   (noise suppression)
  3. CLAHE           (adaptive histogram equalisation → better contrast)
"""

import cv2
import numpy as np
from ..config import (
    GAUSSIAN_BLUR_KERNEL,
    GAUSSIAN_BLUR_SIGMA,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to single-channel grayscale.

    Args:
        frame: BGR image as numpy array (H, W, 3).

    Returns:
        Grayscale image of shape (H, W).
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray: np.ndarray) -> np.ndarray:
    """Suppress high-frequency noise with a Gaussian filter.

    Args:
        gray: Single-channel (grayscale) image.

    Returns:
        Blurred grayscale image.
    """
    return cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)


def apply_histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE is preferred over global histogram equalisation because it avoids
    over-amplifying noise in uniform regions.

    Args:
        gray: Single-channel (grayscale) image.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing chain for a single frame.

    Runs grayscale → Gaussian blur → CLAHE in sequence.

    Args:
        frame: Raw BGR video frame.

    Returns:
        Tuple of (preprocessed_gray, original_bgr_frame).
        The caller keeps the original for visualisation; the preprocessed
        version feeds into downstream stages.
    """
    gray      = to_grayscale(frame)
    blurred   = apply_gaussian_blur(gray)
    enhanced  = apply_histogram_equalization(blurred)
    return enhanced, frame
