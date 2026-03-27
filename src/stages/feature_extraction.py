"""
feature_extraction.py
---------------------
Stage 2 of the pipeline.

Extracts low-level visual features from the preprocessed (grayscale) frame:
  1. Canny edge detection  → edge map overlaid on the final output
  2. HOG feature vector    → diagnostic / future ML use
"""

import cv2
import numpy as np
from skimage.feature import hog
from ..config import (
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
)


def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection to produce a binary edge map.

    Args:
        gray: Preprocessed single-channel (grayscale) frame.

    Returns:
        Binary edge map (same H×W as input, dtype uint8).
    """
    edges = cv2.Canny(gray, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    return edges


def extract_hog_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute HOG (Histogram of Oriented Gradients) features.

    Args:
        gray: Preprocessed single-channel (grayscale) frame.

    Returns:
        Tuple of (feature_vector, hog_visualisation_image).
        The visualisation image can be overlaid on frames for inspection.
    """
    feature_vector, hog_image = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=True,
        channel_axis=None,   # already grayscale
    )
    # Rescale HOG image to [0, 255] for visualisation
    hog_image_rescaled = (hog_image / hog_image.max() * 255).astype(np.uint8)
    return feature_vector, hog_image_rescaled


def extract_features(gray: np.ndarray) -> dict:
    """Run the complete feature extraction stage.

    Args:
        gray: Preprocessed single-channel frame.

    Returns:
        Dictionary with keys:
          - 'edges'             : binary edge map (np.ndarray)
          - 'hog_features'     : 1-D HOG feature vector (np.ndarray)
          - 'hog_visualization': grayscale HOG visualisation (np.ndarray)
    """
    edges                         = extract_edges(gray)
    hog_features, hog_vis         = extract_hog_features(gray)

    return {
        "edges"           : edges,
        "hog_features"    : hog_features,
        "hog_visualization": hog_vis,
    }
