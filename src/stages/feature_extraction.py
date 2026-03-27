"""Extract visual features: Canny edges and HOG descriptors."""

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
    """Apply Canny edge detection."""
    edges = cv2.Canny(gray, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    return edges


def extract_hog_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute HOG (Histogram of Oriented Gradients) features."""
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
    """Extract edge and HOG features from a preprocessed frame."""
    edges                         = extract_edges(gray)
    hog_features, hog_vis         = extract_hog_features(gray)

    return {
        "edges"           : edges,
        "hog_features"    : hog_features,
        "hog_visualization": hog_vis,
    }
