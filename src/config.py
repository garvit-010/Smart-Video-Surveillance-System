"""
config.py
---------
Central configuration for the Smart Video Surveillance Pipeline.
All tunable parameters live here — no hardcoded values anywhere else.
"""

# ── Preprocessing ──────────────────────────────────────────────────────────────
GAUSSIAN_BLUR_KERNEL = (5, 5)       # Kernel size for Gaussian blur
GAUSSIAN_BLUR_SIGMA  = 0            # 0 → auto-compute from kernel size
CLAHE_CLIP_LIMIT     = 2.0          # CLAHE contrast limit
CLAHE_TILE_GRID      = (8, 8)       # CLAHE tile grid size

# ── Feature Extraction ─────────────────────────────────────────────────────────
CANNY_THRESHOLD_LOW  = 50           # Lower hysteresis threshold for Canny
CANNY_THRESHOLD_HIGH = 150          # Upper hysteresis threshold for Canny

HOG_ORIENTATIONS     = 9            # Number of gradient orientation bins
HOG_PIXELS_PER_CELL  = (8, 8)       # Cell size for HOG
HOG_CELLS_PER_BLOCK  = (2, 2)       # Block size (in cells) for HOG

# ── Segmentation (Background Subtraction) ─────────────────────────────────────
MOG2_HISTORY         = 500          # Number of frames for background model
MOG2_VAR_THRESHOLD   = 16           # Variance threshold for foreground mask
MOG2_DETECT_SHADOWS  = False        # Shadow detection (disabled for speed)
MORPH_KERNEL_SIZE    = (5, 5)       # Morphological op kernel size

# ── Detection ─────────────────────────────────────────────────────────────────
MIN_CONTOUR_AREA     = 800          # Ignore contours smaller than this (px²)
BBOX_COLOR           = (0, 255, 0)  # Bounding box colour (BGR)
BBOX_THICKNESS       = 2            # Bounding box line thickness

# ── Tracking ──────────────────────────────────────────────────────────────────
MAX_DISAPPEARED      = 30           # Frames before a track is dropped
MAX_DISTANCE         = 100          # Max centroid distance to continue a track

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR           = "outputs"
OUTPUT_FPS           = 25           # Frames-per-second for the output video
FOURCC               = "mp4v"       # Video codec FourCC code
OVERLAY_FONT_SCALE   = 0.55
OVERLAY_FONT_COLOR   = (0, 255, 255)   # Cyan text (BGR)
OVERLAY_FONT_THICKNESS = 1
