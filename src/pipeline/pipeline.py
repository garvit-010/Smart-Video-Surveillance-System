"""
pipeline.py
-----------
Orchestrates the full Smart Video Surveillance pipeline end-to-end:

  Input Video
       │
       ▼
  Preprocessing   (grayscale, blur, CLAHE)
       │
       ▼
  Feature Extraction  (Canny edges, HOG)
       │
       ▼
  Segmentation    (MOG2 background subtraction)
       │
       ▼
  Detection       (contour → bounding boxes)
       │
       ▼
  Tracking        (centroid tracker)
       │
       ▼
  Output Video    (annotated frames written to disk)
"""

import cv2
import numpy as np

from ..stages.preprocessing      import preprocess_frame
from ..stages.feature_extraction import extract_features
from ..stages.segmentation       import BackgroundSubtractor, segment_frame
from ..stages.detection          import detect_objects, draw_detections
from ..stages.tracking           import CentroidTracker
from ..utils.utils               import (
    log,
    open_video,
    get_video_properties,
    create_video_writer,
    put_overlay_text,
    draw_tracking_ids,
    Timer,
    print_summary,
)


def run_pipeline(input_path: str, output_path: str) -> None:
    """Execute the full CV pipeline on an input video.

    Args:
        input_path:  Path to the source video file.
        output_path: Destination path for the annotated output video.

    Raises:
        FileNotFoundError: If input_path does not exist.
        RuntimeError:      If OpenCV cannot open the video.
    """
    # ── 1. Open video ─────────────────────────────────────────────────────────
    log.info(f"Opening video: {input_path}")
    cap   = open_video(input_path)
    props = get_video_properties(cap)

    log.info(
        f"Resolution: {props['width']}×{props['height']} | "
        f"FPS: {props['fps']:.1f} | "
        f"Frames: {props['total_frames']}"
    )

    # ── 2. Prepare writer + stage objects ─────────────────────────────────────
    writer      = create_video_writer(output_path, props)
    subtractor  = BackgroundSubtractor()
    tracker     = CentroidTracker()
    timer       = Timer()

    frame_count      = 0
    total_detections = 0

    log.info("Starting frame-by-frame processing …")

    # ── 3. Main processing loop ───────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        frame_count += 1

        # Stage A – Preprocessing
        preprocessed, original = preprocess_frame(frame)

        # Stage B – Feature Extraction (edges used for overlay; HOG computed)
        features = extract_features(preprocessed)
        edges    = features["edges"]

        # Stage C – Segmentation: foreground mask
        fg_mask   = segment_frame(subtractor, frame)

        # Stage D – Detection: bounding boxes from mask
        bboxes          = detect_objects(fg_mask)
        total_detections += len(bboxes)

        # Stage E – Tracking: assign persistent IDs
        tracked = tracker.update(bboxes)

        # ── Compose annotated output frame ────────────────────────────────────
        # Start from the original colour frame
        output = original.copy()

        # Overlay a subtle edge channel (green tint)
        edge_overlay              = np.zeros_like(output)
        edge_overlay[:, :, 1]    = edges          # green channel
        output = cv2.addWeighted(output, 1.0, edge_overlay, 0.25, 0)

        # Draw bounding boxes
        output = draw_detections(output, bboxes)

        # Draw tracking IDs
        output = draw_tracking_ids(output, tracked)

        # HUD overlay: frame counter + detection count
        put_overlay_text(output, f"Frame : {frame_count}",     (10, 20))
        put_overlay_text(output, f"Detections : {len(bboxes)}", (10, 40))
        put_overlay_text(output, f"Tracked IDs: {len(tracked)}",  (10, 60))

        # Write annotated frame
        writer.write(output)

        # Progress every 100 frames
        if frame_count % 100 == 0:
            log.info(
                f"  Frame {frame_count}/{props['total_frames']} | "
                f"Detections this frame: {len(bboxes)} | "
                f"Active tracks: {len(tracked)} | "
                f"Elapsed: {timer.elapsed_str()}"
            )

    # ── 4. Cleanup + summary ──────────────────────────────────────────────────
    cap.release()
    writer.release()

    print_summary(
        total_frames=frame_count,
        total_detections=total_detections,
        elapsed=timer.elapsed(),
        output_path=output_path,
    )
