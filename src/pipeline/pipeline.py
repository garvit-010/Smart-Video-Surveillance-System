"""Main pipeline orchestrating all processing stages end-to-end."""

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
    """Process video through all pipeline stages."""
    log.info(f"Opening video: {input_path}")
    cap   = open_video(input_path)
    props = get_video_properties(cap)

    log.info(
        f"Resolution: {props['width']}×{props['height']} | "
        f"FPS: {props['fps']:.1f} | "
        f"Frames: {props['total_frames']}"
    )

    writer      = create_video_writer(output_path, props)
    subtractor  = BackgroundSubtractor()
    tracker     = CentroidTracker()
    timer       = Timer()

    frame_count      = 0
    total_detections = 0

    log.info("Starting frame-by-frame processing…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        frame_count += 1

        preprocessed, original = preprocess_frame(frame)
        features = extract_features(preprocessed)
        edges    = features["edges"]

        fg_mask   = segment_frame(subtractor, frame)
        bboxes          = detect_objects(fg_mask)
        total_detections += len(bboxes)

        tracked = tracker.update(bboxes)

        output = original.copy()
        edge_overlay              = np.zeros_like(output)
        edge_overlay[:, :, 1]    = edges
        output = cv2.addWeighted(output, 1.0, edge_overlay, 0.25, 0)

        output = draw_detections(output, bboxes)
        output = draw_tracking_ids(output, tracked)

        put_overlay_text(output, f"Frame : {frame_count}",     (10, 20))
        put_overlay_text(output, f"Detections : {len(bboxes)}", (10, 40))
        put_overlay_text(output, f"Tracked IDs: {len(tracked)}",  (10, 60))

        writer.write(output)

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
