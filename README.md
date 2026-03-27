# Smart Video Surveillance
## Multi-Stage Computer Vision Pipeline

> **Roll No:** 23BAI10142  
> **Course:** Computer Vision  

---

## Project Overview

This project implements a **real-time, CLI-based video surveillance pipeline** that
processes raw video footage through a chain of classical Computer Vision stages. It performs
end-to-end processing from raw frames to annotated video output with object detection and
tracking—all executable from a single CLI command with no GUI interaction.

### Stage Breakdown

| Stage | Module | Techniques | Output |
|-------|--------|------------|--------|
| **1. Preprocessing** | `preprocessing.py` | Grayscale conversion, Gaussian Blur (5×5 kernel), CLAHE contrast enhancement | Enhanced grayscale frame |
| **2. Feature Extraction** | `feature_extraction.py` | Canny Edge Detection (thresholds: 50–150), HOG descriptors (9 orientations) | Edge map + feature vector |
| **3. Segmentation** | `segmentation.py` | MOG2 background subtraction (500-frame history), morphological post-processing | Binary foreground mask |
| **4. Detection** | `detection.py` | Contour extraction, area filtering (≥800 px²), bounding box generation | List of bounding boxes |
| **5. Tracking** | `tracking.py` | Centroid-based multi-object tracker, Euclidean distance matching | Persistent object IDs |

**Output Visualization**: Green bounding boxes (detections) + red centroid dots with track IDs + green edge overlay + HUD (frame #, detection count, active tracks)

---

## Pipeline Architecture

The complete processing pipeline follows a sequential, modular design:

```
┌─────────────┐
│ Input Video │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│ Stage 1: Preprocessing       │  (grayscale → blur → CLAHE)
│ preprocessing.py             │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Stage 2: Feature Extraction  │  (Canny edges, HOG)
│ feature_extraction.py        │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Stage 3: Segmentation        │  (MOG2 + morphological ops)
│ segmentation.py              │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Stage 4: Detection           │  (contours → bounding boxes)
│ detection.py                 │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Stage 5: Tracking            │  (centroid matching → IDs)
│ tracking.py                  │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Composition & Annotation     │  (draw boxes, dots, labels, HUD)
│ pipeline.py                  │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Output Video (annotated)     │
│ outputs/processed_video.mp4  │
└──────────────────────────────┘
```

**Key Design**: Each frame is processed sequentially through all 5 stages in-order. Intermediate results (masks, features, boxes) are passed to the next stage. The final composition step overlays all annotations and writes the frame to disk.

---

## Utilities & Support Modules

### `utils.py` — Shared Helper Functions

Central utilities used throughout the pipeline:

| Function | Purpose |
|----------|----------|
| `Logger` class | Lightweight console logging with `[INFO]`, `[WARN]`, `[ERROR]` prefixes |
| `open_video(path)` | Safely open and validate video files; raises `FileNotFoundError` or `RuntimeError` on failure |
| `get_video_properties(cap)` | Extract metadata: width, height, FPS, total frame count |
| `create_video_writer(output_path, props)` | Initialize `cv2.VideoWriter` with correct codec, resolution, and FPS |
| `put_overlay_text(frame, text, position)` | Render HUD text (frame counter, detection count, track count) on frame |
| `draw_tracking_ids(frame, tracked_objects)` | Draw red centroid dots and persistent ID labels for each tracked object |
| `Timer` class | Elapsed time tracking with formatted string output |
| `print_summary(...)` | Terminal summary: total frames processed, total detections, total time, average FPS, output path |

### `pipeline.py` — Main Orchestrator

The [src/pipeline/pipeline.py](src/pipeline/pipeline.py) orchestrates the entire workflow:

1. Opens input video and extracts metadata
2. Creates output video writer and initializes all stage objects (subtractor, tracker, etc.)
3. Iterates through each frame:
   - Passes frame through Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
   - Composes final annotated frame (overlays, bounding boxes, tracking IDs, HUD)
   - Writes frame to output video file
   - Logs progress every 100 frames
4. Releases resources and prints final summary

### `config.py` — Centralized Configuration

All 30+ tunable parameters are defined in a single file:

- **Preprocessing**: Gaussian kernel, CLAHE limits
- **Feature Extraction**: Canny thresholds, HOG parameters
- **Segmentation**: MOG2 history, morphological kernel
- **Detection**: Min contour area, bbox styling
- **Tracking**: Max disappeared frames, max distance
- **Output**: Video codec, FPS, overlay font styling

No hardcoded values exist elsewhere; all parameters are imported from `config.py`.

---

## Project Structure

```
Computer_Vision_Project_23BAI10142/
├── main.py               # CLI entry point
├── config.py             # All tunable parameters
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_video.mp4  # Place your input video here
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── segmentation.py
│   ├── detection.py
│   ├── tracking.py
│   ├── pipeline.py
│   └── utils.py
│
└── outputs/
    └── processed_video.mp4
```

---

##  Setup

### 1. Clone / navigate to the project

```bash
cd Computer_Vision_Project_23BAI10142
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py --input data/sample_video.mp4
```

With a custom output path:

```bash
python main.py --input data/sample_video.mp4 --output outputs/result.mp4
```

---

## Expected Output

After processing, the terminal prints a summary like:

```
──────────────────────────────────────────────────
  PROCESSING SUMMARY
──────────────────────────────────────────────────
  Frames processed  : 450
  Total detections  : 1238
  Processing time   : 18.34s
  Avg FPS (pipeline): 24.5
  Output saved to   : outputs/processed_video.mp4
──────────────────────────────────────────────────
```

The output video (`outputs/processed_video.mp4`) contains:
- **Green bounding boxes** around detected moving objects
- **Red centroid dots** with persistent **track IDs**
- **Subtle green edge overlay** (Canny)
- **HUD** showing frame number, detection count, and active tracks

---

## Configuration

All parameters are in `config.py` — change them without touching any other file:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GAUSSIAN_BLUR_KERNEL` | `(5, 5)` | Noise suppression strength |
| `CANNY_THRESHOLD_LOW/HIGH` | `50 / 150` | Edge sensitivity |
| `MOG2_HISTORY` | `500` | Background model memory |
| `MIN_CONTOUR_AREA` | `800` | Smallest detectable object |
| `MAX_DISAPPEARED` | `30` | Frames before a track is dropped |
| `MAX_DISTANCE` | `100` | Max centroid movement per frame |

---

## Syllabus Coverage

| CV Module | Implemented |
|-----------|--------------|
| Image Processing | Gaussian blur, CLAHE |
| Feature Extraction | Canny edges, HOG descriptors |
| Segmentation | MOG2 background subtraction |
| Object Detection | Contour analysis, bounding boxes |
| Motion / Tracking | Centroid-based multi-object tracking |

---

## Notes

- No GUI frameworks are used.
- All parameters are configurable via `config.py`.
- The pipeline is designed to run on any standard MP4/AVI input video.
