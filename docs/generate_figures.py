import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_intermediate_frames():
    images_dir = os.path.join("docs", "images")
    os.makedirs(images_dir, exist_ok=True)

    input_path = os.path.normpath("data/sample_video.mp4")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Could not open {input_path}")
        return

    # Extract frame 300
    target_frame = 300
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        return
    cap.release()

    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(images_dir, "gray_sample.jpg"), gray)

    # 2. Blurred
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(images_dir, "blurred_sample.jpg"), blurred)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    cv2.imwrite(os.path.join(images_dir, "clahe_sample.jpg"), enhanced)

    # 4. Canny Edges
    edges = cv2.Canny(enhanced, 50, 150)
    cv2.imwrite(os.path.join(images_dir, "edges_sample.jpg"), edges)

    print("Intermediate CV frames generated.")

def generate_metrics_graph():
    images_dir = os.path.join("docs", "images")
    os.makedirs(images_dir, exist_ok=True)

    # Data from processing_summary.md
    frames = [100, 200, 300, 400, 500]
    detections = [90, 28, 19, 28, 29]
    active_tracks = [167, 109, 87, 101, 97]
    elapsed = [654.86, 1297.84, 1940.32, 2581.84, 3223.93]

    # Plot 1: Detections and Tracks over time
    plt.figure(figsize=(10, 5))
    plt.plot(frames, active_tracks, marker='o', linestyle='-', color='b', label='Active Tracks')
    plt.plot(frames, detections, marker='s', linestyle='--', color='r', label='New Detections per Frame')
    plt.title("Object Tracking Metrics over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "tracking_metrics.png"), dpi=300)
    plt.close()

    # Plot 2: Processing Time
    plt.figure(figsize=(10, 5))
    plt.plot(frames, elapsed, marker='^', linestyle='-', color='g', label='Cumulative Processing Time (s)')
    plt.fill_between(frames, elapsed, color='g', alpha=0.1)
    plt.title("Pipeline Computational Cost (Elapsed Time)")
    plt.xlabel("Frame Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "processing_time.png"), dpi=300)
    plt.close()

    print("Metrics graphs generated.")

if __name__ == "__main__":
    generate_intermediate_frames()
    generate_metrics_graph()
