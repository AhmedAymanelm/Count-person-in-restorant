import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolo11x.pt")   # YOLO detection
print("Model loaded!")

# -----------------------------
# Load DeepSORT tracker with OSNet
# -----------------------------
tracker = DeepSort(
    max_age=18000,         
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=100,
    override_track_class=None,
    embedder="torchreid",                 
    half=True,
    bgr=True,
    embedder_model_name="osnet_x0_25",  
    polygon=False
)

# -----------------------------
# Video Path
# -----------------------------
video_path = "/content/2025-09-17 14.52.00.mp4"
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))
if fps <= 0:
    fps = 30

# -----------------------------
# VideoWriter
# -----------------------------
video_writer = cv2.VideoWriter("combined_inout_output.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

# -----------------------------
# Trackers colors & dwell time
# -----------------------------
id_colors = {}
id_frame_count = {}
accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

# -----------------------------
# In/Out line
# -----------------------------
line_y = int(h*0.9)
in_count, out_count = 0, 0
last_positions = {}

# -----------------------------
# Run detection + DeepSORT
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.3, classes=[0])  # persons only

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw thinner In/Out line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 1)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w_box, h_box = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, w_box, h_box])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Assign random color
        if track_id not in id_colors:
            id_colors[track_id] = (random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255))

        # Track dwell time
        if track_id not in id_frame_count:
            id_frame_count[track_id] = 0
        id_frame_count[track_id] += 1

        elapsed_seconds = int(id_frame_count[track_id] / fps)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        dwell_text = f"ID {track_id} | {minutes}m {seconds}s" if minutes > 0 else f"ID {track_id} | {seconds}s"

        color = id_colors[track_id]

        # Transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Label
        cv2.putText(frame, dwell_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Heatmap (أهدى شوية)
        cv2.circle(accumulated_heatmap, (cx, cy), 10, 0.03, -1)

        # -----------------------------
        # In/Out counting logic
        # -----------------------------
        if track_id in last_positions:
            prev_y = last_positions[track_id]
            if prev_y < line_y and cy >= line_y:
                in_count += 1
            elif prev_y > line_y and cy <= line_y:
                out_count += 1

        last_positions[track_id] = cy

    # Overlay heatmap
    norm_heatmap = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_faded = cv2.convertScaleAbs(heatmap_color, alpha=0.6, beta=0)
    combined_frame = cv2.addWeighted(frame, 0.9, heatmap_faded, 0.1, 0)

    # Display in/out counts
    cv2.putText(combined_frame, f"IN: {in_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, f"OUT: {out_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(combined_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing Complete!")
print("Output saved as: combined_inout_output.mp4")

print("\nFinal Dwell Times (min:sec):")
for pid, frames in id_frame_count.items():
    elapsed_seconds = int(frames / fps)
    m = elapsed_seconds // 60
    s = elapsed_seconds % 60
    print(f"ID {pid}: {m}m {s}s")

print(f"\nTotal IN: {in_count}")
print(f"Total OUT: {out_count}")
