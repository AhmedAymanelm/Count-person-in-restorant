import cv2
import numpy as np
from ultralytics import YOLO
import random
import time

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolo11x.pt")
print("Model loaded!")

# -----------------------------
# Video Path
# -----------------------------
video_path = "/content/ahmed.mp4"
cap = cv2.VideoCapture(video_path)

# -----------------------------
# VideoWriter
# -----------------------------
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))
if fps <= 0:
    fps = 30

video_writer = cv2.VideoWriter("person_bytetrack_dwell_heatmap_natural.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

if not video_writer.isOpened():
    print("Error: Could not open video writer")
    exit()

# -----------------------------
# ID Colors + Dwell Time
# -----------------------------
id_colors = {}
id_entry_time = {}
id_dwell_time = {}

# Heatmap (float32 للتراكُم)
accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

# -----------------------------
# Ignore Zones (rotated rectangles)
# -----------------------------
rot_rect1 = ((426, 270), (150, 350), 3)
rot_rect2 = ((750, 240), (170, 400), -13)

zone1 = cv2.boxPoints(rot_rect1).astype(int)
zone2 = cv2.boxPoints(rot_rect2).astype(int)

# -----------------------------
# Run YOLO + ByteTrack
# -----------------------------
results = model.track(source=video_path,
                      conf=0.5,
                      classes=[0],   # persons only
                      tracker="bytetrack.yaml",
                      stream=True,
                      verbose=False)

for frame_result in results:
    frame = frame_result.orig_img.copy()
    current_time = time.time()

    if frame_result.boxes is not None:
        for box in frame_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if track_id == -1:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Ignore points in forbidden zones
            if cv2.pointPolygonTest(zone1, (cx, cy), False) >= 0 or \
               cv2.pointPolygonTest(zone2, (cx, cy), False) >= 0:
                continue

            # Assign random color for ID
            if track_id not in id_colors:
                id_colors[track_id] = (random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))

            # Track dwell time
            if track_id not in id_entry_time:
                id_entry_time[track_id] = current_time

            elapsed = int(current_time - id_entry_time[track_id])
            id_dwell_time[track_id] = elapsed

            minutes = elapsed // 60
            seconds = elapsed % 60
            dwell_text = f"ID {track_id} | {minutes}m {seconds}s" if minutes > 0 else f"ID {track_id} | {seconds}s"

            color = id_colors[track_id]

            # Transparent box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Label
            cv2.putText(frame, dwell_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Heatmap update لكل حركة
            # القيمة الصغيرة تتجمع مع مرور الوقت → الأماكن الأكثر مرور تظهر بوضوح
            # cv2.circle(accumulated_heatmap, (cx, cy), 15, 0.05, -1)
            cv2.circle(accumulated_heatmap, (cx, cy), 15, 0.1, -1)


    # -----------------------------
    # Heatmap visualization
    # -----------------------------
    norm_heatmap = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # تخفيف قوة ألوان الهيت ماب
    heatmap_faded = cv2.convertScaleAbs(heatmap_color, alpha=0.6, beta=0)

    # دمج طبيعي مع الفيديو بدون ما يطغى على الإضاءة
    overlay = cv2.addWeighted(frame, 0.85, heatmap_faded, 0.15, 0)


    video_writer.write(overlay)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing Complete!")
print("Output saved as: person_bytetrack_dwell_heatmap_natural.mp4")

# Print summary dwell times
print("\nFinal Dwell Times (min:sec):")
for pid, dwell in id_dwell_time.items():
    m = dwell // 60
    s = dwell % 60
    print(f"ID {pid}: {m}m {s}s")
