import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import json
import time
from datetime import datetime

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded!")

# -----------------------------
# Load DeepSORT tracker
# -----------------------------
tracker = DeepSort(
    max_age=18000,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=100,
    embedder="mobilenet",
    embedder_model_name=None,
    half=True,
    bgr=True,
    polygon=False
)

# -----------------------------
# Real-Time Camera
# -----------------------------
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

id_colors = {}
id_frame_count = {}
id_times = {}  # 🟢 {id: {"entry": time, "exit": time}}
in_count, out_count = 0, 0
last_positions = {}

line_y = 400

last_json_save = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, classes=[0])  # persons only
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w_box, h_box = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, w_box, h_box])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if track_id not in id_colors:
            id_colors[track_id] = (random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255))

        if track_id not in id_frame_count:
            id_frame_count[track_id] = 0
        id_frame_count[track_id] += 1

        elapsed_seconds = int(id_frame_count[track_id] / fps)
        dwell_text = f"ID {track_id} | {elapsed_seconds}s"

        color = id_colors[track_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, dwell_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -----------------------------
        # Entry/Exit time tracking
        # -----------------------------
        if track_id in last_positions:
            prev_y = last_positions[track_id]

            # دخول
            if prev_y < line_y and cy >= line_y:
                in_count += 1
                id_times[track_id] = {"entry": datetime.now().strftime("%H:%M:%S"), "exit": None}

            # خروج
            elif prev_y > line_y and cy <= line_y:
                out_count += 1
                if track_id in id_times and id_times[track_id]["exit"] is None:
                    id_times[track_id]["exit"] = datetime.now().strftime("%H:%M:%S")

        last_positions[track_id] = cy

    # -----------------------------
    # Save JSON every 2 seconds
    # -----------------------------
    if time.time() - last_json_save > 2:
        results_data = {
            "summary": {
                "total_in": in_count,
                "total_out": out_count
            },
            "tracks": []
        }
        for pid, frames in id_frame_count.items():
            elapsed_seconds = int(frames / fps)
            entry_time = id_times.get(pid, {}).get("entry")
            exit_time = id_times.get(pid, {}).get("exit")
            results_data["tracks"].append({
                "id": pid,
                "dwell_time_s": elapsed_seconds,
                "frames": frames,
                "entry_time": entry_time,
                "exit_time": exit_time
            })

        with open("realtime_results.json", "w") as f:
            json.dump(results_data, f, indent=4)

        last_json_save = time.time()

    # -----------------------------
    # Overlay IN/OUT on screen
    # -----------------------------
    cv2.putText(frame, f"IN: {in_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {out_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Real-Time Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
