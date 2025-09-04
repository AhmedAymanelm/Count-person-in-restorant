import cv2
import random
import csv
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolo11x.pt")
print("Model loaded!")

# -----------------------------
# DeepSORT + Re-ID
# -----------------------------
tracker = DeepSort(
    max_age=200,
    n_init=3,
    max_iou_distance=0.6,
    embedder="torchreid",
    embedder_model_name="osnet_x0_5",
    half=True
)

# -----------------------------
# Lines for Enter & Exit
# -----------------------------
limitEnter = [700, 300, 900, 700]  # line Enter
limitExit = [600, 300, 700, 600]   # line Exit

# -----------------------------
# Ignore Zone (x1, y1, x2, y2)
# -----------------------------
ignore_zone = (0, 0, 200, 500)

# -----------------------------
# Video Path
# -----------------------------
video_path = "WhatsApp Video 2025-09-01 at 16.56.14.mp4"
cap = cv2.VideoCapture(video_path)

CounttrackEnter = []
counttrackExit = []

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

print(f"Video dimensions: {w}x{h}, FPS: {fps}")

video_writer = cv2.VideoWriter("person_output.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

if not video_writer.isOpened():
   print("Error: Could not open video writer")
   exit()

id_colors = {}
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames to process: {total_frames}")

# -----------------------------
# Active Status Dictionary + Trajectories
# -----------------------------
active_status = {}
trajectories = {}

# -----------------------------
# CSV Logging
# -----------------------------
csv_file = "tracking_log.csv"
csv_fields = ["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "status"]
if os.path.exists(csv_file):
    os.remove(csv_file)
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_fields)

# -----------------------------
# Process video
# -----------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0


    results = model(frame, classes=[0], verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0 and conf > 0.5:
                # check ignore zone
                if (ignore_zone[0] <= x1 and x2 <= ignore_zone[2] and
                    ignore_zone[1] <= y1 and y2 <= ignore_zone[3]):
                    continue

                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])

    # -----------------------------
    # DeepSORT Tracking
    # -----------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # -----------------------------
    # Draw lines
    # -----------------------------
    cv2.line(frame, (limitEnter[0], limitEnter[1]), (limitEnter[2], limitEnter[3]), (0, 255, 0), 2)
    cv2.line(frame, (limitExit[0], limitExit[1]), (limitExit[2], limitExit[3]), (0, 0, 255), 2)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        active_status[track_id] = "Active"

        Ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, Ltrb)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if track_id not in id_colors:
            id_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        color = id_colors[track_id]

        # Draw bbox with transparency
        mask = frame.copy()
        cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(mask, 0.2, frame, 0.8, 0)

        # Label
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Trajectory
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))
        for point in trajectories[track_id]:
            cv2.circle(frame, point, 2, color, -1)

        # Count Enter / Exit
        if limitEnter[0] < cx < limitEnter[2] and limitEnter[1] < cy < limitEnter[3]:
            if track_id not in CounttrackEnter:
                CounttrackEnter.append(track_id)

        if limitExit[0] < cx < limitExit[2] and limitExit[1] < cy < limitExit[3]:
            if track_id not in counttrackExit:
                counttrackExit.append(track_id)


    # -----------------------------
    # Show counters
    # -----------------------------
    cv2.putText(frame, f"Enter: {len(CounttrackEnter)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {len(counttrackExit)}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    active_count = sum(1 for track in tracks if track.is_confirmed() and track.time_since_update == 0)
    cv2.putText(frame, f"Active: {active_count}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)



    video_writer.write(frame)
    # cv2.imshow("YOLO + DeepSORT Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break


cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing Complete with Re-ID!")
print(f" Output saved as: person_output.mp4")
print(f" Total frames processed: {frame_count}")
print(f" Final count - Enter: {len(CounttrackEnter)}")
print(f" Final count - Exit: {len(counttrackExit)}")
print(f" Total unique persons detected: {len(set(CounttrackEnter + counttrackExit))}")
