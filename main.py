import cv2
import random
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolov8s.pt")
print("Model loaded!")

# -----------------------------
# DeepSORT + Re-ID
# -----------------------------
tracker = DeepSort(
    max_age=18000,
    n_init=4,
    max_iou_distance=0.3,
    max_cosine_distance=0.2,
    nn_budget=150,
    embedder="torchreid",
    embedder_model_name="osnet_x0_5",
    embedder_wts=None,
    half=True
)

# -----------------------------
# Simple ID Stabilizer
# -----------------------------
class SimpleIDStabilizer:
    def __init__(self):
        self.id_map = {}
        self.stable_counter = 1
        self.last_positions = {}
        self.lost_ids = {}

    def get_stable_id(self, track_id, center):
        if track_id in self.id_map:
            stable_id = self.id_map[track_id]
            self.last_positions[stable_id] = center
            return stable_id

        closest_stable_id = None
        min_distance = float('inf')
        for stable_id, lost_data in self.lost_ids.items():
            distance = math.sqrt((center[0] - lost_data['center'][0]) ** 2 +
                                 (center[1] - lost_data['center'][1]) ** 2)
            if distance < 100 and distance < min_distance:
                min_distance = distance
                closest_stable_id = stable_id

        if closest_stable_id:
            self.id_map[track_id] = closest_stable_id
            self.last_positions[closest_stable_id] = center
            del self.lost_ids[closest_stable_id]
            return closest_stable_id

        stable_id = self.stable_counter
        self.stable_counter += 1
        self.id_map[track_id] = stable_id
        self.last_positions[stable_id] = center
        return stable_id

    def handle_lost_track(self, track_id):
        if track_id in self.id_map:
            stable_id = self.id_map[track_id]
            if stable_id in self.last_positions:
                self.lost_ids[stable_id] = {
                    'center': self.last_positions[stable_id],
                    'timestamp': cv2.getTickCount()
                }
            del self.id_map[track_id]

# Initialize stabilizer
stabilizer = SimpleIDStabilizer()

# -----------------------------
# Vertical counting line
# -----------------------------
line_x = 600
line_color = (0, 255, 255)
line_thickness = 2

# -----------------------------
# Ignore Zone (x1, y1, x2, y2)
# -----------------------------
ignore_zone = (0, 0, 200, 500)

# -----------------------------
# Video Path
# -----------------------------
video_path = "/content/WhatsApp Video 2025-09-06 at 23.58.36_3bc30005.mp4"
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

video_writer = cv2.VideoWriter("person_heatmap_output.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

if not video_writer.isOpened():
    print("Error: Could not open video writer")
    exit()

id_colors = {}
frame_count = 0
previous_track_ids = {}
last_positions_line = {}

# -----------------------------
# Heatmap متراكم
# -----------------------------
accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

# -----------------------------
# Process Video
# -----------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    results = model(frame, classes=[0], verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0 and conf > 0.5:
                if (ignore_zone[0] <= x1 <= ignore_zone[2] and
                        ignore_zone[1] <= y1 <= ignore_zone[3]):
                    continue
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])

    tracks = tracker.update_tracks(detections, frame=frame)
    current_track_ids = set()

    # Draw vertical counting line
    cv2.line(frame, (line_x, 0), (line_x, h), line_color, line_thickness)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        current_track_ids.add(track_id)

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        head_x = (x1 + x2) // 2
        head_y = y1

        stable_id = stabilizer.get_stable_id(track_id, (head_x, head_y))

        if stable_id not in id_colors:
            id_colors[stable_id] = (random.randint(0, 255),
                                    random.randint(0, 255),
                                    random.randint(0, 255))
        color = id_colors[stable_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{stable_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.circle(frame, (head_x, head_y), 5, color, -1)

        # Count Enter/Exit
        if stable_id not in last_positions_line:
            last_positions_line[stable_id] = head_x
        else:
            prev_head_x = last_positions_line[stable_id]

            if prev_head_x > line_x and head_x <= line_x and stable_id not in CounttrackEnter:
                CounttrackEnter.append(stable_id)

            if prev_head_x < line_x and head_x >= line_x and stable_id not in counttrackExit:
                counttrackExit.append(stable_id)

            last_positions_line[stable_id] = head_x

        # -----------------------------
        # Update Heatmap
        # -----------------------------
        accumulated_heatmap[y1:y2, x1:x2] += 1

    # Handle lost tracks
    lost_tracks = set(previous_track_ids.keys()) - current_track_ids
    for lost_id in lost_tracks:
        stabilizer.handle_lost_track(lost_id)
        previous_track_ids.pop(lost_id, None)

    previous_track_ids = {tid: True for tid in current_track_ids}

    # Remove old lost_ids
    current_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    old_ids = [sid for sid, ld in stabilizer.lost_ids.items()
               if (current_time - ld['timestamp']) / freq > 5]
    for old_id in old_ids:
        del stabilizer.lost_ids[old_id]

    # Normalize heatmap and apply colormap
    norm_heatmap = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend heatmap مع الفريم
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Show counts
    cv2.putText(overlay, f"Enter: {len(CounttrackEnter)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Exit: {len(counttrackExit)}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(overlay)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing Complete!")
print(f"Output saved as: person_heatmap_output.mp4")
print(f"Final Enter: {len(CounttrackEnter)}")
print(f"Final Exit: {len(counttrackExit)}")
print(f"Total unique persons: {len(set(CounttrackEnter + counttrackExit))}")
print(f"Stable IDs used: {stabilizer.stable_counter - 1}")
