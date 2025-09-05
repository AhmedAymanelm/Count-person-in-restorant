import cv2
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# -----------------------------
# Download YOLO
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolo11x.pt")
print("Model loaded!")

# -----------------------------
#  DeepSORT + Re-ID
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
        self.id_map = {}  # original_id -> stable_id
        self.stable_counter = 1
        self.last_positions = {}  # stable_id -> last_center
        self.lost_ids = {}  # stable_id -> lost_data
        
    def get_stable_id(self, track_id, center):
        
        if track_id in self.id_map:
            stable_id = self.id_map[track_id]
            self.last_positions[stable_id] = center
            return stable_id
        

        closest_stable_id = None
        min_distance = float('inf')
        
        for stable_id, lost_data in self.lost_ids.items():
            distance = math.sqrt((center[0] - lost_data['center'][0])**2 + 
                               (center[1] - lost_data['center'][1])**2)
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

# Initialize the stabilizer
stabilizer = SimpleIDStabilizer()

# -----------------------------
# draw line in video
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

# Track existing paths
previous_track_ids = set()

# -----------------------------
# open cam
# -----------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

    results = model(frame, classes=[0], verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0 and conf > 0.5:  # person show %
                if (ignore_zone[0] <= x1 <= ignore_zone[2] and
                    ignore_zone[1] <= y1 <= ignore_zone[3]):
                    continue  # Cutting  a part  in Video

                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])

    # -----------------------------
    # DeepSORT Tracking with Re-ID
    # -----------------------------
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Collect current tracks
    current_track_ids = set()

    # -----------------------------
    # Draw line in Video
    # -----------------------------
    cv2.line(frame, (limitEnter[0], limitEnter[1]), (limitEnter[2], limitEnter[3]), (0, 255, 0), 2)
    cv2.line(frame, (limitExit[0], limitExit[1]), (limitExit[2], limitExit[3]), (0, 0, 255), 2)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        current_track_ids.add(track_id)  
        
        Ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, Ltrb)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Get stable ID
        stable_id = stabilizer.get_stable_id(track_id, (cx, cy))
        
        if stable_id not in id_colors:
            id_colors[stable_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

        color = id_colors[stable_id]

        # Draw Box in person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{stable_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.circle(frame, (cx, cy), 5, color, -1)

        # -----------------------------
        # Calc - stable_id
        # -----------------------------
        if limitEnter[0] < cx < limitEnter[2] and limitEnter[1] < cy < limitEnter[3]:
            if stable_id not in CounttrackEnter: 
                CounttrackEnter.append(stable_id)

        if limitExit[0] < cx < limitExit[2] and limitExit[1] < cy < limitExit[3]:
            if stable_id not in counttrackExit:  
                counttrackExit.append(stable_id)

    lost_tracks = previous_track_ids - current_track_ids
    for lost_track_id in lost_tracks:
        stabilizer.handle_lost_track(lost_track_id)
    
    previous_track_ids = current_track_ids.copy()

    if frame_count % 100 == 0:
        current_time = cv2.getTickCount()
        freq = cv2.getTickFrequency()
        old_ids = []
        for stable_id, lost_data in stabilizer.lost_ids.items():
            if (current_time - lost_data['timestamp']) / freq > 5: 
                old_ids.append(stable_id)
        for old_id in old_ids:
            del stabilizer.lost_ids[old_id]

    # -----------------------------
    # show counter in video
    # -----------------------------
    cv2.putText(frame, f"Enter: {len(CounttrackEnter)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {len(counttrackExit)}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(frame)
    # cv2.imshow("YOLO + DeepSORT Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break


cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing Complete with Stable Re-ID!")
print(f" Output saved as: person_output.mp4")
print(f" Total frames processed: {frame_count}")
print(f" Final count - Enter: {len(CounttrackEnter)}")
print(f" Final count - Exit: {len(counttrackExit)}")
print(f" Total unique persons detected: {len(set(CounttrackEnter + counttrackExit))}")
print(f" Stable IDs used: {stabilizer.stable_counter - 1}")
