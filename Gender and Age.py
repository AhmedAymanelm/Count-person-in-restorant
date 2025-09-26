import cv2
import time
import json
import os
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------------------
# دالة كشف الوجوه
# ----------------------------
def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 3.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight / 150)), 3)
    return frameOpencvDnn, faceBoxes

# ----------------------------
# الموديلات
# ----------------------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

required_files = [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing model file: {file}")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(3-6)', '(7-12)', '(13-19)', '(20-29)',
 '(30-39)', '(40-49)', '(50-59)', '(60-74)', '(75-100)']

genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# ----------------------------
# Tracker
# ----------------------------
tracker = DeepSort(max_age=5)

# ----------------------------
# مجلد الصور و JSON
# ----------------------------
faces_dir = "faces"
os.makedirs(faces_dir, exist_ok=True)

data_file = "detections.json"
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        detections_data = json.load(f)
else:
    detections_data = []

captured_ids = set(entry["id"] for entry in detections_data)
current_id = max(captured_ids) + 1 if captured_ids else 1

# ----------------------------
# الكاميرا
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
time.sleep(2)
padding = 20

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    detections_for_tracker = []
    for box in faceBoxes:
        x1, y1, x2, y2 = box
        detections_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], 1.0, "face"))

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # لو الشخص ده متسجلش قبل كده
        if track_id not in captured_ids:
            face = frame[max(0, y1-padding):min(y2+padding, frame.shape[0]-1),
                         max(0, x1-padding):min(x2+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            filename = os.path.join(faces_dir, f"person_{current_id}.jpg")
            cv2.imwrite(filename, face)

            detections_data.append({
                "id": current_id,
                "image": filename,
                "gender": gender,
                "age": age,
                "entry_time": entry_time
            })

            with open(data_file, "w") as f:
                json.dump(detections_data, f, indent=4)

            print(f"[✔] Captured Person {current_id}: Gender={gender}, Age={age}, Entry={entry_time}")

            captured_ids.add(track_id)
            current_id += 1

        cv2.putText(resultImg, f'{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Face Detection + Tracker", resultImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
