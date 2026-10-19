import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------------------------------
# Load YOLOv8
# ----------------------------------------
model = YOLO("yolov8n.pt")

# ----------------------------------------
# DeepSORT Tracker
# ----------------------------------------
tracker = DeepSort(
    max_age=10,
    n_init=3,
    max_cosine_distance=0.3
)

# ----------------------------------------
# Open Video
# ----------------------------------------
#video_path = "fdoor.avi"
video_path = "Bus_3.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

# ----------------------------------------
# Counters
# ----------------------------------------
people_in = 0    # left -> right
people_out = 0   # right -> left

# Store previous centroid of every track
track_history = {}

# Store counted IDs separately per direction
# (so a person can be counted once for IN and once for OUT,
#  e.g. if they enter and later leave)
counted_in = set()
counted_out = set()

# ----------------------------------------
# Read first frame
# ----------------------------------------
ret, frame = cap.read()

if not ret:
    print("Video empty")
    exit()

height, width = frame.shape[:2]

# Counting line (VERTICAL)
line_x = int(width * 0.7)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ----------------------------------------
# Create resizable window
# ----------------------------------------
WINDOW_NAME = "Bus Passenger Counter"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)

# ----------------------------------------
# Main Loop
# ----------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []

    # -------------------------------
    # Detect Persons
    # -------------------------------
    for box in results.boxes:

        cls = int(box.cls[0])

        if cls != 0:       # person class
            continue

        conf = float(box.conf[0])

        if conf < 0.6:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        detections.append(
            (
                [x1, y1, x2 - x1, y2 - y1],
                conf,
                "person"
            )
        )

    # -------------------------------
    # Tracking
    # -------------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # -------------------------------
    # Process Tracks
    # -------------------------------
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, r, b = track.to_ltrb()

        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)

        cx = (l + r) // 2
        cy = (t + b) // 2

        # Draw box
        cv2.rectangle(frame, (l, t), (r, b), (0,255,0), 4)
        cv2.circle(frame, (cx, cy), 8, (0,0,255), -1)

        cv2.putText(
            frame,
            str(track_id),
            (l, t-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

        # Previous centroid
        if track_id in track_history:

            prev_x, prev_y = track_history[track_id]

            # Crossing LEFT -> RIGHT  (ENTRY)
            if prev_x < line_x and cx >= line_x:
                if track_id not in counted_in:
                    counted_in.add(track_id)
                    people_in += 1
                    # allow this ID to be counted again for exit later
                    counted_out.discard(track_id)

            # Crossing RIGHT -> LEFT  (EXIT)
            elif prev_x > line_x and cx <= line_x:
                if track_id not in counted_out:
                    counted_out.add(track_id)
                    people_out += 1
                    # allow this ID to be counted again for entry later
                    counted_in.discard(track_id)

        track_history[track_id] = (cx, cy)

    # Draw counting line (VERTICAL)
    cv2.line(frame, (line_x, 0), (line_x, width), (255,0,0), 3)

    cv2.putText(
        frame,
        f"IN  : {people_in}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        3
    )

    cv2.putText(
        frame,
        f"OUT : {people_out}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        3
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()