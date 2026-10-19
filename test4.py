import cv2
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

# Video file path
VIDEO_PATH = "Bus_2.mp4"

# YOLO model
MODEL_PATH = "best_2.pt"

# Detection confidence
CONFIDENCE = 0.3

# ------------------------------------------------------------
# COUNTING LINE
# ------------------------------------------------------------
# Change this according to your video.
LINE_Y = 350

# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    print(VIDEO_PATH)
    exit()

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("--------------------------------")
print("Video opened successfully")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps}")
print("--------------------------------")

# ============================================================
# PASSENGER COUNTERS
# ============================================================

passengers_in = 0
passengers_out = 0

# ============================================================
# STORE PREVIOUS POSITION OF EACH PERSON
# ============================================================

previous_y = {}

# Store IDs that have already crossed
# This prevents the same crossing from being counted repeatedly
crossed_ids = set()

# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    # --------------------------------------------------------
    # END OF VIDEO
    # --------------------------------------------------------

    if not ret:
        print("Video completed")
        break

    # --------------------------------------------------------
    # YOLO DETECTION + TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        classes=[0],       # COCO class 0 = person
        verbose=False
    )

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        track_ids = (
            results[0]
            .boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id in zip(boxes, track_ids):

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(int, box)

            # ------------------------------------------------
            # PERSON CENTER
            # ------------------------------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW TRACK ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW PERSON CENTER
            # ------------------------------------------------

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # =================================================
            # FIRST TIME WE SEE THIS PERSON
            # =================================================

            if track_id not in previous_y:

                previous_y[track_id] = center_y

                continue

            # ------------------------------------------------
            # PREVIOUS POSITION
            # ------------------------------------------------

            old_y = previous_y[track_id]

            # ------------------------------------------------
            # CHECK WHETHER PERSON CROSSED COUNTING LINE
            # ------------------------------------------------

            if track_id not in crossed_ids:

                # =================================================
                # TOP -> BOTTOM
                # =================================================

                if old_y < LINE_Y and center_y >= LINE_Y:

                    passengers_in += 1

                    crossed_ids.add(track_id)

                    print(
                        f"Person ID {track_id} ENTERED BUS "
                        f"| IN = {passengers_in}"
                    )

                # =================================================
                # BOTTOM -> TOP
                # =================================================

                elif old_y > LINE_Y and center_y <= LINE_Y:

                    passengers_out += 1

                    crossed_ids.add(track_id)

                    print(
                        f"Person ID {track_id} EXITED BUS "
                        f"| OUT = {passengers_out}"
                    )

            # ------------------------------------------------
            # UPDATE PREVIOUS POSITION
            # ------------------------------------------------

            previous_y[track_id] = center_y

    # ============================================================
    # DRAW COUNTING LINE
    # ============================================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 0, 0),
        3
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # ============================================================
    # CURRENT PASSENGER COUNT
    # ============================================================

    current_passengers = passengers_in - passengers_out

    # ============================================================
    # DISPLAY COUNTERS
    # ============================================================

    cv2.putText(
        frame,
        f"PASSENGERS IN  : {passengers_in}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"PASSENGERS OUT : {passengers_out}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"INSIDE BUS     : {current_passengers}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # ============================================================
    # SHOW VIDEO
    # ============================================================

    cv2.imshow(
        "Bus Passenger Counting",
        frame
    )

    # ------------------------------------------------------------
    # PRESS Q TO EXIT
    # ------------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user")
        break

# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()
cv2.destroyAllWindows()

# ============================================================
# FINAL RESULT
# ============================================================

print()
print("========================================")
print("       PASSENGER COUNTING RESULT")
print("========================================")
print(f"Passengers IN      : {passengers_in}")
print(f"Passengers OUT     : {passengers_out}")
print(f"Current Passengers : {passengers_in - passengers_out}")
print("========================================")
