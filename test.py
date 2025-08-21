import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video input
cap = cv2.VideoCapture("invideo-ai-1080 You Won't Believe This Highway Jam! One  2025-08-18.mp4")

# Define lanes (for 2-lane example, adjust as needed)
lanes = {
    "Lane A": (0, 0, 640, 720),
    "Lane B": (640, 0, 1280, 720)
}

# Timing control
current_green_lane = None
last_switch_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    # Vehicle counts
    counts = {lane: 0 for lane in lanes}

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Assign to lane
                for lane, (lx1, ly1, lx2, ly2) in lanes.items():
                    if lx1 < cx < lx2 and ly1 < cy < ly2:
                        counts[lane] += 1

    # Decide which lane gets green
    busiest_lane = max(counts, key=counts.get)

    # Switch lane only after 60s delay
    if current_green_lane is None:
        current_green_lane = busiest_lane
        last_switch_time = time.time()
    elif busiest_lane != current_green_lane:
        if time.time() - last_switch_time >= 60:  # 1 min delay
            current_green_lane = busiest_lane
            last_switch_time = time.time()

    # Draw texts only (smaller font, no lines)
    for lane, count in counts.items():
        lx1, ly1, lx2, ly2 = lanes[lane]
        cv2.putText(frame, f"{lane}: {count}", (lx1 + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Show signal
        color = (0, 255, 0) if lane == current_green_lane else (0, 0, 255)
        cv2.putText(frame, "GREEN" if lane == current_green_lane else "RED",
                    (lx1 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display
    cv2.imshow("Traffic System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
