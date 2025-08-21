"""
    Date: 18th August 2025
    Author: Rosey

    Aim: A traffic light system controlled with openCV. It counts vehicles in a lane then 
    comaperes to the other lane then contols the taffic with the higher number of count.
    The traffic light then turns green for the lane with higher count.

    Status: Completed!
    Version: 0.4
"""

#import necessary libraries
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8m.pt") # Any yolo version available in the folder.

# Load traffic video
cap = cv2.VideoCapture("invideo-ai-1080 You Won't Believe This Highway Jam! One  2025-08-18.mp4")

# Lane  
lanes = {
    "Lane A": (0, 0, 640, 720),    # Left lane
    "Lane B": (640, 0, 1280, 720)  # Right lane
}

# Set to track mode
tracked_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Run detection with tracking
    results = model.track(frame, persist=True)

    # Reset lane counts
    lane_counts = {lane: 0 for lane in lanes}

    # Process detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Only count vehicles
            if label in ["car", "bus", "truck", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Assign detection to a lane
                for lane, (lx1, ly1, lx2, ly2) in lanes.items():
                    if lx1 < cx < lx2 and ly1 < cy < ly2:
                        lane_counts[lane] += 1

    # Decide which lane gets green
    max_lane = max(lane_counts, key=lane_counts.get)

    # Draw lane rectangles + counts
    for lane, (lx1, ly1, lx2, ly2) in lanes.items():
        color = (0, 255, 0) if lane == max_lane else (0, 0, 255)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, 1)
        cv2.putText(frame, f"{lane}: {lane_counts[lane]}",
        (lx1 + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    # Show overall decision
    cv2.putText(frame, f"GREEN LIGHT: {max_lane}", (400, 680),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 1) 

    cv2.imshow("YOLO Traffic Light System", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
