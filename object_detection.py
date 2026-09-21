import time
import cv2
from ultralytics import YOLO

# 1. Load the pre-trained lightweight YOLOv8 model
model = YOLO("yolov8s.pt")

# 2. Initialize the default webcam (0 is typically the built-in laptop camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional, 640x480 balances speed and accuracy)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera feed unavailable.")
        break

    # Calculate Frames Per Second (FPS)
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Run object detection (stream=True optimizes memory for live streams)
    results = model(frame, stream=True)

    detected_count = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            
            # Display detections with confidence above 50%
            if conf > 0.65:
                detected_count += 1
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                label = f"{class_name} {conf * 100:.1f}%"

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box (Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label background for clear contrast
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    frame,
                    (x1, y1 - 22),
                    (x1 + text_size[0] + 6, y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    # Project HUD: Display FPS and active object count
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Objects: {detected_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Show live stream
    cv2.imshow("Real-Time Object Detection & Recognition", frame)

    # Press 'q' on the keyboard while focused on the window to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release hardware and close application windows
cap.release()
cv2.destroyAllWindows()