from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8s.pt")

# Open webcam
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

prev_time = 0

# Latest detection statistics
latest_stats = {
    "count": 0,
    "fps": 0,
    "objects": []
}

detection_history = []

def generate_frames():

    global prev_time

    frame_count = 0
    last_results = []

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Count frames
        frame_count += 1

        # Run YOLO on every 2nd frame
        if frame_count % 2 == 0:

            last_results = model(
                frame,
                verbose=False,
                conf=0.50
            )

        detected_count = 0
        detected_objects = []

        # Draw latest detection results
        for result in last_results:

            for box in result.boxes:

                conf = float(box.conf[0])

                if conf >= 0.50:

                    detected_count += 1

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    detected_objects.append({
                        "name": class_name,
                        "confidence": round(conf * 100, 1)
                    })

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label
                    label = f"{class_name} {conf * 100:.1f}%"

                    (text_width, text_height), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        2
                    )

                    # Label background
                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - text_height - 12)),
                        (x1 + text_width + 8, y1),
                        (0, 255, 0),
                        -1
                    )

                    # Label text
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

        # Calculate FPS
        current_time = time.time()

        if prev_time != 0:
            fps = 1 / (current_time - prev_time)
        else:
            fps = 0

        prev_time = current_time

        # Update latest statistics
        latest_stats["count"] = detected_count
        latest_stats["fps"] = int(fps)
        latest_stats["objects"] = detected_objects
        if detected_objects:
             for obj in detected_objects:

        # Same object ko baar-baar history mein add na kare
                 if not detection_history or detection_history[-1]["name"] != obj["name"]:
                   detection_history.append({
                "time": time.strftime("%H:%M:%S"),
                "name": obj["name"],
                "confidence": obj["confidence"]
            })

        if len(detection_history) > 50:
            del detection_history[:-50]
        

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Display object count
        cv2.putText(
            frame,
            f"Objects: {detected_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Convert frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# Statistics API
@app.route("/stats")
def stats():
    return jsonify(latest_stats)


@app.route("/history")
def history():
    return jsonify(detection_history)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    detection_history.clear()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )