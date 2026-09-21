from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8s.pt")


prev_time = 0

# Latest detection statistics
latest_stats = {
    "count": 0,
    "fps": 0,
    "objects": []
}

detection_history = []



@app.route("/")
def home():
    return render_template("index.html")


 
@app.route("/detection")
def detection():
    return render_template("detection.html")



@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    global detection_history

    if "frame" not in request.files:
        return jsonify({
            "success": False,
            "error": "No frame received"
        }), 400

    file = request.files["frame"]

    # Image bytes ko NumPy array mein convert karo
    image_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )

    # JPEG ko OpenCV image mein decode karo
    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "success": False,
            "error": "Invalid image"
        }), 400

    # YOLO detection
    results = model(
        frame,
        verbose=False,
        conf=0.50
    )

    detections = []

    for result in results:

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf >= 0.50:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "name": class_name,
                    "confidence": round(conf * 100, 1),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
    # Update latest statistics
    latest_stats["count"] = len(detections)
    latest_stats["fps"] = 0
    latest_stats["objects"] = detections

     # Update detection history
    if detections:
        for obj in detections:
            if not detection_history or detection_history[-1]["name"] != obj["name"]:
                detection_history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "name": obj["name"],
                    "confidence": obj["confidence"]
                })

    # Keep maximum 50 history items
    if len(detection_history) > 50:
        del detection_history[:-50]

    return jsonify({
        "success": True,
        "count": len(detections),
        "objects": detections,
        "width": frame.shape[1],
        "height": frame.shape[0]
})

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