import os
import cv2
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from deepface_antispoofing import DeepFaceAntiSpoofing

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Config for anti-spoofing
UPLOAD_FOLDER = "static/uploads"
TEMP_FOLDER = "static/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

camera = cv2.VideoCapture(0)

# Initialize DeepFaceAntiSpoofing for deepfake analysis
try:
    deepface_analyzer = DeepFaceAntiSpoofing()
except Exception as e:
    print(f"Failed to initialize DeepFaceAntiSpoofing: {str(e)}")
    deepface_analyzer = None

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("deepface.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    """Capture current webcam frame and analyze using DeepFaceAntiSpoofing."""
    if deepface_analyzer is None:
        return jsonify({"success": False, "error": "DeepFaceAntiSpoofing not initialized"})
    success, frame = camera.read()
    if not success:
        return jsonify({"success": False, "error": "Failed to capture frame"})

    temp_path = os.path.join(TEMP_FOLDER, "capture.jpg")
    cv2.imwrite(temp_path, frame)

    result = deepface_analyzer.analyze_deepface(temp_path)
    return jsonify(result)

@app.route("/upload_anti", methods=["POST"])
def upload_anti_file():
    if deepface_analyzer is None:
        return jsonify({"success": False, "error": "DeepFaceAntiSpoofing not initialized"})
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"})

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    result = deepface_analyzer.analyze_deepface(path)
    print(result)
    return jsonify(result)

@app.route("/upload_deep", methods=["POST"])
def upload_deep_image():
    if deepface_analyzer is None:
        return jsonify({"error": "DeepFaceAntiSpoofing not initialized", "success": False}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded", "success": False}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected", "success": False}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    result = deepface_analyzer.analyze_image(image_path)

    if "error" in result or not result.get("success", True):
        return jsonify(result), 500

    print(result)
    return jsonify(result)

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to start main Flask app: {str(e)}")
        raise