# main.py
import base64
import cv2
import numpy as np
from flask import Flask, render_template_string, Response, request
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")  # лёгкая модель для CPU

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8 WebCam Stream</title>
</head>
<body>
    <h1>YOLOv8 + WebCam</h1>
    <video id="video" autoplay muted playsinline width="320" height="240"></video>
    <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
    <img id="output" width="320" height="240">
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const output = document.getElementById('output');

navigator.mediaDevices.getUserMedia({video:{width:320, height:240}})
.then(stream => {
    video.srcObject = stream;
    video.play();
    sendFrame();
}).catch(e => alert('Error accessing camera: ' + e));

function sendFrame(){
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            fetch('/process_frame', {
                method:'POST',
                headers:{'Content-Type':'application/octet-stream'},
                body:reader.result
            })
            .then(res => res.blob())
            .then(blob => {
                output.src = URL.createObjectURL(blob);
                setTimeout(sendFrame, 100); // ~10 FPS для CPU
            });
        };
        reader.readAsArrayBuffer(blob);
    }, 'image/jpeg', 0.6);
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # получаем бинарные данные кадра
    img_bytes = request.data
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO детекция
    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)