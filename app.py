from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import os
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Save the uploaded file
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Perform object detection
    image = Image.open(file_path)
    results = model(image)

    # Save the results with bounding boxes
    results.render()  # Draw bounding boxes on the image
    output_path = os.path.join(UPLOAD_FOLDER, 'output_' + file.filename)
    Image.fromarray(results.ims[0]).save(output_path)

    # Get detection results (class, confidence, bounding box)
    detections = results.pandas().xyxy[0].to_dict(orient='records')

    # Return the results
    return jsonify({
        'detections': detections,
        'output_image': output_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)