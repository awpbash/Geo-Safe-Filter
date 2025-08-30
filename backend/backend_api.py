# backend/backend_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from core.redactor import (
    load_image, load_gdino, detect_gdino, try_ocr, detect_ocr_boxes,
    union_masks, redact
)

app = Flask(__name__)
CORS(app) # Enable CORS for development

# Load models outside the request handler to improve performance
device = "cuda" if torch.cuda.is_available() else "cpu"
processor, model = load_gdino(device)
ocr = try_ocr()

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400
    
    try:
        img_pil = load_image(image_url)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Redaction settings received from the request (could be from the ReactLynx app)
        queries = ["street name sign. road name sign. flag. national flag. country flag.",
                   "landmark. monument. famous building.",
                   "person", "child"]
        
        # GroundingDINO detection
        boxes_gd, _, _ = detect_gdino(img_pil, processor, model, device, 0.25, 0.20, queries)
        
        # OCR detection
        boxes_ocr = detect_ocr_boxes(img_bgr, ocr) if ocr else np.empty((0, 4), dtype=int)
        
        # Combine and redact
        mask = union_masks(img_bgr.shape, [boxes_gd, boxes_ocr])
        redacted = redact(img_bgr, mask, method="blur", blur_ksize=151)
        
        # Convert redacted image to base64 string for a data URI
        _, buffer = cv2.imencode('.png', redacted)
        redacted_base64 = base64.b64encode(buffer).decode('utf-8')
        redacted_data_uri = f"data:image/png;base64,{redacted_base64}"
        
        return jsonify({'redacted_image': redacted_data_uri})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)