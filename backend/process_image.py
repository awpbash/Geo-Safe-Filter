import sys
import json
import argparse
import base64
import numpy as np
import cv2
import io
import os
import torch
from PIL import Image

# Import all core functions from the redactor.py file
from core.redactor import (
    load_image, load_gdino, detect_gdino, try_ocr, detect_ocr_boxes,
    union_masks, redact
)

def main():
    try:
        # Use argparse to handle command-line arguments correctly
        parser = argparse.ArgumentParser(description="Process and redact an image.")
        parser.add_argument("--method", type=str, default="blur", help="Redaction method (blur or pixelate)")
        parser.add_argument("--blur_ksize", type=int, default=151, help="Kernel size for blur method")
        parser.add_argument("--mosaic_scale", type=float, default=0.06, help="Scale for pixelate method")
        parser.add_argument("--debug", action="store_true", help="Save intermediate images for debugging.")
        args = parser.parse_args()

        # Log start of process
        print("Starting image processing...", file=sys.stderr)
        
        # Read the image data from stdin (sent from FastAPI)
        image_bytes = sys.stdin.buffer.read()
        image_stream = io.BytesIO(image_bytes)
        
        # Decode the image using OpenCV
        image_np = np.frombuffer(image_stream.read(), np.uint8)
        img_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img_bgr is None:
            # Handle the case where the image data is invalid
            response = {"error": "Could not decode image from input. The input may not be a valid image file."}
            json.dump(response, sys.stdout)
            sys.stdout.flush()
            sys.exit(1)

        # Log successful image decoding
        print("Image decoded successfully.", file=sys.stderr)

        # Debugging: Save original image
        if args.debug:
            debug_dir = "./debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "step1_original.jpg"), img_bgr)
            print(f"Saved original image to {debug_dir}/step1_original.jpg", file=sys.stderr)

        # Convert the image to PIL format for Grounding DINO
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        # Load and run models
        print("Loading Grounding DINO and OCR models...", file=sys.stderr)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, model = load_gdino(device)
        ocr = try_ocr()
        print("Models loaded successfully.", file=sys.stderr)
        
        # Detect objects and text
        print("Starting object and text detection...", file=sys.stderr)
        queries = ["street name sign", "road name sign", "flag", "landmark", "monument", "person", "child"]
        boxes_gd, _, _ = detect_gdino(img_pil, processor, model, device, 0.25, 0.20, queries)
        boxes_ocr = detect_ocr_boxes(img_bgr, ocr) if ocr else np.empty((0, 4), dtype=int)
        print("Detection complete.", file=sys.stderr)

        # Debugging: Save image with detections
        if args.debug:
            img_with_boxes = img_bgr.copy()
            for box in boxes_gd:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for box in boxes_ocr:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, "step2_detections.jpg"), img_with_boxes)
            print(f"Saved image with detections to {debug_dir}/step2_detections.jpg", file=sys.stderr)
        
        # Combine the detection masks
        print("Combining masks and redacting image...", file=sys.stderr)
        mask = union_masks(img_bgr.shape, [boxes_gd, boxes_ocr])
        
        # Redact the image
        redacted_image = redact(
            img_bgr,
            mask,
            method=args.method,
            blur_ksize=args.blur_ksize,
            mosaic_scale=args.mosaic_scale
        )
        print("Image redacted successfully.", file=sys.stderr)

        # Debugging: Save redacted image
        if args.debug:
            cv2.imwrite(os.path.join(debug_dir, "step3_redacted.jpg"), redacted_image)
            print(f"Saved redacted image to {debug_dir}/step3_redacted.jpg", file=sys.stderr)
        
        # Encode the redacted image back to bytes in JPEG format
        _, img_encoded = cv2.imencode('.jpeg', redacted_image)
        redacted_image_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # Create a JSON response
        response = {
            "redacted_image": f"data:image/jpeg;base64,{redacted_image_base64}"
        }
        
        # Print the JSON to stdout, which FastAPI will capture
        json.dump(response, sys.stdout)
        sys.stdout.flush()
        
    except Exception as e:
        # Catch any errors and return a JSON error message
        error_response = {"error": f"An unexpected error occurred: {e}"}
        json.dump(error_response, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
