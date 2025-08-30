import io
import cv2
import base64
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from geoclip import GeoCLIP
import tempfile
import json


# Core functions from redactor.py, combined here for a single-file backend
def load_image(image_bytes):
    try:
        image_stream = io.BytesIO(image_bytes)
        image_np = np.frombuffer(image_stream.read(), np.uint8)
        img_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from input.")
        return img_bgr
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

def load_gdino(device: str = "cpu"):
    print("Loading Grounding DINO model...", file=sys.stderr)
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("Grounding DINO model loaded.", file=sys.stderr)
    return processor, model

def detect_gdino(img_pil, processor, model, device, box_threshold, text_threshold, queries):
    # The new model requires text queries to be a single string, lowercase, and end with a period.
    text = ". ".join([q.lower() for q in queries]) + "."
    inputs = processor(images=img_pil, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img_pil.size[::-1]]
    )
    
    boxes = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"]
    #phrases = results[0]["phrases"]

    return boxes, labels

def try_ocr():
    try:
        from paddleocr import PaddleOCR
        print("Loading PaddleOCR...", file=sys.stderr)
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        print("PaddleOCR loaded.", file=sys.stderr)
        return ocr
    except ImportError:
        print("PaddleOCR not found. Skipping OCR detection.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading PaddleOCR: {e}. Skipping OCR detection.", file=sys.stderr)
        return None

def detect_ocr_boxes(image_bgr, ocr):
    results = ocr.ocr(image_bgr, cls=True)
    boxes = []
    if results and results[0]:
        for line in results[0]:
            points = line[0]
            if points:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                boxes.append([x_min, y_min, x_max, y_max])
    return np.array(boxes)

def union_masks(image_shape, box_lists):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for boxes in box_lists:
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x_min, y_min, x_max, y_max = [int(v) for v in box]
                mask[y_min:y_max, x_min:x_max] = 255
    return mask

def redact(image, mask, method="blur", blur_ksize=151, mosaic_scale=0.06):
    if method == "blur":
        # Ensure ksize is odd
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
        return np.where(mask[:, :, None] == 255, blurred, image)
    elif method == "pixelate":
        h, w = image.shape[:2]
        small_h = int(h * mosaic_scale)
        small_w = int(w * mosaic_scale)
        if small_h <= 0: small_h = 1
        if small_w <= 0: small_w = 1
        
        resized = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask[:, :, None] == 255, pixelated, image)
    return image

# Initialize the FastAPI app
app = FastAPI()

# Global variables for models
device = "cuda" if torch.cuda.is_available() else "cpu"
processor, gdino_model = None, None
ocr_model = None
models_loaded = False

async def load_models_async():
    global models_loaded, processor, gdino_model, ocr_model
    if not models_loaded:
        processor, gdino_model = load_gdino(device)
        ocr_model = try_ocr()
        models_loaded = True

@app.on_event("startup")
async def startup_event():
    await load_models_async()

class ProcessImageResponse(BaseModel):
    redacted_image: str

@app.post("/process_image", response_model=ProcessImageResponse)
async def process_image_endpoint(
    image: UploadFile = File(...),
    method: str = Form("blur"),
    blur_ksize: int = Form(151),
    mosaic_scale: float = Form(0.06),
    query: str = Form([])
):
    try:
        # Load models if not already loaded
        await load_models_async()
        # Parse the JSON string back into a Python list
        received_query = json.loads(query)
        print(f"Received query from frontend: {received_query}")
        geo_model = GeoCLIP()
        # Read the uploaded image file's content
        image_bytes = await image.read()
        print("Received image data from frontend.", file=sys.stderr)
        
        # Process the image
        img_bgr = load_image(image_bytes)
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        top_pred_gps, top_pred_prob = geo_model.predict(tmp_path, top_k=1)
        # queries = ["street name sign", "road name sign", "flag", "landmark", "monument", "person", "child"]
        queries = []
        if "flag" in received_query:
            queries.extend(["flag", "country flags", "state flags"])
        if "sign" in received_query:
            queries.extend(["street name sign", "road name sign"])
        if 'faces' in received_query:
            queries.extend(["human faces", "faces", "people faces", "child faces", "human head", "people head"])
        if 'landmark' in received_query:
            queries.extend(["famous landmark", "monument", "historical site", "tourist attraction"])
        print(queries)
        boxes_gd, _ = detect_gdino(img_pil, processor, gdino_model, device, 0.25, 0.20, queries)
        boxes_ocr = detect_ocr_boxes(img_bgr, ocr_model) if ocr_model else np.empty((0, 4), dtype=int)
        print(top_pred_gps, top_pred_prob)
        mask = union_masks(img_bgr.shape, [boxes_gd, boxes_ocr])
        redacted_image = redact(img_bgr, mask, method, blur_ksize, mosaic_scale)
        k = top_pred_gps.tolist()  # First, convert the tensor to a list
        print(k)
        gps = [round(item, 3) for item in k[0]]
        print(gps)
        l = top_pred_prob.tolist()
        print(l)
        prob = [round(l[0], 3)*100]
        print(prob)
        # Encode the redacted image to base64
        _, img_encoded = cv2.imencode('.jpeg', redacted_image)
        redacted_image_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        # save the image
        cv2.imwrite("redacted_output.jpg", redacted_image)
        return JSONResponse(content={
                  "redacted_image": f"data:image/jpeg;base64,{redacted_image_base64}",
                  "predicted_location": {
                        "gps": gps,
                        "probability": prob
                  }
            })

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
