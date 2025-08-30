# backend/core/redactor.py

import os
import requests
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- Core Logic Functions (from your original code) ---
def load_image(path_or_url):
    if str(path_or_url).startswith(("http://", "https://")):
        resp = requests.get(path_or_url, stream=True)
        resp.raise_for_status()
        img = Image.open(resp.raw).convert("RGB")
    else:
        if not os.path.isfile(path_or_url):
            raise FileNotFoundError(path_or_url)
        img = Image.open(path_or_url).convert("RGB")
    return img

def load_gdino(device):
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    model.eval()
    return processor, model

def detect_gdino(img_pil, processor, model, device, box_thresh, text_thresh, queries):
    w, h = img_pil.size
    inputs = processor(
        images=img_pil,
        text=[q.lower().strip() for q in queries],
        padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[(h, w)]
    )[0]
    boxes = result["boxes"].cpu().numpy().astype(int) if len(result["boxes"]) else np.empty((0, 4), dtype=int)
    labels = result["labels"]
    scores = result["scores"].cpu().numpy() if len(result["scores"]) else np.empty((0,))
    
    keep = []
    for i, lab in enumerate(labels):
        lab_l = lab.lower()
        if any(k in lab_l for k in ["sign", "flag", "board", "landmark", "monument", "person", "child"]):
            keep.append(i)
    boxes = boxes[keep]
    labels = [labels[i] for i in keep]
    scores = scores[keep] if len(keep) else np.empty((0,))
    return boxes, labels, scores

def try_ocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    except Exception as e:
        print(f"PaddleOCR not available ({e}); continuing without OCR.")
        return None

def detect_ocr_boxes(img_bgr, ocr, min_area=4000, merge=True):
    if ocr is None:
        return np.empty((0, 4), dtype=int)
    h, w = img_bgr.shape[:2]
    res = ocr.ocr(img_bgr[..., ::-1], cls=True)
    boxes = []
    if res and isinstance(res, list):
        for page in res:
            if page is None: continue
            for det in page:
                poly = np.array(det[0]).astype(int)
                x1, y1 = poly.min(axis=0)
                x2, y2 = poly.max(axis=0)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                area = (x2 - x1) * (y2 - y1)
                if area >= min_area:
                    boxes.append([x1, y1, x2, y2])
    boxes = np.array(boxes, dtype=int) if boxes else np.empty((0, 4), dtype=int)
    if merge and len(boxes) > 0:
        boxes = merge_overlaps(boxes, iou_thresh=0.2)
    return boxes

def merge_overlaps(boxes, iou_thresh=0.2):
    boxes = boxes.tolist()
    merged = []
    while boxes:
        b = boxes.pop(0)
        group = [b]
        rest = []
        for c in boxes:
            if iou(b, c) > iou_thresh:
                group.append(c)
            else:
                rest.append(c)
        x1 = min(t[0] for t in group)
        y1 = min(t[1] for t in group)
        x2 = max(t[2] for t in group)
        y2 = max(t[3] for t in group)
        merged.append([x1, y1, x2, y2])
        boxes = rest
    return np.array(merged, dtype=int)

def iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / float(area_a + area_b - inter)

def union_masks(img_shape, list_of_boxes_lists):
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for boxes in list_of_boxes_lists:
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 1
    return mask

def redact(img_bgr, mask, method="blur", blur_ksize=151, mosaic_scale=0.06):
    out = img_bgr.copy()
    if mask.sum() == 0:
        return out
    if method == "pixelate":
        ys, xs = np.where(mask == 1)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        roi = out[y1:y2, x1:x2]
        if roi.size > 0:
            h, w = roi.shape[:2]
            small = cv2.resize(roi, (max(1, int(w * mosaic_scale)), max(1, int(h * mosaic_scale))), interpolation=cv2.INTER_LINEAR)
            pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            roi_mask = mask[y1:y2, x1:x2]
            out[y1:y2, x1:x2][roi_mask == 1] = pix[roi_mask == 1]
    else:  # blur
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)
        out[mask == 1] = blurred[mask == 1]
    return out