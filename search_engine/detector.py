"""
Object Detection and Focus-Crop Module
Uses YOLOv8 to detect objects/fashion items, crop the focused region, and eliminate background noise.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

_detector_instance = None


import cv2

def detect_salient_contour_box(image_rgb: Image.Image) -> Optional[List[float]]:
    """
    Locates the primary salient fashion product using computer vision (Otsu + Canny contours).
    Works on standalone product photos (shoes, dresses, shirts, watches, sunglasses)
    where standard COCO YOLO lacks dedicated category labels.
    """
    try:
        np_img = np.array(image_rgb)
        h, w = np_img.shape[:2]
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Dual thresholding for both light and dark backgrounds
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, 40, 140)

        # Choose the threshold mask whose borders are cleanest (background)
        # Check border pixel density
        border_pixels = np.concatenate([thresh_inv[0, :], thresh_inv[-1, :], thresh_inv[:, 0], thresh_inv[:, -1]])
        mask = thresh_inv if np.mean(border_pixels) < 128 else thresh_norm
        combined = cv2.bitwise_or(mask, edges)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        total_area = w * h
        best_box = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (total_area * 0.04) < area < (total_area * 0.95):
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Ignore extreme edge strips
                if cw > 20 and ch > 20 and (cw * ch) > best_area:
                    best_area = cw * ch
                    best_box = [float(x), float(y), float(x + cw), float(y + ch)]

        return best_box
    except Exception as e:
        print(f"[ObjectDetector] Salient detection notice: {e}")
        return None


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        from ultralytics import YOLO
        print(f"[ObjectDetector] Loading YOLO model '{model_name}'...")
        self.model = YOLO(model_name)
        print("[ObjectDetector] YOLO model loaded successfully.")

    def detect_and_crop(
        self,
        image_input: Any,
        target_box_index: Optional[int] = None,
        conf_threshold: float = 0.25,
        padding_ratio: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detects objects in the image using Hybrid YOLOv8 + Salient Contour detection.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be file path or PIL Image")

        width, height = image.size
        # 1. Run YOLO inference
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        
        detected_boxes = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, f"object_{cls_id}")
                detected_boxes.append({
                    "box": [round(c, 1) for c in xyxy],
                    "confidence": round(conf, 3),
                    "class_name": cls_name,
                    "is_selected": False
                })

        # Non-fashion COCO classes (animals, vehicles, foods, sports equipment, appliances, etc.)
        NON_FASHION_CLASSES = {
            "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "sports ball", "baseball bat", "baseball glove", "tennis racket", "frisbee", "skateboard", "surfboard", "skis", "snowboard", "kite",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        }
        FASHION_CLASSES = {"backpack", "umbrella", "handbag", "tie", "suitcase"}
        PERSON_CLASSES = {"person"}

        # Check for out-of-domain entities detected by YOLO
        detected_non_fashion = [b for b in detected_boxes if b["class_name"].lower() in NON_FASHION_CLASSES and b["confidence"] >= 0.35]
        detected_fashion = [b for b in detected_boxes if b["class_name"].lower() in FASHION_CLASSES]
        detected_person = [b for b in detected_boxes if b["class_name"].lower() in PERSON_CLASSES]

        is_out_of_domain = False
        detected_entity = None
        out_of_domain_reason = None

        if detected_non_fashion and not detected_fashion and not detected_person:
            # High confidence non-fashion object detected without any person/fashion item
            best_non_fashion = max(detected_non_fashion, key=lambda x: x["confidence"])
            is_out_of_domain = True
            detected_entity = best_non_fashion["class_name"].lower()
            out_of_domain_reason = f"Detected non-fashion subject: '{detected_entity}' (confidence: {best_non_fashion['confidence']})"

        # Cross-validate domain with CLIP zero-shot domain classification
        if not is_out_of_domain and not detected_fashion and not detected_person:
            try:
                from search_engine.clip_service import get_clip_service
                clip = get_clip_service()
                domain_res = clip.classify_domain(image)
                if domain_res.get("is_out_of_domain"):
                    is_out_of_domain = True
                    detected_entity = domain_res.get("top_domain", "non-fashion item").replace("_", " ")
                    out_of_domain_reason = domain_res.get("out_of_domain_reason")
            except Exception as e:
                print(f"[ObjectDetector] Domain classification check notice: {e}")

        # 2. If YOLO didn't detect specific fashion items and image is NOT out of domain:
        # run Salient Product Contour Detection to focus-crop the item and remove background clutter!
        has_fashion = len(detected_fashion) > 0
        if not has_fashion and not is_out_of_domain:
            salient_box = detect_salient_contour_box(image)
            if salient_box:
                detected_boxes.append({
                    "box": [round(c, 1) for c in salient_box],
                    "confidence": 0.88,
                    "class_name": "focused_product",
                    "is_selected": False
                })

        # Selection logic:
        selected_box = None
        if target_box_index is not None and 0 <= target_box_index < len(detected_boxes):
            selected_box = detected_boxes[target_box_index]["box"]
            detected_boxes[target_box_index]["is_selected"] = True
        elif detected_boxes and not is_out_of_domain:
            # Prioritize fashion-like items (handbag, backpack, tie, suitcase, umbrella) or person/largest item
            fashion_priorities = {"handbag", "backpack", "tie", "suitcase", "shoe", "dress", "shirt", "focused_product"}
            best_idx = 0
            best_score = -1

            for idx, item in enumerate(detected_boxes):
                cname = item["class_name"].lower()
                b = item["box"]
                area = (b[2] - b[0]) * (b[3] - b[1])
                priority_bonus = 2.0 if cname in fashion_priorities else (1.2 if cname == "person" else 0.5)
                score = area * priority_bonus * item["confidence"]
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_box = detected_boxes[best_idx]["box"]
            detected_boxes[best_idx]["is_selected"] = True

        crop_applied = False
        crop_box = [0, 0, width, height]
        cropped_image = image

        if selected_box and not is_out_of_domain:
            x1, y1, x2, y2 = selected_box
            box_w = x2 - x1
            box_h = y2 - y1

            # Only crop if the detected box is smaller than 95% of full image
            if (box_w * box_h) < (width * height * 0.95):
                # Add padding
                pad_x = box_w * padding_ratio
                pad_y = box_h * padding_ratio
                cx1 = max(0, int(x1 - pad_x))
                cy1 = max(0, int(y1 - pad_y))
                cx2 = min(width, int(x2 + pad_x))
                cy2 = min(height, int(y2 + pad_y))

                if (cx2 - cx1 > 20) and (cy2 - cy1 > 20):
                    crop_box = [cx1, cy1, cx2, cy2]
                    cropped_image = image.crop((cx1, cy1, cx2, cy2))
                    crop_applied = True

        return {
            "original_image": image,
            "cropped_image": cropped_image,
            "detected_boxes": detected_boxes,
            "crop_applied": crop_applied,
            "crop_box": crop_box,
            "dimensions": {"width": width, "height": height},
            "is_out_of_domain": is_out_of_domain,
            "detected_entity": detected_entity,
            "out_of_domain_reason": out_of_domain_reason
        }


def get_detector() -> ObjectDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ObjectDetector()
    return _detector_instance
