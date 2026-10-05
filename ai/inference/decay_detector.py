#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Core Inference Engine (DecayDetector)
Integrates:
1. Image preprocessing (Median Denoising + CLAHE enhancement)
2. Phase 3B SSDLite320 Monument Decay Detection Model
3. Coordinate transformation mapping back to native resolution
4. Surface defect coverage calculation
5. Structural Risk Assessment (Low, Medium, High)
6. Preservation Recommendation Generation
"""

import os
import sys
import numpy as np
import torch
from torchvision.ops import batched_nms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")

sys.path.insert(0, os.path.join(BASE_DIR, "training"))
sys.path.insert(0, os.path.join(BASE_DIR, "preprocessing"))
sys.path.insert(0, CURRENT_DIR)

from phase3b_framework import build_phase3b_ssd_model, CLASS_NAMES, CLASS_MAP
from image_enhancement import preprocess_monument_image
from preservation_recommender import generate_preservation_plan, format_markdown_report

CLASS_COLORS = {
    "Normal": (34, 197, 94),     # Green #22c55e
    "Crack": (239, 68, 68),      # Red #ef4444
    "Moss": (16, 185, 129),      # Emerald #10b981
    "Seepage": (2, 132, 199)     # Blue #0284c7
}

DEFAULT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "ssd_monument_decay_phase3b_best.pth")

class MonumentDecayDetector:
    def __init__(self, checkpoint_path=None, score_thresh=0.25, nms_thresh=0.45, device=None):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = build_phase3b_ssd_model(num_classes=4)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def predict(self, image_input, apply_enhancement=True, crack_boost_thresh=0.18):
        """
        Runs complete detection, risk assessment, and preservation recommendation pipeline.
        
        Args:
            image_input: File path, PIL.Image, or numpy BGR array
            apply_enhancement: Whether to run denoising and CLAHE before inference
            crack_boost_thresh: Sensitive threshold specifically for Crack detection (default 0.18)
        
        Returns:
            dict containing:
                - detections: list of dicts with box (native scale), box_norm, class_name, confidence
                - risk_assessment: risk level, coverage %, description, timeline
                - preservation_plan: complete conservation protocols
                - annotated_image: PIL Image with visual bounding boxes & status banner
                - preprocessing_info: details of applied enhancements
        """
        # Load and verify input
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
            img_name = os.path.basename(image_input)
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
            img_name = "image_input.jpg"
        else:
            orig_pil = Image.fromarray(image_input).convert("RGB")
            img_name = "image_input.jpg"

        orig_w, orig_h = orig_pil.size

        # 1. Preprocessing & Enhancement
        if apply_enhancement:
            _, enhanced_pil, prep_info = preprocess_monument_image(orig_pil, enable_clahe=True, enable_denoise=True)
            inference_pil = enhanced_pil
        else:
            inference_pil = orig_pil
            prep_info = {"operations_applied": ["None (Raw Input)"], "original_dimensions": [orig_w, orig_h]}

        # 2. Resize to 320x320 for SSDLite model
        resized_pil = inference_pil.resize((320, 320), Image.BILINEAR)
        img_tensor = TF.to_tensor(resized_pil)
        img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)
        img_tensor = img_tensor.to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        pred_boxes = outputs["boxes"].cpu()
        pred_scores = outputs["scores"].cpu()
        pred_labels = outputs["labels"].cpu()

        # Multi-threshold filter (crack gets sensitivity boost)
        keep = []
        for idx in range(len(pred_scores)):
            s = float(pred_scores[idx])
            lbl = int(pred_labels[idx])
            min_thresh = crack_boost_thresh if lbl == 1 else self.score_thresh
            if s >= min_thresh:
                keep.append(idx)

        if keep:
            keep = torch.tensor(keep, dtype=torch.long)
            p_boxes = pred_boxes[keep]
            p_scores = pred_scores[keep]
            p_labels = pred_labels[keep]

            # Batched NMS
            nms_idx = batched_nms(p_boxes, p_scores, p_labels, self.nms_thresh)
            p_boxes = p_boxes[nms_idx].numpy()
            p_scores = p_scores[nms_idx].numpy()
            p_labels = p_labels[nms_idx].numpy()
        else:
            p_boxes = np.empty((0, 4))
            p_scores = np.empty((0,))
            p_labels = np.empty((0,), dtype=int)

        # 4. Map coordinates back to native resolution
        scale_x = orig_w / 320.0
        scale_y = orig_h / 320.0

        detections = []
        detections_320 = []

        for b, s, l in zip(p_boxes, p_scores, p_labels):
            cls_name = CLASS_NAMES[l] if l < len(CLASS_NAMES) else "Unknown"
            if cls_name == "Normal":
                continue # Normal is background

            native_box = [
                round(float(b[0] * scale_x), 1),
                round(float(b[1] * scale_y), 1),
                round(float(b[2] * scale_x), 1),
                round(float(b[3] * scale_y), 1)
            ]
            detections.append({
                "class_name": cls_name,
                "class_id": int(l),
                "confidence": round(float(s), 4),
                "box": native_box,
                "box_normalized": [
                    round(native_box[0] / orig_w, 4),
                    round(native_box[1] / orig_h, 4),
                    round(native_box[2] / orig_w, 4),
                    round(native_box[3] / orig_h, 4)
                ]
            })
            detections_320.append({
                "class_name": cls_name,
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "confidence": float(s)
            })

        # 5. Risk Assessment & Preservation Protocols
        preservation_plan = generate_preservation_plan(img_name, detections_320, 320, 320)
        risk_assessment = preservation_plan["risk_assessment"]

        # 6. Render Visualization with Bounding Boxes & Risk Banner
        annotated_img = orig_pil.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Draw defect bounding boxes
        box_width = max(2, int(min(orig_w, orig_h) / 300))
        for det in detections:
            c_name = det["class_name"]
            color = CLASS_COLORS.get(c_name, (255, 0, 0))
            b = det["box"]
            draw.rectangle(b, outline=color, width=box_width)
            
            # Label banner
            label_text = f"{c_name} {det['confidence']:.2f}"
            tx1, ty1 = b[0], max(0, b[1] - 22)
            draw.rectangle([tx1, ty1, tx1 + len(label_text) * 11, ty1 + 20], fill=color)
            draw.text((tx1 + 4, ty1 + 2), label_text, fill=(255, 255, 255))

        # Draw Top Status Banner
        banner_h = max(40, int(orig_h * 0.045))
        risk_col = (239, 68, 68) if risk_assessment["risk_level"] == "HIGH" else (
            (234, 179, 8) if risk_assessment["risk_level"] == "MEDIUM" else (34, 197, 94)
        )
        draw.rectangle([0, 0, orig_w, banner_h], fill=(15, 23, 42)) # Slate dark header
        draw.rectangle([0, banner_h - 4, orig_w, banner_h], fill=risk_col) # Accent line

        banner_text = (
            f"HERITAGE AI | CONDITION: {risk_assessment['risk_level']} RISK | "
            f"COVERAGE: {risk_assessment['coverage_percentage']}% | "
            f"DEFECTS: {len(detections)} ({', '.join([f'{k}:{v}' for k, v in risk_assessment['defect_counts'].items()]) or 'None'})"
        )
        draw.text((15, int(banner_h * 0.25)), banner_text, fill=(255, 255, 255))

        return {
            "image_name": img_name,
            "original_dimensions": [orig_w, orig_h],
            "total_detections": len(detections),
            "detections": detections,
            "risk_assessment": risk_assessment,
            "preservation_plan": preservation_plan,
            "preprocessing_info": prep_info,
            "annotated_image": annotated_img
        }

if __name__ == "__main__":
    print("Testing MonumentDecayDetector...")
    det = MonumentDecayDetector()
    print("Model loaded successfully into detector on device:", det.device)
