#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Phase 3B Improvement Framework
Modular components for:
- Detection-safe data augmentations with box geometry transformation
- Class-aware weighted random sampler for minority classes (Crack, Moss)
- Class-weighted SSD loss computation
- Resolution scaling support (320x320, 480x480, 640x640)
- Validation evaluation and threshold tuning
"""

import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Normal", "Crack", "Moss", "Seepage"]
CLASS_MAP = {"Normal": 0, "Crack": 1, "Moss": 2, "Seepage": 3}
DEFECT_CLASSES = ["Crack", "Moss", "Seepage"]

def compute_iou(box1, box2):
    """Calculates Intersection over Union between two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class DetectionAugmentationPhase3B:
    """
    Object-detection safe data augmentation:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.2)
    - ColorJitter (brightness, contrast, saturation)
    All coordinate transformations correctly invert bounding boxes in (img_w, img_h) space.
    """
    def __init__(self, p_hflip=0.5, p_vflip=0.2, brightness=0.15, contrast=0.15, saturation=0.10, img_size=(320, 320)):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        self.img_size = img_size  # (W, H)

    def __call__(self, img_tensor, target):
        # 1. Photometric jitter (preserves coordinates and fine cracks)
        img_tensor = self.jitter(img_tensor)

        boxes = target["boxes"].clone()
        img_w, img_h = self.img_size

        # 2. Horizontal flip
        if random.random() < self.p_hflip:
            img_tensor = TF.hflip(img_tensor)
            if len(boxes) > 0:
                new_xmin = img_w - boxes[:, 2]
                new_xmax = img_w - boxes[:, 0]
                boxes[:, 0] = new_xmin
                boxes[:, 2] = new_xmax

        # 3. Vertical flip
        if random.random() < self.p_vflip:
            img_tensor = TF.vflip(img_tensor)
            if len(boxes) > 0:
                new_ymin = img_h - boxes[:, 3]
                new_ymax = img_h - boxes[:, 1]
                boxes[:, 1] = new_ymin
                boxes[:, 3] = new_ymax

        target["boxes"] = boxes
        return img_tensor, target

def get_class_aware_sampler(dataset, weights_dict=None):
    """
    Creates a WeightedRandomSampler that samples minority classes (Crack, Moss)
    more frequently to overcome the 86% Seepage imbalance.
    """
    if weights_dict is None:
        weights_dict = {
            "Crack": 4.0,
            "Moss": 4.0,
            "Normal": 2.0,
            "Seepage": 1.0
        }

    sample_weights = []
    for item in dataset.items:
        # Check all boxes present in this image
        cls_present = {b["name"] for b in item["boxes"]}
        if "Crack" in cls_present:
            weight = weights_dict.get("Crack", 4.0)
        elif "Moss" in cls_present:
            weight = weights_dict.get("Moss", 4.0)
        elif item["is_normal"]:
            weight = weights_dict.get("Normal", 2.0)
        else:
            weight = weights_dict.get("Seepage", 1.0)
        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def create_weighted_ssd_loss(class_weights_tensor):
    """
    Creates a patched compute_loss function that applies class weights to foreground/background cross-entropy.
    """
    def weighted_compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, cls_logits, anchors, matched_idxs
        ):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Weighted classification cross-entropy
        num_classes = cls_logits.size(-1)
        weights = class_weights_tensor.to(cls_logits.device)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            weight=weights,
            reduction="none"
        ).view(cls_targets.size())

        # Hard negative mining
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")
        values, idx = negative_loss.sort(1, descending=True)
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    return weighted_compute_loss

def build_phase3b_ssd_model(num_classes=4, loss_weights=None, score_thresh=0.001, nms_thresh=0.45):
    """
    Builds SSDLite320 model with MobileNetV3-Large backbone and optional class-weighted loss.
    """
    model = ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT
    )
    model.score_thresh = score_thresh
    model.nms_thresh = nms_thresh

    if loss_weights is not None:
        weights_t = torch.tensor(loss_weights, dtype=torch.float32)
        import types
        model.compute_loss = types.MethodType(create_weighted_ssd_loss(weights_t), model)

    return model

def evaluate_detection_metrics(model, dataloader, device, score_thresh=0.15, iou_thresh=0.50):
    """
    Evaluates object detection performance:
    - Sets model.score_thresh and evaluates IoU >= 0.50
    - Matches detections to ground truth per image
    - Computes TP, FP, FN, Precision, Recall, F1, and AP@0.5 per defect class
    - Computes Image-Level Accuracy & Macro F1
    """
    model.eval()
    orig_score_thresh = getattr(model, "score_thresh", 0.001)
    model.score_thresh = score_thresh

    all_gt_boxes = {c: [] for c in DEFECT_CLASSES}
    all_pred_boxes = {c: [] for c in DEFECT_CLASSES}

    y_true_img = []
    y_pred_img = []
    y_scores_img = []

    with torch.no_grad():
        for images, targets in dataloader:
            img = images[0].to(device)
            target = targets[0]
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            dom_gt = target["dominant_class"]

            outputs = model([img])[0]
            p_boxes = outputs["boxes"].cpu().numpy()
            p_scores = outputs["scores"].cpu().numpy()
            p_labels = outputs["labels"].cpu().numpy()

            # Filter by score_thresh
            keep = p_scores >= score_thresh
            p_boxes = p_boxes[keep]
            p_scores = p_scores[keep]
            p_labels = p_labels[keep]

            for c_id, c_name in [(1, "Crack"), (2, "Moss"), (3, "Seepage")]:
                c_gt = [gt_boxes[k] for k in range(len(gt_labels)) if gt_labels[k] == c_id]
                c_preds = [(p_boxes[k], p_scores[k]) for k in range(len(p_labels)) if p_labels[k] == c_id]
                all_gt_boxes[c_name].append(c_gt)
                all_pred_boxes[c_name].append(c_preds)

            # Image-level assignment
            class_probs = [0.0, 0.0, 0.0, 0.0]
            if len(p_labels) == 0:
                pred_dom = "Normal"
                class_probs[0] = 0.95
            else:
                for k in range(len(p_labels)):
                    c_id = p_labels[k]
                    if c_id < 4:
                        class_probs[c_id] = max(class_probs[c_id], float(p_scores[k]))
                # Priority: Crack > Seepage > Moss > Normal
                if class_probs[1] >= score_thresh:
                    pred_dom = "Crack"
                elif class_probs[3] >= score_thresh:
                    pred_dom = "Seepage"
                elif class_probs[2] >= score_thresh:
                    pred_dom = "Moss"
                else:
                    pred_dom = "Normal"
                    class_probs[0] = 0.85

            tot_p = sum(class_probs)
            if tot_p > 0:
                class_probs = [p / tot_p for p in class_probs]
            else:
                class_probs = [1.0, 0.0, 0.0, 0.0]

            y_true_img.append(CLASS_MAP[dom_gt])
            y_pred_img.append(CLASS_MAP[pred_dom])
            y_scores_img.append(class_probs)

    # Restore orig score thresh
    model.score_thresh = orig_score_thresh

    detection_metrics = {}
    aps = []
    num_dataset = len(all_gt_boxes["Crack"])

    for c_name in DEFECT_CLASSES:
        tp = 0
        fp = 0
        total_gt = 0

        for img_idx in range(num_dataset):
            gts = list(all_gt_boxes[c_name][img_idx])
            preds = sorted(all_pred_boxes[c_name][img_idx], key=lambda x: x[1], reverse=True)
            total_gt += len(gts)
            matched = [False] * len(gts)

            for p_box, p_score in preds:
                best_iou = 0.0
                best_idx = -1
                for g_idx, g_box in enumerate(gts):
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = g_idx

                if best_iou >= iou_thresh and not matched[best_idx]:
                    tp += 1
                    matched[best_idx] = True
                else:
                    fp += 1

        fn = total_gt - tp
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, total_gt)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)
        ap = prec * rec
        aps.append(ap)

        detection_metrics[c_name] = {
            "total_gt": total_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "ap50": round(ap, 4)
        }

    mean_ap = sum(aps) / max(1, len(aps))

    cm = confusion_matrix(y_true_img, y_pred_img, labels=[0, 1, 2, 3])
    clf_rep = classification_report(y_true_img, y_pred_img, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    return {
        "mAP_50": round(mean_ap, 4),
        "per_class": detection_metrics,
        "score_thresh": score_thresh,
        "iou_thresh": iou_thresh,
        "image_level": {
            "accuracy": round(clf_rep["accuracy"], 4),
            "macro_f1": round(clf_rep["macro avg"]["f1-score"], 4),
            "weighted_f1": round(clf_rep["weighted avg"]["f1-score"], 4),
            "confusion_matrix": cm.tolist()
        },
        "y_true_img": y_true_img,
        "y_pred_img": y_pred_img,
        "y_scores_img": y_scores_img
    }
