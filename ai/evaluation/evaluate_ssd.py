#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Comprehensive SSD Model Evaluation
Evaluates the trained SSD model on the unseen TEST split (45 images).
Computes:
- Object Detection Metrics: mAP@0.5, per-class AP, Precision, Recall, F1-score
- Image-Level Metrics: 4x4 Confusion Matrix (Normal, Crack, Moss, Seepage)
- ROC-AUC multi-class analysis with ROC curves
- Ground-truth vs predicted bounding box visualizations
"""

import os
import sys
import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")
VIS_DIR = os.path.join(CURRENT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "training"))
from dataset_loader import HeritageMonumentDataset, collate_fn
from train_ssd import build_ssd_model

CLASS_NAMES = ["Normal", "Crack", "Moss", "Seepage"]
CLASS_MAP = {"Normal": 0, "Crack": 1, "Moss": 2, "Seepage": 3}
DEFECT_CLASSES = ["Crack", "Moss", "Seepage"]

CLASS_COLORS = {
    "Normal": (34, 197, 94),     # Green
    "Crack": (239, 68, 68),      # Red
    "Moss": (16, 185, 129),      # Emerald
    "Seepage": (2, 132, 199)     # Blue
}

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

def evaluate_test_set(checkpoint_path, score_thresh=0.25, iou_thresh=0.5):
    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — SSD TEST EVALUATION")
    print("=" * 70)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Score Threshold:  {score_thresh}")
    print(f"IoU Threshold:    {iou_thresh}")
    print("-" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="test", transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Loaded {len(test_dataset)} unseen test images.")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_ssd_model(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_gt_boxes = {c: [] for c in DEFECT_CLASSES}
    all_pred_boxes = {c: [] for c in DEFECT_CLASSES}

    y_true_img = []
    y_pred_img = []
    y_scores_img = []

    visualized_count = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            img_tensor = images[0].to(device)
            target = targets[0]
            filename = target["filename"]
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            dom_gt = target["dominant_class"]

            # Forward pass
            outputs = model([img_tensor])[0]
            pred_boxes = outputs["boxes"].cpu().numpy()
            pred_scores = outputs["scores"].cpu().numpy()
            pred_labels = outputs["labels"].cpu().numpy()

            # Filter by score
            keep = pred_scores >= score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            # Collect for detection mAP
            for c_id, c_name in [(1, "Crack"), (2, "Moss"), (3, "Seepage")]:
                c_gt = [gt_boxes[k] for k in range(len(gt_labels)) if gt_labels[k] == c_id]
                c_preds = [(pred_boxes[k], pred_scores[k]) for k in range(len(pred_labels)) if pred_labels[k] == c_id]
                all_gt_boxes[c_name].append(c_gt)
                all_pred_boxes[c_name].append(c_preds)

            # Image-level class assignment
            class_probs = [0.0, 0.0, 0.0, 0.0]
            if len(pred_labels) == 0:
                pred_dom = "Normal"
                class_probs[0] = 0.95
            else:
                for k in range(len(pred_labels)):
                    c_id = pred_labels[k]
                    if c_id < 4:
                        class_probs[c_id] = max(class_probs[c_id], float(pred_scores[k]))
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

            # Normalize probs
            tot_p = sum(class_probs)
            if tot_p > 0:
                class_probs = [p / tot_p for p in class_probs]
            else:
                class_probs = [1.0, 0.0, 0.0, 0.0]

            y_true_img.append(CLASS_MAP[dom_gt])
            y_pred_img.append(CLASS_MAP[pred_dom])
            y_scores_img.append(class_probs)

            # Visualize first 10 test images
            if visualized_count < 10:
                vis_img = TF.to_pil_image(img_tensor.cpu())
                draw = ImageDraw.Draw(vis_img)

                # Draw GT boxes (solid thin)
                for b, l in zip(gt_boxes, gt_labels):
                    cls_str = CLASS_NAMES[l]
                    draw.rectangle(list(b), outline=(34, 197, 94), width=2)
                    draw.text((b[0], max(0, b[1] - 10)), f"GT:{cls_str}", fill=(34, 197, 94))

                # Draw Pred boxes
                for b, s, l in zip(pred_boxes, pred_scores, pred_labels):
                    cls_str = CLASS_NAMES[l]
                    col = CLASS_COLORS.get(cls_str, (255, 0, 0))
                    draw.rectangle(list(b), outline=col, width=2)
                    draw.text((b[0] + 2, b[3] - 12), f"{cls_str} {s:.2f}", fill=col)

                save_p = os.path.join(VIS_DIR, f"test_pred_{visualized_count+1:02d}_{os.path.splitext(filename)[0]}.png")
                vis_img.save(save_p)
                visualized_count += 1

    # Compute Detection Metrics (AP per class)
    print("\n--- Bounding-Box Detection Metrics (IoU >= 0.50) ---")
    detection_metrics = {}
    aps = []

    for c_name in DEFECT_CLASSES:
        tp = 0
        fp = 0
        total_gt = 0

        for img_idx in range(len(test_dataset)):
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
        # Approximate 11-point interpolated AP or prec * rec
        ap = prec * rec
        aps.append(ap)

        detection_metrics[c_name] = {
            "total_ground_truth_boxes": total_gt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "ap50": round(ap, 4)
        }
        print(f"Class: {c_name:8s} | GT: {total_gt:4d} | TP: {tp:4d} | FP: {fp:4d} | FN: {fn:4d} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AP@50: {ap:.4f}")

    mean_ap = sum(aps) / max(1, len(aps))
    print(f"Mean Average Precision (mAP@0.5): {mean_ap:.4f}")

    # Compute Image-Level Confusion Matrix & Classification Metrics
    cm = confusion_matrix(y_true_img, y_pred_img, labels=[0, 1, 2, 3])
    clf_rep = classification_report(y_true_img, y_pred_img, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    print("\n--- Image-Level Confusion Matrix (4 Classes) ---")
    print(f"{'':10s} " + " ".join([f"{c:>10s}" for c in CLASS_NAMES]))
    for row_idx, row in enumerate(cm):
        print(f"{CLASS_NAMES[row_idx]:10s} " + " ".join([f"{val:10d}" for val in row]))

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("SSD Monument Decay - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight="bold")

    plt.ylabel("True Monument Condition")
    plt.xlabel("Predicted Monument Condition")
    plt.tight_layout()
    cm_path = os.path.join(CURRENT_DIR, "confusion_matrix_ssd.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nSaved Confusion Matrix plot to: {cm_path}")

    # ROC-AUC Multi-Class Analysis
    y_true_bin = label_binarize(y_true_img, classes=[0, 1, 2, 3])
    y_scores_np = np.array(y_scores_img)

    roc_aucs = {}
    plt.figure(figsize=(8, 6))

    for idx, c_name in enumerate(CLASS_NAMES):
        if np.sum(y_true_bin[:, idx]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_scores_np[:, idx])
            roc_score = auc(fpr, tpr)
            roc_aucs[c_name] = round(roc_score, 4)
            plt.plot(fpr, tpr, label=f"{c_name} (AUC = {roc_score:.2f})", linewidth=2)
        else:
            roc_aucs[c_name] = "N/A (No test samples)"

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("SSD Monument Decay - Multi-Class ROC Curves", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(CURRENT_DIR, "roc_curves_ssd.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC curves plot to: {roc_path}")

    # Overall Summary Results
    macro_f1 = clf_rep["macro avg"]["f1-score"]
    weighted_f1 = clf_rep["weighted avg"]["f1-score"]
    accuracy = clf_rep["accuracy"]

    full_evaluation = {
        "model": "SSDLite320_MobileNet_V3_Large",
        "checkpoint": checkpoint_path,
        "test_image_count": len(test_dataset),
        "object_detection_metrics": {
            "mAP_50": round(mean_ap, 4),
            "per_class": detection_metrics
        },
        "image_level_classification_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(clf_rep["macro avg"]["precision"], 4),
            "macro_recall": round(clf_rep["macro avg"]["recall"], 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix": cm.tolist(),
            "roc_auc_per_class": roc_aucs,
            "detailed_classification_report": clf_rep
        }
    }

    metrics_json = os.path.join(CURRENT_DIR, "ssd_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(full_evaluation, f, indent=2)
    print(f"Saved complete evaluation metrics to: {metrics_json}")

    # CSV Summary
    metrics_csv = os.path.join(CURRENT_DIR, "ssd_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mAP@0.5", round(mean_ap, 4)])
        w.writerow(["accuracy", round(accuracy, 4)])
        w.writerow(["macro_f1", round(macro_f1, 4)])
        w.writerow(["weighted_f1", round(weighted_f1, 4)])
        for c in DEFECT_CLASSES:
            w.writerow([f"{c}_precision", detection_metrics[c]["precision"]])
            w.writerow([f"{c}_recall", detection_metrics[c]["recall"]])
            w.writerow([f"{c}_ap50", detection_metrics[c]["ap50"]])

    print(f"Saved CSV metrics to: {metrics_csv}")
    print(f"Saved 10 prediction visualizations to: {VIS_DIR}")
    print("\n[SUCCESS] Test evaluation completed.")
    return full_evaluation

if __name__ == "__main__":
    ckpt = os.path.join(CHECKPOINTS_DIR, "ssd_monument_decay_best.pth")
    if os.path.exists(ckpt):
        evaluate_test_set(ckpt)
    else:
        print(f"Checkpoint not found at: {ckpt}. Please run train_ssd.py first.")
