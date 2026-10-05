#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Sliced/Tiled High-Resolution Inference (Candidate B)
Implements:
- Sliced / overlapping crop inference on 640x640 high-resolution images
- Tile size: 320x320, Stride: 240x240 (25% overlap = 80px)
- Precise coordinate shifting back to full image space
- Global Batched NMS to deduplicate boxes across overlapping tiles
- Evaluates detection metrics against ground truth
"""

import os
import sys
import numpy as np
import torch
torch.backends.mkldnn.enabled = False
from torchvision.ops import batched_nms
from PIL import Image
import torchvision.transforms.functional as TF

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")

sys.path.insert(0, os.path.join(BASE_DIR, "training"))
from dataset_loader import HeritageMonumentDataset, collate_fn
from phase3b_framework import build_phase3b_ssd_model, compute_iou, CLASS_MAP, CLASS_NAMES, DEFECT_CLASSES

def predict_tiled_image(model, img_pil, tile_size=320, stride=240, target_res=(640, 640), score_thresh=0.15, nms_thresh=0.45, device="cpu"):
    """
    Performs sliced/tiled inference:
    1. Resizes input PIL image to high-res target (e.g. 640x640).
    2. Slices into 320x320 overlapping crops with specified stride.
    3. Runs model on each tile.
    4. Offsets coordinates back to 640x640 space, then scales to 320x320 GT coordinate space.
    5. Applies global batched NMS to suppress duplicate overlapping detections.
    """
    orig_w, orig_h = img_pil.size
    high_res_img = img_pil.resize(target_res, Image.BILINEAR)
    high_w, high_h = target_res

    # Scale factor from 640x640 to 320x320 canonical GT space
    scale_x = 320.0 / high_w
    scale_y = 320.0 / high_h

    # Tile offsets
    x_starts = list(range(0, high_w - tile_size + 1, stride))
    if x_starts[-1] != high_w - tile_size:
        x_starts.append(high_w - tile_size)

    y_starts = list(range(0, high_h - tile_size + 1, stride))
    if y_starts[-1] != high_h - tile_size:
        y_starts.append(high_h - tile_size)

    all_boxes = []
    all_scores = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                tile = high_res_img.crop((x0, y0, x0 + tile_size, y0 + tile_size))
                tile_tensor = TF.to_tensor(tile).to(device)

                out = model([tile_tensor])[0]
                tb = out["boxes"].cpu()
                ts = out["scores"].cpu()
                tl = out["labels"].cpu()

                keep = ts >= score_thresh
                tb = tb[keep]
                ts = ts[keep]
                tl = tl[keep]

                if len(tb) > 0:
                    # Map from tile coords to 640 coords, then to 320 GT coords
                    mapped_b = torch.empty_like(tb)
                    mapped_b[:, 0] = (tb[:, 0] + x0) * scale_x
                    mapped_b[:, 1] = (tb[:, 1] + y0) * scale_y
                    mapped_b[:, 2] = (tb[:, 2] + x0) * scale_x
                    mapped_b[:, 3] = (tb[:, 3] + y0) * scale_y

                    all_boxes.append(mapped_b)
                    all_scores.append(ts)
                    all_labels.append(tl)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Global batched NMS to deduplicate overlapping predictions
    keep_nms = batched_nms(all_boxes, all_scores, all_labels, nms_thresh)

    final_boxes = all_boxes[keep_nms].numpy()
    final_scores = all_scores[keep_nms].numpy()
    final_labels = all_labels[keep_nms].numpy()

    return final_boxes, final_scores, final_labels

def evaluate_tiled_dataset(model, dataset, split_name="val", score_thresh=0.15, nms_thresh=0.45, device="cpu"):
    """
    Evaluates Candidate B (tiled inference) on validation or test dataset.
    """
    print(f"\n--- Evaluating Candidate B Tiled Inference on {split_name} ({len(dataset)} images) ---", flush=True)
    print(f"Tile Size: 320x320 | Stride: 240px (Overlap: 25% = 80px) | High-Res Target: 640x640", flush=True)
    print(f"Score Thresh: {score_thresh} | NMS Thresh: {nms_thresh}", flush=True)

    all_gt_boxes = {c: [] for c in DEFECT_CLASSES}
    all_pred_boxes = {c: [] for c in DEFECT_CLASSES}
    y_true_img = []
    y_pred_img = []

    for item in dataset.items:
        img_path = os.path.join(dataset.img_dir, item["processed_image"])
        # Open image
        img_pil = Image.open(img_path).convert("RGB")
        dom_gt = item["dominant_class"]

        gt_boxes = []
        gt_labels = []
        for b in item["boxes"]:
            cls_name = b["name"]
            if cls_name in CLASS_MAP and cls_name != "Normal":
                gt_boxes.append(b["bbox"])
                gt_labels.append(CLASS_MAP[cls_name])

        gt_boxes = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
        gt_labels = np.array(gt_labels) if gt_labels else np.empty((0,), dtype=int)

        pred_boxes, pred_scores, pred_labels = predict_tiled_image(
            model, img_pil, tile_size=320, stride=240, target_res=(640, 640),
            score_thresh=score_thresh, nms_thresh=nms_thresh, device=device
        )

        for c_id, c_name in [(1, "Crack"), (2, "Moss"), (3, "Seepage")]:
            c_gt = [gt_boxes[k] for k in range(len(gt_labels)) if gt_labels[k] == c_id]
            c_preds = [(pred_boxes[k], pred_scores[k]) for k in range(len(pred_labels)) if pred_labels[k] == c_id]
            all_gt_boxes[c_name].append(c_gt)
            all_pred_boxes[c_name].append(c_preds)

        class_probs = [0.0, 0.0, 0.0, 0.0]
        if len(pred_labels) == 0:
            pred_dom = "Normal"
        else:
            for k in range(len(pred_labels)):
                c_id = pred_labels[k]
                if c_id < 4:
                    class_probs[c_id] = max(class_probs[c_id], float(pred_scores[k]))
            if class_probs[1] >= score_thresh:
                pred_dom = "Crack"
            elif class_probs[3] >= score_thresh:
                pred_dom = "Seepage"
            elif class_probs[2] >= score_thresh:
                pred_dom = "Moss"
            else:
                pred_dom = "Normal"

        y_true_img.append(CLASS_MAP[dom_gt])
        y_pred_img.append(CLASS_MAP[pred_dom])

    det_metrics = {}
    aps = []
    num_dataset = len(dataset)

    for c_name in DEFECT_CLASSES:
        tp, fp, total_gt = 0, 0, 0
        for img_idx in range(num_dataset):
            gts = list(all_gt_boxes[c_name][img_idx])
            preds = sorted(all_pred_boxes[c_name][img_idx], key=lambda x: x[1], reverse=True)
            total_gt += len(gts)
            matched = [False] * len(gts)
            for pb, ps in preds:
                best_iou, best_idx = 0.0, -1
                for gi, gb in enumerate(gts):
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
                if best_iou >= 0.50 and not matched[best_idx]:
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
        det_metrics[c_name] = {
            "tp": tp, "fp": fp, "fn": fn, "gt": total_gt,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "ap": round(ap, 4)
        }
        print(f"Class: {c_name:8s} | GT:{total_gt:3d} | TP:{tp:3d} | FP:{fp:3d} | FN:{fn:3d} | Prec:{prec:.4f} | Rec:{rec:.4f} | F1:{f1:.4f} | AP:{ap:.4f}", flush=True)

    mAP = sum(aps) / len(aps)
    correct = sum(1 for yt, yp in zip(y_true_img, y_pred_img) if yt == yp)
    acc = correct / len(y_true_img)
    macro_f1 = sum(det_metrics[c]["f1"] for c in DEFECT_CLASSES) / len(DEFECT_CLASSES)

    print(f"Mean Average Precision (mAP@0.5): {mAP:.4f}", flush=True)
    print(f"Image-Level Accuracy:             {acc*100:.2f}%", flush=True)
    print(f"Macro F1-Score:                   {macro_f1:.4f}", flush=True)

    return {
        "mAP_50": round(mAP, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "det_metrics": det_metrics
    }

if __name__ == "__main__":
    device = torch.device("cpu")
    val_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="val", transforms=None)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "ssd_phase3b_Candidate_A.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_phase3b_ssd_model(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    res = evaluate_tiled_dataset(model, val_dataset, split_name="val", score_thresh=0.20, nms_thresh=0.45)
