#!/usr/bin/env python3
"""
Fast 1-pass cached validation threshold grid search for SSD models.
Runs model forward pass ONCE over validation set (47 images),
then evaluates score and NMS threshold combinations in memory.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "checkpoints")

sys.path.insert(0, CURRENT_DIR)
from dataset_loader import HeritageMonumentDataset, collate_fn
from phase3b_framework import build_phase3b_ssd_model, compute_iou, CLASS_MAP, CLASS_NAMES, DEFECT_CLASSES

def run_fast_val_tuning(ckpt_name):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return None

    device = torch.device("cpu")
    val_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="val", transforms=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"\n=======================================================", flush=True)
    print(f" FAST VALIDATION GRID TUNING: {ckpt_name}", flush=True)
    print(f"=======================================================", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_phase3b_ssd_model(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Base inference with minimal score_thresh so all candidate anchors are available
    model.score_thresh = 0.01
    model.nms_thresh = 0.80
    model.topk_candidates = 500
    model.detections_per_img = 500

    print("Running 1-pass inference over 47 validation images...", flush=True)
    cached_preds = []
    cached_gts = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            img = images[0].to(device)
            tgt = targets[0]
            out = model([img])[0]

            cached_preds.append({
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu()
            })
            cached_gts.append({
                "boxes": tgt["boxes"].cpu().numpy(),
                "labels": tgt["labels"].cpu().numpy(),
                "dominant_class": tgt["dominant_class"]
            })

    print(f"Inference cached! Evaluating grid combinations...", flush=True)

    score_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    nms_grid = [0.35, 0.45, 0.55]

    results = []

    for nms_t in nms_grid:
        for s_t in score_grid:
            all_gt_boxes = {c: [] for c in DEFECT_CLASSES}
            all_pred_boxes = {c: [] for c in DEFECT_CLASSES}
            y_true_img = []
            y_pred_img = []

            for p_dict, g_dict in zip(cached_preds, cached_gts):
                boxes = p_dict["boxes"]
                scores = p_dict["scores"]
                labels = p_dict["labels"]

                # Filter by score
                keep_score = scores >= s_t
                boxes_f = boxes[keep_score]
                scores_f = scores[keep_score]
                labels_f = labels[keep_score]

                # Apply per-class NMS
                if len(boxes_f) > 0:
                    keep_nms = batched_nms(boxes_f, scores_f, labels_f, nms_t)
                    boxes_f = boxes_f[keep_nms].numpy()
                    scores_f = scores_f[keep_nms].numpy()
                    labels_f = labels_f[keep_nms].numpy()
                else:
                    boxes_f = np.empty((0, 4))
                    scores_f = np.empty((0,))
                    labels_f = np.empty((0,), dtype=int)

                # Ground truth
                gt_boxes = g_dict["boxes"]
                gt_labels = g_dict["labels"]
                dom_gt = g_dict["dominant_class"]

                for c_id, c_name in [(1, "Crack"), (2, "Moss"), (3, "Seepage")]:
                    c_gt = [gt_boxes[k] for k in range(len(gt_labels)) if gt_labels[k] == c_id]
                    c_preds = [(boxes_f[k], scores_f[k]) for k in range(len(labels_f)) if labels_f[k] == c_id]
                    all_gt_boxes[c_name].append(c_gt)
                    all_pred_boxes[c_name].append(c_preds)

                # Image-level assignment
                class_probs = [0.0, 0.0, 0.0, 0.0]
                if len(labels_f) == 0:
                    pred_dom = "Normal"
                else:
                    for k in range(len(labels_f)):
                        c_id = labels_f[k]
                        if c_id < 4:
                            class_probs[c_id] = max(class_probs[c_id], float(scores_f[k]))
                    if class_probs[1] >= s_t:
                        pred_dom = "Crack"
                    elif class_probs[3] >= s_t:
                        pred_dom = "Seepage"
                    elif class_probs[2] >= s_t:
                        pred_dom = "Moss"
                    else:
                        pred_dom = "Normal"

                y_true_img.append(CLASS_MAP[dom_gt])
                y_pred_img.append(CLASS_MAP[pred_dom])

            # Calculate metrics
            det_metrics = {}
            aps = []
            for c_name in DEFECT_CLASSES:
                tp, fp, total_gt = 0, 0, 0
                for img_idx in range(len(cached_preds)):
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

            mAP = sum(aps) / len(aps)
            correct = sum(1 for yt, yp in zip(y_true_img, y_pred_img) if yt == yp)
            acc = correct / len(y_true_img)

            results.append({
                "nms_thresh": nms_t,
                "score_thresh": s_t,
                "mAP_50": round(mAP, 4),
                "accuracy": round(acc, 4),
                "crack_tp": det_metrics["Crack"]["tp"],
                "crack_prec": det_metrics["Crack"]["precision"],
                "crack_rec": det_metrics["Crack"]["recall"],
                "crack_f1": det_metrics["Crack"]["f1"],
                "moss_tp": det_metrics["Moss"]["tp"],
                "moss_prec": det_metrics["Moss"]["precision"],
                "moss_rec": det_metrics["Moss"]["recall"],
                "moss_f1": det_metrics["Moss"]["f1"],
                "seepage_tp": det_metrics["Seepage"]["tp"],
                "seepage_prec": det_metrics["Seepage"]["precision"],
                "seepage_rec": det_metrics["Seepage"]["recall"],
                "seepage_f1": det_metrics["Seepage"]["f1"],
                "det_metrics": det_metrics
            })

    # Sort results primarily by mAP_50, then by Crack F1 + Moss F1
    results = sorted(results, key=lambda x: (x["mAP_50"], x["crack_f1"] + x["moss_f1"]), reverse=True)

    print("\n--- TOP 10 VALIDATION THRESHOLD CONFIGURATIONS ---", flush=True)
    for rank, r in enumerate(results[:10]):
        print(f"#{rank+1:02d} | NMS: {r['nms_thresh']:.2f}, Score: {r['score_thresh']:.2f} | "
              f"mAP@0.5: {r['mAP_50']:.4f} | Acc: {r['accuracy']*100:.1f}% | "
              f"Crack: F1={r['crack_f1']:.4f} (TP:{r['crack_tp']}/{r['det_metrics']['Crack']['gt']}, Rec:{r['crack_rec']:.4f}) | "
              f"Moss: F1={r['moss_f1']:.4f} (TP:{r['moss_tp']}/{r['det_metrics']['Moss']['gt']}, Rec:{r['moss_rec']:.4f}) | "
              f"Seepage: F1={r['seepage_f1']:.4f} (TP:{r['seepage_tp']}/{r['det_metrics']['Seepage']['gt']}, Rec:{r['seepage_rec']:.4f})", flush=True)

    best = results[0]
    print(f"\n[WINNING CONFIG] NMS={best['nms_thresh']}, Score={best['score_thresh']} -> mAP@0.5={best['mAP_50']:.4f}", flush=True)
    return best, results

if __name__ == "__main__":
    best, all_res = run_fast_val_tuning("ssd_phase3b_Candidate_A.pth")
