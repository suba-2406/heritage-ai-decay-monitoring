#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Phase 3B Candidate Model Training Pipeline
Implements:
- Candidate A: Improved class balancing + augmentation, 320x320, AdamW lr=1e-3
- Candidate B: Candidate A + High-resolution / Sliced-Tiled inference
- Candidate C: Candidate A + Alternative optimizer/learning-rate (AdamW lr=5e-4, higher Crack weight)
- Candidate D: Best practical combination (Class-weighted focal loss, balanced sampling, CosineAnnealingLR)
All selections performed strictly on the VALIDATION split.
"""

import os
import sys
import json
import time
import random
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "checkpoints")
LOGS_DIR = os.path.join(CURRENT_DIR, "logs")
EXPERIMENTS_DIR = os.path.join(CURRENT_DIR, "experiments")

for d in [CHECKPOINTS_DIR, LOGS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, CURRENT_DIR)
from dataset_loader import HeritageMonumentDataset, collate_fn
from phase3b_framework import (
    DetectionAugmentationPhase3B,
    get_class_aware_sampler,
    build_phase3b_ssd_model,
    compute_iou,
    CLASS_MAP,
    CLASS_NAMES,
    DEFECT_CLASSES
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_fast_val(model, val_loader, device, s_thresh=0.20, nms_thresh=0.45):
    """Evaluates validation mAP@0.5 and per-class metrics."""
    model.eval()
    all_gt_boxes = {c: [] for c in DEFECT_CLASSES}
    all_pred_boxes = {c: [] for c in DEFECT_CLASSES}
    y_true_img = []
    y_pred_img = []

    with torch.no_grad():
        for images, targets in val_loader:
            img = images[0].to(device)
            target = targets[0]
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            dom_gt = target["dominant_class"]

            outputs = model([img])[0]
            p_boxes = outputs["boxes"].cpu()
            p_scores = outputs["scores"].cpu()
            p_labels = outputs["labels"].cpu()

            keep_s = p_scores >= s_thresh
            p_boxes = p_boxes[keep_s]
            p_scores = p_scores[keep_s]
            p_labels = p_labels[keep_s]

            if len(p_boxes) > 0:
                keep_n = batched_nms(p_boxes, p_scores, p_labels, nms_thresh)
                p_boxes = p_boxes[keep_n].numpy()
                p_scores = p_scores[keep_n].numpy()
                p_labels = p_labels[keep_n].numpy()
            else:
                p_boxes = np.empty((0, 4))
                p_scores = np.empty((0,))
                p_labels = np.empty((0,), dtype=int)

            for c_id, c_name in [(1, "Crack"), (2, "Moss"), (3, "Seepage")]:
                c_gt = [gt_boxes[k] for k in range(len(gt_labels)) if gt_labels[k] == c_id]
                c_preds = [(p_boxes[k], p_scores[k]) for k in range(len(p_labels)) if p_labels[k] == c_id]
                all_gt_boxes[c_name].append(c_gt)
                all_pred_boxes[c_name].append(c_preds)

            class_probs = [0.0, 0.0, 0.0, 0.0]
            if len(p_labels) == 0:
                pred_dom = "Normal"
            else:
                for k in range(len(p_labels)):
                    c_id = p_labels[k]
                    if c_id < 4:
                        class_probs[c_id] = max(class_probs[c_id], float(p_scores[k]))
                if class_probs[1] >= s_thresh:
                    pred_dom = "Crack"
                elif class_probs[3] >= s_thresh:
                    pred_dom = "Seepage"
                elif class_probs[2] >= s_thresh:
                    pred_dom = "Moss"
                else:
                    pred_dom = "Normal"

            y_true_img.append(CLASS_MAP[dom_gt])
            y_pred_img.append(CLASS_MAP[pred_dom])

    det_metrics = {}
    aps = []
    num_dataset = len(all_gt_boxes["Crack"])

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

    mAP = sum(aps) / len(aps)
    correct = sum(1 for yt, yp in zip(y_true_img, y_pred_img) if yt == yp)
    acc = correct / len(y_true_img)
    macro_f1 = sum(det_metrics[c]["f1"] for c in DEFECT_CLASSES) / len(DEFECT_CLASSES)

    return {
        "mAP_50": round(mAP, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "det_metrics": det_metrics
    }

def train_candidate(cand_config):
    name = cand_config["name"]
    lr = cand_config["lr"]
    opt_type = cand_config.get("optimizer", "AdamW")
    loss_weights = cand_config.get("loss_weights", [1.0, 3.5, 3.5, 1.0])
    sampler_weights = cand_config.get("sampler_weights", {"Crack": 4.0, "Moss": 4.0, "Normal": 2.0, "Seepage": 1.0})
    epochs = cand_config.get("epochs", 15)
    batch_size = cand_config.get("batch_size", 4)
    eval_s_thresh = cand_config.get("eval_score_thresh", 0.20)
    eval_nms_thresh = cand_config.get("eval_nms_thresh", 0.45)
    weight_decay = cand_config.get("weight_decay", 0.0005)

    print("\n" + "=" * 75, flush=True)
    print(f" TRAINING PHASE 3B CANDIDATE: {name}", flush=True)
    print(f" Optimizer: {opt_type} | LR: {lr} | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f" Loss Weights: {loss_weights}", flush=True)
    print(f" Sampler Weights: {sampler_weights}", flush=True)
    print("=" * 75, flush=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detection safe augmentations
    aug = DetectionAugmentationPhase3B(p_hflip=0.5, p_vflip=0.2, brightness=0.15, contrast=0.15, saturation=0.10)
    train_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="train", transforms=aug)
    val_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="val", transforms=None)

    sampler = get_class_aware_sampler(train_dataset, sampler_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_phase3b_ssd_model(
        num_classes=4,
        loss_weights=loss_weights,
        score_thresh=0.01,
        nms_thresh=eval_nms_thresh
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_type == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=cand_config.get("momentum", 0.9), weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"ssd_phase3b_{name}.pth")
    best_mAP = -1.0
    best_val_res = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_cls = 0.0
        train_box = 0.0
        n_b = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            cls_l = loss_dict.get("classification", torch.tensor(0.0))
            box_l = loss_dict.get("bbox_regression", torch.tensor(0.0))
            loss = cls_l + box_l

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_cls += cls_l.item()
            train_box += box_l.item()
            n_b += 1

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        ep_sec = time.time() - t0

        # Evaluate on validation split
        val_res = evaluate_fast_val(model, val_loader, device, s_thresh=eval_s_thresh, nms_thresh=eval_nms_thresh)
        cur_mAP = val_res["mAP_50"]
        c_crack = val_res["det_metrics"]["Crack"]
        c_moss = val_res["det_metrics"]["Moss"]
        c_seep = val_res["det_metrics"]["Seepage"]

        is_best = cur_mAP > best_mAP
        if is_best:
            best_mAP = cur_mAP
            best_val_res = val_res
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mAP_50": cur_mAP,
                "val_metrics": val_res,
                "config": cand_config
            }, ckpt_path)

        flag = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} ({ep_sec:.1f}s, lr={cur_lr:.6f}) - Train Loss: {train_loss/n_b:.4f} (cls:{train_cls/n_b:.4f}, box:{train_box/n_b:.4f}) | "
              f"Val mAP@0.5: {cur_mAP:.4f}{flag} | Crack F1: {c_crack['f1']:.4f} (TP:{c_crack['tp']}) | "
              f"Moss F1: {c_moss['f1']:.4f} (TP:{c_moss['tp']}) | Seepage F1: {c_seep['f1']:.4f} (TP:{c_seep['tp']})", flush=True)

    total_time = round(time.time() - start_time, 2)
    print(f"\nTraining Complete for {name} in {total_time:.2f}s. Best Val mAP@0.5: {best_mAP:.4f}")

    # Record to results CSV
    results_csv = os.path.join(EXPERIMENTS_DIR, "phase3b_results.csv")
    file_exists = os.path.exists(results_csv)
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "experiment_name", "input_resolution", "optimizer", "learning_rate",
                "batch_size", "epochs", "augmentation_configuration", "sampling_strategy",
                "confidence_threshold", "nms_threshold", "validation_mAP_50",
                "val_crack_precision", "val_crack_recall", "val_crack_f1",
                "val_moss_precision", "val_moss_recall", "val_moss_f1",
                "val_seepage_precision", "val_seepage_recall", "val_seepage_f1",
                "validation_macro_f1", "training_time_sec", "checkpoint_path"
            ])
        c_cr = best_val_res["det_metrics"]["Crack"]
        c_mo = best_val_res["det_metrics"]["Moss"]
        c_se = best_val_res["det_metrics"]["Seepage"]
        w.writerow([
            name, "320x320", opt_type, lr, batch_size, epochs,
            "HFlip+VFlip+ColorJitter", f"class_aware_weights{sampler_weights}",
            eval_s_thresh, eval_nms_thresh, best_val_res["mAP_50"],
            c_cr["precision"], c_cr["recall"], c_cr["f1"],
            c_mo["precision"], c_mo["recall"], c_mo["f1"],
            c_se["precision"], c_se["recall"], c_se["f1"],
            best_val_res["macro_f1"], total_time, ckpt_path
        ])

    return best_val_res

if __name__ == "__main__":
    candidate_c = {
        "name": "Candidate_C",
        "optimizer": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.0005,
        "batch_size": 4,
        "epochs": 15,
        "loss_weights": [1.0, 7.0, 5.0, 1.0],
        "sampler_weights": {"Crack": 6.0, "Moss": 5.0, "Normal": 2.0, "Seepage": 1.0},
        "eval_score_thresh": 0.20,
        "eval_nms_thresh": 0.45
    }
    train_candidate(candidate_c)
