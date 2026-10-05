#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - SSD Training & Hyperparameter Tuning
Model: SSDLite320 with MobileNetV3 Backbone (Pretrained on ImageNet)
Classes: 0 = Normal (Background/Negative), 1 = Crack, 2 = Moss, 3 = Seepage
Logs training/validation metrics, saves best checkpoint, and plots loss curves.
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
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "checkpoints")
LOGS_DIR = os.path.join(CURRENT_DIR, "logs")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

for d in [CHECKPOINTS_DIR, LOGS_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, CURRENT_DIR)
from dataset_loader import HeritageMonumentDataset, collate_fn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TrainAugmentation:
    """Random horizontal flip and subtle color jitter for object detection."""
    def __init__(self, p_flip=0.5):
        self.p_flip = p_flip

    def __call__(self, img, target):
        if random.random() < self.p_flip:
            img = TF.hflip(img)
            boxes = target["boxes"]
            if len(boxes) > 0:
                # Width is 320 (img width)
                # xmin, ymin, xmax, ymax -> (width - xmax), ymin, (width - xmin), ymax
                new_xmin = 320.0 - boxes[:, 2]
                new_xmax = 320.0 - boxes[:, 0]
                boxes[:, 0] = new_xmin
                boxes[:, 2] = new_xmax
                target["boxes"] = boxes
        return img, target

def build_ssd_model(num_classes=4):
    """Builds SSDLite320 with pretrained MobileNetV3 backbone for 4 classes."""
    model = ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT
    )
    return model

def evaluate_loss(model, dataloader, device):
    """Computes validation loss over validation dataloader."""
    model.train()  # In torchvision detection models, model.train() is required to output losses
    total_val_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
            total_batches += 1
    return total_val_loss / max(1, total_batches)

def run_hyperparameter_tuning(train_dataset, val_dataset, device):
    """Controlled hyperparameter search across learning rates and optimizers."""
    print("\n" + "=" * 70)
    print(" STEP 1: HYPERPARAMETER TUNING & EXPLORATION")
    print("=" * 70)

    candidates = [
        {"name": "AdamW_lr1e-3_b4", "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0005, "batch_size": 4},
        {"name": "AdamW_lr5e-4_b4", "optimizer": "AdamW", "lr": 0.0005, "weight_decay": 0.0001, "batch_size": 4},
        {"name": "SGD_lr5e-3_b4", "optimizer": "SGD", "lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005, "batch_size": 4},
    ]

    best_cand = None
    best_pilot_loss = float("inf")
    tuning_results = []

    for cand in candidates:
        print(f"\n--- Testing Candidate: {cand['name']} ---")
        train_loader = DataLoader(
            train_dataset,
            batch_size=cand["batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cand["batch_size"],
            shuffle=False,
            collate_fn=collate_fn
        )

        model = build_ssd_model(num_classes=4).to(device)
        params = [p for p in model.parameters() if p.requires_grad]

        if cand["optimizer"] == "AdamW":
            opt = torch.optim.AdamW(params, lr=cand["lr"], weight_decay=cand["weight_decay"])
        else:
            opt = torch.optim.SGD(params, lr=cand["lr"], momentum=cand.get("momentum", 0.9), weight_decay=cand["weight_decay"])

        # 2-epoch pilot test
        pilot_loss = 0.0
        for ep in range(2):
            model.train()
            ep_loss = 0.0
            n_b = 0
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                n_b += 1
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"  Epoch {ep+1}/2 - Train Loss: {ep_loss/n_b:.4f}, Val Loss: {val_loss:.4f}")
            pilot_loss = val_loss

        cand_result = dict(cand)
        cand_result["pilot_val_loss"] = round(pilot_loss, 4)
        tuning_results.append(cand_result)

        if pilot_loss < best_pilot_loss:
            best_pilot_loss = pilot_loss
            best_cand = cand

    print(f"\nHyperparameter Search Complete. Best Candidate: {best_cand['name']} (Val Loss: {best_pilot_loss:.4f})")
    tuning_log_path = os.path.join(LOGS_DIR, "hyperparameter_tuning.json")
    with open(tuning_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "candidates": tuning_results,
            "selected_best": best_cand,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    return best_cand

def train_final_ssd(best_config, train_dataset, val_dataset, device, epochs=20):
    print("\n" + "=" * 70)
    print(" STEP 2: FULL SSD MODEL TRAINING")
    print("=" * 70)
    print(f"Selected Configuration: {best_config['name']}")
    print(f"Optimizer:              {best_config['optimizer']}")
    print(f"Learning Rate:          {best_config['lr']}")
    print(f"Batch Size:             {best_config['batch_size']}")
    print(f"Total Epochs:           {epochs}")
    print(f"Device:                 {device}")
    print("-" * 70)

    train_loader = DataLoader(
        train_dataset,
        batch_size=best_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_ssd_model(num_classes=4).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    if best_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=best_config["lr"], weight_decay=best_config["weight_decay"])
    else:
        optimizer = torch.optim.SGD(params, lr=best_config["lr"], momentum=best_config.get("momentum", 0.9), weight_decay=best_config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(CHECKPOINTS_DIR, "ssd_monument_decay_best.pth")

    csv_log_path = os.path.join(LOGS_DIR, "training_history.csv")
    csv_file = open(csv_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_cls_loss", "train_bbox_loss", "val_loss", "lr", "epoch_time_sec"])

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_total_loss = 0.0
        train_cls_loss = 0.0
        train_bbox_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            cls_l = loss_dict.get("classification", torch.tensor(0.0))
            box_l = loss_dict.get("bbox_regression", torch.tensor(0.0))
            losses = cls_l + box_l

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            train_total_loss += losses.item()
            train_cls_loss += cls_l.item()
            train_bbox_loss += box_l.item()
            n_batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        val_loss = evaluate_loss(model, val_loader, device)

        avg_train_loss = train_total_loss / max(1, n_batches)
        avg_cls_loss = train_cls_loss / max(1, n_batches)
        avg_box_loss = train_bbox_loss / max(1, n_batches)
        ep_time = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": best_config,
                "num_classes": 4,
                "class_names": ["Normal", "Crack", "Moss", "Seepage"]
            }, best_checkpoint_path)

        csv_writer.writerow([epoch, round(avg_train_loss, 4), round(avg_cls_loss, 4), round(avg_box_loss, 4), round(val_loss, 4), round(current_lr, 6), round(ep_time, 2)])
        csv_file.flush()

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_cls_loss": avg_cls_loss,
            "train_bbox_loss": avg_box_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "time_sec": ep_time
        })

        best_flag = " [*BEST SAVED*]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} (Cls: {avg_cls_loss:.4f}, BBox: {avg_box_loss:.4f}) | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {ep_time:.1f}s{best_flag}")

    csv_file.close()
    total_elapsed = time.time() - start_train_time
    print(f"\nTraining Complete in {total_elapsed/60:.2f} minutes.")
    print(f"Best Checkpoint: {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    # Save training configuration
    cfg_save_path = os.path.join(CHECKPOINTS_DIR, "training_config.json")
    with open(cfg_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_architecture": "SSDLite320_MobileNet_V3_Large",
            "pretrained_backbone": "MobileNet_V3_Large_Weights.DEFAULT",
            "num_classes": 4,
            "class_mapping": {"Normal": 0, "Crack": 1, "Moss": 2, "Seepage": 3},
            "input_resolution": [320, 320, 3],
            "best_hyperparameters": best_config,
            "total_epochs": epochs,
            "best_val_loss": round(best_val_loss, 4),
            "final_train_loss": round(history[-1]["train_loss"], 4),
            "checkpoint_path": best_checkpoint_path,
            "total_training_time_sec": round(total_elapsed, 2)
        }, f, indent=2)

    # Plot Training & Validation Curves
    plot_curves(history)

    return best_checkpoint_path

def plot_curves(history):
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    cls_losses = [h["train_cls_loss"] for h in history]
    box_losses = [h["train_bbox_loss"] for h in history]

    plt.figure(figsize=(12, 5))

    # Total loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-o", label="Train Total Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r--s", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("SSD Monument Decay - Total Loss Curves", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Loss components
    plt.subplot(1, 2, 2)
    plt.plot(epochs, cls_losses, "g-^", label="Train Classification Loss", linewidth=2)
    plt.plot(epochs, box_losses, "m-d", label="Train BBox Regression Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss Component", fontsize=11)
    plt.title("SSD Loss Breakdown (Cls vs BBox)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(EVAL_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curves plot to: {curve_path}")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="train", transforms=TrainAugmentation(p_flip=0.5))
    val_dataset = HeritageMonumentDataset(PROCESSED_DIR, split="val", transforms=None)

    print(f"Train Dataset: {len(train_dataset)} images")
    print(f"Val Dataset:   {len(val_dataset)} images")

    # Step 1: Hyperparameter Search
    best_config = run_hyperparameter_tuning(train_dataset, val_dataset, device)

    # Step 2: Full SSD Training
    train_final_ssd(best_config, train_dataset, val_dataset, device, epochs=15)

if __name__ == "__main__":
    main()
