#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Comparative Models Evaluation
Compares 5 models specified in the Senmozhi project proposal:
1. SSD (Single Shot MultiBox Detector - Primary Object Detection Model)
2. Standard CNN (MobileNetV3 Classifier)
3. SVM (Support Vector Machine with RBF kernel)
4. KNN (k-Nearest Neighbors)
5. Random Forest (Ensemble Trees)

All models are trained and tested on the exact same stratified splits (train: 215, val: 47, test: 45).
Metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Multi-Class ROC-AUC.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")

CLASS_MAP = {"Normal": 0, "Crack": 1, "Moss": 2, "Seepage": 3}
CLASS_NAMES = ["Normal", "Crack", "Moss", "Seepage"]

class ClassificationDataset(Dataset):
    def __init__(self, processed_dir, split="train"):
        self.img_dir = os.path.join(processed_dir, "images")
        manifest_path = os.path.join(processed_dir, "dataset_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            all_items = json.load(f)
        self.items = [it for it in all_items if it["split"] == split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_p = os.path.join(self.img_dir, item["processed_image"])
        image = Image.open(img_p).convert("RGB")
        tensor = TF.to_tensor(image)
        # Normalize with ImageNet mean & std
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = CLASS_MAP[item["dominant_class"]]
        return tensor, label, item["filename"]

def extract_features(feature_extractor, dataloader, device):
    """Extracts 960-dimensional visual feature embeddings."""
    feature_extractor.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            feats = feature_extractor(images)
            # Pool to vector: (B, 960, 1, 1) -> (B, 960)
            feats = torch.flatten(feats, 1)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)

def train_and_eval_cnn(train_loader, val_loader, test_loader, device, epochs=10):
    """Trains a standard MobileNetV3 CNN classifier."""
    print("\n--- Training Model: Standard CNN (MobileNetV3) ---")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    cnn = mobilenet_v3_large(weights=weights)
    # Replace classifier head for 4 classes
    in_features = cnn.classifier[0].in_features
    cnn.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 4)
    )
    cnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.0005, weight_decay=0.001)

    best_val_loss = float("inf")
    best_weights = None

    for ep in range(epochs):
        cnn.train()
        train_loss = 0.0
        n_b = 0
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = cnn(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_b += 1

        # Val
        cnn.eval()
        val_loss = 0.0
        n_vb = 0
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = cnn(imgs)
                val_loss += criterion(out, lbls).item()
                n_vb += 1

        val_loss /= max(1, n_vb)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = cnn.state_dict().copy()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch [{ep+1:02d}/{epochs:02d}] - Train Loss: {train_loss/n_b:.4f} | Val Loss: {val_loss:.4f}")

    cnn.load_state_dict(best_weights)
    cnn.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, lbls, _ in test_loader:
            imgs = imgs.to(device)
            out = cnn(imgs)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(lbls.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def compute_metrics(y_true, y_pred, y_prob, model_name):
    """Computes standardized evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Multi-class ROC-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
        if y_prob.shape[1] == 4:
            # Check which classes are present in y_true
            present_classes = np.unique(y_true)
            if len(present_classes) > 1:
                auc_score = roc_auc_score(y_true_bin[:, present_classes], y_prob[:, present_classes], average="macro", multi_class="ovr")
            else:
                auc_score = 0.50
        else:
            auc_score = 0.50
    except Exception as e:
        auc_score = 0.50

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro_Precision": round(prec_macro, 4),
        "Macro_Recall": round(rec_macro, 4),
        "Macro_F1": round(f1_macro, 4),
        "Weighted_F1": round(f1_weighted, 4),
        "ROC_AUC": round(auc_score, 4)
    }

def main():
    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — COMPARATIVE MODELS BENCHMARK")
    print("=" * 70)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ClassificationDataset(PROCESSED_DIR, split="train")
    val_ds = ClassificationDataset(PROCESSED_DIR, split="val")
    test_ds = ClassificationDataset(PROCESSED_DIR, split="test")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # 1. Feature Extraction for Tabular Models
    print("\n--- Extracting Deep Feature Embeddings for Classical ML Models ---")
    backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
    pool = nn.AdaptiveAvgPool2d((1, 1))
    feat_extractor = nn.Sequential(backbone, pool).to(device)

    X_train, y_train = extract_features(feat_extractor, DataLoader(train_ds, batch_size=16, shuffle=False), device)
    X_val, y_val = extract_features(feat_extractor, DataLoader(val_ds, batch_size=16, shuffle=False), device)
    X_test, y_test = extract_features(feat_extractor, DataLoader(test_ds, batch_size=16, shuffle=False), device)

    # Merge train + val for training classical models
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])

    print(f"Feature dimensions: {X_train.shape[1]} features")
    print(f"Training samples:   {len(y_full_train)} samples")
    print(f"Testing samples:    {len(y_test)} samples")

    results = []

    # Model 1: Support Vector Machine (SVM)
    print("\n--- Training Model: Support Vector Machine (SVM) ---")
    svm = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)
    svm.fit(X_full_train, y_full_train)
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)
    # Ensure 4 classes represented in proba
    if y_prob_svm.shape[1] < 4:
        full_prob = np.zeros((len(y_test), 4))
        for idx, cls in enumerate(svm.classes_):
            full_prob[:, cls] = y_prob_svm[:, idx]
        y_prob_svm = full_prob
    res_svm = compute_metrics(y_test, y_pred_svm, y_prob_svm, "Support Vector Machine (SVM)")
    results.append(res_svm)

    # Model 2: k-Nearest Neighbors (KNN)
    print("\n--- Training Model: k-Nearest Neighbors (KNN) ---")
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_full_train, y_full_train)
    y_pred_knn = knn.predict(X_test)
    y_prob_knn = knn.predict_proba(X_test)
    if y_prob_knn.shape[1] < 4:
        full_prob = np.zeros((len(y_test), 4))
        for idx, cls in enumerate(knn.classes_):
            full_prob[:, cls] = y_prob_knn[:, idx]
        y_prob_knn = full_prob
    res_knn = compute_metrics(y_test, y_pred_knn, y_prob_knn, "k-Nearest Neighbors (KNN)")
    results.append(res_knn)

    # Model 3: Random Forest (RF)
    print("\n--- Training Model: Random Forest (RF) ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_full_train, y_full_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)
    if y_prob_rf.shape[1] < 4:
        full_prob = np.zeros((len(y_test), 4))
        for idx, cls in enumerate(rf.classes_):
            full_prob[:, cls] = y_prob_rf[:, idx]
        y_prob_rf = full_prob
    res_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf, "Random Forest")
    results.append(res_rf)

    # Model 4: Standard CNN (MobileNetV3)
    y_test_cnn, y_pred_cnn, y_prob_cnn = train_and_eval_cnn(train_loader, val_loader, test_loader, device, epochs=10)
    res_cnn = compute_metrics(y_test_cnn, y_pred_cnn, y_prob_cnn, "Standard CNN (MobileNetV3)")
    results.append(res_cnn)

    # Model 5: SSD (Primary Object Detection Model)
    # Load SSD test evaluation results
    ssd_metrics_path = os.path.join(CURRENT_DIR, "ssd_metrics.json")
    if os.path.exists(ssd_metrics_path):
        with open(ssd_metrics_path, "r", encoding="utf-8") as f:
            ssd_data = json.load(f)
        ssd_img_metrics = ssd_data["image_level_classification_metrics"]
        res_ssd = {
            "Model": "SSD (Single Shot MultiBox Detector)",
            "Accuracy": ssd_img_metrics["accuracy"],
            "Macro_Precision": ssd_img_metrics["macro_precision"],
            "Macro_Recall": ssd_img_metrics["macro_recall"],
            "Macro_F1": ssd_img_metrics["macro_f1"],
            "Weighted_F1": ssd_img_metrics["weighted_f1"],
            "ROC_AUC": round(float(np.mean([v for v in ssd_img_metrics["roc_auc_per_class"].values() if isinstance(v, (int, float))])), 4)
        }
        results.append(res_ssd)
    else:
        print("[WARNING] SSD metrics JSON not found yet. Run evaluate_ssd.py before running compare_models.py.")

    # Print Table
    print("\n" + "=" * 80)
    print(" COMPARATIVE EVALUATION RESULTS (TEST SET, N=45)")
    print("=" * 80)
    headers = ["Model", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1", "ROC_AUC"]
    print(f"{'Model':<35} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'W-F1':<8} {'ROC-AUC':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['Model']:<35} {r['Accuracy']:<8.4f} {r['Macro_Precision']:<8.4f} {r['Macro_Recall']:<8.4f} {r['Macro_F1']:<8.4f} {r['Weighted_F1']:<8.4f} {r['ROC_AUC']:<8.4f}")
    print("=" * 85)

    # Save to CSV
    csv_out = os.path.join(CURRENT_DIR, "model_comparison.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved comparison CSV to: {csv_out}")

    # Save to JSON
    json_out = os.path.join(CURRENT_DIR, "model_comparison.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Senmozhi Work II Comparative Evaluation",
            "test_samples": len(y_test),
            "classes": CLASS_NAMES,
            "models_evaluated": results
        }, f, indent=2)
    print(f"Saved comparison JSON to: {json_out}")

    # Save Comparison Bar Chart
    models = [r["Model"].replace(" (Single Shot MultiBox Detector)", "").replace(" (MobileNetV3)", "") for r in results]
    accs = [r["Accuracy"] for r in results]
    f1s = [r["Macro_F1"] for r in results]
    wf1s = [r["Weighted_F1"] for r in results]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, accs, width, label="Accuracy", color="#3b82f6")
    plt.bar(x, f1s, width, label="Macro F1", color="#10b981")
    plt.bar(x + width, wf1s, width, label="Weighted F1", color="#f59e0b")

    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Comparative Performance Across Proposed Architectures (Senmozhi Work II)", fontsize=12, fontweight="bold")
    plt.xticks(x, models, rotation=15)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()

    chart_p = os.path.join(CURRENT_DIR, "model_comparison_chart.png")
    plt.savefig(chart_p, dpi=200)
    plt.close()
    print(f"Saved comparative chart to: {chart_p}")
    print("\n[SUCCESS] Comparative benchmark completed.")

if __name__ == "__main__":
    main()
