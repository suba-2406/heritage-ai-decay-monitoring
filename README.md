# Heritage AI Decay Monitoring and Preservation Recommendation System

An AI-driven system for automated decay detection, structural damage monitoring, and conservation recommendation for ancient Indian monuments and stone heritage structures.

Based on the research proposal:
> **Work II: AI-Based Decay Monitoring and Preservation Recommendation System for Ancient Monuments** (Senmozhi Project).

---

## 1. Project Overview

Ancient monuments face progressive deterioration driven by environmental weathering, humidity, air pollution ($CO_2$, $SO_2$), biological growth, and structural stress. This project implements an automated computer vision pipeline using **Single Shot MultiBox Detector (SSD)** deep learning architecture to localize and categorize surface defects, assess structural risk, and deliver actionable preservation recommendations.

### Core Objectives
1. **Defect Detection & Localization**: Real-time object detection using SSD across four target classes.
2. **Preprocessing Pipeline**: Noise removal, image enhancement, and denoising (Gaussian, Median, Otsu/CLAHE).
3. **Model Optimization**: Hyperparameter optimization via random search and manual fine-tuning.
4. **Risk Categorization**: Quantify defect coverage and classify monuments into **Low**, **Medium**, or **High** risk levels.
5. **Preservation Recommendations**: Map detected defect types and severity levels to intervention actions.

---

## 2. Standardized Project Classes

The system operates on four standardized project classes:

| Class ID | Class Name | Representation in Annotation | Default Risk Level | Recommended Action |
|:---:|:---|:---|:---:|:---|
| **0** | **Normal** | Undamaged masonry surface (0 defect boxes) | Low | Routine periodic monitoring |
| **1** | **Crack** | Bounding box enclosing fissure or crack | High | Immediate structural inspection & grouting |
| **2** | **Moss** | Bounding box enclosing biological colony | Medium | Biocidal cleaning & dampness remediation |
| **3** | **Seepage** | Bounding box enclosing dampness/efflorescence | High | Waterproofing & drainage inspection |

*Note on Internal SSD Background:* The SSD object detection network internally manages background anchors during training (e.g. mapping background to an internal index or offset). Background is not treated as a project decay class.

---

## 3. Dataset Status (Phase 1 & 2 Findings)

A comprehensive recursive scan and audit of the local monument image dataset was executed:

- **Dataset Location**: `../TEMPLE` (`c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE`)
- **Total Images**: 325 images (100% JPEG, `.jpg`)
- **Canonical Unique Images**: **307 images** cataloged in [`ai/dataset/annotation_manifest.csv`](file:///c:/Users/subaj/OneDrive/Desktop/intern/heritage-ai-decay-monitoring/ai/dataset/annotation_manifest.csv)
- **Identified Duplicates**: **18 exact duplicate pairs** (36 files total, 18 redundant copies prefixed `Copy of ` in `sakkarathu alwar near mrng star schl_`) cataloged in [`ai/dataset/duplicates_manifest.csv`](file:///c:/Users/subaj/OneDrive/Desktop/intern/heritage-ai-decay-monitoring/ai/dataset/duplicates_manifest.csv). These are quarantined from the annotation and training pipeline to prevent cross-split leakage.
- **Corrupted / Unreadable Images**: 0 (all 325 images verified and loadable)
- **High Resolutions**: `3072x4096` (286 images) and `4096x3072` (39 images)
- **Folder Organization**: Grouped geographically by temple site (6 locations), **not** by damage class.
- **Bounding-box Annotations**: **NONE** currently exist.

---

## 4. Annotation Status & Verification (100% Complete)

All 307 canonical unique images have been visually inspected and verified with zero missing annotations:

- **Target Unique Images:** 307
- **Annotated Images:** 307 (100%)
- **Missing Annotations:** 0
- **Total Defect Bounding Boxes:** 4,593
- **Validation Errors / Warnings:** 0 / 0

To re-verify annotation integrity at any time:
```bash
python scripts/validate_annotations.py
```

---

## 5. Dataset Preparation & Stratified Splits

The dataset preparation pipeline pre-resizes images to 320x320, proportionally scales bounding boxes, and generates reproducible, leakage-free splits:

- **Train Split (70%):** 215 images (3,194 defect boxes, 2 Normal images)
- **Validation Split (15.3%):** 47 images (734 defect boxes, 1 Normal image)
- **Test Split (14.7%):** 45 images (665 defect boxes, 1 Normal image)

To re-run dataset preparation:
```bash
python ai/preprocessing/prepare_dataset.py
```

---

## 6. SSD Model Training & Hyperparameter Tuning

- **Architecture:** SSDLite320 with MobileNetV3-Large Backbone (Pretrained on ImageNet)
- **Classes:** 4 (0: Normal/Background, 1: Crack, 2: Moss, 3: Seepage)
- **Hyperparameter Exploration:** Tested AdamW vs. SGD across multiple learning rates.
- **Winning Configuration:** AdamW (LR: `0.001`, Weight Decay: `0.0005`, Batch Size: `4`, CosineAnnealingLR)
- **Best Model Checkpoint:** `ai/training/checkpoints/ssd_monument_decay_best.pth` (Best Val Loss: `5.7030`)
- **Loss Curves:** Saved to `ai/evaluation/training_curves.png`

To train the SSD model:
```bash
python ai/training/train_ssd.py
```

---

## 7. Model Evaluation & Comparative Benchmark

### 7.1 SSD Test Set Evaluation (Unseen Test Set, N=45)
- **mAP@0.5:** 0.0139 (IoU >= 0.50)
- **Confusion Matrix:** Saved to `ai/evaluation/confusion_matrix_ssd.png`
- **ROC Curves:** Saved to `ai/evaluation/roc_curves_ssd.png`
- **Prediction Visualizations:** Saved to `ai/evaluation/visualizations/`

To run SSD evaluation:
```bash
python ai/evaluation/evaluate_ssd.py
```

### 7.2 Comparative Models Benchmark (Senmozhi Work II)
Evaluated across all 5 models on the identical stratified test set (N=45):

| Model Architecture | Accuracy | Macro Prec | Macro Rec | Macro F1 | Weighted F1 | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard CNN (MobileNetV3)** | **75.56%** | **0.5256** | 0.4413 | 0.4469 | 0.7159 | 0.5989 |
| **k-Nearest Neighbors (KNN)** | **75.56%** | 0.4768 | **0.4667** | **0.4679** | **0.7367** | 0.5849 |
| **Support Vector Machine (SVM)** | 68.89% | 0.4247 | 0.4333 | 0.4290 | 0.6801 | **0.7109** |
| **SSD (Single Shot MultiBox Detector)** | 66.67% | 0.1667 | 0.2500 | 0.2000 | 0.5333 | 0.5048 |
| **Random Forest** | 62.22% | 0.3873 | 0.4000 | 0.3919 | 0.6233 | 0.6711 |

To run the comparative benchmark:
```bash
python ai/evaluation/compare_models.py
```

---

## 8. Repository Structure

```text
heritage-ai-decay-monitoring/
│
├── ai/
│   ├── dataset/          # Manifests (annotation_manifest.csv, duplicates_manifest.csv)
│   ├── annotations/      # Annotation workflow (raw/, converted/, predefined_classes.txt)
│   ├── preprocessing/    # Denoising, filtering, contrast enhancement, resizing
│   ├── training/         # SSD training loop, loss functions, optimizer setup
│   ├── evaluation/       # Confusion matrix, ROC-AUC, Precision, Recall, F1
│   ├── inference/        # Real-time inference engine and visualization
│   └── configs/          # YAML configs for classes, dataset, and SSD model
│
├── backend/
│   └── app/              # FastAPI / REST API service for model serving
│
├── frontend/
│   └── src/              # Dashboard UI for decay monitoring & report generation
│
├── docs/                 # Architecture, Phase reports, annotation guidelines
│
├── scripts/              # Dataset inspection and annotation validation scripts
│
├── .env.example          # Environment configuration template
├── .gitignore            # Git ignore rules for datasets, models, and cache
└── README.md             # Project documentation
```
