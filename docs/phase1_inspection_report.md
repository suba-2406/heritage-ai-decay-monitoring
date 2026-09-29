# Phase 1: Project & Dataset Inspection Report

**Date:** 2026-09-29  
**Project:** Heritage AI Decay Monitoring and Preservation Recommendation System  
**Primary Reference:** Senmozhi Project Proposal — Work II (Pages 8–13)

---

## 1. Project Source Analysis (Senmozhi Project File)

### 1.1 Project Objective
To develop an automated, deep learning and image processing-based decay monitoring system for ancient Indian monuments and heritage structures, capable of detecting and localizing physical damage, classifying risk levels (Low, Medium, High), and proposing targeted preservation and conservation actions.

### 1.2 Problem Statement
Ancient monuments face severe degradation caused by environmental pollution ($CO_2$, $SO_2$), climate fluctuations (humidity, extreme weather), biological colonization, and inadequate conservation. Existing manual inspection methods are slow, subjective, and unable to detect early-stage decay continuously.

### 1.3 Required Methodology & Pipeline
1. **Image Acquisition**: Collection of high-resolution monument imagery (via field cameras/drones).
2. **Preprocessing**:
   - Resizing to model dimensions.
   - Noise removal using **Gaussian and Median filtering**.
   - Contrast enhancement & binarization/enhancement (e.g. Otsu / histogram equalization).
3. **Deep Learning Model (Mandatory: SSD)**:
   - **Single Shot MultiBox Detector (SSD)** is explicitly mandated as the primary detector due to its balance between computational efficiency and suitability for real-time edge/field monitoring.
   - Task: Identify and localize defect classes within monument imagery.
4. **Hyperparameter Optimization**:
   - Systematic tuning of learning rate, batch size, activation functions, and layer configurations using **random search and manual fine-tuning**.
5. **Model Training**:
   - **Stratified training approach** to preserve class balance across batches.
   - Supervised training using ground-truth annotated bounding boxes.
6. **Evaluation Framework**:
   - Confusion Matrix across all four classes.
   - ROC-AUC analysis.
   - Precision, Recall, and F1-score per class.
   - Comparative evaluation of SSD against baseline algorithms (**SVM, KNN, standard CNN, Random Forest**).
7. **Post-Processing & Risk Categorization**:
   - Measure damage coverage and categorize monument condition into **Low**, **Medium**, or **High** risk.
   - Generate preservation actions mapped to detected defect categories (e.g. Crack -> Severe damage / immediate structural stabilization; Moss -> Moderate damage / biocidal treatment).

### 1.4 Explicit Requirements vs. Implementation Decisions
| Component | Senmozhi Proposal Specification | Implementation Decision / Extension |
|:---|:---|:---|
| **Detection Architecture** | Explicitly SSD (Single Shot MultiBox Detector) | Backbone selection (e.g., MobileNetV2 for speed or VGG16/ResNet50 for capacity) |
| **Classes** | Exactly 4 classes: Normal, Crack, Moss, Seepage | Background class (ID 0) added for SSD anchor matching |
| **Input Resolution** | Resizing required | $300\times300$ (SSD300 standard) or $512\times512$ |
| **Hyperparameter Search** | Random search and manual fine-tuning | Specific learning rate schedule (StepLR/CosineAnnealing) |
| **Risk Scoring Formula** | Conceptual Low/Medium/High categorization | Defect bounding box count & surface area percentage thresholds |

---

## 2. Local Monument Image Dataset Inspection

The local dataset directory was recursively scanned on the filesystem:

### 2.1 Dataset Root & Directory Structure
- **Dataset Path**: `c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE`
- **Total Images**: **325 images**
- **Total Unique Images**: **307 images**
- **Total Storage Size**: 1,383.52 MB (~1.38 GB)
- **Image Format**: 100% JPEG (`.jpg`)

### 2.2 Directory Breakdown (Temple Sites)
The existing directory structure groups images by **geographic temple site / field location**, not by defect condition:

| Folder / Site Name | Image Count | Unique Images | Duplicates |
|:---|:---:|:---:|:---:|
| `big temple` (Thanjavur Brihadisvara Temple) | 217 | 217 | 0 |
| `sakkarathu alwar near mrng star schl_` | 36 | 18 | 18 |
| `kasiviswa nadhar therku vedhi rd , next to micheal tea shop` | 25 | 25 | 0 |
| `market hanuman` | 17 | 17 | 0 |
| `rathnagirishwarar kovil near peter school` | 16 | 16 | 0 |
| `Ramar kovil therku vidhi , next to kasi viswa kovil` | 14 | 14 | 0 |
| **Total** | **325** | **307** | **18** |

### 2.3 Image Dimensions & Aspect Ratios
All 325 images are high-resolution camera captures conforming to two standard orientations:
- **Portrait ($3072 \times 4096$)**: 286 images (88.0%)
- **Landscape ($4096 \times 3072$)**: 39 images (12.0%)
- Aspect Ratio: Fixed $3:4$ / $4:3$ standard digital camera sensor ratio.
- File sizes range from **1.70 MB to 6.94 MB** (average ~4.26 MB).

---

## 3. Four Damage Classes Verification

The project mandates four classes:
1. `Normal`
2. `Crack`
3. `Moss`
4. `Seepage`

### Current Dataset Class Status:
- **Explicit Class Folders**: **None**. (No folders named `Normal`, `Crack`, `Moss`, or `Seepage`).
- **File Name Labels**: **None**. Filenames are camera timestamps (e.g., `IMG_20260609_073544-2.jpg`).
- **Existing Class Counts**:
  - `Normal`: Unlabeled
  - `Crack`: Unlabeled
  - `Moss`: Unlabeled
  - `Seepage`: Unlabeled
- While the photos depict real ancient stone structures with visible weathering, moss patches, fissures, and intact sections, **they are currently unclassified raw site captures**.

---

## 4. Object Detection Annotations Inspection

A full recursive search was performed for annotation files:
- Pascal VOC XML (`.xml`): **0 found**
- COCO JSON (`.json`): **0 found**
- YOLO TXT (`.txt`): **0 found**
- CSV Annotations (`.csv`): **0 found**
- YAML / LabelMe / CVAT files: **0 found**

### Bounding Boxes:
- **DO NOT EXIST**.
- There are no coordinates, bounding boxes, or polygons associated with any image in the local dataset.

> **Crucial Finding:**  
> Bounding-box annotations are missing and annotation is required before SSD training.  
> Under no circumstances should fake bounding boxes or synthetic bounding coordinates be fabricated.

---

## 5. Data Quality Check

| Check Item | Result | Details |
|:---|:---:|:---|
| **Corrupted Files** | 0 | All 325 files were successfully opened, verified, and loaded via PIL raster decoder. |
| **Unsupported Formats** | 0 | 100% valid JPEG files. |
| **Zero-byte / Tiny Images** | 0 | Minimum file size is 1.70 MB; all images are full resolution ($3072 \times 4096$ or $4096 \times 3072$). |
| **Exact Duplicate Files** | 18 | Exactly 18 files in `sakkarathu alwar near mrng star schl_` are identical byte-for-byte duplicates of other files in that folder, prefixed with `Copy of `. |
| **Missing Labels** | 325 | All 325 raw images require bounding box annotations. |
| **Data Integrity** | Protected | Original dataset files remain completely untouched and unmodified. |

---

## 6. Train / Validation / Test Split

- **Existing Split**: **None**. No `train/`, `validation/`, or `test/` subdirectories exist.
- **Protocol**: As per Phase 1 guidelines, splitting has **not** been prematurely performed on raw unannotated data.
- **Future Target Split**:
  - **70% Training** (~215 unique images)
  - **20% Validation** (~61 unique images)
  - **10% Test** (~31 unique images)
  - The split should be stratified across the four classes and across temple sites once bounding-box annotations are created.

---

## 7. Project Setup & Basic Configuration

The following repository architecture and configurations have been initialized in `heritage-ai-decay-monitoring/`:

1. **Repository Structure**:
   - `ai/`: Modules for `dataset`, `annotations`, `preprocessing`, `training`, `evaluation`, `inference`, and `configs`.
   - `backend/app/`: Foundation for FastAPI REST serving.
   - `frontend/src/`: Foundation for monitoring dashboard.
   - `docs/`: Architecture documentation and Phase 1 report.
   - `scripts/`: Reproducible dataset inspection tool (`scripts/inspect_dataset.py`).
2. **Configuration Files**:
   - `.env.example`: Configurable `DATASET_ROOT=../TEMPLE`, model architecture, and hyperparameters.
   - `.gitignore`: Prevents accidental commits of 1.38 GB dataset, checkpoints, cache, and virtual environments.
   - `README.md`: System overview, methodology, class definitions, and setup guide.
   - `ai/configs/classes.yaml`: Mapping of `Normal`, `Crack`, `Moss`, `Seepage` IDs, default risk levels, and preservation actions.
   - `ai/configs/dataset_config.yaml`: Dataset metadata, dimensions, site distributions, and target 70/20/10 split.
   - `ai/configs/model_config.yaml`: SSD300 specifications, hyperparameter optimization targets, and evaluation metrics.

---

## 8. Next Required Step

```text
ANNOTATION REQUIRED BEFORE PHASE 2
```

Ground-truth bounding-box annotations identifying and localizing `Normal`, `Crack`, `Moss`, and `Seepage` must be prepared (e.g., in Pascal VOC XML or COCO JSON format) before SSD model training, hyperparameter optimization, and evaluation can proceed in Phase 2.
