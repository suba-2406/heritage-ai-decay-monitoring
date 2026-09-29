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

## 4. Annotation Requirement & Current Status

> **CRITICAL REQUIREMENT:**  
> **SSD training cannot begin until human-verified bounding-box annotations are available and validated.**  
> Synthetic bounding boxes, pseudo-labels, and full-image pseudo-annotations are strictly prohibited.

**Current Phase 2 Status:**
```text
ANNOTATION READY — HUMAN ANNOTATION REQUIRED
```

---

## 5. Annotation Setup & Workflow

### Directory Layout
```text
ai/annotations/
├── raw/                      # Place exported Pascal VOC XML or COCO JSON files here
├── converted/                # Converted format for SSD data loaders
├── predefined_classes.txt    # Predefined class list for LabelImg
├── flagged_for_review.csv    # Logging table for ambiguous regions
└── README.md                 # Detailed instructions
```

### Recommended Tool: **LabelImg** (Simplest Offline Tool)
1. Install LabelImg:
   ```bash
   pip install labelimg
   labelimg
   ```
2. Open dataset directory: `../TEMPLE/<site>` (e.g., `../TEMPLE/big temple`).
3. Set save directory: `./ai/annotations/raw/`.
4. Ensure format is set to **PascalVOC**.
5. Use shortcut `W` to draw bounding boxes around `Crack`, `Moss`, or `Seepage`.
6. For **Normal (healthy)** images, save without drawing any boxes (or register in manifest).
7. Save with `Ctrl + S`, navigate with `D` (next) and `A` (previous).

### Annotation Rules:
- **Crack**: Tight rectangular box enclosing visible fissures or fracture paths.
- **Moss**: Bounding box enclosing distinct patches of biological/algal growth.
- **Seepage**: Bounding box enclosing damp patches, tide marks, and efflorescence.
- **Normal**: **Do NOT draw artificial bounding boxes covering the entire image.** Intact images are recorded with 0 defect boxes.
- **Multi-defect Images**: Multiple bounding boxes are drawn for each distinct defect instance.
- **Ambiguous Regions**: Log in `ai/annotations/flagged_for_review.csv` instead of guessing.

---

## 6. Annotation Validation

Once annotations are saved in `ai/annotations/raw/`, run the automated validation script:

```bash
python scripts/validate_annotations.py
```

The script verifies:
- Completeness (flags any missing or orphaned annotations).
- Label correctness (ensures classes match `Normal`, `Crack`, `Moss`, `Seepage`).
- Box geometry (flags zero-area boxes, $x_{min} \ge x_{max}$, $y_{min} \ge y_{max}$, or out-of-bounds coordinates).
- Duplicates (flags redundant boxes on the same image).
- Outputs structured distribution statistics to `ai/annotations/validation_report.json`.

---

## 7. Future Train / Validation / Test Split Strategy

The dataset split will be performed **only after** all annotations have been completed and validated.

- **Target Split**:
  - **70% Training** (~215 unique images)
  - **20% Validation** (~61 unique images)
  - **10% Test** (~31 unique images)
- **Leakage Prevention**:
  - All 18 duplicate files are strictly excluded.
  - Stratification across both defect classes and temple sites ensures robust generalization across different stone masonry textures.

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
