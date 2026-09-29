# Bounding-Box Annotation Workflow & Specifications

This directory houses the ground-truth bounding-box annotations for the **Heritage AI Decay Monitoring and Preservation Recommendation System** (Senmozhi Project, Work II).

> **CRITICAL REQUIREMENT:**  
> **SSD training cannot begin until human-verified bounding-box annotations are available and validated.**  
> Under no circumstances should fake, synthetic, or full-image bounding boxes be generated.

---

## 1. Directory Structure

```text
ai/annotations/
├── raw/                      # Place exported Pascal VOC XML or COCO JSON files here
│   └── .gitkeep
├── converted/                # Preprocessed / converted annotations for model input
│   └── .gitkeep
├── predefined_classes.txt    # Class list loaded automatically by LabelImg
├── validation_report.json    # Auto-generated report from scripts/validate_annotations.py
└── README.md                 # This document
```

---

## 2. Standardized Project Classes & IDs

All annotation tools, configuration files, and downstream SSD data loaders adhere to the following project class IDs:

| Class ID | Class Name | Representation | Risk Category |
|:---:|:---|:---|:---:|
| **0** | **Normal** | Intact masonry surface, no active decay | Low Risk |
| **1** | **Crack** | Structural fracture, fissure, or micro-crack | High Risk |
| **2** | **Moss** | Biological growth, algae, lichen colonization | Medium Risk |
| **3** | **Seepage** | Moisture infiltration, damp stains, efflorescence | High Risk |

*Note on Internal SSD Background:* The SSD object detection network internally manages background anchors during training (e.g. mapping background to an internal index or offset). Background is not treated as a project decay class.

---

## 3. Representation of the `Normal` Class

### Proposed Strategy for Human Review:
1. **Never draw full-image bounding boxes:**  
   Drawing an artificial box spanning $[0, 0, W, H]$ labeled "Normal" damages SSD training by corrupting multi-scale default anchor priors and penalizing spatial feature extractors.
2. **Defect-Free Images as Pure Negative Samples:**  
   When an image depicts an intact heritage surface with zero visible deterioration:
   - In **LabelImg / Pascal VOC**: Save the XML file with an empty `<annotation>` element (zero `<object>` tags), or leave unannotated while tagged as `Normal` in `ai/dataset/annotation_manifest.csv`.
   - In **SSD Training**: Negative images are passed directly to the hard negative mining stage to teach the network what clean monument stone looks like.
3. **Image-Level Risk Output:**  
   During inference, an image is classified as `Normal` (Low Risk) whenever the SSD detector finds 0 defect detections above the confidence threshold.

---

## 4. Annotation Rules by Defect Category

### 4.1 Crack (`id: 1`)
- **What to annotate:** Visible fractures, structural cracks, joint openings, and surface fissures.
- **Bounding Box Rule:** Draw a tight rectangular box enclosing the crack path.
- **Long/Branching Cracks:** If a crack spans an extensive area or branches, draw separate tight bounding boxes over connected segments rather than one massive box covering healthy stone.

### 4.2 Moss (`id: 2`)
- **What to annotate:** Patches of green moss, black/grey lichen, vegetative growth, or algal colonization on stone or mortar.
- **Bounding Box Rule:** Draw bounding boxes around distinct clusters of biological growth.
- **Dispersed Patches:** Group nearby small clusters into a single coherent bounding box if they form a contiguous colony.

### 4.3 Seepage (`id: 3`)
- **What to annotate:** Visible dampness, dark water marks, moisture seepage paths, and white efflorescence (salt blooms).
- **Bounding Box Rule:** Enclose the active moisture discoloration and damp stain boundaries.

### 4.4 Co-occurring Defects (Multi-Label)
- Monuments frequently display multiple deterioration types simultaneously.
- **Rule:** If an image contains a crack running through a patch of moss with water staining, create separate bounding boxes for each respective defect:
  ```text
  Image: IMG_20260609_073544.jpg
   ├── Bounding Box 1: Crack
   ├── Bounding Box 2: Moss
   └── Bounding Box 3: Seepage
  ```

### 4.5 Ambiguity Handling
- If a surface mark is ambiguous (e.g., carving shadow, historic mortar variation, intentional chisel mark), **do not guess**.
- Note the filename and description in `flagged_for_review.csv` for domain expert consensus.

---

## 5. Tool Setup & Step-by-Step Instructions

### Recommended Tool: **LabelImg** (Simplest, Zero-Setup Offline Tool)
LabelImg is an open-source graphical image annotation tool that saves directly in **Pascal VOC XML** format (the native format for SSD).

#### Step 1: Launch LabelImg
In your Python environment:
```bash
pip install labelme labelimg
labelimg
```

#### Step 2: Open Dataset & Predefined Classes
1. In LabelImg, click **Open Dir** $\rightarrow$ select `c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE\big temple` (or any site subfolder).
2. Click **Change Save Dir** $\rightarrow$ select `c:\Users\subaj\OneDrive\Desktop\intern\heritage-ai-decay-monitoring\ai\annotations\raw`.
3. Ensure the format button below Save is set to **PascalVOC** (NOT YOLO).
4. Predefined classes from `ai/annotations/predefined_classes.txt` (`Normal`, `Crack`, `Moss`, `Seepage`) will appear in the label selection dropdown.

#### Step 3: Drawing & Saving Annotations
1. Press `W` to activate the bounding-box drawing tool.
2. Click and drag tightly around the defect (`Crack`, `Moss`, or `Seepage`).
3. Select the appropriate label from the popup list and click **OK**.
4. Press `Ctrl + S` to save the `.xml` file into `ai/annotations/raw/`.
5. Press `D` to navigate to the next image (`A` for previous).
6. For **Normal (healthy)** images, simply save without drawing boxes.

#### Alternative: **CVAT (Computer Vision Annotation Tool)**
- If performing collaborative web-based annotation, create a task in CVAT, import the 307 unique images from `ai/dataset/annotation_manifest.csv`, and export the task as **Pascal VOC 1.1** or **COCO 1.0**.

---

## 6. Validating Annotations

Once annotations are saved in `ai/annotations/raw/`, run the automated validation script:

```bash
python scripts/validate_annotations.py
```

The script verifies:
1. Every unique image has a corresponding annotation file.
2. Coordinates are within image boundaries ($0 \le x_{min} < x_{max} \le W$, $0 \le y_{min} < y_{max} \le H$).
3. No zero-area or negative boxes exist.
4. Only valid project labels are used (`Normal`, `Crack`, `Moss`, `Seepage`).
5. No duplicate bounding boxes exist on the same image.
6. Class distribution statistics are generated and saved to `ai/annotations/validation_report.json`.
