#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Dataset Preparation & Preprocessing Pipeline
Reads verified Pascal VOC XML annotations from ai/annotations/raw/ and generates:
- Clean 70/15/15 train/val/test stratified splits (preventing data leakage)
- 320x320 cached processed images with proportionally scaled bounding boxes
- Normal images handled with ZERO bounding boxes (negative/background samples)
- Comprehensive dataset statistics, class distribution, split statistics
- QC sample annotation preview visualizations
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_ROOT = os.environ.get("DATASET_ROOT", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
RAW_ANN_DIR = os.path.join(BASE_DIR, "annotations", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset", "annotation_manifest.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
PROCESSED_IMG_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_ANN_DIR = os.path.join(PROCESSED_DIR, "annotations")
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
SAMPLES_DIR = os.path.join(PROCESSED_DIR, "sample_visualizations")
QC_DIR = os.path.join(PROJECT_ROOT, "docs", "qc_previews")

TARGET_SIZE = (320, 320)  # Standard SSDLite input resolution

CLASS_MAP = {
    "Normal": 0,
    "Crack": 1,
    "Moss": 2,
    "Seepage": 3
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
CLASS_COLORS = {
    "Normal": (34, 197, 94),     # Green
    "Crack": (239, 68, 68),      # Red
    "Moss": (16, 185, 129),      # Emerald
    "Seepage": (2, 132, 199)     # Blue
}

def parse_voc_xml(xml_path):
    """Parses Pascal VOC XML into image metadata and bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fn = root.find("filename").text if root.find("filename") is not None else os.path.splitext(os.path.basename(xml_path))[0] + ".jpg"
    folder = root.find("folder").text if root.find("folder") is not None else ""
    path_elem = root.find("path")
    full_path = path_elem.text if path_elem is not None else None

    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 3072
    height = int(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 4096

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue
        if name == "Normal":
            # Normal should never have boxes
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Coordinate clamping
        xmin = max(0.0, min(float(width), xmin))
        ymin = max(0.0, min(float(height), ymin))
        xmax = max(0.0, min(float(width), xmax))
        ymax = max(0.0, min(float(height), ymax))

        if xmax > xmin and ymax > ymin:
            boxes.append({
                "name": name,
                "class_id": CLASS_MAP[name],
                "bbox": [xmin, ymin, xmax, ymax]
            })

    return {
        "filename": fn,
        "folder": folder,
        "orig_path": full_path,
        "orig_width": width,
        "orig_height": height,
        "boxes": boxes
    }

def determine_dominant_class(boxes):
    """Determines image-level dominant category for stratification."""
    if not boxes:
        return "Normal"
    # Order of priority if multiple defects: Crack > Seepage > Moss
    classes_present = {b["name"] for b in boxes}
    if "Crack" in classes_present:
        return "Crack"
    if "Seepage" in classes_present:
        return "Seepage"
    if "Moss" in classes_present:
        return "Moss"
    return "Normal"

def prepare_dataset(random_seed=42):
    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — DATASET PREPARATION PIPELINE")
    print("=" * 70)
    np.random.seed(random_seed)

    for d in [PROCESSED_DIR, PROCESSED_IMG_DIR, PROCESSED_ANN_DIR, SPLITS_DIR, SAMPLES_DIR, QC_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Load canonical manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    canonical_images = {item["filename"]: item for item in manifest_data["images"]}
    print(f"Loaded {len(canonical_images)} canonical unique images from manifest.")

    # 2. Parse all 307 raw annotations
    xml_files = [f for f in os.listdir(RAW_ANN_DIR) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} raw XML annotations.")
    assert len(xml_files) == 307, f"Expected 307 annotations, found {len(xml_files)}"

    dataset_records = []
    class_box_counts = defaultdict(int)
    image_class_counts = defaultdict(int)

    for xml_file in sorted(xml_files):
        xml_p = os.path.join(RAW_ANN_DIR, xml_file)
        parsed = parse_voc_xml(xml_p)
        fn = parsed["filename"]

        if fn not in canonical_images:
            # Match by base name if extension discrepancy
            base_fn = os.path.splitext(xml_file)[0]
            for cand_fn in canonical_images:
                if os.path.splitext(cand_fn)[0] == base_fn:
                    fn = cand_fn
                    break

        cat_item = canonical_images[fn]
        raw_full_path = os.path.join(DATASET_ROOT, cat_item["rel_path"])
        parsed["image_source_path"] = raw_full_path
        parsed["site"] = cat_item["site"]
        parsed["rel_path"] = cat_item["rel_path"]

        dom_class = determine_dominant_class(parsed["boxes"])
        parsed["dominant_class"] = dom_class
        image_class_counts[dom_class] += 1

        for b in parsed["boxes"]:
            class_box_counts[b["name"]] += 1

        dataset_records.append(parsed)

    print(f"Parsed {len(dataset_records)} records successfully.")
    print("Total Defect Bounding Boxes:", sum(class_box_counts.values()))
    for c, cnt in sorted(class_box_counts.items()):
        print(f"  - {c}: {cnt} boxes")
    print("\nImage-Level Dominant Class Counts:")
    for c, cnt in sorted(image_class_counts.items()):
        print(f"  - {c}: {cnt} images")

    # 3. Stratified Split (70% Train, 15% Val, 15% Test)
    # Target counts: ~215 train, ~46 val, ~46 test
    train_records, val_records, test_records = [], [], []

    # Group records by dominant class
    by_class = defaultdict(list)
    for rec in dataset_records:
        by_class[rec["dominant_class"]].append(rec)

    for cls, recs in by_class.items():
        np.random.shuffle(recs)
        n = len(recs)
        if n == 4:
            # Special case for Normal: 2 train, 1 val, 1 test
            n_train, n_val, n_test = 2, 1, 1
        else:
            n_train = int(round(n * 0.70))
            n_val = int(round(n * 0.15))
            n_test = n - n_train - n_val

        train_records.extend(recs[:n_train])
        val_records.extend(recs[n_train:n_train + n_val])
        test_records.extend(recs[n_train + n_val:])

    # Ensure no leakage
    train_fns = {r["filename"] for r in train_records}
    val_fns = {r["filename"] for r in val_records}
    test_fns = {r["filename"] for r in test_records}
    assert len(train_fns.intersection(val_fns)) == 0, "Data leakage between train and val!"
    assert len(train_fns.intersection(test_fns)) == 0, "Data leakage between train and test!"
    assert len(val_fns.intersection(test_fns)) == 0, "Data leakage between val and test!"
    assert len(train_records) + len(val_records) + len(test_records) == 307, "Total split counts do not equal 307!"

    print("\nStratified Splits Formed:")
    print(f"  - Train: {len(train_records)} images ({len(train_records)/307*100:.1f}%)")
    print(f"  - Val:   {len(val_records)} images ({len(val_records)/307*100:.1f}%)")
    print(f"  - Test:  {len(test_records)} images ({len(test_records)/307*100:.1f}%)")

    # 4. Save Split Text Files
    with open(os.path.join(SPLITS_DIR, "train.txt"), "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(f"{r['filename']}\n")
    with open(os.path.join(SPLITS_DIR, "val.txt"), "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(f"{r['filename']}\n")
    with open(os.path.join(SPLITS_DIR, "test.txt"), "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(f"{r['filename']}\n")

    # 5. Process & Resize Images and Scale Bounding Boxes
    print(f"\nProcessing images to {TARGET_SIZE} and scaling bounding box coordinates...")
    processed_manifest = []

    for idx, rec in enumerate(dataset_records):
        src_path = rec["image_source_path"]
        assert os.path.exists(src_path), f"Source image does not exist: {src_path}"

        # Load image with PIL
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            scale_x = TARGET_SIZE[0] / orig_w
            scale_y = TARGET_SIZE[1] / orig_h

            # Resize with Lanczos filtering
            resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            out_img_name = f"{os.path.splitext(rec['filename'])[0]}.jpg"
            out_img_path = os.path.join(PROCESSED_IMG_DIR, out_img_name)
            resized_img.save(out_img_path, quality=95)

        # Scale bounding boxes
        scaled_boxes = []
        for b in rec["boxes"]:
            oxmin, oymin, oxmax, oymax = b["bbox"]
            sxmin = round(oxmin * scale_x, 2)
            symin = round(oymin * scale_y, 2)
            sxmax = round(oxmax * scale_x, 2)
            symax = round(oymax * scale_y, 2)
            # Ensure valid area
            if sxmax > sxmin and symax > symin:
                scaled_boxes.append({
                    "name": b["name"],
                    "class_id": b["class_id"],
                    "bbox": [sxmin, symin, sxmax, symax],
                    "orig_bbox": [oxmin, oymin, oxmax, oymax]
                })

        split_assignment = "train" if rec["filename"] in train_fns else ("val" if rec["filename"] in val_fns else "test")

        proc_item = {
            "id": idx + 1,
            "filename": rec["filename"],
            "processed_image": out_img_name,
            "site": rec["site"],
            "split": split_assignment,
            "width": TARGET_SIZE[0],
            "height": TARGET_SIZE[1],
            "orig_width": orig_w,
            "orig_height": orig_h,
            "dominant_class": rec["dominant_class"],
            "boxes": scaled_boxes,
            "box_count": len(scaled_boxes),
            "is_normal": len(scaled_boxes) == 0
        }
        processed_manifest.append(proc_item)

        # Also write Pascal VOC XML in processed annotations folder
        xml_root = ET.Element("annotation")
        ET.SubElement(xml_root, "filename").text = out_img_name
        size_tag = ET.SubElement(xml_root, "size")
        ET.SubElement(size_tag, "width").text = str(TARGET_SIZE[0])
        ET.SubElement(size_tag, "height").text = str(TARGET_SIZE[1])
        ET.SubElement(size_tag, "depth").text = "3"

        for sb in scaled_boxes:
            obj_tag = ET.SubElement(xml_root, "object")
            ET.SubElement(obj_tag, "name").text = sb["name"]
            bnd = ET.SubElement(obj_tag, "bndbox")
            ET.SubElement(bnd, "xmin").text = str(int(round(sb["bbox"][0])))
            ET.SubElement(bnd, "ymin").text = str(int(round(sb["bbox"][1])))
            ET.SubElement(bnd, "xmax").text = str(int(round(sb["bbox"][2])))
            ET.SubElement(bnd, "ymax").text = str(int(round(sb["bbox"][3])))

        xml_tree = ET.ElementTree(xml_root)
        out_xml_path = os.path.join(PROCESSED_ANN_DIR, f"{os.path.splitext(out_img_name)[0]}.xml")
        xml_tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

        if (idx + 1) % 50 == 0 or idx == len(dataset_records) - 1:
            print(f"  Processed {idx + 1}/{len(dataset_records)} images.")

    # 6. Save Complete Processed Dataset Manifest
    manifest_out = os.path.join(PROCESSED_DIR, "dataset_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(processed_manifest, f, indent=2)
    print(f"\nSaved processed dataset manifest to: {manifest_out}")

    # 7. Generate Split and Class Statistics
    split_stats = {}
    for s_name, recs in [("train", train_records), ("val", val_records), ("test", test_records)]:
        s_fns = {r["filename"] for r in recs}
        s_items = [p for p in processed_manifest if p["filename"] in s_fns]
        b_counts = defaultdict(int)
        for it in s_items:
            for b in it["boxes"]:
                b_counts[b["name"]] += 1
        split_stats[s_name] = {
            "image_count": len(s_items),
            "normal_images": sum(1 for it in s_items if it["is_normal"]),
            "defect_images": sum(1 for it in s_items if not it["is_normal"]),
            "total_boxes": sum(b_counts.values()),
            "box_distribution": dict(b_counts)
        }

    dataset_stats = {
        "dataset_name": "Heritage AI Decay Monitoring - Processed Dataset",
        "target_resolution": list(TARGET_SIZE),
        "total_canonical_images": len(processed_manifest),
        "total_boxes": sum(class_box_counts.values()),
        "classes": CLASS_MAP,
        "class_box_distribution": dict(class_box_counts),
        "image_level_dominant_distribution": dict(image_class_counts),
        "splits": split_stats
    }

    stats_out = os.path.join(PROCESSED_DIR, "dataset_stats.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"Saved dataset stats to: {stats_out}")

    # 8. Generate Sample Visualizations (with bounding boxes)
    print("\nGenerating sample annotation visualizations...")
    sample_indices = [0, 10, 25, 50, 100, 150, 200, 250, 300, 305]
    for s_idx in sample_indices:
        if s_idx >= len(processed_manifest):
            continue
        item = processed_manifest[s_idx]
        img_p = os.path.join(PROCESSED_IMG_DIR, item["processed_image"])
        with Image.open(img_p) as simg:
            draw = ImageDraw.Draw(simg)
            for b in item["boxes"]:
                cls_name = b["name"]
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                xmin, ymin, xmax, ymax = b["bbox"]
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
                draw.text((xmin + 2, max(0, ymin - 10)), cls_name, fill=color)

            if item["is_normal"]:
                draw.rectangle([5, 5, 120, 25], fill=(34, 197, 94))
                draw.text((10, 8), "NORMAL (0 BOXES)", fill=(255, 255, 255))

            save_name = f"sample_{s_idx:03d}_{os.path.splitext(item['processed_image'])[0]}.png"
            simg.save(os.path.join(SAMPLES_DIR, save_name))
            simg.save(os.path.join(QC_DIR, save_name))

    print(f"Saved sample visualizations to {SAMPLES_DIR} and {QC_DIR}")
    print("\n[SUCCESS] Dataset preparation finished successfully.")

if __name__ == "__main__":
    prepare_dataset(random_seed=42)
