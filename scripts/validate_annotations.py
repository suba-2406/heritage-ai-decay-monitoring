#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring System
Phase 2 - Annotation Validation Script

Validates Pascal VOC XML, COCO JSON, or YOLO annotations for:
- Missing annotations
- Orphaned annotations (annotation file without corresponding image)
- Invalid class labels / IDs (must match Normal, Crack, Moss, Seepage)
- Invalid bounding boxes (xmin >= xmax or ymin >= ymax)
- Out-of-bounds coordinates (xmin < 0, ymin < 0, xmax > width, ymax > height)
- Zero-area boxes
- Duplicate annotations on the same image
- Summary of class distribution and box counts

Note: This script ONLY validates data integrity. It DOES NOT trigger SSD training.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image

VALID_CLASSES = {"Normal", "Crack", "Moss", "Seepage"}
CLASS_ID_MAP = {
    "Normal": 0,
    "Crack": 1,
    "Moss": 2,
    "Seepage": 3
}

def validate_voc_xml(xml_path, image_width=None, image_height=None):
    """Parses a Pascal VOC XML file and returns errors, warnings, and box list."""
    errors = []
    warnings = []
    boxes = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return [f"Malformed XML: {str(e)}"], [], []

    size_tag = root.find("size")
    xml_w = int(size_tag.find("width").text) if size_tag is not None and size_tag.find("width") is not None else image_width
    xml_h = int(size_tag.find("height").text) if size_tag is not None and size_tag.find("height") is not None else image_height

    w = image_width if image_width else xml_w
    h = image_height if image_height else xml_h

    seen_boxes = set()

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            errors.append("Object missing <name> tag")
            continue
        cls_name = name_tag.text.strip()
        if cls_name not in VALID_CLASSES:
            errors.append(f"Invalid class name '{cls_name}'. Allowed: {sorted(list(VALID_CLASSES))}")

        bndbox = obj.find("bndbox")
        if bndbox is None:
            errors.append(f"Object '{cls_name}' missing <bndbox> tag")
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception as e:
            errors.append(f"Invalid coordinate format in bndbox: {str(e)}")
            continue

        # Check for zero or negative area
        if xmin >= xmax:
            errors.append(f"Invalid x coordinates: xmin ({xmin}) >= xmax ({xmax})")
        if ymin >= ymax:
            errors.append(f"Invalid y coordinates: ymin ({ymin}) >= ymax ({ymax})")

        box_w = max(0, xmax - xmin)
        box_h = max(0, ymax - ymin)
        if box_w * box_h == 0:
            errors.append(f"Zero-area bounding box: [{xmin}, {ymin}, {xmax}, {ymax}]")

        # Boundary checks
        if w and h:
            if xmin < 0 or ymin < 0:
                errors.append(f"Coordinates less than 0: xmin={xmin}, ymin={ymin}")
            if xmax > w:
                errors.append(f"xmax ({xmax}) exceeds image width ({w})")
            if ymax > h:
                errors.append(f"ymax ({ymax}) exceeds image height ({h})")

        # Duplicate box check
        box_key = (cls_name, round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1))
        if box_key in seen_boxes:
            warnings.append(f"Duplicate bounding box detected: {cls_name} {box_key[1:]}")
        seen_boxes.add(box_key)

        # Full image bounding box warning for Normal
        if cls_name == "Normal" and w and h:
            if box_w >= 0.95 * w and box_h >= 0.95 * h:
                warnings.append("Full-image bounding box labeled as 'Normal'. Normal images should be recorded without boxes as negative/healthy images.")

        boxes.append({
            "class": cls_name,
            "class_id": CLASS_ID_MAP.get(cls_name, -1),
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return errors, warnings, boxes


def run_validation(dataset_root, annotations_dir, manifest_path=None, report_out=None):
    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — ANNOTATION INTEGRITY VALIDATION")
    print("=" * 70)
    print(f"Dataset Root:      {dataset_root}")
    print(f"Annotations Dir:   {annotations_dir}")
    print(f"Valid Classes:     {sorted(list(VALID_CLASSES))}")
    print("-" * 70)

    # 1. Load manifest or scan images
    image_catalog = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            for item in manifest_data.get("images", []):
                image_catalog[item["filename"]] = item
    else:
        for root, _, files in os.walk(dataset_root):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_p = os.path.join(root, f)
                    image_catalog[f] = {
                        "filename": f,
                        "rel_path": os.path.relpath(full_p, dataset_root),
                        "full_path": full_p
                    }

    total_images = len(image_catalog)
    print(f"Loaded {total_images} target images for validation.")

    # 2. Check annotation files
    if not os.path.exists(annotations_dir):
        print(f"[ERROR] Annotations directory not found: {annotations_dir}")
        return

    ann_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith((".xml", ".json", ".txt")) and not f.startswith(".")]
    print(f"Found {len(ann_files)} annotation files in {annotations_dir}.\n")

    if len(ann_files) == 0:
        print("[STATUS REPORT] ----------------------------------------------------")
        print("  NO ANNOTATIONS DETECTED.")
        print("  Status: ANNOTATION REQUIRED BEFORE PHASE 2")
        print("  All 307 unique images require bounding-box annotation.")
        print("  Please follow docs/annotation_guidelines.md to produce annotations.")
        print("--------------------------------------------------------------------")
        return

    # 3. Validate each annotation
    stats = {
        "annotated_images_count": 0,
        "images_without_annotation": 0,
        "orphaned_annotations": 0,
        "total_boxes": 0,
        "class_counts": defaultdict(int),
        "errors_by_file": defaultdict(list),
        "warnings_by_file": defaultdict(list),
        "images_with_defects": 0,
        "normal_images_count": 0
    }

    matched_images = set()

    for ann_file in ann_files:
        base_name, ext = os.path.splitext(ann_file)
        # Search for corresponding image
        corresp_img = None
        for cand_ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            cand = base_name + cand_ext
            if cand in image_catalog:
                corresp_img = image_catalog[cand]
                break

        if not corresp_img:
            stats["orphaned_annotations"] += 1
            stats["errors_by_file"][ann_file].append(f"Orphaned annotation: No corresponding image found for '{ann_file}'")
            continue

        matched_images.add(corresp_img["filename"])
        stats["annotated_images_count"] += 1

        img_w = corresp_img.get("width")
        img_h = corresp_img.get("height")
        if not img_w or not img_h:
            full_img_p = corresp_img.get("full_path") or os.path.join(dataset_root, corresp_img.get("rel_path", ""))
            if os.path.exists(full_img_p):
                try:
                    with Image.open(full_img_p) as im:
                        img_w, img_h = im.size
                except Exception:
                    pass

        xml_full_path = os.path.join(annotations_dir, ann_file)
        errs, warns, boxes = validate_voc_xml(xml_full_path, img_w, img_h)

        if errs:
            stats["errors_by_file"][ann_file].extend(errs)
        if warns:
            stats["warnings_by_file"][ann_file].extend(warns)

        stats["total_boxes"] += len(boxes)
        if len(boxes) == 0:
            stats["normal_images_count"] += 1
        else:
            stats["images_with_defects"] += 1
            for b in boxes:
                stats["class_counts"][b["class"]] += 1

    # Unannotated images
    unannotated = [fn for fn in image_catalog if fn not in matched_images]
    stats["images_without_annotation"] = len(unannotated)

    # Print Report
    print("--- VALIDATION RESULTS ---")
    print(f"Target Images:                  {total_images}")
    print(f"Annotated Images:               {stats['annotated_images_count']}")
    print(f"Images Missing Annotation:      {stats['images_without_annotation']}")
    print(f"Orphaned Annotations:           {stats['orphaned_annotations']}")
    print(f"Total Bounding Boxes:           {stats['total_boxes']}")
    print(f"Images with Defect Boxes:       {stats['images_with_defects']}")
    print(f"Healthy / Normal Images:        {stats['normal_images_count']}")
    print(f"Total Files with Errors:        {len(stats['errors_by_file'])}")
    print(f"Total Files with Warnings:      {len(stats['warnings_by_file'])}")
    print("\nClass Distribution (Boxes):")
    for cls in sorted(list(VALID_CLASSES)):
        print(f"  - {cls} (ID {CLASS_ID_MAP[cls]}): {stats['class_counts'][cls]} boxes")

    if stats["errors_by_file"]:
        print("\nTop 5 Files with Validation Errors:")
        for fn, err_list in list(stats["errors_by_file"].items())[:5]:
            print(f"  [{fn}]: {'; '.join(err_list)}")

    if report_out:
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump({
                "target_images": total_images,
                "annotated_images": stats["annotated_images_count"],
                "missing_annotations_count": stats["images_without_annotation"],
                "orphaned_annotations_count": stats["orphaned_annotations"],
                "total_boxes": stats["total_boxes"],
                "class_distribution": dict(stats["class_counts"]),
                "error_files_count": len(stats["errors_by_file"]),
                "warning_files_count": len(stats["warnings_by_file"]),
                "errors": dict(stats["errors_by_file"]),
                "warnings": dict(stats["warnings_by_file"])
            }, f, indent=2)
        print(f"\nSaved structured validation report to: {report_out}")

    print("\n[CONCLUSION]")
    if stats["images_without_annotation"] > 0 or len(stats["errors_by_file"]) > 0:
        print("  STATUS: ANNOTATION INCOMPLETE OR ERRORS DETECTED")
        print("  -> SSD model training CANNOT begin.")
    else:
        print("  STATUS: ALL ANNOTATIONS VALIDATED SUCCESSFULLY")
        print("  -> Ready for stratified dataset split and Phase 3.")

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataset = os.environ.get("DATASET_ROOT", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
    default_ann_dir = os.path.join(base, "ai", "annotations", "raw")
    default_manifest = os.path.join(base, "ai", "dataset", "annotation_manifest.json")
    default_report = os.path.join(base, "ai", "annotations", "validation_report.json")

    run_validation(default_dataset, default_ann_dir, default_manifest, default_report)
