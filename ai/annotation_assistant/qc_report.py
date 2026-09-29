#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Quality Control & Statistics Reporter
Generates visual preview images with overlaid annotations, audits for suspicious boxes,
and compiles comprehensive dataset statistics.
"""

import os
import sys
import json
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from annotation_schema import ImageAnnotationRecord, ReviewStatus

CLASS_COLORS = {
    "Crack": (239, 68, 68),     # Red
    "Moss": (34, 197, 94),      # Green
    "Seepage": (2, 132, 199),   # Blue
    "Normal": (16, 185, 129)    # Emerald
}

def generate_qc_report(dataset_root: str, review_dir: str, manifest_path: str, output_dir: str, num_previews: int = 5):
    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — QUALITY CONTROL & DATASET STATS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest_data = json.load(fp)

    total_unique = len(manifest_data.get("images", []))
    image_catalog = {img["filename"]: img for img in manifest_data.get("images", [])}

    stats = {
        "total_unique_images": total_unique,
        "verified_normal_images": 0,
        "verified_defect_images": 0,
        "pending_review_images": 0,
        "flagged_ambiguous_images": 0,
        "total_bounding_boxes": 0,
        "boxes_by_class": {
            "Crack": 0,
            "Moss": 0,
            "Seepage": 0
        },
        "images_with_multiple_classes": 0,
        "images_with_multiple_boxes": 0,
        "suspicious_cases": []
    }

    verified_records = []

    for fn, img_info in image_catalog.items():
        base_name = os.path.splitext(fn)[0]
        review_json = os.path.join(review_dir, f"{base_name}.json")

        if not os.path.exists(review_json):
            stats["pending_review_images"] += 1
            continue

        try:
            with open(review_json, "r", encoding="utf-8") as fp:
                rdata = json.load(fp)
            record = ImageAnnotationRecord.from_dict(rdata)
        except Exception:
            stats["pending_review_images"] += 1
            continue

        status = record.status
        if status == ReviewStatus.VERIFIED_NORMAL.value:
            stats["verified_normal_images"] += 1
            verified_records.append(record)
        elif status == ReviewStatus.VERIFIED_DEFECT.value:
            stats["verified_defect_images"] += 1
            verified_records.append(record)
            box_count = len(record.boxes)
            stats["total_bounding_boxes"] += box_count
            if box_count > 1:
                stats["images_with_multiple_boxes"] += 1

            classes_in_img = set()
            for b in record.boxes:
                stats["boxes_by_class"][b.label] = stats["boxes_by_class"].get(b.label, 0) + 1
                classes_in_img.add(b.label)

                # Suspicious box audit
                bw = b.width
                bh = b.height
                img_area = record.width * record.height
                box_area = b.area
                if box_area >= 0.85 * img_area:
                    stats["suspicious_cases"].append({
                        "filename": fn,
                        "class": b.label,
                        "issue": f"Extremely large box covering {round(box_area/img_area*100, 1)}% of image"
                    })
                elif bw < 30 or bh < 30:
                    stats["suspicious_cases"].append({
                        "filename": fn,
                        "class": b.label,
                        "issue": f"Extremely tiny box: {bw}x{bh} pixels"
                    })

            if len(classes_in_img) > 1:
                stats["images_with_multiple_classes"] += 1
        elif status == ReviewStatus.FLAGGED_AMBIGUOUS.value:
            stats["flagged_ambiguous_images"] += 1
        else:
            stats["pending_review_images"] += 1

    # Print Summary
    print(f"Total Unique Images:               {stats['total_unique_images']}")
    print(f"Verified Normal Images (0 boxes):  {stats['verified_normal_images']}")
    print(f"Verified Defect Images:            {stats['verified_defect_images']}")
    print(f"Pending Review Images:             {stats['pending_review_images']}")
    print(f"Flagged for Expert Review:         {stats['flagged_ambiguous_images']}")
    print(f"Total Verified Bounding Boxes:     {stats['total_bounding_boxes']}")
    print("Verified Boxes by Class:")
    for cls_name, cnt in stats["boxes_by_class"].items():
        print(f"  - {cls_name}: {cnt} boxes")
    print(f"Images with Multiple Defect Classes: {stats['images_with_multiple_classes']}")
    print(f"Images with Multiple Boxes:          {stats['images_with_multiple_boxes']}")
    print(f"Suspicious Cases Flagged:            {len(stats['suspicious_cases'])}")

    # Generate visual preview images for random verified samples
    if verified_records:
        sample_records = random.sample(verified_records, min(num_previews, len(verified_records)))
        print(f"\nGenerating {len(sample_records)} visual preview overlays in {output_dir}...")
        for rec in sample_records:
            full_img_p = os.path.join(dataset_root, rec.rel_path)
            if not os.path.exists(full_img_p):
                continue
            with Image.open(full_img_p) as im:
                preview = im.copy()
                draw = ImageDraw.Draw(preview)
                for b in rec.boxes:
                    color = CLASS_COLORS.get(b.label, (255, 0, 0))
                    draw.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=color, width=8)
                    draw.text((b.xmin + 10, max(10, b.ymin - 30)), f"{b.label}", fill=color)

                # Downsample preview for quick inspection
                preview.thumbnail((1200, 1200))
                out_p = os.path.join(output_dir, f"qc_preview_{os.path.splitext(rec.filename)[0]}.jpg")
                preview.save(out_p, "JPEG")
                print(f"  Saved preview: {out_p}")
    else:
        print("\n[NOTE] No verified records yet to render previews. Human verification is pending.")

    # Save stats JSON
    stats_out = os.path.join(review_dir, "dataset_stats.json")
    with open(stats_out, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    print(f"\nSaved QC statistics to: {stats_out}")

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_dataset = os.environ.get("DATASET_ROOT", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
    default_review = os.path.join(base, "ai", "annotations", "review")
    default_manifest = os.path.join(base, "ai", "dataset", "annotation_manifest.json")
    default_qc_out = os.path.join(base, "docs", "qc_previews")
    generate_qc_report(default_dataset, default_review, default_manifest, default_qc_out)
