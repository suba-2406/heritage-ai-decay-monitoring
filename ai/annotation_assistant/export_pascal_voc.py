#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Pascal VOC XML Exporter
Exports verified human annotations from JSON into standard Pascal VOC XML format
into ai/annotations/raw/ for SSD training and validation.

RULES:
- Normal images are exported with ZERO defect <object> tags.
- Coordinates are validated before writing.
- Never creates fake annotations.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, List

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from annotation_schema import BoundingBox, ImageAnnotationRecord, ReviewStatus

def prettify_xml(elem: ET.Element) -> str:
    """Returns a pretty-printed XML string."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def export_record_to_voc_xml(record: ImageAnnotationRecord, output_dir: str, dataset_root: str = None) -> str:
    """Converts a verified ImageAnnotationRecord into Pascal VOC XML."""
    os.makedirs(output_dir, exist_ok=True)
    xml_filename = f"{os.path.splitext(record.filename)[0]}.xml"
    out_xml_path = os.path.join(output_dir, xml_filename)

    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = record.site if record.site else "TEMPLE"

    filename = ET.SubElement(root, "filename")
    filename.text = record.filename

    path = ET.SubElement(root, "path")
    path.text = os.path.join(dataset_root, record.rel_path) if dataset_root and record.rel_path else record.filename

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Tamil Nadu Heritage Monuments"

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(record.width)
    height = ET.SubElement(size, "height")
    height.text = str(record.height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    # Write defect objects (if Normal, record.boxes is empty -> ZERO object tags)
    for box in record.boxes:
        # Ignore Normal label if accidentally drawn as box; Normal should have 0 boxes
        if box.label == "Normal":
            continue

        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = box.label

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(round(box.xmin)))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(round(box.ymin)))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(round(box.xmax)))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(round(box.ymax)))

    xml_str = prettify_xml(root)

    # Remove extra blank lines introduced by minidom
    lines = [line for line in xml_str.splitlines() if line.strip()]
    cleaned_xml = "\n".join(lines) + "\n"

    with open(out_xml_path, "w", encoding="utf-8") as fp:
        fp.write(cleaned_xml)

    return out_xml_path

def export_all_verified(review_dir: str, output_dir: str, dataset_root: str = None) -> int:
    """Exports all verified JSON records in review_dir to Pascal VOC XML."""
    if not os.path.exists(review_dir):
        print(f"[WARN] Review directory does not exist: {review_dir}")
        return 0

    json_files = [f for f in os.listdir(review_dir) if f.endswith(".json") and not f.startswith(".")]
    count = 0
    for jf in json_files:
        jp = os.path.join(review_dir, jf)
        try:
            with open(jp, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            record = ImageAnnotationRecord.from_dict(data)
            # Only export verified records
            if record.status in [ReviewStatus.VERIFIED_DEFECT.value, ReviewStatus.VERIFIED_NORMAL.value]:
                export_record_to_voc_xml(record, output_dir, dataset_root)
                count += 1
        except Exception as e:
            print(f"[ERROR] Failed exporting {jf}: {str(e)}")

    print(f"Exported {count} verified Pascal VOC XML annotations to {output_dir}")
    return count

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_review = os.path.join(base, "ai", "annotations", "review")
    default_output = os.path.join(base, "ai", "annotations", "raw")
    default_dataset = os.environ.get("DATASET_ROOT", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
    export_all_verified(default_review, default_output, default_dataset)
