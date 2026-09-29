#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring System
Phase 1 - Local Monument Dataset Inspection Script
"""

import os
import sys
import hashlib
import json
from collections import defaultdict
from PIL import Image

def run_inspection(dataset_root: str, output_json: str = None):
    print(f"[INFO] Scanning dataset at: {dataset_root}")
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Directory not found: {dataset_root}")
        sys.exit(1)

    results = {
        "dataset_root": os.path.abspath(dataset_root),
        "total_files": 0,
        "total_images": 0,
        "image_formats": defaultdict(int),
        "file_extensions": defaultdict(int),
        "folder_file_counts": defaultdict(int),
        "subfolders": [],
        "annotations_found": [],
        "corrupted_files": [],
        "dimensions": {
            "min_width": float("inf"),
            "max_width": 0,
            "min_height": float("inf"),
            "max_height": 0,
            "distribution": defaultdict(int)
        },
        "file_sizes": {
            "min_bytes": float("inf"),
            "max_bytes": 0,
            "total_bytes": 0,
            "zero_byte_files": []
        },
        "hash_to_files": defaultdict(list),
        "duplicates": [],
        "split_folders_found": []
    }

    target_keywords = ["normal", "crack", "moss", "seepage"]
    split_keywords = ["train", "val", "validation", "test"]

    for root, dirs, files in os.walk(dataset_root):
        rel_folder = os.path.relpath(root, dataset_root)
        results["subfolders"].append(rel_folder)

        for d in dirs:
            if d.lower() in split_keywords:
                results["split_folders_found"].append(os.path.join(rel_folder, d))

        for f in files:
            results["total_files"] += 1
            file_path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            results["file_extensions"][ext] += 1
            results["folder_file_counts"][rel_folder] += 1

            if ext in [".xml", ".json", ".txt", ".csv", ".yaml", ".yml"]:
                results["annotations_found"].append(file_path)

            try:
                size = os.path.getsize(file_path)
                results["file_sizes"]["total_bytes"] += size
                if size == 0:
                    results["file_sizes"]["zero_byte_files"].append(file_path)
                results["file_sizes"]["min_bytes"] = min(results["file_sizes"]["min_bytes"], size)
                results["file_sizes"]["max_bytes"] = max(results["file_sizes"]["max_bytes"], size)

                hasher = hashlib.md5()
                with open(file_path, "rb") as fp:
                    while chunk := fp.read(65536):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
                results["hash_to_files"][file_hash].append(os.path.join(rel_folder, f))
            except Exception as e:
                results["corrupted_files"].append((file_path, f"File read error: {str(e)}"))
                continue

            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                results["total_images"] += 1
                try:
                    with Image.open(file_path) as img:
                        fmt = img.format
                        results["image_formats"][fmt] += 1
                        w, h = img.size
                        results["dimensions"]["distribution"][f"{w}x{h}"] += 1
                        results["dimensions"]["min_width"] = min(results["dimensions"]["min_width"], w)
                        results["dimensions"]["max_width"] = max(results["dimensions"]["max_width"], w)
                        results["dimensions"]["min_height"] = min(results["dimensions"]["min_height"], h)
                        results["dimensions"]["max_height"] = max(results["dimensions"]["max_height"], h)
                        img.verify()
                    with Image.open(file_path) as img:
                        img.load()
                except Exception as e:
                    results["corrupted_files"].append((file_path, f"Image corruption error: {str(e)}"))

    for hsh, paths in results["hash_to_files"].items():
        if len(paths) > 1:
            results["duplicates"].append(paths)

    print("\n--- PHASE 1 DATASET INSPECTION SUMMARY ---")
    print(f"Dataset Root: {results['dataset_root']}")
    print(f"Total Files Scanned: {results['total_files']}")
    print(f"Total Images: {results['total_images']}")
    print(f"Image Formats: {dict(results['image_formats'])}")
    print(f"Corrupted Files: {len(results['corrupted_files'])}")
    print(f"Duplicate Groups: {len(results['duplicates'])}")
    print(f"Annotation Files Detected: {len(results['annotations_found'])}")
    print(f"Train/Val/Test Split Detected: {'Yes' if results['split_folders_found'] else 'No'}")
    print(f"Subfolders and Counts: {dict(results['folder_file_counts'])}")
    print(f"Dimension Distribution: {dict(results['dimensions']['distribution'])}")

    if output_json:
        serializable = {
            "dataset_root": results["dataset_root"],
            "total_files": results["total_files"],
            "total_images": results["total_images"],
            "image_formats": dict(results["image_formats"]),
            "folder_file_counts": dict(results["folder_file_counts"]),
            "annotations_found": results["annotations_found"],
            "corrupted_files": results["corrupted_files"],
            "dimensions": dict(results["dimensions"]["distribution"]),
            "duplicates_count": len(results["duplicates"]),
            "split_folders_found": results["split_folders_found"]
        }
        with open(output_json, "w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2)
        print(f"\n[INFO] Detailed inspection report written to: {output_json}")

if __name__ == "__main__":
    default_path = os.environ.get("DATASET_ROOT", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
    run_inspection(default_path)
