#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Standalone Inference & Preservation Runner
Command-line tool to analyze ancient monument imagery for surface decay,
quantify defect coverage, evaluate structural risk, and generate actionable
ASI preservation recommendation reports.

Usage:
    python ai/inference/run_inference.py --image path/to/monument.jpg
    python ai/inference/run_inference.py --input-dir path/to/images/ --output-dir outputs/
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.insert(0, CURRENT_DIR)
from decay_detector import MonumentDecayDetector
from preservation_recommender import format_markdown_report

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "inference_outputs")

def run_single_inference(detector, image_path, output_dir, apply_enhancement=True):
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]

    # Predict
    result = detector.predict(image_path, apply_enhancement=apply_enhancement)

    # 1. Save annotated image
    vis_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    result["annotated_image"].save(vis_path, quality=95)

    # 2. Save JSON report
    report_dict = {
        "image_file": img_name,
        "dimensions": result["original_dimensions"],
        "preprocessing": result["preprocessing_info"],
        "total_detections": result["total_detections"],
        "detections": result["detections"],
        "risk_assessment": result["risk_assessment"],
        "preservation_plan": result["preservation_plan"]
    }
    json_path = os.path.join(output_dir, f"{base_name}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # 3. Save Markdown conservation report
    md_report = format_markdown_report(result["preservation_plan"])
    md_path = os.path.join(output_dir, f"{base_name}_conservation_protocol.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    return {
        "image": img_name,
        "risk_level": result["risk_assessment"]["risk_level"],
        "coverage_pct": result["risk_assessment"]["coverage_percentage"],
        "defect_counts": result["risk_assessment"]["defect_counts"],
        "vis_path": vis_path,
        "json_path": json_path,
        "md_path": md_path
    }

def main():
    parser = argparse.ArgumentParser(description="Monument Decay Detection & Preservation Recommendation Runner")
    parser.add_argument("--image", type=str, help="Path to a single monument image file")
    parser.add_argument("--input-dir", type=str, help="Directory containing monument images to process in batch")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save inspection outputs")
    parser.add_argument("--score-thresh", type=float, default=0.25, help="Confidence score threshold (default: 0.25)")
    parser.add_argument("--nms-thresh", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable CLAHE and median denoising preprocessing")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom model checkpoint path")
    args = parser.parse_args()

    print("=" * 80)
    print(" HERITAGE AI DECAY MONITORING — INFERENCE & PRESERVATION PIPELINE")
    print("=" * 80)

    # Collect images
    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Specified image file not found: {args.image}")
            sys.exit(1)
        image_paths.append(args.image)
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Specified input directory not found: {args.input_dir}")
            sys.exit(1)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend([str(p) for p in Path(args.input_dir).glob(ext)])
        image_paths = sorted(list(set(image_paths)))
        if not image_paths:
            print(f"No valid image files found in: {args.input_dir}")
            sys.exit(1)
    else:
        # Default demo: Run on first 3 test set images
        test_dir = os.path.join(BASE_DIR, "dataset", "processed", "images")
        manifest_p = os.path.join(BASE_DIR, "dataset", "processed", "dataset_manifest.json")
        if os.path.exists(manifest_p):
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            test_items = [it for it in manifest if it["split"] == "test"][:3]
            for it in test_items:
                image_paths.append(os.path.join(test_dir, it["processed_image"]))
        else:
            print("Please specify --image or --input-dir.")
            sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to analyze.")
    print(f"Output directory: {args.output_dir}")
    print(f"Score threshold:  {args.score_thresh}")
    print(f"NMS threshold:    {args.nms_thresh}")
    print(f"Preprocessing:    {'Disabled' if args.no_enhance else 'Median Denoising + CLAHE (Enabled)'}")
    print("-" * 80)

    # Initialize Detector
    detector = MonumentDecayDetector(
        checkpoint_path=args.checkpoint,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh
    )

    results = []
    for img_p in tqdm(image_paths, desc="Analyzing Monuments"):
        res = run_single_inference(detector, img_p, args.output_dir, apply_enhancement=not args.no_enhance)
        results.append(res)

    print("\n" + "=" * 80)
    print(" INSPECTION SUMMARY & RISK CLASSIFICATION")
    print("=" * 80)
    print(f"{'Image':<35s} | {'Risk':^8s} | {'Coverage':^10s} | {'Identified Defects':<25s}")
    print("-" * 85)
    for r in results:
        defects_str = ", ".join([f"{k}:{v}" for k, v in r["defect_counts"].items()]) or "Normal"
        print(f"{r['image'][:35]:<35s} | {r['risk_level']:^8s} | {r['coverage_pct']:>8.1f}% | {defects_str:<25s}")
    print("-" * 85)
    print(f"\n[SUCCESS] Analyzed {len(results)} image(s). All reports saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
