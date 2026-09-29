#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - AI-Assisted Proposal Generator
Scans the 307 unique monument images and produces candidate bounding-box proposals
using computer vision feature saliency (color-space, gradient/ridge, dampness ratios).

CRITICAL POLICY:
- All generated proposals are labeled with status 'REVIEW_REQUIRED'.
- Proposals are suggestions to reduce human effort; they are NOT ground truth.
- Normal images receive ZERO defect bounding boxes.
- Original images are NEVER modified.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import List, Dict, Any, Tuple
import yaml

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from annotation_schema import BoundingBox, ImageAnnotationRecord, ReviewStatus, CLASS_MAP

def load_config() -> Dict[str, Any]:
    cfg_path = os.path.join(current_dir, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def connected_components_bboxes(binary_mask: np.ndarray, min_area: int = 150) -> List[Tuple[int, int, int, int, float]]:
    """
    Fast connected component bounding box extractor using standard NumPy labeling.
    Returns list of (xmin, ymin, xmax, ymax, saliency_score).
    """
    h, w = binary_mask.shape
    # Simple grid-based connected component clustering
    # Downsample mask by 8x for fast blob grouping
    grid_h = h // 16
    grid_w = w // 16
    if grid_h == 0 or grid_w == 0:
        return []

    # Downsampled density grid
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            patch = binary_mask[r*16:(r+1)*16, c*16:(c+1)*16]
            grid[r, c] = np.mean(patch)

    density_threshold = 0.20
    active = grid > density_threshold

    # Find bounding boxes of connected active grid cells
    visited = np.zeros_like(active, dtype=bool)
    boxes = []

    for r in range(grid_h):
        for c in range(grid_w):
            if active[r, c] and not visited[r, c]:
                # BFS/Flood fill
                queue = [(r, c)]
                visited[r, c] = True
                min_r, max_r = r, r
                min_c, max_c = c, c
                cells = 0
                total_density = 0.0

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    cells += 1
                    total_density += grid[curr_r, curr_c]
                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < grid_h and 0 <= nc < grid_w and active[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                if cells >= 2:
                    xmin = max(0, min_c * 16 - 8)
                    ymin = max(0, min_r * 16 - 8)
                    xmax = min(w, (max_c + 1) * 16 + 8)
                    ymax = min(h, (max_r + 1) * 16 + 8)
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    if box_w * box_h >= min_area and (box_w * box_h) < (w * h * 0.75):
                        avg_density = total_density / cells
                        boxes.append((xmin, ymin, xmax, ymax, avg_density))

    return boxes

def detect_moss_proposals(arr_rgb: np.ndarray, orig_w: int, orig_h: int) -> List[BoundingBox]:
    """Detects biological moss growth using color saliency (green vs red/blue chromaticity)."""
    r = arr_rgb[:, :, 0].astype(np.float32)
    g = arr_rgb[:, :, 1].astype(np.float32)
    b = arr_rgb[:, :, 2].astype(np.float32)

    # Green Excess Index (2G - R - B)
    green_excess = 2.0 * g - r - b
    # Also check that green channel is prominent
    green_ratio = (g + 1e-5) / (r + b + 1e-5)

    moss_mask = (green_excess > 20.0) & (green_ratio > 0.62) & (g > 35) & (g < 220)
    raw_boxes = connected_components_bboxes(moss_mask, min_area=300)

    cur_h, cur_w = arr_rgb.shape[:2]
    scale_x = orig_w / cur_w
    scale_y = orig_h / cur_h

    proposals = []
    for xmin, ymin, xmax, ymax, score in raw_boxes:
        conf = round(min(0.92, max(0.55, 0.50 + float(score) * 0.40)), 2)
        bbox = BoundingBox(
            xmin=round(xmin * scale_x, 1),
            ymin=round(ymin * scale_y, 1),
            xmax=round(xmax * scale_x, 1),
            ymax=round(ymax * scale_y, 1),
            label="Moss",
            class_id=CLASS_MAP["Moss"],
            confidence=conf,
            source="AI_PROPOSAL"
        )
        proposals.append(bbox)
    return proposals

def detect_crack_proposals(gray_img: Image.Image, orig_w: int, orig_h: int) -> List[BoundingBox]:
    """Detects structural cracks using dark ridge and directional gradient filters."""
    # Find fine dark edges
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    arr_edges = np.array(edges, dtype=np.float32)
    arr_gray = np.array(gray_img, dtype=np.float32)

    # Local dark lines with sharp gradient contrast
    mean_val = np.mean(arr_gray)
    std_val = np.std(arr_gray)
    dark_threshold = mean_val - 0.75 * std_val
    crack_mask = (arr_edges > 35.0) & (arr_gray < dark_threshold)

    raw_boxes = connected_components_bboxes(crack_mask, min_area=250)
    cur_w, cur_h = gray_img.size
    scale_x = orig_w / cur_w
    scale_y = orig_h / cur_h

    proposals = []
    for xmin, ymin, xmax, ymax, score in raw_boxes:
        bw = (xmax - xmin) * scale_x
        bh = (ymax - ymin) * scale_y
        # Cracks are typically elongated (aspect ratio > 1.4) or diagonal
        aspect = max(bw / (bh + 1e-5), bh / (bw + 1e-5))
        if aspect > 1.3:
            conf = round(min(0.88, max(0.52, 0.50 + float(score) * 0.35)), 2)
            bbox = BoundingBox(
                xmin=round(xmin * scale_x, 1),
                ymin=round(ymin * scale_y, 1),
                xmax=round(xmax * scale_x, 1),
                ymax=round(ymax * scale_y, 1),
                label="Crack",
                class_id=CLASS_MAP["Crack"],
                confidence=conf,
                source="AI_PROPOSAL"
            )
            proposals.append(bbox)
    return proposals

def detect_seepage_proposals(arr_rgb: np.ndarray, orig_w: int, orig_h: int) -> List[BoundingBox]:
    """Detects dampness and water staining using low-luminance moisture gradients."""
    r = arr_rgb[:, :, 0].astype(np.float32)
    g = arr_rgb[:, :, 1].astype(np.float32)
    b = arr_rgb[:, :, 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Moisture causes significant local darkening without high color saturation
    mean_lum = np.mean(lum)
    std_lum = np.std(lum)
    damp_mask = (lum < mean_lum - 1.1 * std_lum) & (lum > 20) & (np.abs(r - g) < 25)

    raw_boxes = connected_components_bboxes(damp_mask, min_area=400)
    cur_h, cur_w = arr_rgb.shape[:2]
    scale_x = orig_w / cur_w
    scale_y = orig_h / cur_h

    proposals = []
    for xmin, ymin, xmax, ymax, score in raw_boxes:
        conf = round(min(0.85, max(0.50, 0.48 + float(score) * 0.35)), 2)
        bbox = BoundingBox(
            xmin=round(xmin * scale_x, 1),
            ymin=round(ymin * scale_y, 1),
            xmax=round(xmax * scale_x, 1),
            ymax=round(ymax * scale_y, 1),
            label="Seepage",
            class_id=CLASS_MAP["Seepage"],
            confidence=conf,
            source="AI_PROPOSAL"
        )
        proposals.append(bbox)
    return proposals

def nms_filter(boxes: List[BoundingBox], iou_thresh: float = 0.40) -> List[BoundingBox]:
    """Applies class-aware Non-Maximum Suppression to eliminate overlapping redundant proposals."""
    if not boxes:
        return []

    # Group by class
    by_class = {}
    for b in boxes:
        by_class.setdefault(b.label, []).append(b)

    filtered = []
    for cls_name, cls_boxes in by_class.items():
        # Sort by confidence descending
        cls_boxes.sort(key=lambda x: x.confidence, reverse=True)
        keep = []
        while cls_boxes:
            best = cls_boxes.pop(0)
            keep.append(best)
            cls_boxes = [b for b in cls_boxes if best.iou(b) < iou_thresh]
        filtered.extend(keep)
    return filtered

def process_single_image(image_info: Dict[str, Any], dataset_root: str, cfg: Dict[str, Any]) -> ImageAnnotationRecord:
    rel_path = image_info["rel_path"]
    full_path = os.path.join(dataset_root, rel_path)
    orig_w = image_info["width"]
    orig_h = image_info["height"]

    record = ImageAnnotationRecord(
        filename=image_info["filename"],
        rel_path=rel_path,
        site=image_info["site"],
        width=orig_w,
        height=orig_h,
        status=ReviewStatus.REVIEW_REQUIRED.value,
        boxes=[]
    )

    if not os.path.exists(full_path):
        record.notes = f"Image file not found at {full_path}"
        return record

    try:
        with Image.open(full_path) as img:
            # Downsample for fast feature analysis (e.g. max dim 1024)
            downsample_max = cfg.get("proposal_generator", {}).get("downsample_max_dim", 1024)
            scale = downsample_max / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            small_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            arr_rgb = np.array(small_img.convert("RGB"))
            gray_img = small_img.convert("L")

            # Extract defect candidates
            moss_boxes = detect_moss_proposals(arr_rgb, orig_w, orig_h)
            crack_boxes = detect_crack_proposals(gray_img, orig_w, orig_h)
            seepage_boxes = detect_seepage_proposals(arr_rgb, orig_w, orig_h)

            all_candidates = moss_boxes + crack_boxes + seepage_boxes

            # Clip to image bounds
            for b in all_candidates:
                b.clip_to_bounds(orig_w, orig_h)

            # Apply NMS
            nms_thresh = cfg.get("proposal_generator", {}).get("nms_iou_threshold", 0.40)
            final_boxes = nms_filter(all_candidates, iou_thresh=nms_thresh)

            # Limit max proposals per image to top 15 most salient
            final_boxes.sort(key=lambda x: x.confidence, reverse=True)
            final_boxes = final_boxes[:15]

            record.boxes = final_boxes

            if len(final_boxes) == 0:
                record.status = ReviewStatus.PROPOSED_NORMAL.value
            else:
                record.status = ReviewStatus.REVIEW_REQUIRED.value

    except Exception as e:
        record.notes = f"Error during proposal extraction: {str(e)}"
        record.status = ReviewStatus.REVIEW_REQUIRED.value

    return record

def run_proposals(max_images: int = None):
    cfg = load_config()
    dataset_root = cfg.get("dataset", {}).get("raw_root", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
    manifest_path = cfg.get("dataset", {}).get("manifest_path")
    proposals_dir = cfg.get("paths", {}).get("proposals_dir")

    os.makedirs(proposals_dir, exist_ok=True)

    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — AI PROPOSAL GENERATOR")
    print("=" * 70)
    print(f"Dataset Root:      {dataset_root}")
    print(f"Manifest Path:     {manifest_path}")
    print(f"Proposals Dir:     {proposals_dir}")
    print("=" * 70)

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    images = manifest_data.get("images", [])
    if max_images:
        images = images[:max_images]

    total = len(images)
    print(f"Processing {total} unique images for proposal generation...\n")

    summary_stats = {
        "total_images": total,
        "proposed_normal": 0,
        "review_required_with_proposals": 0,
        "total_proposals": 0,
        "proposals_by_class": {
            "Crack": 0,
            "Moss": 0,
            "Seepage": 0
        }
    }

    for idx, img_info in enumerate(images, start=1):
        record = process_single_image(img_info, dataset_root, cfg)
        out_json = os.path.join(proposals_dir, f"{os.path.splitext(img_info['filename'])[0]}.json")
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(record.to_dict(), fp, indent=2)

        if len(record.boxes) == 0:
            summary_stats["proposed_normal"] += 1
        else:
            summary_stats["review_required_with_proposals"] += 1
            summary_stats["total_proposals"] += len(record.boxes)
            for b in record.boxes:
                summary_stats["proposals_by_class"][b.label] += 1

        if idx % 25 == 0 or idx == total:
            print(f"  [{idx:3d}/{total:3d}] Processed: {img_info['filename']} | Boxes proposed: {len(record.boxes)} ({record.status})")

    # Save summary report
    summary_path = os.path.join(proposals_dir, "proposals_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary_stats, fp, indent=2)

    print("\n" + "=" * 70)
    print(" PROPOSAL GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total Unique Images Processed:       {summary_stats['total_images']}")
    print(f"Images with Proposed Defects:        {summary_stats['review_required_with_proposals']}")
    print(f"Images Proposed Normal (0 boxes):    {summary_stats['proposed_normal']}")
    print(f"Total Candidate Bounding Boxes:      {summary_stats['total_proposals']}")
    print("Proposal Counts by Class:")
    for cls_name, cnt in summary_stats["proposals_by_class"].items():
        print(f"  - {cls_name}: {cnt} candidate boxes")
    print(f"\nAll proposals saved to: {proposals_dir}")
    print("CRITICAL: All proposals are marked 'REVIEW_REQUIRED' for human verification.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate AI-assisted defect proposals for monument images")
    parser.add_argument("--max", type=int, default=None, help="Max images to process for quick testing")
    args = parser.parse_args()
    run_proposals(max_images=args.max)
