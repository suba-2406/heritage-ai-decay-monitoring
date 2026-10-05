"""
Monuments, Dataset Overview, and Model Benchmark Endpoints
"""

import os
import json
from fastapi import APIRouter
from backend.app.schemas.monuments import DatasetOverviewResponse, MonumentSite, ModelBenchmarkResponse
from backend.app.config import settings

router = APIRouter(prefix="/monuments", tags=["Monuments & Benchmarks"])

SITES_METADATA = [
    MonumentSite(
        id="kasiviswanathar",
        name="Kasiviswanathar Temple",
        location="Thanjavur, Tamil Nadu, India",
        total_images=187,
        dominant_defects=["Seepage", "Moss", "Crack"]
    ),
    MonumentSite(
        id="brihadisvara",
        name="Brihadisvara (Big) Temple",
        location="Thanjavur, Tamil Nadu, India",
        total_images=120,
        dominant_defects=["Seepage", "Crack", "Moss"]
    )
]

@router.get("", response_model=list[MonumentSite], summary="List Monitored Monument Sites")
async def list_monuments():
    """
    Returns geographical and structural decay metadata for monitored ancient monument complexes.
    """
    return SITES_METADATA

@router.get("/overview", summary="Dataset Class & Split Overview")
async def get_dataset_overview():
    """
    Returns dataset statistics, 4-class distributions, and train/val/test splits.
    """
    stats_p = settings.STATS_PATH
    if os.path.exists(stats_p):
        with open(stats_p, "r", encoding="utf-8") as f:
            stats = json.load(f)
        stats["monument_sites"] = [s.model_dump() for s in SITES_METADATA]
        return stats
    
    return {
        "dataset_name": "Heritage AI Monument Decay Dataset",
        "total_canonical_images": 307,
        "total_boxes": 4593,
        "classes": {"Normal": 0, "Crack": 1, "Moss": 2, "Seepage": 3},
        "class_box_distribution": {"Crack": 332, "Moss": 321, "Seepage": 3940, "Normal": 0},
        "monument_sites": [s.model_dump() for s in SITES_METADATA]
    }

@router.get("/benchmarks", summary="Model Performance Benchmarks (Baseline vs Phase 3B)")
async def get_model_benchmarks():
    """
    Returns comparative evaluation metrics demonstrating Phase 3B model improvements.
    """
    comp_json = os.path.join(settings.PROJECT_ROOT, "ai", "evaluation", "phase3b_vs_baseline_comparison.json")
    if os.path.exists(comp_json):
        with open(comp_json, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        return comp_data
    
    return {
        "status": "Comparison metrics not found. Run ai/evaluation/compare_phase3_phase3b.py"
    }
