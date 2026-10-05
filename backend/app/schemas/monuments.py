"""
Pydantic Schemas for Monument Sites and Dataset Overview
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

class MonumentSite(BaseModel):
    id: str
    name: str
    location: str
    total_images: int
    dominant_defects: List[str]

class DatasetOverviewResponse(BaseModel):
    total_canonical_images: int
    total_boxes: int
    classes: Dict[str, int]
    class_box_distribution: Dict[str, int]
    image_level_distribution: Dict[str, int]
    monument_sites: List[MonumentSite]

class ModelBenchmarkResponse(BaseModel):
    architecture: str
    input_resolution: List[int]
    optimizer: str
    learning_rate: float
    mAP_50: float
    image_accuracy: float
    overall_roc_auc: float
    per_class_metrics: Dict[str, Any]
    baseline_vs_phase3b: Dict[str, Any]
