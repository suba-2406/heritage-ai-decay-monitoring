"""
Pydantic Schemas for Decay Detection and Preservation Recommendation Responses
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class DetectionItem(BaseModel):
    class_name: str = Field(..., example="Crack", description="Name of detected decay class (Crack, Moss, Seepage)")
    class_id: int = Field(..., example=1, description="Numeric class ID")
    confidence: float = Field(..., example=0.42, description="Model prediction confidence score")
    box: List[float] = Field(..., example=[120.5, 340.0, 280.2, 450.8], description="Bounding box [xmin, ymin, xmax, ymax] in native image coordinates")
    box_normalized: List[float] = Field(..., example=[0.12, 0.34, 0.28, 0.45], description="Normalized bounding box [xmin/W, ymin/H, xmax/W, ymax/H]")

class RiskAssessmentResponse(BaseModel):
    risk_level: str = Field(..., example="HIGH", description="Risk level (LOW, MEDIUM, HIGH)")
    severity_score: int = Field(..., example=3, description="Numeric severity score (1=Low, 2=Medium, 3=High)")
    color_hex: str = Field(..., example="#ef4444", description="Hex color representing risk level")
    description: str = Field(..., example="Structural fractures or active moisture seepage.")
    action_timeline: str = Field(..., example="URGENT intervention required within 15 to 30 days.")
    coverage_percentage: float = Field(..., example=18.4, description="Percentage of stone surface covered by defect boxes")
    defect_counts: Dict[str, int] = Field(..., example={"Crack": 2, "Seepage": 5}, description="Count of detections per class")
    defect_max_confidences: Dict[str, float] = Field(..., example={"Crack": 0.38, "Seepage": 0.45})
    risk_rationale: str = Field(..., example="HIGH RISK triggered: Structural Crack detected.")

class ConservationProtocol(BaseModel):
    defect_type: str = Field(..., example="Crack")
    urgency: str = Field(..., example="High")
    summary: str = Field(..., example="Structural Fracture & Fissure Intervention")
    count: int = Field(..., example=2)
    max_confidence: float = Field(..., example=0.38)
    diagnostic_steps: List[str] = Field(...)
    treatment_steps: List[str] = Field(...)
    contraindications: List[str] = Field(...)

class PreservationSummary(BaseModel):
    monument_condition: str = Field(..., example="HIGH")
    primary_concern: str = Field(..., example="Crack")
    recommended_action_summary: str = Field(...)
    supervising_authority: str = Field(default="Archaeological Survey of India (ASI) / State Archaeology Dept")

class PreservationPlanResponse(BaseModel):
    report_id: str = Field(..., example="REP-DECAY-20260930-161015")
    timestamp: str = Field(...)
    analyzed_image: str = Field(...)
    risk_assessment: RiskAssessmentResponse
    total_detections: int = Field(...)
    recommended_protocols: List[ConservationProtocol]
    preservation_summary: PreservationSummary

class PreprocessingInfo(BaseModel):
    original_dimensions: List[int] = Field(..., example=[3072, 4096])
    operations_applied: List[str] = Field(..., example=["MedianDenoise(k=3)", "CLAHE(clip=2.5, grid=8x8)"])
    sharpness_laplacian_var: Optional[float] = None
    mean_brightness: Optional[float] = None
    is_underexposed: Optional[bool] = None
    is_overexposed: Optional[bool] = None

class AnalysisResponse(BaseModel):
    image_name: str
    original_dimensions: List[int]
    total_detections: int
    detections: List[DetectionItem]
    risk_assessment: RiskAssessmentResponse
    preservation_plan: PreservationPlanResponse
    preprocessing_info: PreprocessingInfo
