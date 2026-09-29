"""
Heritage AI Decay Monitoring - Annotation Data Schema
Defines structured classes and serialization utilities for AI proposals and human-verified annotations.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json

class ReviewStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PROPOSED_NORMAL = "PROPOSED_NORMAL"
    VERIFIED_DEFECT = "VERIFIED_DEFECT"
    VERIFIED_NORMAL = "VERIFIED_NORMAL"
    FLAGGED_AMBIGUOUS = "FLAGGED_AMBIGUOUS"

CLASS_MAP = {
    "Normal": 0,
    "Crack": 1,
    "Moss": 2,
    "Seepage": 3
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str
    class_id: int
    confidence: float = 1.0
    source: str = "AI_PROPOSAL" # "AI_PROPOSAL", "HUMAN_VERIFIED", "MANUAL_DRAWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clip_to_bounds(self, img_w: int, img_h: int):
        self.xmin = max(0.0, min(float(img_w), self.xmin))
        self.ymin = max(0.0, min(float(img_h), self.ymin))
        self.xmax = max(0.0, min(float(img_w), self.xmax))
        self.ymax = max(0.0, min(float(img_h), self.ymax))

    def iou(self, other: "BoundingBox") -> float:
        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)
        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

@dataclass
class ImageAnnotationRecord:
    filename: str
    rel_path: str
    site: str
    width: int
    height: int
    status: str = ReviewStatus.UNREVIEWED.value
    boxes: List[BoundingBox] = field(default_factory=list)
    reviewer: Optional[str] = None
    review_timestamp: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "rel_path": self.rel_path,
            "site": self.site,
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "boxes": [b.to_dict() for b in self.boxes],
            "reviewer": self.reviewer,
            "review_timestamp": self.review_timestamp,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageAnnotationRecord":
        boxes = []
        for b_data in data.get("boxes", []):
            label = b_data.get("label", b_data.get("class", "Crack"))
            cid = b_data.get("class_id", CLASS_MAP.get(label, 1))
            boxes.append(BoundingBox(
                xmin=float(b_data["xmin"] if "xmin" in b_data else b_data["bbox"][0]),
                ymin=float(b_data["ymin"] if "ymin" in b_data else b_data["bbox"][1]),
                xmax=float(b_data["xmax"] if "xmax" in b_data else b_data["bbox"][2]),
                ymax=float(b_data["ymax"] if "ymax" in b_data else b_data["bbox"][3]),
                label=label,
                class_id=cid,
                confidence=float(b_data.get("confidence", 1.0)),
                source=b_data.get("source", "HUMAN_VERIFIED")
            ))
        return cls(
            filename=data["filename"],
            rel_path=data.get("rel_path", ""),
            site=data.get("site", ""),
            width=int(data["width"]),
            height=int(data["height"]),
            status=data.get("status", ReviewStatus.UNREVIEWED.value),
            boxes=boxes,
            reviewer=data.get("reviewer"),
            review_timestamp=data.get("review_timestamp"),
            notes=data.get("notes")
        )
