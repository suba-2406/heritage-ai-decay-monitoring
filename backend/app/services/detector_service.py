"""
Service Layer: Singleton Detector Service for FastAPI Backend
"""

import os
import io
import sys
from PIL import Image
from backend.app.config import settings

# Add AI paths to sys.path
sys.path.insert(0, os.path.join(settings.PROJECT_ROOT, "ai", "inference"))
# pyrefly: ignore [missing-import]
from decay_detector import MonumentDecayDetector
# pyrefly: ignore [missing-import]
from preservation_recommender import format_markdown_report

class DetectorService:
    _instance = None
    _detector = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print(f"[DetectorService] Initializing MonumentDecayDetector with checkpoint: {settings.CHECKPOINT_PATH}")
        self._detector = MonumentDecayDetector(
            checkpoint_path=settings.CHECKPOINT_PATH,
            score_thresh=settings.DEFAULT_SCORE_THRESH,
            nms_thresh=settings.DEFAULT_NMS_THRESH
        )
        print(f"[DetectorService] Model loaded successfully on device: {self._detector.device}")

    @property
    def detector(self) -> MonumentDecayDetector:
        return self._detector

    def analyze_image_bytes(self, image_bytes: bytes, filename: str,
                            score_thresh: float = None,
                            nms_thresh: float = None,
                            enable_enhancement: bool = True):
        """
        Runs complete analysis on raw image bytes.
        """
        # Set dynamic thresholds if provided
        prev_score = self._detector.score_thresh
        prev_nms = self._detector.nms_thresh
        if score_thresh is not None:
            self._detector.score_thresh = score_thresh
        if nms_thresh is not None:
            self._detector.nms_thresh = nms_thresh

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = self._detector.predict(
                pil_img,
                apply_enhancement=enable_enhancement,
                crack_boost_thresh=settings.CRACK_BOOST_THRESH
            )
            result["image_name"] = filename
            return result
        finally:
            self._detector.score_thresh = prev_score
            self._detector.nms_thresh = prev_nms

    def get_annotated_image_bytes(self, result: dict, format: str = "JPEG") -> bytes:
        """
        Converts the annotated PIL image into raw bytes for streaming.
        """
        annotated_pil = result["annotated_image"]
        buf = io.BytesIO()
        annotated_pil.save(buf, format=format, quality=92)
        buf.seek(0)
        return buf.getvalue()

    def get_markdown_report(self, result: dict) -> str:
        """
        Generates markdown conservation protocol text.
        """
        return format_markdown_report(result["preservation_plan"])

    def get_status(self) -> dict:
        return {
            "status": "ready",
            "model_architecture": "SSDLite320_MobileNet_V3_Large_Weighted",
            "device": str(self._detector.device),
            "checkpoint_path": self._detector.checkpoint_path,
            "default_score_thresh": self._detector.score_thresh,
            "default_nms_thresh": self._detector.nms_thresh
        }

def get_detector_service() -> DetectorService:
    return DetectorService.get_instance()
