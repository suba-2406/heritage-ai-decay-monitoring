"""
Heritage AI Decay Monitoring - Backend Configuration Settings
"""

import os
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

class Settings(BaseModel):
    PROJECT_NAME: str = "Heritage AI - Ancient Monument Decay Monitoring & Preservation System"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_ROOT: str = PROJECT_ROOT
    DESCRIPTION: str = (
        "REST API service powered by PyTorch and SSDLite320 for automated stone decay detection, "
        "structural risk assessment, and ASI preservation protocol recommendations for ancient Indian monuments."
    )
    
    # Model configuration
    CHECKPOINT_PATH: str = os.path.join(
        PROJECT_ROOT, "ai", "training", "checkpoints", "ssd_monument_decay_phase3b_best.pth"
    )
    CONFIG_JSON_PATH: str = os.path.join(
        PROJECT_ROOT, "ai", "training", "checkpoints", "phase3b_training_config.json"
    )
    
    # Detection defaults
    DEFAULT_SCORE_THRESH: float = 0.25
    DEFAULT_NMS_THRESH: float = 0.45
    CRACK_BOOST_THRESH: float = 0.18
    ENABLE_PREPROCESSING: bool = True
    
    # Data paths
    MANIFEST_PATH: str = os.path.join(
        PROJECT_ROOT, "ai", "dataset", "processed", "dataset_manifest.json"
    )
    STATS_PATH: str = os.path.join(
        PROJECT_ROOT, "ai", "dataset", "processed", "dataset_stats.json"
    )
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]

settings = Settings()
