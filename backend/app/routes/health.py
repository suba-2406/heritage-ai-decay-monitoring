"""
Health Check and System Metadata Routes
"""

from fastapi import APIRouter, Depends
from backend.app.services.detector_service import DetectorService, get_detector_service
from backend.app.config import settings

router = APIRouter(prefix="/health", tags=["Health & Status"])

@router.get("", summary="System Health and Model Status")
async def health_check(service: DetectorService = Depends(get_detector_service)):
    """
    Returns system health, model readiness, device placement, and configuration parameters.
    """
    model_status = service.get_status()
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": model_status
    }
