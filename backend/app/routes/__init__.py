from .health import router as health_router
from .analysis import router as analysis_router
from .monuments import router as monuments_router

__all__ = ["health_router", "analysis_router", "monuments_router"]
