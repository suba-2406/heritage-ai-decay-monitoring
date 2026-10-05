"""
Heritage AI Decay Monitoring and Preservation Recommendation System
FastAPI Backend Application Entrypoint
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Ensure root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai", "inference"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai", "preprocessing"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai", "training"))

from backend.app.config import settings
from backend.app.services.detector_service import DetectorService
from backend.app.routes import health_router, analysis_router, monuments_router

SWAGGER_DESCRIPTION = """
### Research Context & Scope
This API powers the **Heritage AI Decay Monitoring & Preservation Recommendation System** (*Senmozhi Project, Work II*).
It automates the detection and structural monitoring of stone degradation across ancient Indian temples (such as the **Brihadisvara Temple** and **Kasiviswanathar Temple** in Thanjavur).

### Core Capabilities
* **Automated Defect Detection:** SSDLite320 + MobileNetV3-Large deep learning model for 4 standardized classes:
  * `0: Normal` (Baseline undamaged masonry)
  * `1: Crack` (Structural fissure & fracture)
  * `2: Moss` (Biological colonization & lichen)
  * `3: Seepage` (Moisture infiltration & damp staining)
* **Image Preprocessing:** Median noise reduction and CLAHE (Contrast Limited Adaptive Histogram Equalization) to accentuate micro-fractures in granite stone.
* **Surface Coverage & Risk Assessment:** Computes defect percentage coverage and classifies monument stability into **LOW**, **MEDIUM**, or **HIGH** risk levels.
* **Preservation Prescriptions:** Generates domain-specific conservation protocols mapped to **Archaeological Survey of India (ASI)** and **ICOMOS** guidelines.

### Interactive Documentation
Explore endpoints below. You can directly upload monument photos via the `/api/v1/analyze` and `/api/v1/analyze/visualize` endpoints.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load model on startup to avoid cold start
    print("[Server Startup] Pre-loading MonumentDecayDetector...")
    DetectorService.get_instance()
    print("[Server Startup] Detector loaded and ready for inference.")
    yield
    print("[Server Shutdown] Cleaning up resources.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=SWAGGER_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Include API Routers under /api/v1
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis_router, prefix=settings.API_V1_PREFIX)
app.include_router(monuments_router, prefix=settings.API_V1_PREFIX)

# Mount Frontend Dashboard Static Files
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend", "src")
if os.path.exists(FRONTEND_DIR):
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/", include_in_schema=False)
async def root():
    """Redirects base root to interactive Web Dashboard."""
    return RedirectResponse(url="/dashboard/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8008, reload=True)
