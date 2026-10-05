"""
Decay Analysis, Detection, and Preservation Report Endpoints
"""

import os
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from backend.app.services.detector_service import DetectorService, get_detector_service
from backend.app.schemas.detection import AnalysisResponse
from backend.app.config import settings

router = APIRouter(prefix="/analyze", tags=["Decay Detection & Preservation"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def validate_image_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

@router.post("", response_model=AnalysisResponse, summary="Analyze Monument Image")
async def analyze_monument(
    file: UploadFile = File(..., description="Monument image file (JPEG, PNG)"),
    score_thresh: float = Query(
        default=settings.DEFAULT_SCORE_THRESH,
        ge=0.01,
        le=1.0,
        description="Confidence score threshold for defect detection"
    ),
    nms_thresh: float = Query(
        default=settings.DEFAULT_NMS_THRESH,
        ge=0.05,
        le=0.95,
        description="Non-Maximum Suppression (NMS) IoU threshold"
    ),
    enable_enhancement: bool = Query(
        default=settings.ENABLE_PREPROCESSING,
        description="Enable CLAHE and median denoising preprocessing"
    ),
    service: DetectorService = Depends(get_detector_service)
):
    """
    Performs complete automated inspection on an uploaded monument image:
    1. Preprocesses image (Median Denoising + CLAHE enhancement).
    2. Runs Phase 3B SSDLite320 detector to localize **Crack**, **Moss**, and **Seepage**.
    3. Calculates total defect surface coverage percentage.
    4. Stratifies structural risk into **LOW**, **MEDIUM**, or **HIGH**.
    5. Formulates actionable, step-by-step ASI conservation preservation prescriptions.
    """
    validate_image_file(file)
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = service.analyze_image_bytes(
        image_bytes=contents,
        filename=file.filename,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        enable_enhancement=enable_enhancement
    )

    return AnalysisResponse(
        image_name=result["image_name"],
        original_dimensions=result["original_dimensions"],
        total_detections=result["total_detections"],
        detections=result["detections"],
        risk_assessment=result["risk_assessment"],
        preservation_plan=result["preservation_plan"],
        preprocessing_info=result["preprocessing_info"]
    )

@router.post("/visualize", summary="Get Annotated Image with Bounding Boxes & Risk Banner")
async def visualize_monument(
    file: UploadFile = File(..., description="Monument image file"),
    score_thresh: float = Query(default=settings.DEFAULT_SCORE_THRESH, ge=0.01, le=1.0),
    nms_thresh: float = Query(default=settings.DEFAULT_NMS_THRESH, ge=0.05, le=0.95),
    enable_enhancement: bool = Query(default=settings.ENABLE_PREPROCESSING),
    service: DetectorService = Depends(get_detector_service)
):
    """
    Returns the processed image with visual color-coded bounding boxes and top condition banner:
    - Red boxes: Crack (High Risk)
    - Emerald boxes: Moss (Medium Risk)
    - Blue boxes: Seepage (High Risk)
    - Top banner: Monument risk category, coverage %, and defect summary
    """
    validate_image_file(file)
    contents = await file.read()
    
    result = service.analyze_image_bytes(
        image_bytes=contents,
        filename=file.filename,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        enable_enhancement=enable_enhancement
    )

    img_bytes = service.get_annotated_image_bytes(result, format="JPEG")
    base_name = os.path.splitext(file.filename)[0]

    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="{base_name}_annotated.jpg"'}
    )

@router.post("/report", summary="Generate Downloadable Conservation Protocol Report")
async def get_conservation_report(
    file: UploadFile = File(..., description="Monument image file"),
    format: str = Query(default="markdown", pattern="^(markdown|json)$", description="Report format (markdown or json)"),
    score_thresh: float = Query(default=settings.DEFAULT_SCORE_THRESH, ge=0.01, le=1.0),
    nms_thresh: float = Query(default=settings.DEFAULT_NMS_THRESH, ge=0.05, le=0.95),
    enable_enhancement: bool = Query(default=settings.ENABLE_PREPROCESSING),
    service: DetectorService = Depends(get_detector_service)
):
    """
    Generates a formal, printable conservation recommendation report following Archaeological
    Survey of India (ASI) manuals and ICOMOS standards.
    """
    validate_image_file(file)
    contents = await file.read()
    
    result = service.analyze_image_bytes(
        image_bytes=contents,
        filename=file.filename,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        enable_enhancement=enable_enhancement
    )

    base_name = os.path.splitext(file.filename)[0]

    if format == "markdown":
        md_text = service.get_markdown_report(result)
        return PlainTextResponse(
            content=md_text,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{base_name}_conservation_protocol.md"'}
        )
    else:
        return JSONResponse(
            content=result["preservation_plan"],
            headers={"Content-Disposition": f'attachment; filename="{base_name}_preservation_plan.json"'}
        )
