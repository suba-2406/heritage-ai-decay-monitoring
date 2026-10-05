# Heritage AI Decay Monitoring — FastAPI Backend Service

This directory contains the production-grade **FastAPI REST API service** for automated ancient monument decay monitoring, structural risk assessment, and preservation recommendation generation.

---

## 1. Quickstart — Running the Server

Activate the virtual environment and start the Uvicorn server:

```powershell
# From the repository root
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
```

Once running, access the interactive documentation:

* **Interactive Swagger UI:** [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8008/redoc](http://127.0.0.1:8008/redoc)
* **OpenAPI JSON Schema:** [http://127.0.0.1:8008/openapi.json](http://127.0.0.1:8008/openapi.json)

---

## 2. API Endpoints Reference

### Health & System Status

* `GET /api/v1/health`  
  Returns model status, architecture version, device (CPU/GPU), and checkpoint path.

### Decay Detection & Preservation (Primary Workflow)

* `POST /api/v1/analyze`  
  **Payload:** `multipart/form-data` with `file: <image>`  
  **Query Parameters:**
  * `score_thresh` (float, default: `0.25`): Confidence score threshold
  * `nms_thresh` (float, default: `0.45`): Non-Maximum Suppression IoU threshold
  * `enable_enhancement` (bool, default: `true`): Runs median denoising and CLAHE
  
  **Response:** Complete structured JSON with:
  * Detected bounding boxes in native coordinates (`Crack`, `Moss`, `Seepage`)
  * Surface defect coverage percentage
  * Structural risk assessment (`LOW`, `MEDIUM`, or `HIGH`)
  * Step-by-step ASI conservation protocols and prohibited actions

* `POST /api/v1/analyze/visualize`  
  **Payload:** `multipart/form-data` with `file: <image>`  
  **Response:** Streams an annotated JPEG image with color-coded bounding boxes and top condition banner:
  * Red: Crack (High Risk)
  * Emerald: Moss (Medium Risk)
  * Blue: Seepage (High Risk)

* `POST /api/v1/analyze/report`  
  **Payload:** `multipart/form-data` with `file: <image>`  
  **Query Parameter:** `format` (`markdown` or `json`)  
  **Response:** Formatted printable conservation report ready for archaeological field teams.

### Monuments & Benchmarks

* `GET /api/v1/monuments`  
  Lists monitored temple complexes (Kasiviswanathar, Brihadisvara) and decay summaries.
* `GET /api/v1/monuments/overview`  
  Returns 4-class distribution and dataset split statistics.
* `GET /api/v1/monuments/benchmarks`  
  Returns comparative evaluation metrics (Phase 3 Baseline vs Phase 3B Improved).

---

## 3. Running Automated Tests

To run the complete automated test suite verifying all endpoints:

```powershell
python backend/tests/test_api.py
```

---

## 4. Architecture Overview

```
backend/
├── app/
│   ├── main.py              # FastAPI application entrypoint, CORS, lifespan eager loading
│   ├── config.py            # Pydantic configuration settings
│   ├── schemas/
│   │   ├── detection.py     # Pydantic models for detections & preservation plans
│   │   └── monuments.py     # Pydantic models for monument sites & benchmarks
│   ├── services/
│   │   └── detector_service.py # Singleton wrapping MonumentDecayDetector
│   └── routes/
│       ├── health.py        # /health endpoint
│       ├── analysis.py      # /analyze, /visualize, /report endpoints
│       └── monuments.py     # /monuments, /overview, /benchmarks endpoints
└── tests/
    └── test_api.py          # Automated test suite
```
