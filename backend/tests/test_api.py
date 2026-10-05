"""
Automated Test Suite for Heritage AI FastAPI Backend Endpoints
"""

import os
import sys
import io
from PIL import Image
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, PROJECT_ROOT)
from backend.app.main import app

client = TestClient(app)

def create_dummy_image_bytes():
    img = Image.new("RGB", (320, 320), color=(128, 120, 110))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [302, 307]
    assert response.headers["location"] == "/dashboard/"

def test_dashboard_endpoint():
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "HERITAGE AI" in response.text

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert data["model"]["status"] == "ready"

def test_monuments_endpoints():
    # List sites
    res1 = client.get("/api/v1/monuments")
    assert res1.status_code == 200
    sites = res1.json()
    assert len(sites) >= 2
    assert any(s["id"] == "kasiviswanathar" for s in sites)

    # Overview
    res2 = client.get("/api/v1/monuments/overview")
    assert res2.status_code == 200
    overview = res2.json()
    assert "classes" in overview

    # Benchmarks
    res3 = client.get("/api/v1/monuments/benchmarks")
    assert res3.status_code == 200

def test_analyze_endpoint():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test_temple.jpg", img_bytes, "image/jpeg")},
        params={"score_thresh": 0.25, "nms_thresh": 0.45}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "risk_assessment" in data
    assert "preservation_plan" in data
    assert data["risk_assessment"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "coverage_percentage" in data["risk_assessment"]

def test_visualize_endpoint():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/analyze/visualize",
        files={"file": ("test_temple.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000

def test_report_endpoint_markdown():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/analyze/report?format=markdown",
        files={"file": ("test_temple.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "MONUMENT DECAY & PRESERVATION RECOMMENDATION REPORT" in response.text

def test_report_endpoint_json():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/analyze/report?format=json",
        files={"file": ("test_temple.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert "recommended_protocols" in data

def test_invalid_file_extension():
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("malicious.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

if __name__ == "__main__":
    print("Running API tests manually...")
    test_root_redirect()
    print("[PASS] test_root_redirect")
    test_dashboard_endpoint()
    print("[PASS] test_dashboard_endpoint")
    test_health_endpoint()
    print("[PASS] test_health_endpoint")
    test_monuments_endpoints()
    print("[PASS] test_monuments_endpoints")
    test_analyze_endpoint()
    print("[PASS] test_analyze_endpoint")
    test_visualize_endpoint()
    print("[PASS] test_visualize_endpoint")
    test_report_endpoint_markdown()
    print("[PASS] test_report_endpoint_markdown")
    test_report_endpoint_json()
    print("[PASS] test_report_endpoint_json")
    test_invalid_file_extension()
    print("[PASS] test_invalid_file_extension")
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
