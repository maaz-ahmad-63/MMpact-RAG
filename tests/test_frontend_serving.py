"""Tests for production frontend serving and API isolation in app.main."""

import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app, FRONTEND_DIST_DIR, FRONTEND_INDEX_FILE, FRONTEND_ASSETS_DIR


@pytest.fixture
def client():
    """Test client without running background lifespan workers."""
    return TestClient(app, raise_server_exceptions=False)


def test_health_check_returns_expected_json(client):
    """GET /health must return the expected status and service JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert data == {
        "status": "healthy",
        "service": "MMpact RAG Service",
    }


def test_root_serves_frontend_index_when_built(client):
    """GET / serves index.html when the frontend build exists."""
    if FRONTEND_INDEX_FILE.is_file():
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<div id=\"root\">" in response.text or "<html" in response.text
    else:
        # If build is missing, it returns fallback JSON
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "MMpact RAG Service",
        }


def test_root_fallback_when_frontend_missing(client, monkeypatch):
    """GET / gracefully falls back to JSON when index.html does not exist."""
    fake_path = Path("/nonexistent/path/to/dist/index.html")
    with patch("app.main.FRONTEND_INDEX_FILE", fake_path):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "status": "healthy",
            "service": "MMpact RAG Service",
        }


def test_built_frontend_assets_served(client):
    """Assets inside frontend/dist/assets/ and static SVGs are served correctly."""
    if FRONTEND_ASSETS_DIR.is_dir():
        asset_files = list(FRONTEND_ASSETS_DIR.iterdir())
        if asset_files:
            sample_asset = asset_files[0].name
            response = client.get(f"/assets/{sample_asset}")
            assert response.status_code == 200

    favicon_path = FRONTEND_DIST_DIR / "favicon.svg"
    if favicon_path.is_file():
        response = client.get("/favicon.svg")
        assert response.status_code == 200
        assert "image/svg+xml" in response.headers["content-type"]

    icons_path = FRONTEND_DIST_DIR / "icons.svg"
    if icons_path.is_file():
        response = client.get("/icons.svg")
        assert response.status_code == 200
        assert "image/svg+xml" in response.headers["content-type"]


def test_existing_api_routes_functional(client):
    """Existing API routes remain functional and take precedence over frontend."""
    # /api/rag/health endpoint
    response = client.get("/api/rag/health")
    assert response.status_code in (200, 503)  # 200 or 503 depending on vector store availability
    assert response.headers["content-type"].startswith("application/json")


def test_unknown_api_endpoints_return_404_json(client):
    """Unknown /api/* endpoints must return normal 404 JSON, NEVER index.html."""
    unknown_api_paths = [
        "/api/nonexistent",
        "/api/rag/does-not-exist",
        "/api/v1/invalid/endpoint",
        "/api/",
    ]
    for path in unknown_api_paths:
        response = client.get(path)
        assert response.status_code == 404, f"Path {path} did not return 404"
        assert response.headers["content-type"].startswith("application/json"), f"Path {path} returned non-JSON"
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Not Found"


def test_frontend_fallback_does_not_override_docs_and_openapi(client):
    """OpenAPI and Swagger docs must not be overridden by the SPA fallback."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    assert "Swagger UI" in docs_resp.text or "swagger" in docs_resp.text.lower()

    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    assert openapi_resp.headers["content-type"].startswith("application/json")
    openapi_data = openapi_resp.json()
    assert "openapi" in openapi_data or "paths" in openapi_data


def test_spa_client_side_routing(client):
    """Non-API frontend routes return index.html for client-side routing."""
    if FRONTEND_INDEX_FILE.is_file():
        spa_routes = ["/documents", "/search", "/analytics", "/settings/system"]
        for route in spa_routes:
            response = client.get(route)
            assert response.status_code == 200, f"Route {route} did not return 200"
            assert "text/html" in response.headers["content-type"], f"Route {route} did not return HTML"
            assert "<div id=\"root\">" in response.text or "<html" in response.text
