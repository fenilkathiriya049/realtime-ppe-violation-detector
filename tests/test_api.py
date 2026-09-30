# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_provider" in data

def test_audit_json_without_file():
    response = client.post("/api/v1/audit/json")
    assert response.status_code == 422  # Missing required multipart file