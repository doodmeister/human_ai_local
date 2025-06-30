"""
Test cases for the metacognitive reflection API endpoints (FastAPI)
"""
import pytest
from fastapi.testclient import TestClient
from src.interfaces.api.reflection_api import app

client = TestClient(app)

def test_reflect():
    response = client.post("/reflect")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "report" in data
    assert "timestamp" in data["report"]

def test_reflection_status():
    response = client.get("/reflection/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_reflection_report():
    # Should succeed after /reflect
    client.post("/reflect")
    response = client.get("/reflection/report")
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "timestamp" in data["report"]

def test_reflection_report_404():
    # Reset state via test endpoint
    client.post("/test/reset")
    response = client.get("/reflection/report")
    assert response.status_code == 404

def test_reflection_start_stop():
    # Start scheduler
    response = client.post("/reflection/start", json={"interval": 1})
    assert response.status_code == 200
    assert response.json()["status"] == "started"
    # Stop scheduler
    response = client.post("/reflection/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
