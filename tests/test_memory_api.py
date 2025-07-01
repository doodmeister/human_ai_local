import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Test cases for STM and LTM memory API endpoints (FastAPI)
"""
import pytest
from fastapi.testclient import TestClient
from src.interfaces.api.memory_api import app

client = TestClient(app)

@pytest.mark.parametrize("system", ["stm", "ltm"])
def test_store_and_retrieve_memory(system):
    # Store
    response = client.post(f"/memory/{system}/store", json={"content": f"test-{system}-memory"})
    assert response.status_code == 200
    memory_id = response.json()["memory_id"]
    # Retrieve
    response = client.get(f"/memory/{system}/retrieve/{memory_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == f"test-{system}-memory"

@pytest.mark.parametrize("system", ["stm", "ltm"])
def test_search_memory(system):
    # Store
    client.post(f"/memory/{system}/store", json={"content": f"searchable-{system}"})
    # Search
    response = client.post(f"/memory/{system}/search", json={"query": f"searchable-{system}"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert any(f"searchable-{system}" in str(r) for r in results)

@pytest.mark.parametrize("system", ["stm", "ltm"])
def test_delete_memory(system):
    # Store
    response = client.post(f"/memory/{system}/store", json={"content": f"delete-{system}"})
    memory_id = response.json()["memory_id"]
    # Delete
    response = client.delete(f"/memory/{system}/delete/{memory_id}")
    assert response.status_code == 200
    # Confirm deleted
    response = client.get(f"/memory/{system}/retrieve/{memory_id}")
    assert response.status_code == 404

def test_ltm_feedback():
    # Store in LTM
    response = client.post("/memory/ltm/store", json={"content": "feedback-ltm"})
    memory_id = response.json()["memory_id"]
    # Add feedback
    response = client.post(f"/memory/ltm/feedback/{memory_id}", json={"feedback_type": "relevance", "value": 5})
    if response.status_code == 200:
        assert response.json()["status"] == "feedback added"
    elif response.status_code == 501:
        detail = response.json().get("detail", "").lower()
        assert (
            "not implemented" in detail or "not supported" in detail
        ), f"Unexpected 501 detail: {detail}"
    else:
        assert False, f"Unexpected status code: {response.status_code}"
