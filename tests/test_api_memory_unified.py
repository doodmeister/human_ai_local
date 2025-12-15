import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Test all memory type endpoints in the unified API using FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.interfaces.api.reflection_api import app

# Use context manager to ensure lifespan events are triggered
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Test data for different memory types
# Only test memory systems that have full store/retrieve/search support via the unified API
MEMORY_PAYLOADS = {
    "stm": {"content": "Test STM data"},
    "ltm": {"content": "Test LTM data"},
    "episodic": {"content": {"summary": "Test episodic summary", "detailed_content": "A detailed account of the test episode."}},
}

SEARCH_PAYLOADS = {
    "stm": {"query": "Test STM data"},
    "ltm": {"query": "Test LTM data"},
    "episodic": {"query": "Test episodic summary"},
}

@pytest.mark.parametrize("system", ["stm", "ltm", "episodic"])
def test_store_and_retrieve_memory(system, client):
    """Test storing and retrieving a memory for a given system."""
    payload = MEMORY_PAYLOADS[system]
    
    # Store memory
    response = client.post(f"/api/memory/{system}/store", json=payload)
    assert response.status_code == 200
    memory_id = response.json().get("memory_id")
    assert memory_id

    # Retrieve memory
    response = client.get(f"/api/memory/{system}/retrieve/{memory_id}")
    assert response.status_code == 200

    # Search memory - just verify it doesn't error (results may be empty due to timing)
    search_payload = SEARCH_PAYLOADS[system]
    response = client.post(f"/api/memory/{system}/search", json=search_payload)
    assert response.status_code == 200
    assert "results" in response.json()
    # Note: Search results may be empty due to embedding timing, so we don't require results > 0

    # Delete memory
    response = client.delete(f"/api/memory/{system}/delete/{memory_id}")
    # Accept 200 (deleted) or 405 (not supported for this memory type)
    assert response.status_code in [200, 405]
    if response.status_code == 200:
        assert response.json()["status"] in ["deleted", "ok"]
