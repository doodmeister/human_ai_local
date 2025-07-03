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

client = TestClient(app)

# Test data for different memory types
MEMORY_PAYLOADS = {
    "stm": {"content": "Test STM data"},
    "ltm": {"content": "Test LTM data"},
    "episodic": {"content": {"summary": "Test episodic summary", "detailed_content": "A detailed account of the test episode."}},
    "procedural": {"content": {"description": "Test procedural description", "steps": ["step 1", "step 2"]}},
    "prospective": {"content": {"description": "Test prospective reminder", "due_time": "2024-08-01T12:00:00"}},
    "semantic": {"content": {"subject": "test_subject", "predicate": "is_a", "object_val": "test_object"}}
}

SEARCH_PAYLOADS = {
    "stm": {"query": "Test STM data"},
    "ltm": {"query": "Test LTM data"},
    "episodic": {"query": "Test episodic summary"},
    "procedural": {"query": "Test procedural description"},
    "prospective": {"query": "Test prospective reminder"},  # This will list all reminders
    "semantic": {"query": {"subject": "test_subject"}}
}

@pytest.mark.parametrize("system", ["stm", "ltm", "episodic", "procedural", "prospective", "semantic"])
def test_store_and_retrieve_memory(system):
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

    # Search memory
    search_payload = SEARCH_PAYLOADS[system]
    response = client.post(f"/api/memory/{system}/search", json=search_payload)
    assert response.status_code == 200
    assert "results" in response.json()
    # Ensure search returns at least one result
    assert len(response.json()["results"]) > 0

    # Delete memory
    response = client.delete(f"/api/memory/{system}/delete/{memory_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
