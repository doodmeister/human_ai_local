import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from src.interfaces.api.reflection_api import app
from src.core.agent_singleton import create_agent

# Initialize the agent in app state for testing
@pytest.fixture(scope="module", autouse=True)
def setup_agent():
    """Set up the agent in the app state before tests run."""
    agent = create_agent()
    app.state.agent = agent
    yield

client = TestClient(app)

@pytest.mark.parametrize("memory_type", ["stm", "ltm"])
def test_store_and_retrieve_procedure(memory_type):
    # Store
    response = client.post("/api/procedure/store", json={
        "description": f"test-procedure-{memory_type}",
        "steps": ["step1", "step2"],
        "tags": ["test", memory_type],
        "memory_type": memory_type
    })
    assert response.status_code == 200
    proc_id = response.json()["procedure_id"]
    # Retrieve
    response = client.get(f"/api/procedure/retrieve/{proc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == f"test-procedure-{memory_type}"
    assert "step1" in data["steps"]

@pytest.mark.parametrize("memory_type", ["stm", "ltm"])
def test_search_procedure(memory_type):
    # Store
    client.post("/api/procedure/store", json={
        "description": f"searchable-proc-{memory_type}",
        "steps": ["search-step"],
        "tags": ["search", memory_type],
        "memory_type": memory_type
    })
    # Search
    response = client.post("/api/procedure/search", json={
        "query": f"searchable-proc-{memory_type}",
        "memory_type": memory_type
    })
    assert response.status_code == 200
    results = response.json()["results"]
    assert any(f"searchable-proc-{memory_type}" in str(r) for r in results)

def test_use_and_delete_procedure():
    # Store
    response = client.post("/api/procedure/store", json={
        "description": "use-proc",
        "steps": ["do this", "do that"],
        "tags": ["use"],
        "memory_type": "ltm"
    })
    proc_id = response.json()["procedure_id"]
    # Use
    response = client.post(f"/api/procedure/use/{proc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "use-proc"
    # Delete
    response = client.delete(f"/api/procedure/delete/{proc_id}")
    assert response.status_code == 200
    # Confirm deleted
    response = client.get(f"/api/procedure/retrieve/{proc_id}")
    assert response.status_code == 404

def test_clear_procedures():
    # Store a procedure
    client.post("/api/procedure/store", json={
        "description": "clear-proc",
        "steps": ["step"],
        "tags": ["clear"],
        "memory_type": "stm"
    })
    # Clear all STM procedures
    response = client.post("/api/procedure/clear", params={"memory_type": "stm"})
    assert response.status_code == 200
    # Search should return nothing
    response = client.post("/api/procedure/search", json={"query": "clear-proc", "memory_type": "stm"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert not results
