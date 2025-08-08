"""
REST API for Human-AI Cognition Framework: Metacognitive Reflection & Reporting

Endpoints:
- POST /reflect: Trigger agent-level reflection, return report
- GET /reflection/status: Get current reflection scheduler status (running/stopped)
- GET /reflection/report: Get last reflection report (404 if none)
- POST /reflection/start: Start periodic reflection (optional interval_minutes)
- POST /reflection/stop: Stop periodic reflection

Usage:
- Start server: 
    c:/dev/human_ai_local/venv/Scripts/python.exe -m uvicorn src.interfaces.api.reflection_api:app --reload --port 8000
- Example curl:
    curl -X POST http://localhost:8000/reflect
    curl http://localhost:8000/reflection/status
    curl http://localhost:8000/reflection/report
    curl -X POST http://localhost:8000/reflection/start -H "Content-Type: application/json" -d '{"interval": 5}'
    curl -X POST http://localhost:8000/reflection/stop

Returns:
- JSON with status, report, or error details.

Requirements: fastapi, uvicorn
"""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import threading
from contextlib import asynccontextmanager

# Import the cognitive agent and reflection logic

from src.core.agent_singleton import create_agent


# Import routers instead of FastAPI app for unified API registration
from src.interfaces.api.memory_api import router as memory_router
from src.interfaces.api.procedural_api import router as procedural_router
from src.interfaces.api.prospective_api import router as prospective_router
from src.interfaces.api.semantic_api import router as semantic_router
from src.interfaces.api.agent_api import router as agent_router
from src.interfaces.api.executive_api import router as executive_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager to create and shut down the agent.
    """
    print("Application startup: Initializing CognitiveAgent...")
    app.state.agent = create_agent()
    yield
    # Clean up the agent and its resources
    print("Application shutdown: Cleaning up CognitiveAgent...")
    # If the agent has a cleanup method, call it here.
    # For example: app.state.agent.shutdown()
    app.state.agent = None


app = FastAPI(lifespan=lifespan)

# Health check endpoint for launcher verification
@app.get("/health")
async def health_check():
    """Health check endpoint for verifying server is running."""
    return {"status": "healthy", "service": "George Cognitive API"}

# Register all routers under /api
app.include_router(memory_router, prefix="/api")
app.include_router(procedural_router, prefix="/api")
app.include_router(prospective_router, prefix="/api")
app.include_router(semantic_router, prefix="/api")
app.include_router(agent_router, prefix="/api/agent")
app.include_router(executive_router, prefix="/api/executive")

# Reflection state (for status/report endpoints)
reflection_lock = threading.Lock()
last_reflection_report = None

class ReflectionStartRequest(BaseModel):
    interval: Optional[int] = None  # seconds

@app.post("/reflect")
def reflect(request: Request):
    """Trigger agent-level metacognitive reflection and return the report."""
    global last_reflection_report
    agent = request.app.state.agent
    with reflection_lock:
        report = agent.reflect()
        last_reflection_report = report
    return {"status": "ok", "report": report}

@app.get("/reflection/status")
def reflection_status(request: Request):
    agent = request.app.state.agent
    status = agent.get_reflection_status()
    return {"status": status}

@app.get("/reflection/report")
def reflection_report():
    # Defensive: treat None or empty dict as no report (for test isolation)
    import sys
    this_module = sys.modules[__name__]
    report = getattr(this_module, "last_reflection_report", None)
    if report is None or report == {}:
        raise HTTPException(status_code=404, detail="No reflection report available.")
    return {"report": report}

@app.post("/reflection/start")
def reflection_start(req: ReflectionStartRequest, request: Request):
    agent = request.app.state.agent
    interval = req.interval
    if interval is not None:
        agent.start_reflection_scheduler(interval_minutes=interval)
    else:
        agent.start_reflection_scheduler()
    return {"status": "started", "interval": interval}

@app.post("/reflection/stop")
def reflection_stop(request: Request):
    agent = request.app.state.agent
    agent.stop_reflection_scheduler()
    return {"status": "stopped"}

# TEST-ONLY: Reset reflection state for test isolation
@app.post("/test/reset")
def test_reset():
    global last_reflection_report
    last_reflection_report = None
    return {"status": "reset"}
