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
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import threading

# Import the cognitive agent and reflection logic
from src.core.cognitive_agent import CognitiveAgent

app = FastAPI()
agent = CognitiveAgent()

# Reflection state (for status/report endpoints)
reflection_lock = threading.Lock()
last_reflection_report = None

class ReflectionStartRequest(BaseModel):
    interval: Optional[int] = None  # seconds

@app.post("/reflect")
def reflect():
    """Trigger agent-level metacognitive reflection and return the report."""
    global last_reflection_report
    with reflection_lock:
        report = agent.reflect()
        last_reflection_report = report
    return {"status": "ok", "report": report}

@app.get("/reflection/status")
def reflection_status():
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
def reflection_start(req: ReflectionStartRequest):
    interval = req.interval
    if interval is not None:
        agent.start_reflection_scheduler(interval_minutes=interval)
    else:
        agent.start_reflection_scheduler()
    return {"status": "started", "interval": interval}

@app.post("/reflection/stop")
def reflection_stop():
    agent.stop_reflection_scheduler()
    return {"status": "stopped"}

# TEST-ONLY: Reset reflection state for test isolation
@app.post("/test/reset")
def test_reset():
    global last_reflection_report
    last_reflection_report = None
    return {"status": "reset"}
