"""
George API v2 - Extended for redesigned dashboard with WebSocket streaming
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from pathlib import Path
import sys
import asyncio

# -----------------------------------------
# Path setup & FastAPI initialization
# -----------------------------------------
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

app = FastAPI(title="George Cognitive API v2", version="2.0.0")

# -----------------------------------------
# Lazy Agent Initialization
# -----------------------------------------
_agent = None
_agent_initializing = False
_initialization_error = None

def get_agent():
    """Lazy agent initialization"""
    global _agent, _agent_initializing, _initialization_error
    
    if _agent is not None:
        return _agent
    if _agent_initializing:
        raise HTTPException(status_code=503, detail="Agent is initializing...")
    if _initialization_error:
        raise HTTPException(status_code=500, detail=f"Agent init failed: {_initialization_error}")
    
    try:
        _agent_initializing = True
        print("🧠 Initializing George cognitive agent...")
        from src.core.agent_singleton import create_agent
        _agent = create_agent()
        print("✅ Agent initialized")
        _agent_initializing = False
        return _agent
    except Exception as e:
        _agent_initializing = False
        _initialization_error = str(e)
        print(f"❌ Agent initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------
# Health & Init Status
# -----------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "George Cognitive API v2"}

@app.get("/api/agent/init-status")
async def initialization_status():
    global _agent, _agent_initializing, _initialization_error
    if _agent is not None:
        return {"status": "ready", "message": "Agent initialized"}
    elif _agent_initializing:
        return {"status": "initializing", "message": "Agent initializing..."}
    elif _initialization_error:
        return {"status": "error", "message": f"Init failed: {_initialization_error}"}
    else:
        return {"status": "not_started", "message": "Initialization not started"}

# -----------------------------------------
# Models
# -----------------------------------------
class ProcessInputRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str

# -----------------------------------------
# Core Agent Endpoints
# -----------------------------------------
@app.get("/api/agent/status")
async def agent_status():
    try:
        agent = get_agent()
        return agent.get_cognitive_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.post("/api/agent/process")
async def process_input(request: ProcessInputRequest):
    try:
        agent = get_agent()
        response = await agent.process_input(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/agent/chat")
async def chat(request: ChatRequest):
    try:
        agent = get_agent()
        response = await agent.process_input(request.message)
        return {
            "response": response,
            "memory_context": [],
            "cognitive_state": {},
            "rationale": "Stub rationale for now"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# -----------------------------------------
# WebSocket for Streaming Updates
# -----------------------------------------
@app.websocket("/ws/updates")
async def websocket_updates(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                agent = get_agent()
                update_data = {
                    "agent_status": agent.get_cognitive_status(),
                    "dpad_status": getattr(agent, "get_dpad_status", lambda: {"frontier": []})(),
                    "memory_analytics": getattr(agent, "get_memory_analytics", lambda: {"ltm_salience_recency": []})(),
                    "human_likeness": getattr(agent, "get_human_likeness_metrics", lambda: {
                        "memory_fidelity": 0.0,
                        "attentional_adaptation": 0.0,
                        "consolidation_precision": 0.0
                    })()
                }
            except Exception as e:
                update_data = {"error": str(e)}
            await ws.send_json(update_data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

# -----------------------------------------
# DPAD Endpoints
# -----------------------------------------
@app.post("/api/dpad/train")
async def dpad_train(request: dict):
    agent = get_agent()
    action = request.get("action")
    # Use agent's DPAD control methods if available, else stub
    if action == "start":
        start_fn = getattr(agent, "start_dpad_training", None)
        if callable(start_fn):
            start_fn()
            return {"status": "training_started"}
        # fallback: try attribute access if method not found
        dpad = getattr(agent, "dpad", None)
        if dpad and hasattr(dpad, "start_training"):
            dpad.start_training()
            return {"status": "training_started"}
    elif action == "stop":
        stop_fn = getattr(agent, "stop_dpad_training", None)
        if callable(stop_fn):
            stop_fn()
            return {"status": "training_stopped"}
        dpad = getattr(agent, "dpad", None)
        if dpad and hasattr(dpad, "stop_training"):
            dpad.stop_training()
            return {"status": "training_stopped"}
    return {"status": f"Stub: DPAD training {action}"}

@app.post("/api/dpad/nonlinearity")
async def dpad_nonlinearity(request: dict):
    elements = request.get("elements", [])
    agent = get_agent()
    dpad = getattr(agent, "dpad", None)
    if dpad and hasattr(dpad, "set_active_nonlinearities"):
        dpad.set_active_nonlinearities(elements)
    return {"status": "ok", "applied": elements}

@app.get("/api/dpad/status")
async def dpad_status():
    return getattr(get_agent(), "get_dpad_status", lambda: {"frontier": []})()

# -----------------------------------------
# Memory Analytics
# -----------------------------------------
@app.get("/api/memory/analytics")
async def memory_analytics():
    return getattr(get_agent(), "get_memory_analytics", lambda: {"ltm_salience_recency": []})()

@app.post("/api/memory/search")
async def memory_search(request: dict):
    query = request.get("query", "")
    return getattr(get_agent().memory, "search", lambda q: [])(query)

# -----------------------------------------
# Dream State
# -----------------------------------------
@app.post("/api/memory/consolidate")
async def trigger_consolidation():
    return {"status": "ok", "events_generated": 0}

@app.get("/api/memory/consolidation-events")
async def consolidation_events():
    return {"events": ["Stub event: consolidation not yet implemented"]}

# -----------------------------------------
# Meta-Cognition Bias
# -----------------------------------------
@app.post("/api/metacognition/bias")
async def set_bias(request: dict):
    return {"status": "ok", "applied": request}

# -----------------------------------------
# Planner Graph
# -----------------------------------------
@app.get("/api/planner/graph")
async def planner_graph():
    return {
        "nodes": [{"id": "g1", "label": "Stub Goal"}],
        "edges": []
    }

# -----------------------------------------
# Human-Likeness Metrics
# -----------------------------------------
@app.get("/api/analytics/human-likeness")
async def human_likeness():
    return getattr(get_agent(), "get_human_likeness_metrics", lambda: {
        "memory_fidelity": 0.0,
        "attentional_adaptation": 0.0,
        "consolidation_precision": 0.0
    })()

# -----------------------------------------
# Startup
# -----------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting George Cognitive API v2...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
