"""
Simplified George API for testing - delayed agent initialization
"""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create FastAPI app without lifespan for now
app = FastAPI(title="George Cognitive API", version="1.0.0")

# Global agent variable
_agent = None
_agent_initializing = False
_initialization_error = None

def get_agent():
    """Lazy agent initialization"""
    global _agent, _agent_initializing, _initialization_error
    
    if _agent is not None:
        return _agent
    
    if _agent_initializing:
        raise HTTPException(status_code=503, detail="Agent is currently initializing, please wait...")
    
    if _initialization_error:
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {_initialization_error}")
    
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
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for verifying server is running."""
    return {"status": "healthy", "service": "George Cognitive API"}

# Initialization status endpoint
@app.get("/api/agent/init-status")
async def initialization_status():
    """Check agent initialization status"""
    global _agent, _agent_initializing, _initialization_error
    
    if _agent is not None:
        return {"status": "ready", "message": "Agent is fully initialized"}
    elif _agent_initializing:
        return {"status": "initializing", "message": "Agent is currently initializing..."}
    elif _initialization_error:
        return {"status": "error", "message": f"Initialization failed: {_initialization_error}"}
    else:
        return {"status": "not_started", "message": "Agent initialization has not started"}

# Agent status endpoint
@app.get("/api/agent/status")
async def agent_status():
    """Get agent status"""
    try:
        agent = get_agent()
        status = agent.get_cognitive_status()
        return {"status": "ok", "cognitive_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

class ProcessInputRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str

@app.post("/api/agent/process")
async def process_input(request: ProcessInputRequest):
    """Process user input through the cognitive agent"""
    try:
        agent = get_agent()
        response = await agent.process_input(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/agent/chat")
async def chat(request: ChatRequest):
    """Chat endpoint for Streamlit interface"""
    try:
        agent = get_agent()
        response = await agent.process_input(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Memory endpoints
@app.get("/api/agent/memory/list/{system}")
async def list_memories(system: str):
    """List memories from STM or LTM"""
    try:
        agent = get_agent()
        if system == "stm":
            memories = agent.memory.stm.get_all_memories()
            return {"memories": [item.__dict__ for item in memories]}
        elif system == "ltm":
            ltm_collection = agent.memory.ltm.collection
            if ltm_collection:
                result = ltm_collection.get()
                memories = []
                if result.get('ids'):
                    for i, memory_id in enumerate(result['ids']):
                        memory_dict = {
                            'id': memory_id,
                            'content': result.get('documents', [])[i] if i < len(result.get('documents', [])) else '',
                            'metadata': result.get('metadatas', [])[i] if i < len(result.get('metadatas', [])) else {}
                        }
                        memories.append(memory_dict)
                return {"memories": memories}
            return {"memories": []}
        else:
            raise HTTPException(status_code=400, detail="Invalid system. Use 'stm' or 'ltm'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory error: {str(e)}")

# Executive endpoints
@app.get("/api/executive/status")
async def executive_status():
    """Get executive function status"""
    try:
        agent = get_agent()
        # Get basic executive status - safely check for executive functions
        if hasattr(agent, 'executive') and getattr(agent, 'executive', None):
            executive = getattr(agent, 'executive')
            return {
                "status": "active",
                "active_goals": getattr(executive, 'goals', []),
                "active_tasks": getattr(executive, 'tasks', []),
                "decision_count": getattr(executive, 'decision_count', 0)
            }
        else:
            return {
                "status": "inactive", 
                "active_goals": [],
                "active_tasks": [],
                "decision_count": 0,
                "message": "Executive functions not initialized"
            }
    except Exception as e:
        return {
            "status": "error",
            "active_goals": [],
            "active_tasks": [],
            "decision_count": 0,
            "error": str(e)
        }

# Prospective Memory endpoints
@app.get("/api/agent/memory/prospective/list")
async def list_prospective_memories():
    """List prospective memories (reminders/intentions)"""
    try:
        agent = get_agent()
        if hasattr(agent.memory, 'prospective') and agent.memory.prospective:
            reminders = agent.memory.prospective.list_reminders(include_completed=False)
            return {"reminders": reminders}
        return {"reminders": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prospective memory error: {str(e)}")

@app.post("/api/agent/memory/prospective/create")
async def create_prospective_memory(request: dict):
    """Create a new prospective memory"""
    try:
        agent = get_agent()
        if hasattr(agent.memory, 'prospective') and agent.memory.prospective:
            reminder_id = agent.memory.prospective.store(
                description=request.get('description', ''),
                trigger_time=request.get('trigger_time'),
                tags=request.get('tags', [])
            )
            return {"reminder_id": reminder_id, "status": "created"}
        return {"error": "Prospective memory not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prospective memory creation error: {str(e)}")

# Procedural Memory endpoints
@app.get("/api/agent/memory/procedural/list")
async def list_procedural_memories():
    """List procedural memories (procedures/skills)"""
    try:
        agent = get_agent()
        if hasattr(agent.memory, 'procedural') and agent.memory.procedural:
            # Get procedures from both STM and LTM
            procedures = []
            # This is a simplified version - in practice you'd implement a proper search
            return {"procedures": procedures}
        return {"procedures": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Procedural memory error: {str(e)}")

@app.post("/api/agent/memory/procedural/create")
async def create_procedural_memory(request: dict):
    """Create a new procedural memory"""
    try:
        agent = get_agent()
        if hasattr(agent.memory, 'procedural') and agent.memory.procedural:
            proc_id = agent.memory.procedural.store(
                description=request.get('description', ''),
                steps=request.get('steps', []),
                tags=request.get('tags', []),
                memory_type=request.get('memory_type', 'stm'),
                importance=request.get('importance', 0.5)
            )
            return {"procedure_id": proc_id, "status": "created"}
        return {"error": "Procedural memory not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Procedural memory creation error: {str(e)}")

# Neural Activity endpoints
@app.get("/api/neural/status")
async def neural_status():
    """Get neural network status and activity"""
    try:
        agent = get_agent()
        # Check for neural components
        neural_status = {
            "dpad_available": False,
            "lshn_available": False,
            "neural_activity": {
                "attention_enhancement": 0.0,
                "pattern_completion": 0.0,
                "consolidation_activity": 0.0
            },
            "performance_metrics": {
                "response_time": 0.0,
                "accuracy": 0.0,
                "efficiency": 0.0
            }
        }
        
        # Check if neural networks are available
        if hasattr(agent, 'neural_integration'):
            neural_status["dpad_available"] = True
            neural_status["lshn_available"] = True
            
        return neural_status
    except Exception as e:
        return {"error": str(e), "neural_activity": {}, "performance_metrics": {}}

# Performance Analytics endpoint
@app.get("/api/analytics/performance")
async def performance_analytics():
    """Get comprehensive performance analytics"""
    try:
        agent = get_agent()
        status = agent.get_cognitive_status()
        
        analytics = {
            "cognitive_efficiency": {
                "overall": status.get("cognitive_integration", {}).get("overall_efficiency", 0.0),
                "memory": status.get("memory_status", {}).get("system_active", False),
                "attention": status.get("attention_status", {}).get("capacity_utilization", 0.0),
                "processing": status.get("cognitive_integration", {}).get("processing_capacity", 0.0)
            },
            "usage_statistics": {
                "session_duration": status.get("memory_status", {}).get("uptime_seconds", 0),
                "interactions": status.get("conversation_length", 0),
                "memory_operations": status.get("memory_status", {}).get("operation_counts", {}),
                "error_rate": 0.0
            },
            "trends": {
                "cognitive_load_trend": "stable",
                "memory_usage_trend": "increasing",
                "performance_trend": "improving"
            }
        }
        
        return analytics
    except Exception as e:
        return {"error": str(e), "cognitive_efficiency": {}, "usage_statistics": {}, "trends": {}}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting George Cognitive API...")
    print("📍 Health check: http://localhost:8000/health")
    print("📍 Agent status: http://localhost:8000/api/agent/status")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
