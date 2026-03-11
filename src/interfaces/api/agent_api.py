from fastapi import APIRouter, Body, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os
from datetime import datetime

from src.interfaces.api.dependencies import get_request_agent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

router = APIRouter()


# Add /reflect endpoint for manual reflection trigger
@router.post("/reflect")
async def reflect(request: Request):
    """
    Trigger agent-level metacognitive reflection and return the report.
    """
    agent = get_request_agent(request)
    if not hasattr(agent, "reflect"):
        raise HTTPException(status_code=501, detail="Reflection not implemented in agent.")
    report = await agent.reflect() if callable(getattr(agent.reflect, "__await__", None)) else agent.reflect()
    if report is None or report == {}:
        report = {"status": "empty"}
    if not isinstance(report, dict):
        report = {"status": "ok", "value": str(report)}
    report.setdefault("timestamp", datetime.utcnow().isoformat())
    return {"status": "ok", "report": report}


class ProcessInputRequest(BaseModel):
    text: str

class MemorySearchRequest(BaseModel):
    query: str


@router.post("/process")
async def process_input(process_request: ProcessInputRequest, request: Request):
    """
    Processes user input through the cognitive agent and returns the response.
    """
    agent = get_request_agent(request)
    response = await agent.process_input(process_request.text)

    memory_context = await agent.retrieve_memory_context(process_request.text)

    return {"response": response, "memory_context": memory_context}



# New endpoint: search memory directly
@router.post("/memory/search")
async def memory_search(search_request: MemorySearchRequest, request: Request):
    """
    Search all memory systems for a query and return results (STM, LTM, Episodic, Semantic).
    """
    agent = get_request_agent(request)
    memory_context = await agent.retrieve_memory_context(search_request.query)
    return {"memory_context": memory_context}


# List all STM or LTM memories
@router.get("/memory/list/{system}")
async def list_memories(system: str, request: Request):
    """
    List all memories in STM or LTM.
    """
    agent = get_request_agent(request)
    if system == "stm":
        # Return all STM items using get_all_memories method
        stm_memories = agent.memory.stm.get_all_memories()
        return {"memories": [item.__dict__ for item in stm_memories]}
    elif system == "ltm":
        # Return all LTM records directly from collection
        try:
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
            else:
                return {"memories": []}
        except Exception as e:
            return {"error": f"Failed to retrieve LTM memories: {str(e)}"}
    else:
        return {"error": "Invalid system. Use 'stm' or 'ltm'."}


# Trigger a dream state cycle

class DreamRequest(BaseModel):
    cycle_type: Optional[str] = "deep"  # 'light', 'deep', or 'rem'

class ChatRequest(BaseModel):
    text: str
    include_reflection: Optional[bool] = False
    create_goal: Optional[bool] = False
    use_memory_context: Optional[bool] = True

class CognitiveBreakRequest(BaseModel):
    duration_minutes: Optional[float] = 1.0

@router.get("/status")
async def get_agent_status(request: Request):
    """
    Get comprehensive agent cognitive state including attention, memory, and performance metrics.
    """
    try:
        agent = get_request_agent(request)
        status = agent.get_cognitive_status()
        attention_status = status.get("attention_status", {})
        memory_status = status.get("memory_status", {})
        integration = status.get("cognitive_integration", {})

        return {
            "cognitive_mode": "FOCUSED",
            "attention_status": {
                "cognitive_load": attention_status.get("cognitive_load", 0.0),
                "fatigue_level": status.get("fatigue_level", 0.0),
                "current_focus": status.get("attention_focus", []),
                "available_capacity": attention_status.get("available_capacity", 0.0),
            },
            "memory_status": memory_status,
            "performance_metrics": {
                "response_time": None,
                "accuracy": None,
                "efficiency": integration.get("overall_efficiency", 0.0),
                "processing_capacity": integration.get("processing_capacity", 0.0),
            },
            "active_processes": len(status.get("active_goals", [])) if isinstance(status.get("active_goals"), list) else 0,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "error": f"Failed to get agent status: {str(e)}",
            "status": "error"
        }

@router.post("/chat")
async def enhanced_chat(chat_request: ChatRequest, request: Request):
    """
    Enhanced chat interface with full cognitive integration.
    """
    try:
        agent = get_request_agent(request)
        
        # Process input through cognitive agent
        response = await agent.process_input(chat_request.text)
        
        # Get memory context
        memory_context = []
        if chat_request.use_memory_context:
            try:
                memory_context = await agent.retrieve_memory_context(chat_request.text)
            except Exception:
                memory_context = []

        memory_events = ["interaction_processed"]
        status = agent.get_cognitive_status()
        cognitive_state = {
            "cognitive_load": status.get("attention_status", {}).get("cognitive_load", 0.0),
            "attention_focus": len(status.get("attention_focus", [])),
            "memory_operations": len(memory_events),
        }
        
        # Get reasoning rationale (optional)
        rationale = None
        if chat_request.include_reflection:
            rationale = f"Processing: {chat_request.text[:50]}... with memory context and attention focus"
        
        return {
            "response": response,
            "memory_context": memory_context if isinstance(memory_context, list) else [],
            "memory_events": memory_events,
            "cognitive_state": cognitive_state,
            "rationale": rationale,
            "status": "success"
        }
    except Exception as e:
        return {
            "response": f"Error processing request: {str(e)}",
            "memory_context": [],
            "memory_events": [],
            "cognitive_state": {},
            "rationale": None,
            "status": "error"
        }

@router.post("/cognitive_break")
async def take_cognitive_break(break_request: CognitiveBreakRequest, request: Request):
    """
    Take a cognitive break to reduce fatigue and reset attention.
    """
    try:
        agent = get_request_agent(request)
        results = agent.take_cognitive_break(break_request.duration_minutes or 1.0)
        return {
            "cognitive_load_reduction": results.get("cognitive_load_reduction", 0.0),
            "recovery_effective": results.get("recovery_effective", False),
            "new_cognitive_load": max(0.0, agent.get_cognitive_status().get("attention_status", {}).get("cognitive_load", 0.0)),
            "break_duration": results.get("break_duration", break_request.duration_minutes),
            "status": "completed",
        }
    except Exception as e:
        return {
            "error": f"Cognitive break failed: {str(e)}",
            "status": "error"
        }

@router.post("/memory/consolidate")
async def trigger_memory_consolidation(request: Request):
    """
    Trigger dream-state memory consolidation.
    """
    try:
        agent = get_request_agent(request)
        dream_results = await agent.enter_dream_state("deep")
        return {
            "consolidation_events": ["dream_state_consolidation_triggered"],
            "memories_transferred": dream_results.get("memories_consolidated", dream_results.get("memories_transferred", 0)),
            "memories_pruned": dream_results.get("memories_pruned", 0),
            "new_associations": dream_results.get("new_associations", 0),
            "status": "completed"
        }
    except Exception as e:
        return {
            "error": f"Memory consolidation failed: {str(e)}",
            "status": "error"
        }

@router.post("/dream/start")
async def start_dream(request: Request, dream_req: DreamRequest = Body(...)):
    """
    Trigger a dream state cycle (light, deep, rem).
    """
    agent = get_request_agent(request)
    results = await agent.enter_dream_state(dream_req.cycle_type)
    return {"dream_results": results}
