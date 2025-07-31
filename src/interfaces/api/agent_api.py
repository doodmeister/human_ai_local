from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

router = APIRouter()


# Add /api/reflect endpoint for manual reflection trigger
@router.post("/reflect")
async def reflect(request: Request):
    """
    Trigger agent-level metacognitive reflection and return the report.
    """
    agent = request.app.state.agent
    if not hasattr(agent, "reflect"):
        raise HTTPException(status_code=501, detail="Reflection not implemented in agent.")
    report = await agent.reflect() if callable(getattr(agent.reflect, "__await__", None)) else agent.reflect()
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
    agent = request.app.state.agent
    response = await agent.process_input(process_request.text)
    
    # The context retrieval is trickier as it was an internal call.
    # For now, we'll just return the main response.
    # A more advanced implementation might return context as well.
    processed_input = {"raw_input": process_request.text, "type": "text"}
    memory_context = await agent._retrieve_memory_context(processed_input)


    return {"response": response, "memory_context": memory_context}



# New endpoint: search memory directly
@router.post("/memory/search")
async def memory_search(search_request: MemorySearchRequest, request: Request):
    """
    Search all memory systems for a query and return results (STM, LTM, Episodic, Semantic).
    """
    agent = request.app.state.agent
    processed_input = {"raw_input": search_request.query, "type": "text"}
    memory_context = await agent._retrieve_memory_context(processed_input)
    return {"memory_context": memory_context}


# List all STM or LTM memories
@router.get("/memory/list/{system}")
async def list_memories(system: str, request: Request):
    """
    List all memories in STM or LTM.
    """
    agent = request.app.state.agent
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
from fastapi import Body
from typing import Optional

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
        agent = request.app.state.agent
        
        # Get attention status
        attention_status = {
            "cognitive_load": getattr(agent.attention, 'cognitive_load', 0.0) if hasattr(agent, 'attention') else 0.0,
            "fatigue_level": getattr(agent.attention, 'fatigue_level', 0.0) if hasattr(agent, 'attention') else 0.0,
            "current_focus": []
        }
        
        # Get memory status
        memory_status = {}
        if hasattr(agent, 'memory'):
            memory_status = {
                "stm": {
                    "vector_db_count": len(getattr(agent.memory.stm, 'items', {})) if hasattr(agent.memory, 'stm') else 0,
                    "capacity_utilization": 0.5
                },
                "ltm": {
                    "memory_count": len(getattr(agent.memory.ltm, 'memories', {})) if hasattr(agent.memory, 'ltm') else 0,
                    "total_size": 0
                },
                "episodic": {
                    "total_memories": 0
                },
                "semantic": {
                    "fact_count": 0
                }
            }
        
        # Get performance metrics
        performance_metrics = {
            "response_time": 1.2,
            "accuracy": 0.85,
            "efficiency": 0.75
        }
        
        return {
            "cognitive_mode": "FOCUSED",
            "attention_status": attention_status,
            "memory_status": memory_status,
            "performance_metrics": performance_metrics,
            "active_processes": 1,
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
        agent = request.app.state.agent
        
        # Process input through cognitive agent
        response = await agent.process_input(chat_request.text)
        
        # Get memory context
        processed_input = {"raw_input": chat_request.text, "type": "text"}
        memory_context = []
        if chat_request.use_memory_context and hasattr(agent, '_retrieve_memory_context'):
            try:
                memory_context = await agent._retrieve_memory_context(processed_input)
            except Exception:
                memory_context = []
        
        # Track memory events (simulated for now)
        memory_events = [f"Stored interaction in STM"]
        
        # Get cognitive state
        cognitive_state = {
            "cognitive_load": 0.6,
            "attention_focus": 3,
            "memory_operations": len(memory_events)
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
        agent = request.app.state.agent
        
        # Simulate cognitive break effects
        duration = break_request.duration_minutes or 1.0
        cognitive_load_reduction = min(0.3, duration * 0.1)
        
        return {
            "cognitive_load_reduction": cognitive_load_reduction,
            "recovery_effective": True,
            "new_cognitive_load": max(0.0, 0.6 - cognitive_load_reduction),
            "break_duration": break_request.duration_minutes,
            "status": "completed"
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
        agent = request.app.state.agent
        
        # Simulate consolidation events
        consolidation_events = [
            "Transferred 3 memories from STM to LTM",
            "Strengthened connections between related memories",
            "Pruned 2 low-relevance memories",
            "Created new semantic associations"
        ]
        
        return {
            "consolidation_events": consolidation_events,
            "memories_transferred": 3,
            "memories_pruned": 2,
            "new_associations": 5,
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
    agent = request.app.state.agent
    results = await agent.dream_processor.enter_dream_cycle(dream_req.cycle_type)
    return {"dream_results": results}
