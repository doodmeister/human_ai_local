from fastapi import FastAPI, APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union
from src.core.cognitive_agent import CognitiveAgent

router = APIRouter()

class StoreMemoryRequest(BaseModel):
    content: Any
    memory_type: Optional[str] = "episodic"
    importance: Optional[float] = 0.5
    emotional_valence: Optional[float] = 0.0
    source: Optional[str] = "api"
    tags: Optional[list[str]] = None
    associations: Optional[list[str]] = None

class SearchMemoryRequest(BaseModel):
    query: Optional[Union[str, Dict[str, Any]]] = None
    tags: Optional[list[str]] = None
    operator: Optional[str] = "OR"
    min_importance: Optional[float] = 0.0
    max_results: Optional[int] = 10

class FeedbackRequest(BaseModel):
    feedback_type: str
    value: Any
    comment: Optional[str] = None
    user_id: Optional[str] = None

# Helper to get the memory system
def get_system(system: str, agent: CognitiveAgent):
    SYSTEM_MAP = {
        "stm": lambda: agent.memory.stm,
        "ltm": lambda: agent.memory.ltm,
        "episodic": lambda: agent.memory.episodic,
        "procedural": lambda: agent.memory.procedural,
        "prospective": lambda: agent.memory.prospective,
        "semantic": lambda: agent.memory.semantic,
    }
    if system not in SYSTEM_MAP:
        raise HTTPException(status_code=400, detail="Invalid memory system")
    return SYSTEM_MAP[system]()

@router.post("/memory/{system}/store")
def store_memory(system: str, req: StoreMemoryRequest, request: Request):
    import uuid
    from datetime import datetime
    agent = request.app.state.agent
    memsys = get_system(system, agent)
    
    if system == "stm":
        # STM requires a unique memory_id and content
        memory_id = str(uuid.uuid4())
        memsys.store(
            memory_id=memory_id,
            content=req.content,
            importance=req.importance if req.importance is not None else 0.5,
            attention_score=0.0,  # Optionally expose in API if needed
            emotional_valence=req.emotional_valence if req.emotional_valence is not None else 0.0,
            associations=req.associations,
        )
        return {"memory_id": memory_id}
    
    elif system == "ltm":
        # LTM requires a valid string memory_id
        memory_id = uuid.uuid4().hex
        memsys.store(
            memory_id=memory_id,
            content=req.content,
            memory_type=req.memory_type,
            importance=req.importance,
            emotional_valence=req.emotional_valence,
            source=req.source,
            tags=req.tags,
            associations=req.associations,
        )
        return {"memory_id": memory_id}
    
    elif system == "episodic":
        # Episodic memory requires 'content' (as a summary), and optional 'detailed_content'
        if not isinstance(req.content, dict) or "summary" not in req.content:
            raise HTTPException(status_code=422, detail="For episodic memory, 'content' must be a dictionary with a 'summary' key.")
        
        memory_id = memsys.store(
            summary=req.content["summary"],
            detailed_content=req.content.get("detailed_content", ""),
            timestamp=datetime.fromisoformat(req.content["timestamp"]) if "timestamp" in req.content else datetime.now(),
            importance=req.importance,
            emotional_valence=req.emotional_valence,
            tags=req.tags,
            associations=req.associations,
        )
        return {"memory_id": memory_id}
    
    else:
        raise HTTPException(status_code=400, detail=f"Storing memories in '{system}' is not supported via this generic endpoint.")

@router.get("/memory/{system}/retrieve/{memory_id}")
def retrieve_memory(system: str, memory_id: str, request: Request):
    agent = request.app.state.agent
    memsys = get_system(system, agent)
    memory = memsys.retrieve(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory

@router.delete("/memory/{system}/delete/{memory_id}")
def delete_memory(system: str, memory_id: str, request: Request):
    agent = request.app.state.agent
    memsys = get_system(system, agent)
    if not hasattr(memsys, 'delete'):
        raise HTTPException(status_code=405, detail=f"Delete operation not supported for '{system}' memory system.")
    
    try:
        memsys.delete(memory_id)
        return {"status": "ok", "message": f"Memory {memory_id} deleted from {system}."}
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/{system}/search")
def search_memory(system: str, req: SearchMemoryRequest, request: Request):
    agent = request.app.state.agent
    memsys = get_system(system, agent)
    
    # STM search is simple, by content only
    if system == "stm":
        if not isinstance(req.query, str):
            raise HTTPException(status_code=422, detail="STM search query must be a string.")
        results = memsys.search(req.query, max_results=req.max_results)
        return {"results": results}

    # LTM search is more complex
    if system == "ltm":
        results = memsys.search(
            query=req.query,
            tags=req.tags,
            operator=req.operator,
            min_importance=req.min_importance,
            max_results=req.max_results,
        )
        return {"results": results}

    # Episodic search
    if system == "episodic":
        if not isinstance(req.query, str):
            raise HTTPException(status_code=422, detail="Episodic search query must be a string.")
        results = memsys.search(req.query, max_results=req.max_results)
        return {"results": results}

    raise HTTPException(status_code=400, detail=f"Search not supported for '{system}' memory system via this generic endpoint.")

@router.post("/memory/{system}/feedback/{memory_id}")
def add_feedback(system: str, memory_id: str, req: FeedbackRequest, request: Request):
    agent = request.app.state.agent
    memsys = get_system(system, agent)
    if system != "ltm":
        raise HTTPException(status_code=400, detail="Feedback is only supported for LTM")

    try:
        memsys.add_feedback(
            memory_id=memory_id,
            feedback_type=req.feedback_type,
            value=req.value,
            comment=req.comment,
            user_id=req.user_id,
        )
        return {"status": "ok", "message": f"Feedback added to memory {memory_id}."}
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/memory/{system}/list")
def list_memories(system: str, request: Request):
    agent = request.app.state.agent
    memsys = get_system(system, agent)
    
    # LTM, Episodic, Semantic (Chroma-based)
    if hasattr(memsys, 'get_all'):
        return {"memories": memsys.get_all()}

    # STM (dict-based)
    if hasattr(memsys, 'memories') and isinstance(memsys.memories, dict):
        return {"memories": list(memsys.memories.values())}

    # Procedural (dict-based)
    if hasattr(memsys, 'procedures') and isinstance(memsys.procedures, dict):
        return {"memories": list(memsys.procedures.values())}

    # Prospective (dict-based)
    if hasattr(memsys, 'tasks') and isinstance(memsys.tasks, dict):
        return {"memories": list(memsys.tasks.values())}

    raise HTTPException(status_code=405, detail=f"List operation not supported for '{system}' memory system.")

# Create FastAPI app for testing purposes (must be at end of file)
app = FastAPI()
app.include_router(router)
