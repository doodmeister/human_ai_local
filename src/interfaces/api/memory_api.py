"""
REST API for STM and LTM Memory Systems

Endpoints:
- POST /memory/{system}/store: Store a new memory (returns memory ID)
- GET /memory/{system}/retrieve/{memory_id}: Retrieve a memory by ID
- DELETE /memory/{system}/delete/{memory_id}: Delete a memory by ID
- POST /memory/{system}/search: Search for memories (by content/tags)
- POST /memory/{system}/feedback/{memory_id}: Add feedback to a memory (LTM only)

Where {system} is either 'stm' or 'ltm'.
"""
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

from src.interfaces.api.dependencies import get_request_agent, get_request_memory_system

router = APIRouter()


def _serialize_memory_record(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "to_dict"):
        maybe_dict = item.to_dict()
        if isinstance(maybe_dict, dict):
            return dict(maybe_dict)
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return {"value": item}


def _collection_records(collection: Any) -> list[dict[str, Any]]:
    result = collection.get()
    records: list[dict[str, Any]] = []
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    for index, memory_id in enumerate(ids):
        records.append(
            {
                "id": memory_id,
                "content": docs[index] if index < len(docs) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
        )
    return records

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

class ProactiveRecallRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    min_relevance: Optional[float] = 0.7
    use_ai_summary: Optional[bool] = False

# Added FeedbackRequest model
class FeedbackRequest(BaseModel):
    feedback_type: str
    value: float
    comment: Optional[str] = None
    user_id: Optional[str] = None

@router.post("/memory/proactive-recall")
def proactive_recall(req: ProactiveRecallRequest, request: Request):
    """Perform proactive recall of relevant memories based on query"""
    agent = get_request_agent(request)
    
    try:
        # Use the memory system's proactive_recall method
        result = agent.memory.proactive_recall(
            query=req.query,
            max_results=req.max_results,
            min_relevance=req.min_relevance,
            use_ai_summary=req.use_ai_summary
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proactive recall failed: {str(e)}")

# Create FastAPI app for testing purposes (must be at end of file)

# Helper to get the memory system
def get_system(system: str, agent: Any):
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
    agent = get_request_agent(request)
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
        # Episodic memory requires 'content' with detailed_content
        if not isinstance(req.content, dict):
            raise HTTPException(status_code=422, detail="For episodic memory, 'content' must be a dictionary.")
        
        # Support both 'summary' (legacy) and 'detailed_content' (current)
        detailed_content = req.content.get("detailed_content") or req.content.get("summary", "")
        
        memory_id = memsys.store(
            detailed_content=detailed_content,
            importance=req.importance,
            emotional_valence=req.emotional_valence,
        )
        return {"memory_id": memory_id}
    
    else:
        raise HTTPException(status_code=400, detail=f"Storing memories in '{system}' is not supported via this generic endpoint.")

@router.get("/memory/{system}/retrieve/{memory_id}")
def retrieve_memory(system: str, memory_id: str, request: Request):
    agent = get_request_agent(request)
    memsys = get_system(system, agent)
    memory = memsys.retrieve(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory

@router.delete("/memory/{system}/delete/{memory_id}")
def delete_memory(system: str, memory_id: str, request: Request):
    agent = get_request_agent(request)
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
    agent = get_request_agent(request)
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
    agent = get_request_agent(request)
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
    agent = get_request_agent(request)
    memsys = get_system(system, agent)

    if hasattr(memsys, 'get_all_memories'):
        try:
            return {"memories": [_serialize_memory_record(item) for item in memsys.get_all_memories()]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list memories for '{system}': {str(e)}")

    collection = getattr(memsys, 'collection', None)
    if collection is not None and hasattr(collection, 'get'):
        try:
            return {"memories": _collection_records(collection)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list memories for '{system}': {str(e)}")
    
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

@router.get("/status")
async def get_memory_status(request: Request):
    """Get comprehensive memory system status"""
    try:
        memsys = get_request_memory_system(request)
        status = memsys.get_status()
        stm_status = status.get("stm", {}) if isinstance(status, dict) else {}
        ltm_status = status.get("ltm", {}) if isinstance(status, dict) else {}
        return {
            **status,
            "overall_health": "healthy" if status.get("system_active") else "degraded",
            "stm": {
                **stm_status,
                "health": stm_status.get("status", "healthy"),
            },
            "ltm": {
                **ltm_status,
                "health": ltm_status.get("status", "healthy"),
            },
        }
    
    except Exception as e:
        return {
            "stm": {"vector_db_count": 0, "capacity": 7, "capacity_utilization": 0.0, "health": "error"},
            "ltm": {"memory_count": 0, "total_size": 0, "health": "error"},
            "episodic": {"total_memories": 0, "recent_memories": 0, "health": "error"},
            "semantic": {"fact_count": 0, "knowledge_domains": 0, "health": "error"},
            "procedural": {"procedure_count": 0, "health": "error"},
            "prospective": {"active_reminders": 0, "health": "error"},
            "overall_health": "error",
            "error": f"Failed to get memory status: {str(e)}"
        }

# Create FastAPI app for testing purposes (must be at end of file)
app = FastAPI()
app.include_router(router)
