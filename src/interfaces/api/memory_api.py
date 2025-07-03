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
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union
from src.core.cognitive_agent import CognitiveAgent

app = FastAPI()
router = APIRouter()
agent = CognitiveAgent()

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
SYSTEM_MAP = {
    "stm": lambda: agent.memory.stm,
    "ltm": lambda: agent.memory.ltm,
    "episodic": lambda: agent.memory.episodic,
    "procedural": lambda: agent.memory.procedural,
    "prospective": lambda: agent.memory.prospective,
    "semantic": lambda: agent.memory.semantic,
}

def get_system(system: str):
    if system not in SYSTEM_MAP:
        raise HTTPException(status_code=400, detail="Invalid memory system")
    return SYSTEM_MAP[system]()

@router.post("/memory/{system}/store")
def store_memory(system: str, req: StoreMemoryRequest):
    import uuid
    from datetime import datetime
    memsys = get_system(system)
    
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
            importance=req.importance if req.importance is not None else 0.5,
            emotional_valence=req.emotional_valence if req.emotional_valence is not None else 0.0,
            tags=req.tags,
            associations=req.associations,
        )
        return {"memory_id": memory_id}

    elif system == "procedural":
        if not isinstance(req.content, dict) or "description" not in req.content or "steps" not in req.content:
            raise HTTPException(status_code=422, detail="For procedural memory, 'content' must be a dictionary with 'description' and 'steps'.")
        
        memory_id = memsys.store(
            description=req.content["description"],
            steps=req.content["steps"],
            tags=req.tags,
            importance=req.importance,
        )
        return {"memory_id": memory_id}

    elif system == "prospective":
        if not isinstance(req.content, dict) or "description" not in req.content or "due_time" not in req.content:
            raise HTTPException(status_code=422, detail="For prospective memory, 'content' must be a dictionary with 'description' and 'due_time'.")
        
        try:
            due_time = datetime.fromisoformat(req.content["due_time"])
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid 'due_time' format. Please use ISO 8601 format.")

        memory_id = memsys.add_reminder(
            description=req.content["description"],
            due_time=due_time,
        )
        return {"memory_id": memory_id}

    elif system == "semantic":
        if not isinstance(req.content, dict) or "subject" not in req.content or "predicate" not in req.content or "object_val" not in req.content:
            raise HTTPException(status_code=422, detail="For semantic memory, 'content' must be a dictionary with 'subject', 'predicate', and 'object_val'.")

        memory_id = memsys.store_fact(
            subject=req.content["subject"],
            predicate=req.content["predicate"],
            object_val=req.content["object_val"],
        )
        return {"memory_id": memory_id}

    else:
        # Generic store for other systems if they follow the basic pattern
        memory_id = memsys.store(content=req.content)
        return {"memory_id": memory_id}

@router.get("/memory/{system}/retrieve/{memory_id}")
def retrieve_memory(system: str, memory_id: str):
    memsys = get_system(system)
    
    if system == "semantic":
        memory = memsys.retrieve_fact(memory_id)
    else:
        memory = memsys.retrieve(memory_id)

    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"memory": memory}

@router.delete("/memory/{system}/delete/{memory_id}")
def delete_memory(system: str, memory_id: str):
    memsys = get_system(system)
    
    if system == "semantic":
        memsys.delete_fact(memory_id)
    else:
        memsys.delete(memory_id)
        
    return {"status": "deleted"}

@router.post("/memory/{system}/search")
def search_memory(system: str, req: SearchMemoryRequest):
    memsys = get_system(system)

    if system == "semantic":
        # Assuming the query for semantic memory is a dictionary
        if not isinstance(req.query, dict):
            raise HTTPException(status_code=422, detail="Semantic search query must be a dictionary with 'subject', 'predicate', or 'object_val'.")
        results = memsys.find_facts(
            subject=req.query.get("subject"),
            predicate=req.query.get("predicate"),
            object_val=req.query.get("object_val")
        )
    elif system == "procedural":
        results = memsys.search(query=req.query)
    elif system == "prospective":
        # Prospective memory search might list all reminders or based on a query
        results = memsys.list_reminders()
    else:
        results = memsys.search(
            query=req.query,
            tags=req.tags,
            operator=req.operator,
            min_importance=req.min_importance,
            max_results=req.max_results,
        )
        
    return {"results": results}

@router.post("/memory/ltm/feedback/{memory_id}")
def add_feedback(memory_id: str, req: FeedbackRequest):
    ltm = get_system("ltm")
    if hasattr(ltm, "add_feedback") and callable(getattr(ltm, "add_feedback", None)):
        try:
            ltm.add_feedback(
                memory_id=memory_id,
                feedback_type=req.feedback_type,
                value=req.value,
                comment=req.comment,
                user_id=req.user_id,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "feedback added"}
    else:
        raise HTTPException(status_code=501, detail="Feedback not supported for this LTM implementation")
