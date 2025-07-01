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
from typing import Any, Dict, Optional
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
    query: Optional[str] = None
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
}

def get_system(system: str):
    if system not in SYSTEM_MAP:
        raise HTTPException(status_code=400, detail="Invalid memory system")
    return SYSTEM_MAP[system]()

@app.post("/memory/{system}/store")
def store_memory(system: str, req: StoreMemoryRequest):
    import uuid
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
    else:
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

@app.get("/memory/{system}/retrieve/{memory_id}")
def retrieve_memory(system: str, memory_id: str):
    memsys = get_system(system)
    result = memsys.retrieve(memory_id)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result

@app.delete("/memory/{system}/delete/{memory_id}")
def delete_memory(system: str, memory_id: str):
    memsys = get_system(system)
    success = memsys.delete(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or could not be deleted")
    return {"status": "deleted"}

@app.post("/memory/{system}/search")
def search_memory(system: str, req: SearchMemoryRequest):
    memsys = get_system(system)
    if system == "stm":
        # STM: Only pass supported arguments
        if req.query:
            results = memsys.search(query=req.query)
        elif req.tags:
            # If STM supports tag search, otherwise return error
            if hasattr(memsys, "search_by_tags"):
                results = memsys.search_by_tags(tags=req.tags, operator=req.operator)
            else:
                raise HTTPException(status_code=400, detail="Tag search not supported for STM")
        else:
            raise HTTPException(status_code=400, detail="Must provide query or tags")
    else:
        # LTM: Pass all supported arguments
        if req.query:
            results = memsys.search(query=req.query, min_importance=req.min_importance, max_results=req.max_results)
        elif req.tags:
            results = memsys.search_by_tags(tags=req.tags, operator=req.operator)
        else:
            raise HTTPException(status_code=400, detail="Must provide query or tags")
    return {"results": results}

@app.post("/memory/ltm/feedback/{memory_id}")
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
