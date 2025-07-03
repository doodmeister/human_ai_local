from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any
from src.core.agent_singleton import agent

router = APIRouter()

class ProspectiveRequest(BaseModel):
    description: str
    trigger_time: Optional[str] = None  # ISO datetime or cron
    tags: Optional[List[str]] = None
    memory_type: str = "ltm"  # or "stm"

@router.post("/prospective/store")
def store_prospective(req: ProspectiveRequest):
    memsys = agent.memory.prospective
    event_id = memsys.store(
        description=req.description,
        trigger_time=req.trigger_time,
        tags=req.tags or [],
        memory_type=req.memory_type
    )
    return {"event_id": event_id}

@router.get("/prospective/retrieve/{event_id}")
def retrieve_prospective(event_id: str):
    memsys = agent.memory.prospective
    event = memsys.retrieve(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@router.post("/prospective/search")
def search_prospective(query: Optional[str] = None):
    memsys = agent.memory.prospective
    results = memsys.search(query or "")
    return {"results": results}

@router.delete("/prospective/delete/{event_id}")
def delete_prospective(event_id: str):
    memsys = agent.memory.prospective
    success = memsys.delete(event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"status": "deleted"}

@router.post("/prospective/clear")
def clear_prospective():
    memsys = agent.memory.prospective
    memsys.clear()
    return {"status": "cleared"}
