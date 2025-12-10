from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

@router.post("/prospective/process_due")
def process_due_reminders(request: Request):
    """
    Process all due reminders: remove from prospective memory and convert to LTM.
    Returns the number of reminders processed.
    """
    agent = request.app.state.agent
    memsys = agent.memory.prospective
    ltm = agent.memory.ltm
    count = memsys.process_due_reminders(ltm_system=ltm)
    return {"processed": count}
from fastapi import APIRouter, Request

router = APIRouter()

class ProspectiveRequest(BaseModel):
    description: str
    trigger_time: Optional[str] = None  # ISO datetime or cron
    tags: Optional[List[str]] = None
    memory_type: str = "ltm"  # or "stm"

@router.post("/prospective/store")
def store_prospective(req: ProspectiveRequest, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.prospective
    event_id = memsys.store(
        description=req.description,
        trigger_time=req.trigger_time,
        tags=req.tags or [],
        memory_type=req.memory_type
    )
    return {"event_id": event_id}

@router.get("/prospective/retrieve/{event_id}")
def retrieve_prospective(event_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.prospective
    event = memsys.retrieve(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


# New endpoint: List all reminders (for CLI `/reminders` command)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

@router.get("/prospective/search")
def list_prospective_reminders(request: Request, include_completed: bool = False, goal_id: Optional[str] = None):
    """
    List all prospective reminders (optionally include completed).
    Optionally filter by goal_id for auto-generated plan reminders.
    Returns a list of reminders as dicts.
    """
    agent = request.app.state.agent
    memsys = agent.memory.prospective
    reminders = memsys.list_reminders(include_completed=include_completed)
    
    # Filter by goal_id if specified
    if goal_id:
        reminders = [
            r for r in reminders 
            if r.metadata.get("goal_id") == goal_id or f"goal:{goal_id}" in r.tags
        ]
    
    # Convert dataclass objects to dicts for JSON serialization
    reminders_dicts = [jsonable_encoder(r) for r in reminders]
    return JSONResponse(content={"results": reminders_dicts})

@router.delete("/prospective/delete/{event_id}")
def delete_prospective(event_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.prospective
    success = memsys.delete(event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"status": "deleted"}

@router.post("/prospective/clear")
def clear_prospective(request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.prospective
    memsys.clear()
    return {"status": "cleared"}
