from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Any

router = APIRouter()

class SemanticFactRequest(BaseModel):
    subject: str
    predicate: str
    object_val: Any
    tags: Optional[List[str]] = None

@router.post("/semantic/fact/store")
def store_fact(req: SemanticFactRequest, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.semantic
    fact_id = memsys.store_fact(
        subject=req.subject,
        predicate=req.predicate,
        object_val=req.object_val
    )
    return {"fact_id": fact_id}

@router.get("/semantic/fact/retrieve/{fact_id}")
def retrieve_fact(fact_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.semantic
    fact = memsys.retrieve_fact(fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    return fact

@router.post("/semantic/fact/search")
def search_facts(request: Request, subject: Optional[str] = None, predicate: Optional[str] = None, object_val: Optional[Any] = None):
    agent = request.app.state.agent
    memsys = agent.memory.semantic
    results = memsys.find_facts(subject=subject, predicate=predicate, object_val=object_val)
    return {"results": results}

@router.delete("/semantic/fact/delete/{fact_id}")
def delete_fact(fact_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.semantic
    success = memsys.delete(fact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"status": "deleted"}

@router.post("/semantic/clear")
def clear_semantic(request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.semantic
    memsys.clear()
    return {"status": "cleared"}
