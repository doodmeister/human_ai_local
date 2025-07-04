from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any

router = APIRouter()

class ProcedureRequest(BaseModel):
    description: str
    steps: List[str]
    tags: Optional[List[str]] = None
    memory_type: str = "ltm"  # or "stm"

class ProcedureSearchRequest(BaseModel):
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    memory_type: str = "ltm"
    max_results: int = 10

@router.post("/procedure/store")
def store_procedure(req: ProcedureRequest, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    proc_id = memsys.store(
        description=req.description,
        steps=req.steps,
        tags=req.tags or [],
        memory_type=req.memory_type
    )
    return {"procedure_id": proc_id}

@router.get("/procedure/retrieve/{procedure_id}")
def retrieve_procedure(procedure_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    proc = memsys.retrieve(procedure_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return proc

@router.post("/procedure/search")
def search_procedure(req: ProcedureSearchRequest, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    # Only query is supported by procedural.search
    results = memsys.search(req.query or "")
    return {"results": results}

@router.post("/procedure/use/{procedure_id}")
def use_procedure(procedure_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    result = memsys.use(procedure_id)
    if not result:
        raise HTTPException(status_code=404, detail="Procedure not found")
    # If use() returns True/False, fetch the procedure object to return
    if isinstance(result, bool):
        if result:
            proc = memsys.retrieve(procedure_id)
            if not proc:
                raise HTTPException(status_code=404, detail="Procedure not found after use")
            return proc
        else:
            raise HTTPException(status_code=404, detail="Procedure not found or could not be used")
    # If use() returns the procedure object, return it directly
    return result

@router.delete("/procedure/delete/{procedure_id}")
def delete_procedure(procedure_id: str, request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    success = memsys.delete(procedure_id)
    if not success:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return {"status": "deleted"}


# List all procedural memories
@router.get("/procedural/list")
def list_procedures(request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    procs = memsys.all_procedures()
    return procs

@router.post("/procedure/clear")
def clear_procedures(request: Request, memory_type: str = "ltm"):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    memsys.clear()
    return {"status": "cleared"}