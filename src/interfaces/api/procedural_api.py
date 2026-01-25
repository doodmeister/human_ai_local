from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Optional

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
    results = memsys.search(
        req.query or "",
        memory_type=req.memory_type,
        tags=req.tags,
        max_results=req.max_results,
    )
    return {"results": results}


@router.get("/procedure/search")
def search_procedure_get(
    request: Request,
    query: str = "",
    memory_type: str = "ltm",
    max_results: int = 10,
    tags: Optional[List[str]] = Query(default=None),
):
    """Convenience GET endpoint (primarily for CLI/backward compatibility)."""
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    results = memsys.search(
        query,
        memory_type=memory_type,
        tags=tags,
        max_results=max_results,
    )
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


# Canonical: list all procedures
@router.get("/procedure/list")
def list_procedures(request: Request):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    procs = memsys.all_procedures()
    return procs


# Backward-compatible alias routes under `/procedural/*`
@router.post("/procedural/store")
def store_procedure_legacy(req: ProcedureRequest, request: Request):
    return store_procedure(req=req, request=request)


@router.get("/procedural/retrieve/{procedure_id}")
def retrieve_procedure_legacy(procedure_id: str, request: Request):
    return retrieve_procedure(procedure_id=procedure_id, request=request)


@router.get("/procedural/list")
def list_procedures_legacy(request: Request):
    return list_procedures(request=request)


@router.get("/procedural/search")
def search_procedure_legacy(
    request: Request,
    query: str = "",
    memory_type: str = "ltm",
    max_results: int = 10,
    tags: Optional[List[str]] = Query(default=None),
):
    return search_procedure_get(
        request=request,
        query=query,
        memory_type=memory_type,
        max_results=max_results,
        tags=tags,
    )


@router.delete("/procedural/delete/{procedure_id}")
def delete_procedure_legacy(procedure_id: str, request: Request):
    return delete_procedure(procedure_id=procedure_id, request=request)

@router.post("/procedure/clear")
def clear_procedures(request: Request, memory_type: str = "ltm"):
    agent = request.app.state.agent
    memsys = agent.memory.procedural
    memsys.clear(memory_type=memory_type)
    return {"status": "cleared"}


@router.post("/procedural/clear")
def clear_procedures_legacy(request: Request, memory_type: str = "ltm"):
    return clear_procedures(request=request, memory_type=memory_type)