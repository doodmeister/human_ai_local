from fastapi import APIRouter, Request
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

router = APIRouter()


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
        # Return all STM items as dicts
        return {"memories": [item.__dict__ for item in agent.memory.stm.items.values()]}
    elif system == "ltm":
        # Return all LTM records as dicts
        return {"memories": [record.to_dict() for record in agent.memory.ltm.memories.values()]}
    else:
        return {"error": "Invalid system. Use 'stm' or 'ltm'."}


# Trigger a dream state cycle
from fastapi import Body
from typing import Optional

class DreamRequest(BaseModel):
    cycle_type: Optional[str] = "deep"  # 'light', 'deep', or 'rem'

@router.post("/dream/start")
async def start_dream(request: Request, dream_req: DreamRequest = Body(...)):
    """
    Trigger a dream state cycle (light, deep, rem).
    """
    agent = request.app.state.agent
    results = await agent.dream_processor.enter_dream_cycle(dream_req.cycle_type)
    return {"dream_results": results}
