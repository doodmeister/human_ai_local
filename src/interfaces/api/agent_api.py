from fastapi import APIRouter, Request
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

router = APIRouter()

class ProcessInputRequest(BaseModel):
    text: str

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
