"""
A factory function to create the CognitiveAgent.
The singleton instance is managed by the FastAPI application's lifespan.
"""
from src.core.cognitive_agent import CognitiveAgent

def create_agent():
    """
    Creates and returns a new instance of the CognitiveAgent.
    """
    print("Creating a new CognitiveAgent instance...")
    agent = CognitiveAgent()
    print("CognitiveAgent instance created.")
    return agent
