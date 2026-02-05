"""Agent singleton helpers for API integration."""

from typing import Optional
import logging

from src.orchestration.cognitive_agent import CognitiveAgent

# Initialize logger
logger = logging.getLogger(__name__)

_agent_instance: Optional[CognitiveAgent] = None


def create_agent() -> CognitiveAgent:
    """Create or return the global CognitiveAgent instance."""
    global _agent_instance

    if _agent_instance is None:
        logger.info("Creating CognitiveAgent instance...")
        _agent_instance = CognitiveAgent()
        logger.info("CognitiveAgent initialized")

    return _agent_instance


def get_agent_instance() -> Optional[CognitiveAgent]:
    """Return the existing CognitiveAgent instance if available."""
    return _agent_instance


def reset_agent() -> None:
    """Reset the singleton (useful for testing)."""
    global _agent_instance
    _agent_instance = None
