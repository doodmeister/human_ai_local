"""Agent singleton helpers for API integration."""

from typing import Optional

from src.core.cognitive_agent import CognitiveAgent

_agent_instance: Optional[CognitiveAgent] = None


def create_agent() -> CognitiveAgent:
    """Create or return the global CognitiveAgent instance."""
    global _agent_instance

    if _agent_instance is None:
        print("[agent] creating CognitiveAgent instance...")
        _agent_instance = CognitiveAgent()
        print("[agent] CognitiveAgent initialized")

    return _agent_instance


def get_agent_instance() -> Optional[CognitiveAgent]:
    """Return the existing CognitiveAgent instance if available."""
    return _agent_instance


def reset_agent() -> None:
    """Reset the singleton (useful for testing)."""
    global _agent_instance
    _agent_instance = None
