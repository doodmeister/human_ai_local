"""Runtime helpers used by the FastAPI interface layer.

Phase 6: interfaces must depend on orchestration only.
"""

from __future__ import annotations

from typing import Any

from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from src.memory.metrics import metrics_registry  # noqa: F401
from src.orchestration.runtime.app_container import get_runtime

_prospective = get_inmemory_prospective_memory()


def get_agent() -> Any:
    return get_runtime().get_agent()


def get_prospective():
    return get_runtime().get_prospective()


def get_chat_service() -> Any:
    """Return the shared runtime ChatService instance."""
    return get_runtime().get_chat_service()
