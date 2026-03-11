from __future__ import annotations

from typing import Any

from fastapi import Request

from src.orchestration.runtime.app_container import get_runtime


def get_request_agent(request: Request) -> Any:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        request.app.state.agent = get_runtime().get_agent()
        agent = request.app.state.agent
    return agent


def get_request_memory_system(request: Request) -> Any:
    return get_request_agent(request).memory