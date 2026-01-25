"""Orchestration-layer exports for executive APIs.

Phase 6: interfaces must depend on orchestration only.
"""

from __future__ import annotations

from src.executive.executive_agent import ExecutiveAgent
from src.executive.goal_manager import GoalPriority
from src.executive.integration import ExecutiveSystem
from src.executive.planning.world_state import WorldState

try:  # optional / legacy
    from src.executive.tasks import TaskStatus  # type: ignore
except Exception:  # pragma: no cover
    TaskStatus = None  # type: ignore

__all__ = [
    "ExecutiveAgent",
    "ExecutiveSystem",
    "GoalPriority",
    "WorldState",
    "TaskStatus",
]
