"""ExecutiveAgent (Phase 2: Executive Collapse)

This module remains as a backwards-compatible facade for callers that expect
`ExecutiveAgent`, but it no longer owns goal selection, planning, or action
commitment.

Single executive owner:
    - src/executive/executive_core.py (ExecutiveController)

Constraint:
    - No module except the executive may commit actions.
      (All commitment/tool actuation occurs inside ExecutiveController.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional

from ..core.cognitive_tick import CognitiveStep, CognitiveTick

from .adapters import ActuatorAdapter, AttentionAdapter, LLMAdapter, MemoryAdapter
from .executive_core import ExecutiveController, Goal as CoreGoal
from .goal_manager import GoalManager

logger = logging.getLogger(__name__)


class ExecutiveState(Enum):
    """High-level state for UI/API surfaces (advisory)."""

    INITIALIZING = "initializing"
    EXECUTING = "executing"
    IDLE = "idle"
    ERROR = "error"


@dataclass
class ExecutiveContext:
    """Lightweight context tracked by the facade for UI surfaces."""

    current_input: Optional[str] = None
    recent_inputs: List[str] = field(default_factory=list)

    def update(self, new_input: str) -> None:
        self.current_input = new_input
        self.recent_inputs.append(new_input)
        if len(self.recent_inputs) > 10:
            self.recent_inputs.pop(0)


class _InMemoryExecutiveMemory:
    """Tiny in-memory memory system shim for default construction.

    This exists so `ExecutiveAgent()` can be instantiated in lightweight
    chat/testing flows without requiring a full MemorySystem.
    """

    class _STM:
        def __init__(self) -> None:
            self._items: List[Dict[str, Any]] = []

        def store(self, *, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
            self._items.append({"content": content, "metadata": metadata or {}})

        def get_all_memories(self) -> List[Dict[str, Any]]:
            return list(self._items)

    class _LTM:
        def __init__(self) -> None:
            self._docs: List[str] = []

        def store(self, *, content: str, tags: Optional[List[str]] = None) -> str:
            _ = tags
            self._docs.append(content)
            return f"ltm::{len(self._docs) - 1}"

        def search(self, query: str) -> List[Dict[str, Any]]:
            q = query.lower().strip()
            out: List[Dict[str, Any]] = []
            for doc in self._docs:
                if not q or q in doc.lower():
                    out.append({"content": doc, "similarity": 0.1})
            return out

    def __init__(self) -> None:
        self.stm = self._STM()
        self.ltm = self._LTM()

    def store_memory(self, *, memory_id: str, content: str, importance: float = 0.4) -> str:
        _ = memory_id
        self.stm.store(content=content, metadata={"importance": importance})
        return memory_id


class ExecutiveAgent:
    """Backwards-compatible facade.

    - Goal selection is performed by `ExecutiveController.select_goal()`.
    - Planning and action commitment are performed by `ExecutiveController.run_turn()`.
    - This class may hold advisors (GoalManager) for UI/API convenience.
    """

    def __init__(
        self,
        attention_mechanism: Any = None,
        memory_system: Any = None,
        goal_manager: Optional[GoalManager] = None,
        llm_client: Any = None,
        actuator: Any = None,
    ):
        self.state = ExecutiveState.INITIALIZING
        self.context = ExecutiveContext()

        self.goals = goal_manager or GoalManager()
        self.goal_manager = self.goals

        ms = memory_system or _InMemoryExecutiveMemory()
        self._core = ExecutiveController(
            memory=MemoryAdapter(ms),
            attention=AttentionAdapter(attention_mechanism),
            llm=LLMAdapter(llm_client),
            actuator=actuator or ActuatorAdapter(),
        )

        self.state = ExecutiveState.IDLE

    @property
    def mode(self) -> str:
        return self._core.mode.name

    async def process_input(self, input_text: str) -> Dict[str, Any]:
        """Process one input turn.

        All goal selection / planning / commitment is delegated to `executive_core`.
        """
        tick = CognitiveTick(owner="executive_core", kind="executive_turn")
        try:
            tick.assert_step(CognitiveStep.PERCEIVE)
            self.state = ExecutiveState.EXECUTING
            self.context.update(input_text)
            tick.state["input_text"] = input_text
            tick.advance(CognitiveStep.PERCEIVE)

            tick.assert_step(CognitiveStep.UPDATE_STM)
            tick.state["recent_inputs"] = list(self.context.recent_inputs)
            tick.advance(CognitiveStep.UPDATE_STM)

            tick.assert_step(CognitiveStep.RETRIEVE)
            candidate_goals = self._candidate_goals()
            tick.state["candidate_goals"] = [g.description for g in candidate_goals[:5]]
            tick.advance(CognitiveStep.RETRIEVE)

            tick.assert_step(CognitiveStep.DECIDE)
            tick.mark_decided({"delegate": "executive_core.run_turn"})
            tick.advance(CognitiveStep.DECIDE)

            tick.assert_step(CognitiveStep.ACT)
            result = await self._core.run_turn(
                user_input=input_text,
                high_level_goal=None,
                candidate_goals=candidate_goals,
            )
            tick.state["result_ok"] = result.ok
            tick.advance(CognitiveStep.ACT)

            tick.assert_step(CognitiveStep.REFLECT)
            tick.advance(CognitiveStep.REFLECT)

            tick.assert_step(CognitiveStep.CONSOLIDATE)
            tick.finish()

            self.state = ExecutiveState.IDLE
            return {
                "response": result.content,
                "executive_actions": [],
                "goals_updated": [],
                "tasks_created": [],
                "decisions_made": [],
                "cognitive_state": self.get_cognitive_status(),
            }
        except Exception as e:
            self.state = ExecutiveState.ERROR
            logger.error("Error in ExecutiveAgent.process_input: %s", e)
            try:
                tick.finish()
            except Exception:
                pass
            return {
                "response": "I encountered an error processing your request.",
                "error": str(e),
                "executive_actions": [],
                "cognitive_state": self.get_cognitive_status(),
            }

    def _candidate_goals(self) -> List[CoreGoal]:
        candidate_goals: List[CoreGoal] = []
        try:
            active = self.goals.get_active_goals()
        except Exception:
            return []

        for g in active[:20]:
            title = getattr(g, "title", "")
            desc = getattr(g, "description", title) or title

            pr_value = getattr(getattr(g, "priority", None), "value", None)
            pr_norm = 0.5
            if isinstance(pr_value, str):
                if pr_value.lower() == "high":
                    pr_norm = 0.9
                elif pr_value.lower() == "low":
                    pr_norm = 0.3

            candidate_goals.append(
                CoreGoal(
                    id=str(getattr(g, "id", title or desc)),
                    description=f"{title}: {desc}" if title else desc,
                    priority=float(pr_norm),
                )
            )

        return candidate_goals

    def get_cognitive_status(self) -> Dict[str, Any]:
        return {
            "mode": self._core.mode.name,
            "state": self.state.value,
            "timestamp": datetime.now().isoformat(),
        }

    def reflect(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "executive_state": self.state.value,
            "mode": self._core.mode.name,
            "goal_status": self.goals.get_statistics() if hasattr(self.goals, "get_statistics") else {},
        }

    def adapt(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        _ = feedback
        return {"adaptations_made": [], "timestamp": datetime.now().isoformat()}

    def get_executive_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "mode": self._core.mode.name,
            "context": {
                "current_input": self.context.current_input,
                "recent_inputs_count": len(self.context.recent_inputs),
            },
            "configuration": {},
        }

    def shutdown(self) -> None:
        self.state = ExecutiveState.IDLE
