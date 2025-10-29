from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .executive_core import Decision, ActionResult

"""Outcome feedback structures and adaptation logic."""

@dataclass(frozen=True)
class Outcome:
    decision: Decision
    result: ActionResult
    user_satisfaction: Optional[float] = None  # 0..1
    task_completed: bool = False
    error: bool = False

class OutcomeAdapter:
    def compute_reward(self, outcome: Outcome) -> float:
        base = 1.0 if outcome.result.ok else 0.0
        base += min(1.0, len(outcome.result.content) / 600.0)
        if outcome.user_satisfaction is not None:
            base = (base + outcome.user_satisfaction) / 2.0
        if outcome.error:
            base *= 0.3
        if outcome.task_completed:
            base += 0.5
        return max(0.0, min(base, 2.5))