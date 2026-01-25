from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class CognitiveStep(str, Enum):
    PERCEIVE = "perceive"
    UPDATE_STM = "update_stm"
    RETRIEVE = "retrieve"
    DECIDE = "decide"
    ACT = "act"
    REFLECT = "reflect"
    CONSOLIDATE = "consolidate"


_STEP_ORDER: dict[CognitiveStep, CognitiveStep] = {
    CognitiveStep.PERCEIVE: CognitiveStep.UPDATE_STM,
    CognitiveStep.UPDATE_STM: CognitiveStep.RETRIEVE,
    CognitiveStep.RETRIEVE: CognitiveStep.DECIDE,
    CognitiveStep.DECIDE: CognitiveStep.ACT,
    CognitiveStep.ACT: CognitiveStep.REFLECT,
    CognitiveStep.REFLECT: CognitiveStep.CONSOLIDATE,
}


@dataclass
class CognitiveTick:
    """A single canonical cognition cycle.

    Canonical steps:
      Perceive → Update STM → Retrieve → Decide → Act → Reflect → Consolidate

    Constraint enforced:
      Exactly one decision owner per tick (the `owner` field).

    This is intentionally minimal: it provides step ordering + a small shared state bag.
    """

    owner: str
    kind: str
    tick_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None

    step: CognitiveStep = CognitiveStep.PERCEIVE
    decided: bool = False

    decision: Any = None
    state: dict[str, Any] = field(default_factory=dict)

    def assert_step(self, expected: CognitiveStep) -> None:
        if self.step != expected:
            raise RuntimeError(
                f"CognitiveTick step mismatch: expected={expected.value} actual={self.step.value} "
                f"owner={self.owner} kind={self.kind}"
            )

    def advance(self, expected_current: CognitiveStep) -> None:
        """Advance to the next canonical step."""
        self.assert_step(expected_current)
        next_step = _STEP_ORDER.get(expected_current)
        if next_step is None:
            raise RuntimeError(
                f"CognitiveTick cannot advance from terminal step {expected_current.value}"
            )
        self.step = next_step

    def mark_decided(self, decision: Any = None) -> None:
        """Mark that the single owner has made the tick's decision."""
        self.assert_step(CognitiveStep.DECIDE)
        if self.decided:
            raise RuntimeError(
                f"CognitiveTick decision already made: owner={self.owner} kind={self.kind}"
            )
        self.decided = True
        self.decision = decision

    def finish(self) -> None:
        self.assert_step(CognitiveStep.CONSOLIDATE)
        self.finished_at = datetime.now()

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None
