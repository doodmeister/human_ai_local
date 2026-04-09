from __future__ import annotations

import time

from .enums import ScheduledTaskStatus
from .models import CriticReport, ScheduledCognitiveTask, WorkspaceState


class DefaultCognitiveScheduler:
    """Turn critic feedback into deferred cognitive follow-up tasks."""

    def __init__(
        self,
        *,
        contradiction_delay_seconds: float = 5.0,
        uncertainty_delay_seconds: float = 10.0,
        baseline_delay_seconds: float = 30.0,
    ) -> None:
        self._contradiction_delay_seconds = contradiction_delay_seconds
        self._uncertainty_delay_seconds = uncertainty_delay_seconds
        self._baseline_delay_seconds = baseline_delay_seconds

    def schedule(
        self,
        workspace: WorkspaceState,
        report: CriticReport,
    ) -> list[ScheduledCognitiveTask]:
        if workspace.stimulus.metadata.get("background_task"):
            return []

        tasks: list[ScheduledCognitiveTask] = []
        turn_index = workspace.snapshot.turn_index
        session_id = workspace.stimulus.session_id
        now_ts = time.time()

        if report.contradictions_detected:
            tasks.append(
                ScheduledCognitiveTask(
                    task_id=f"conflict-review:{session_id}:{turn_index}",
                    description="Review contradictory context on a later pass",
                    status=ScheduledTaskStatus.PENDING,
                    priority=min(1.0, 0.55 + 0.10 * len(report.contradictions_detected)),
                    due_at=now_ts + self._contradiction_delay_seconds,
                    goal_id=workspace.dominant_goal.goal_id if workspace.dominant_goal else None,
                    metadata={"reason": "contradictions", "cycle_id": report.cycle_id},
                )
            )
        if report.follow_up_recommended and workspace.snapshot.uncertainty >= 0.5:
            tasks.append(
                ScheduledCognitiveTask(
                    task_id=f"memory-followup:{session_id}:{turn_index}",
                    description="Retrieve additional supporting context",
                    status=ScheduledTaskStatus.PENDING,
                    priority=0.60,
                    due_at=now_ts + self._uncertainty_delay_seconds,
                    goal_id=workspace.dominant_goal.goal_id if workspace.dominant_goal else None,
                    metadata={"reason": "uncertainty", "cycle_id": report.cycle_id},
                )
            )
        if not report.follow_up_recommended and not tasks:
            tasks.append(
                ScheduledCognitiveTask(
                    task_id=f"stability-check:{session_id}:{turn_index}",
                    description="Revisit cycle stability if conditions change",
                    status=ScheduledTaskStatus.PENDING,
                    priority=0.25,
                    due_at=now_ts + self._baseline_delay_seconds,
                    goal_id=workspace.dominant_goal.goal_id if workspace.dominant_goal else None,
                    metadata={"reason": "baseline", "cycle_id": report.cycle_id},
                )
            )
        return tasks
