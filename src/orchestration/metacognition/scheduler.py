from __future__ import annotations

import math
import time

from .enums import ScheduledTaskStatus, StimulusMetadataKey, TaskReason
from .interfaces import CognitiveScheduler
from .models import CriticReport, ScheduledCognitiveTask, WorkspaceState
from .thresholds import DEFAULT_COGNITIVE_THRESHOLDS


__all__ = ["DefaultCognitiveScheduler"]


class DefaultCognitiveScheduler(CognitiveScheduler):
    """Turn critic feedback into deferred cognitive follow-up tasks.

    Task IDs are unique per `(session_id, turn_index, cycle_id, reason)` so a
    replayed cycle does not overwrite prior queued work from an earlier cycle.

    The default uncertainty threshold is shared with the workspace builder and
    policy engine through DEFAULT_COGNITIVE_THRESHOLDS.
    """

    # Revisiting contradictions is the highest-priority follow-up work.
    _CONTRADICTION_BASE_PRIORITY = 0.55
    # Each additional contradiction slightly raises urgency up to the clamp.
    _CONTRADICTION_PER_ITEM_PRIORITY = 0.10
    # High uncertainty warrants a stronger deferred context-retrieval task.
    _UNCERTAINTY_PRIORITY = 0.60
    # Baseline stability checks should stay low priority versus explicit issues.
    _BASELINE_PRIORITY = 0.25

    def __init__(
        self,
        *,
        contradiction_delay_seconds: float = 5.0,
        uncertainty_delay_seconds: float = 10.0,
        baseline_delay_seconds: float = 30.0,
        uncertainty_threshold: float = DEFAULT_COGNITIVE_THRESHOLDS.uncertainty_threshold,
    ) -> None:
        for name, value in (
            ("contradiction_delay_seconds", contradiction_delay_seconds),
            ("uncertainty_delay_seconds", uncertainty_delay_seconds),
            ("baseline_delay_seconds", baseline_delay_seconds),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got {value!r}")
        if not math.isfinite(uncertainty_threshold) or not 0.0 <= uncertainty_threshold <= 1.0:
            raise ValueError(f"uncertainty_threshold must be in [0.0, 1.0], got {uncertainty_threshold!r}")
        self._contradiction_delay_seconds = contradiction_delay_seconds
        self._uncertainty_delay_seconds = uncertainty_delay_seconds
        self._baseline_delay_seconds = baseline_delay_seconds
        self._uncertainty_threshold = uncertainty_threshold

    def schedule(
        self,
        workspace: WorkspaceState,
        report: CriticReport,
    ) -> list[ScheduledCognitiveTask]:
        """Produce follow-up tasks from the critic report.

        Scheduling is skipped for background stimuli to prevent recursive task
        generation. Otherwise the scheduler emits contradiction review work,
        uncertainty follow-up work, and a low-priority baseline revisit when no
        more specific task was scheduled.
        """
        if workspace.stimulus.metadata.get(StimulusMetadataKey.BACKGROUND_TASK.value):
            return []

        tasks: list[ScheduledCognitiveTask] = []
        turn_index = workspace.snapshot.turn_index
        session_id = workspace.stimulus.session_id
        dominant_goal_id = workspace.dominant_goal.goal_id if workspace.dominant_goal else None
        # Wall-clock is used because due_at is persisted to disk and compared
        # across process restarts. Callers must tolerate clock skew.
        now_ts = time.time()

        if report.contradictions_detected:
            tasks.append(
                ScheduledCognitiveTask(
                    task_id=f"conflict-review:{session_id}:{turn_index}:{report.cycle_id}",
                    description="Review contradictory context on a later pass",
                    status=ScheduledTaskStatus.PENDING,
                    priority=self._clamp_priority(
                        self._CONTRADICTION_BASE_PRIORITY
                        + self._CONTRADICTION_PER_ITEM_PRIORITY * len(report.contradictions_detected)
                    ),
                    due_at=now_ts + self._contradiction_delay_seconds,
                    goal_id=dominant_goal_id,
                    metadata={
                        "reason": TaskReason.CONTRADICTIONS.value,
                        "cycle_id": report.cycle_id,
                        "scheduled_at_ts": now_ts,
                    },
                )
            )
        if report.follow_up_recommended and workspace.snapshot.uncertainty >= self._uncertainty_threshold:
            tasks.append(
                ScheduledCognitiveTask(
                    task_id=f"memory-followup:{session_id}:{turn_index}:{report.cycle_id}",
                    description="Retrieve additional supporting context",
                    status=ScheduledTaskStatus.PENDING,
                    priority=self._clamp_priority(self._UNCERTAINTY_PRIORITY),
                    due_at=now_ts + self._uncertainty_delay_seconds,
                    goal_id=dominant_goal_id,
                    metadata={
                        "reason": TaskReason.UNCERTAINTY.value,
                        "cycle_id": report.cycle_id,
                        "scheduled_at_ts": now_ts,
                    },
                )
            )
        if not tasks:
            tasks.append(
                ScheduledCognitiveTask(
                    task_id=f"stability-check:{session_id}:{turn_index}:{report.cycle_id}",
                    description="Revisit cycle stability if conditions change",
                    status=ScheduledTaskStatus.PENDING,
                    priority=self._clamp_priority(self._BASELINE_PRIORITY),
                    due_at=now_ts + self._baseline_delay_seconds,
                    goal_id=dominant_goal_id,
                    metadata={
                        "reason": TaskReason.BASELINE.value,
                        "cycle_id": report.cycle_id,
                        "scheduled_at_ts": now_ts,
                        "follow_up_recommended": report.follow_up_recommended,
                    },
                )
            )
        return tasks

    @staticmethod
    def _clamp_priority(value: float) -> float:
        return max(0.0, min(1.0, value))
