from __future__ import annotations

import hashlib
import time
from typing import Any
from uuid import uuid4

from .enums import BackgroundTaskReason, InputType, ScheduledTaskStatus, StimulusMetadataKey
from .critic import DefaultCritic
from .event_bus import InProcessEventBus, MetacognitiveEvent
from .executor import DefaultPlanExecutor
from .goal_manager import HeuristicGoalManager
from .interfaces import (
    CognitiveScheduler,
    Critic,
    CycleTracer,
    EventBus,
    GoalManager,
    PlanExecutor,
    PolicyEngine,
    SelfModelUpdater,
    StateProvider,
    WorkspaceBuilder,
)
from .models import ContradictionRecord, MetacognitiveCycleResult, Stimulus
from .policy_engine import HeuristicPolicyEngine
from .scorecard import build_metacognitive_scorecard
from .scheduler import DefaultCognitiveScheduler
from .self_model_updater import DefaultSelfModelUpdater


class MetacognitiveController:
    """Coordinate one deterministic metacognitive cycle end-to-end."""

    def __init__(
        self,
        *,
        state_provider: StateProvider,
        workspace_builder: WorkspaceBuilder,
        goal_manager: GoalManager | None = None,
        policy_engine: PolicyEngine | None = None,
        executor: PlanExecutor | None = None,
        critic: Critic | None = None,
        self_model_updater: SelfModelUpdater | None = None,
        scheduler: CognitiveScheduler | None = None,
        event_bus: EventBus | None = None,
        cycle_tracer: CycleTracer | None = None,
        max_reflection_episodes: int | None = 200,
    ) -> None:
        self._state_provider = state_provider
        self._workspace_builder = workspace_builder
        self._goal_manager = goal_manager or HeuristicGoalManager()
        self._policy_engine = policy_engine or HeuristicPolicyEngine()
        self._executor = executor or DefaultPlanExecutor()
        self._critic = critic or DefaultCritic()
        self._self_model_updater = self_model_updater or DefaultSelfModelUpdater()
        self._scheduler = scheduler or DefaultCognitiveScheduler()
        self._event_bus = event_bus or InProcessEventBus()
        self._cycle_tracer = cycle_tracer
        self._max_reflection_episodes = max_reflection_episodes

    def run_cycle(self, stimulus: Stimulus) -> MetacognitiveCycleResult:
        cycle_id = f"cycle-{uuid4()}"
        self._event_bus.publish(MetacognitiveEvent.CYCLE_STARTED, {"cycle_id": cycle_id, "session_id": stimulus.session_id})
        snapshot = None
        workspace = None
        ranked_goals = ()
        proposals = ()
        plan = None
        execution_result = None
        critic_report = None
        updated_self_model = None
        scheduled_tasks = ()
        completed_stages: list[str] = []
        failure: dict[str, Any] | None = None

        try:
            snapshot = self._state_provider.snapshot(stimulus)
            completed_stages.append("snapshot")
            self._event_bus.publish(MetacognitiveEvent.SNAPSHOT_CREATED, {"cycle_id": cycle_id, "turn_index": snapshot.turn_index})

            workspace = self._workspace_builder.build(stimulus, snapshot)
            completed_stages.append("workspace")

            ranked_goals = tuple(self._goal_manager.rank_goals(workspace))
            if ranked_goals:
                workspace.dominant_goal = ranked_goals[0]
            completed_stages.append("goals")

            proposals = tuple(self._policy_engine.propose_acts(workspace, list(ranked_goals)))
            plan = self._policy_engine.select_plan(workspace, list(ranked_goals), list(proposals))
            completed_stages.append("policy")

            execution_result = self._executor.execute(workspace, plan)
            completed_stages.append("execution")

            critic_report = self._critic.evaluate(workspace, plan, execution_result, cycle_id=cycle_id)
            completed_stages.append("critic")

            updated_self_model = self._self_model_updater.apply(snapshot.self_model, critic_report)
            completed_stages.append("self_model_updater")

            scheduled_tasks = tuple(self._scheduler.schedule(workspace, critic_report))
            completed_stages.append("scheduler")
        except Exception as exc:
            failure = {
                "stage": self._infer_failed_stage(completed_stages),
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }

        metadata = {
            "proposal_count": len(proposals),
            "completed_stages": completed_stages,
            "failed": failure is not None,
        }
        if failure is not None:
            metadata["error"] = failure

        trace_id = self._record_cycle_artifacts(
            cycle_id=cycle_id,
            stimulus=stimulus,
            workspace=workspace,
            ranked_goals=ranked_goals,
            proposals=proposals,
            plan=plan,
            execution_result=execution_result,
            critic_report=critic_report,
            updated_self_model=updated_self_model,
            scheduled_tasks=scheduled_tasks,
            metadata=metadata,
        )

        result = MetacognitiveCycleResult(
            cycle_id=cycle_id,
            workspace=workspace,
            ranked_goals=tuple(ranked_goals),
            plan=plan,
            execution_result=execution_result,
            critic_report=critic_report,
            updated_self_model=updated_self_model,
            scheduled_tasks=scheduled_tasks,
            trace_id=trace_id,
            metadata=metadata,
        )
        self._event_bus.publish(
            MetacognitiveEvent.CYCLE_COMPLETED,
            {
                "cycle_id": cycle_id,
                "trace_id": trace_id,
                "success_score": critic_report.success_score if critic_report is not None else None,
                "scheduled_task_count": len(scheduled_tasks),
                "failed": failure is not None,
                "failed_stage": failure.get("stage") if failure is not None else None,
            },
        )
        return result

    def get_latest_trace(self, session_id: str) -> dict[str, Any] | None:
        if self._cycle_tracer is None:
            return None
        return self._cycle_tracer.latest_trace_for_session(session_id)

    def get_persisted_self_model(self, session_id: str) -> dict[str, Any] | None:
        if self._cycle_tracer is None:
            return None
        return self._cycle_tracer.load_self_model(session_id)

    def persist_reflection_episode(self, session_id: str, report: dict[str, Any]) -> None:
        if self._cycle_tracer is None:
            return
        self._cycle_tracer.append_reflection_episode(session_id, report)
        if self._max_reflection_episodes is not None:
            self._cycle_tracer.prune_reflection_episodes(session_id, max_episodes=self._max_reflection_episodes)

    def list_reflection_episodes(self, session_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        if self._cycle_tracer is None:
            return []
        return self._cycle_tracer.list_reflection_episodes(session_id, limit=limit)

    def build_status(self, session_id: str, *, history_limit: int = 10) -> dict[str, Any]:
        latest_trace = None if self._cycle_tracer is not None else self.get_latest_trace(session_id)
        persisted_self_model = self.get_persisted_self_model(session_id)
        reflection_episodes = self.list_reflection_episodes(session_id, limit=history_limit)
        queued_tasks = self.list_tasks(session_id)
        now_ts = time.time()
        metrics: dict[str, Any] = {}
        if self._cycle_tracer is not None:
            traces = self._cycle_tracer.list_traces(session_id=session_id, limit=history_limit)
            latest_trace = traces[-1] if traces else None
            metrics = self._cycle_tracer.build_regression_metrics(
                traces=traces,
                session_id=session_id,
                limit=history_limit,
            )
        return {
            "available": any((latest_trace, persisted_self_model, reflection_episodes)),
            "session_id": session_id,
            "last_trace": latest_trace,
            "persisted_self_model": persisted_self_model,
            "reflection_episode_count": len(reflection_episodes),
            "unresolved_contradiction_count": len(self.list_unresolved_contradictions(session_id)),
            "pending_task_count": len([task for task in queued_tasks if task.get("status") == ScheduledTaskStatus.PENDING.value]),
            "due_task_count": len(
                [
                    task
                    for task in queued_tasks
                    if task.get("status") == ScheduledTaskStatus.PENDING.value and float(task.get("due_at") or 0.0) <= now_ts
                ]
            ),
            "regression_metrics": metrics,
        }

    def list_tasks(self, session_id: str) -> list[dict[str, Any]]:
        if self._cycle_tracer is None:
            return []
        return list(self._cycle_tracer.load_task_queue(session_id))

    def build_scorecard(self, session_id: str, *, limit: int = 50) -> dict[str, Any]:
        traces: list[dict[str, Any]] = []
        if self._cycle_tracer is not None:
            traces = list(self._cycle_tracer.list_traces(session_id=session_id, limit=limit))
        return build_metacognitive_scorecard(traces, session_id=session_id).to_dict()

    def list_unresolved_contradictions(self, session_id: str) -> list[ContradictionRecord]:
        try:
            contradictions = self._workspace_builder.contradictions_for_session(session_id)
        except Exception:
            contradictions = []
        normalized: list[ContradictionRecord] = []
        for item in contradictions:
            contradiction = ContradictionRecord.from_value(item)
            if not (contradiction.description or contradiction.kind):
                continue
            normalized.append(contradiction)
        return normalized

    def run_contradiction_audit(
        self,
        session_id: str,
        *,
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        contradictions = self.list_unresolved_contradictions(session_id)
        if not contradictions:
            return {
                "session_id": session_id,
                "contradiction_count": 0,
                "enqueued_count": 0,
                "enqueued_task_ids": [],
            }

        queued_tasks = self.list_tasks(session_id)
        existing_task_ids = {str(task.get("task_id")) for task in queued_tasks if task.get("task_id")}
        current_ts = float(now_ts if now_ts is not None else time.time())
        enqueued_tasks: list[dict[str, Any]] = []

        for index, contradiction in enumerate(contradictions):
            task_id = self._build_contradiction_task_id(session_id, contradiction, index=index)
            if task_id in existing_task_ids:
                continue
            enqueued_tasks.append(
                {
                    "task_id": task_id,
                    "description": self._describe_contradiction(contradiction),
                    "status": ScheduledTaskStatus.PENDING.value,
                    "priority": self._score_contradiction_priority(contradiction, contradiction_count=len(contradictions)),
                    "due_at": current_ts,
                    "metadata": {
                        "reason": BackgroundTaskReason.CONTRADICTION_AUDIT.value,
                        "audit_generated": True,
                        "contradiction": contradiction.to_dict(),
                    },
                }
            )

        if enqueued_tasks and self._cycle_tracer is not None:
            self._cycle_tracer.upsert_task_queue(session_id, enqueued_tasks)

        return {
            "session_id": session_id,
            "contradiction_count": len(contradictions),
            "enqueued_count": len(enqueued_tasks),
            "enqueued_task_ids": [task["task_id"] for task in enqueued_tasks],
        }

    def run_scheduler_tick(
        self,
        session_id: str,
        *,
        now_ts: float | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        queued_tasks = self.list_tasks(session_id)
        if not queued_tasks:
            return {
                "session_id": session_id,
                "executed_count": 0,
                "executed_task_ids": [],
                "pending_task_count": 0,
            }

        now_value = now_ts if now_ts is not None else time.time()
        pending_due = [
            task
            for task in queued_tasks
            if task.get("status") == ScheduledTaskStatus.PENDING.value and float(task.get("due_at") or 0.0) <= now_value
        ]
        pending_due.sort(key=lambda task: (-float(task.get("priority") or 0.0), float(task.get("due_at") or 0.0)))
        selected = pending_due[: max(0, limit)]
        if not selected:
            return {
                "session_id": session_id,
                "executed_count": 0,
                "executed_task_ids": [],
                "pending_task_count": len(
                    [task for task in queued_tasks if task.get("status") == ScheduledTaskStatus.PENDING.value]
                ),
            }

        task_map = {str(task.get("task_id")): dict(task) for task in queued_tasks if task.get("task_id")}
        executed_task_ids: list[str] = []
        execution_cycle_ids: list[str] = []

        for selected_task in selected:
            task_id = str(selected_task.get("task_id"))
            current_task = dict(task_map[task_id])
            current_task["status"] = ScheduledTaskStatus.RUNNING.value
            current_task["metadata"] = dict(current_task.get("metadata") or {})
            current_task["metadata"]["started_at"] = now_value
            task_map[task_id] = current_task
            self._persist_task_map(session_id, task_map)

            try:
                result = self.run_cycle(
                    Stimulus(
                        session_id=session_id,
                        user_input=current_task.get("description") or task_id,
                            input_type=InputType.BACKGROUND_TASK,
                        metadata={
                            StimulusMetadataKey.BACKGROUND_TASK.value: True,
                            StimulusMetadataKey.SCHEDULER_TICK.value: True,
                            StimulusMetadataKey.TASK_ID.value: task_id,
                            StimulusMetadataKey.TASK_REASON.value: current_task["metadata"].get("reason"),
                        },
                    )
                )
                if result.metadata.get("failed"):
                    current_task["status"] = ScheduledTaskStatus.FAILED.value
                    current_task["metadata"]["failed_at"] = time.time()
                    current_task["metadata"]["execution_cycle_id"] = result.cycle_id
                    current_task["metadata"]["last_error"] = (result.metadata.get("error") or {}).get("error")
                    current_task["metadata"]["last_error_stage"] = (result.metadata.get("error") or {}).get("stage")
                else:
                    current_task["status"] = ScheduledTaskStatus.COMPLETED.value
                    current_task["metadata"]["completed_at"] = time.time()
                    current_task["metadata"]["execution_cycle_id"] = result.cycle_id
                    executed_task_ids.append(task_id)
                    execution_cycle_ids.append(result.cycle_id)
                task_map[task_id] = current_task
            except Exception as exc:
                current_task["status"] = ScheduledTaskStatus.FAILED.value
                current_task["metadata"]["failed_at"] = time.time()
                current_task["metadata"]["last_error"] = str(exc)
                current_task["metadata"]["last_error_type"] = type(exc).__name__
                task_map[task_id] = current_task
            finally:
                self._persist_task_map(session_id, task_map)

        return {
            "session_id": session_id,
            "executed_count": len(executed_task_ids),
            "executed_task_ids": executed_task_ids,
            "execution_cycle_ids": execution_cycle_ids,
            "pending_task_count": len(
                [task for task in task_map.values() if task.get("status") == ScheduledTaskStatus.PENDING.value]
            ),
        }

    def _persist_task_map(self, session_id: str, task_map: dict[str, dict[str, Any]]) -> None:
        if self._cycle_tracer is None:
            return
        ordered_tasks = sorted(
            task_map.values(),
            key=lambda task: (
                float(task.get("due_at") or 0.0),
                -float(task.get("priority") or 0.0),
                str(task.get("task_id") or ""),
            ),
        )
        self._cycle_tracer.persist_task_queue(session_id, ordered_tasks)

    @staticmethod
    def _describe_contradiction(contradiction: ContradictionRecord) -> str:
        return contradiction.description or contradiction.kind or "Review contradictory context on a later pass"

    @staticmethod
    def _build_contradiction_task_id(session_id: str, contradiction: ContradictionRecord, *, index: int) -> str:
        stable_source = "|".join(
            [
                str(contradiction.contradiction_set_id or ""),
                contradiction.kind,
                contradiction.description or contradiction.kind,
                str(index),
            ]
        )
        digest = hashlib.sha256(stable_source.encode("utf-8")).hexdigest()[:12]
        return f"contradiction-audit:{session_id}:{digest}"

    def _record_cycle_artifacts(
        self,
        *,
        cycle_id: str,
        stimulus: Stimulus,
        workspace: Any,
        ranked_goals: tuple[Any, ...],
        proposals: tuple[Any, ...],
        plan: Any,
        execution_result: Any,
        critic_report: Any,
        updated_self_model: Any,
        scheduled_tasks: tuple[Any, ...],
        metadata: dict[str, Any],
    ) -> str | None:
        trace_id = None
        if self._cycle_tracer is None:
            return None

        trace_payload = {
            "cycle_id": cycle_id,
            "stimulus": stimulus,
            "workspace": workspace,
            "ranked_goals": list(ranked_goals),
            "candidate_acts": list(proposals),
            "plan": plan,
            "execution_result": execution_result,
            "critic_report": critic_report,
            "updated_self_model": updated_self_model,
            "scheduled_tasks": list(scheduled_tasks),
            "metadata": metadata,
        }

        try:
            trace_id = self._cycle_tracer.write_trace(trace_payload)
        except Exception as exc:
            metadata["trace_error"] = {
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }

        if updated_self_model is not None:
            try:
                self._cycle_tracer.persist_self_model(stimulus.session_id, updated_self_model)
            except Exception as exc:
                metadata["self_model_persist_error"] = {
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
        if scheduled_tasks:
            try:
                self._cycle_tracer.upsert_task_queue(stimulus.session_id, list(scheduled_tasks))
            except Exception as exc:
                metadata["task_queue_persist_error"] = {
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
        return trace_id

    @staticmethod
    def _infer_failed_stage(completed_stages: list[str]) -> str:
        stage_order = [
            "snapshot",
            "workspace",
            "goals",
            "policy",
            "execution",
            "critic",
            "self_model_updater",
            "scheduler",
        ]
        for stage_name in stage_order:
            if stage_name not in completed_stages:
                return stage_name
        return "unknown"

    @staticmethod
    def _score_contradiction_priority(contradiction: ContradictionRecord, *, contradiction_count: int) -> float:
        severity = contradiction.severity if isinstance(contradiction.severity, (int, float)) else contradiction.confidence
        if isinstance(severity, (int, float)):
            normalized_severity = max(0.0, min(1.0, float(severity)))
            return min(1.0, 0.55 + 0.35 * normalized_severity)
        return min(0.9, 0.60 + 0.03 * max(0, contradiction_count - 1))
