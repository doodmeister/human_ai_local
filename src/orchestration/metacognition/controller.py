from __future__ import annotations

from dataclasses import asdict
import hashlib
import time
from typing import Any
from uuid import uuid4

from .enums import ScheduledTaskStatus
from .critic import DefaultCritic
from .event_bus import InProcessEventBus
from .executor import DefaultPlanExecutor
from .goal_manager import HeuristicGoalManager
from .models import MetacognitiveCycleResult, Stimulus
from .policy_engine import HeuristicPolicyEngine
from .scorecard import build_metacognitive_scorecard
from .scheduler import DefaultCognitiveScheduler
from .self_model_updater import DefaultSelfModelUpdater
from .state_provider import DefaultStateProvider
from .workspace_builder import DefaultWorkspaceBuilder


class MetacognitiveController:
    """Coordinate one deterministic metacognitive cycle end-to-end."""

    def __init__(
        self,
        *,
        state_provider: DefaultStateProvider,
        workspace_builder: DefaultWorkspaceBuilder,
        goal_manager: HeuristicGoalManager | None = None,
        policy_engine: HeuristicPolicyEngine | None = None,
        executor: DefaultPlanExecutor | None = None,
        critic: DefaultCritic | None = None,
        self_model_updater: DefaultSelfModelUpdater | None = None,
        scheduler: DefaultCognitiveScheduler | None = None,
        event_bus: InProcessEventBus | None = None,
        cycle_tracer: object | None = None,
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

    def run_cycle(self, stimulus: Stimulus) -> MetacognitiveCycleResult:
        cycle_id = f"cycle-{uuid4()}"
        self._event_bus.publish("metacognition.cycle.started", {"cycle_id": cycle_id, "session_id": stimulus.session_id})

        snapshot = self._state_provider.snapshot(stimulus)
        self._event_bus.publish("metacognition.snapshot.created", {"cycle_id": cycle_id, "turn_index": snapshot.turn_index})
        workspace = self._workspace_builder.build(stimulus, snapshot)
        ranked_goals = self._goal_manager.rank_goals(workspace)
        if ranked_goals:
            workspace.dominant_goal = ranked_goals[0]

        proposals = self._policy_engine.propose_acts(workspace, ranked_goals)
        plan = self._policy_engine.select_plan(workspace, ranked_goals, proposals)
        execution_result = self._executor.execute(workspace, plan)
        critic_report = self._critic.evaluate(workspace, plan, execution_result)
        updated_self_model = self._self_model_updater.apply(snapshot.self_model, critic_report)
        scheduled_tasks = tuple(self._scheduler.schedule(workspace, critic_report))

        trace_payload = {
            "cycle_id": cycle_id,
            "stimulus": asdict(stimulus),
            "workspace": asdict(workspace),
            "ranked_goals": [asdict(goal) for goal in ranked_goals],
            "plan": asdict(plan),
            "execution_result": asdict(execution_result),
            "critic_report": asdict(critic_report),
            "updated_self_model": asdict(updated_self_model),
            "scheduled_tasks": [asdict(task) for task in scheduled_tasks],
        }
        trace_id = None
        if self._cycle_tracer is not None and hasattr(self._cycle_tracer, "write_trace"):
            trace_id = self._cycle_tracer.write_trace(trace_payload)
        if self._cycle_tracer is not None and hasattr(self._cycle_tracer, "persist_self_model"):
            self._cycle_tracer.persist_self_model(stimulus.session_id, asdict(updated_self_model))
        if scheduled_tasks and self._cycle_tracer is not None and hasattr(self._cycle_tracer, "upsert_task_queue"):
            self._cycle_tracer.upsert_task_queue(
                stimulus.session_id,
                [asdict(task) for task in scheduled_tasks],
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
            metadata={"proposal_count": len(proposals)},
        )
        self._event_bus.publish(
            "metacognition.cycle.completed",
            {
                "cycle_id": cycle_id,
                "trace_id": trace_id,
                "success_score": critic_report.success_score,
                "scheduled_task_count": len(scheduled_tasks),
            },
        )
        return result

    def get_latest_trace(self, session_id: str) -> dict[str, Any] | None:
        if self._cycle_tracer is None:
            return None
        if hasattr(self._cycle_tracer, "latest_trace_for_session"):
            return self._cycle_tracer.latest_trace_for_session(session_id)
        if hasattr(self._cycle_tracer, "latest_trace"):
            return self._cycle_tracer.latest_trace()
        return None

    def get_persisted_self_model(self, session_id: str) -> dict[str, Any] | None:
        if self._cycle_tracer is None or not hasattr(self._cycle_tracer, "load_self_model"):
            return None
        return self._cycle_tracer.load_self_model(session_id)

    def persist_reflection_episode(self, session_id: str, report: dict[str, Any]) -> None:
        if self._cycle_tracer is None or not hasattr(self._cycle_tracer, "append_reflection_episode"):
            return
        self._cycle_tracer.append_reflection_episode(session_id, report)

    def list_reflection_episodes(self, session_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        if self._cycle_tracer is None or not hasattr(self._cycle_tracer, "list_reflection_episodes"):
            return []
        return self._cycle_tracer.list_reflection_episodes(session_id, limit=limit)

    def build_status(self, session_id: str, *, history_limit: int = 10) -> dict[str, Any]:
        latest_trace = self.get_latest_trace(session_id)
        persisted_self_model = self.get_persisted_self_model(session_id)
        reflection_episodes = self.list_reflection_episodes(session_id, limit=history_limit)
        queued_tasks = self.list_tasks(session_id)
        now_ts = time.time()
        metrics: dict[str, Any] = {}
        if self._cycle_tracer is not None and hasattr(self._cycle_tracer, "build_regression_metrics"):
            metrics = self._cycle_tracer.build_regression_metrics(session_id=session_id, limit=history_limit)
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
        if self._cycle_tracer is None or not hasattr(self._cycle_tracer, "load_task_queue"):
            return []
        return list(self._cycle_tracer.load_task_queue(session_id))

    def build_scorecard(self, session_id: str, *, limit: int = 50) -> dict[str, Any]:
        traces: list[dict[str, Any]] = []
        if self._cycle_tracer is not None and hasattr(self._cycle_tracer, "list_traces"):
            traces = list(self._cycle_tracer.list_traces(session_id=session_id, limit=limit))
        return build_metacognitive_scorecard(traces, session_id=session_id).to_dict()

    def list_unresolved_contradictions(self, session_id: str) -> list[dict[str, Any]]:
        if not hasattr(self._workspace_builder, "contradictions_for_session"):
            return []
        try:
            contradictions = self._workspace_builder.contradictions_for_session(session_id)
        except Exception:
            contradictions = []
        return [dict(item) for item in contradictions if isinstance(item, dict)]

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
                    "priority": min(1.0, 0.65 + 0.05 * len(contradictions)),
                    "due_at": current_ts,
                    "metadata": {
                        "reason": "background_contradiction_audit",
                        "audit_generated": True,
                        "contradiction": contradiction,
                    },
                }
            )

        if enqueued_tasks and self._cycle_tracer is not None and hasattr(self._cycle_tracer, "upsert_task_queue"):
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

        task_map = {task.get("task_id"): dict(task) for task in queued_tasks if task.get("task_id")}
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

            result = self.run_cycle(
                Stimulus(
                    session_id=session_id,
                    user_input=current_task.get("description") or task_id,
                    input_type="background_task",
                    metadata={
                        "background_task": True,
                        "scheduler_tick": True,
                        "task_id": task_id,
                        "task_reason": current_task["metadata"].get("reason"),
                    },
                )
            )
            current_task["status"] = ScheduledTaskStatus.COMPLETED.value
            current_task["metadata"]["completed_at"] = time.time()
            current_task["metadata"]["execution_cycle_id"] = result.cycle_id
            task_map[task_id] = current_task
            executed_task_ids.append(task_id)
            execution_cycle_ids.append(result.cycle_id)

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
        if self._cycle_tracer is None or not hasattr(self._cycle_tracer, "persist_task_queue"):
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
    def _describe_contradiction(contradiction: dict[str, Any]) -> str:
        description = str(contradiction.get("description") or contradiction.get("summary") or "Review contradictory context on a later pass")
        return description

    @staticmethod
    def _build_contradiction_task_id(session_id: str, contradiction: dict[str, Any], *, index: int) -> str:
        stable_source = "|".join(
            [
                str(contradiction.get("contradiction_set_id") or ""),
                str(contradiction.get("kind") or contradiction.get("type") or ""),
                str(contradiction.get("description") or contradiction.get("summary") or contradiction),
                str(index),
            ]
        )
        digest = hashlib.sha1(stable_source.encode("utf-8")).hexdigest()[:12]
        return f"contradiction-audit:{session_id}:{digest}"
