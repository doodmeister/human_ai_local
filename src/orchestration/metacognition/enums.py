from __future__ import annotations

from enum import Enum


class GoalKind(str, Enum):
    RESPONSE = "response"
    REFLECTION = "reflection"
    MEMORY = "memory"
    ATTENTION = "attention"
    PLANNING = "planning"


class InputType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    VOICE = "voice"
    IMAGE = "image"
    TOOL_RESULT = "tool_result"
    BACKGROUND_TASK = "background_task"


class PolicyName(str, Enum):
    UNSET = "unset"
    HEURISTIC_V1 = "heuristic_policy_v1"


class CognitiveActType(str, Enum):
    RESPOND = "respond"
    RETRIEVE_CONTEXT = "retrieve_context"
    STORE_MEMORY = "store_memory"
    REFOCUS_ATTENTION = "refocus_attention"
    INSPECT_CONFLICT = "inspect_conflict"
    DEFER = "defer"


class CycleStage(str, Enum):
    SNAPSHOT = "snapshot"
    WORKSPACE = "workspace"
    GOALS = "goals"
    POLICY = "policy"
    EXECUTION = "execution"
    CRITIC = "critic"
    SCHEDULING = "scheduling"
    COMPLETE = "complete"


class ScheduledTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StimulusMetadataKey(str, Enum):
    BACKGROUND_TASK = "background_task"
    SCHEDULER_TICK = "scheduler_tick"
    TASK_ID = "task_id"
    TASK_REASON = "task_reason"


class TaskReason(str, Enum):
    CONTRADICTIONS = "contradictions"
    UNCERTAINTY = "uncertainty"
    BASELINE = "baseline"
    BACKGROUND_CONTRADICTION_AUDIT = "background_contradiction_audit"


class BackgroundTaskReason(str, Enum):
    CONTRADICTION_AUDIT = TaskReason.BACKGROUND_CONTRADICTION_AUDIT.value


class ReflectionTrigger(str, Enum):
    IDLE_SCHEDULER = "idle_background_scheduler"
