from __future__ import annotations

from enum import Enum


class GoalKind(str, Enum):
    RESPONSE = "response"
    REFLECTION = "reflection"
    MEMORY = "memory"
    ATTENTION = "attention"
    PLANNING = "planning"


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
    CANCELLED = "cancelled"
