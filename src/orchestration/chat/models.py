from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import uuid


def _ts() -> float:
    return time.time()


@dataclass
class TurnRecord:
    """Represents a single conversational turn."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=_ts)
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    salience: float = 0.0
    emotional_valence: float = 0.0  # -1..1
    importance: float = 0.0
    embedding: Optional[List[float]] = None
    consolidation_status: str = "pending"  # pending|stored|skipped
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextItem:
    """Item included (or considered) for model context."""
    source_id: str
    source_system: str  # turn|stm|ltm|episodic|attention|executive
    content: str
    rank: int = 0
    included: bool = True
    reason: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    forced: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStageResult:
    name: str
    candidates_in: int
    candidates_out: int
    latency_ms: float
    rationale: str = ""
    added: int = 0
    removed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceTrace:
    pipeline_stages: List[PipelineStageResult] = field(default_factory=list)
    scoring_version: str = "v0.1"
    degraded_mode: bool = False
    notes: List[str] = field(default_factory=list)

    def add_stage(self, stage: PipelineStageResult) -> None:
        self.pipeline_stages.append(stage)


@dataclass
class ChatMetrics:
    turn_latency_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    stm_hits: int = 0
    ltm_hits: int = 0
    episodic_hits: int = 0
    attention_boost: float = 0.0
    fatigue_delta: float = 0.0
    consolidation_time_ms: float = 0.0
    fallback_used: bool = False
    consolidated_user_turn: bool = False  # added flag


@dataclass
class BuiltContext:
    session_id: str
    items: List[ContextItem]
    trace: ProvenanceTrace
    metrics: ChatMetrics
    used_turn_ids: List[str] = field(default_factory=list)
