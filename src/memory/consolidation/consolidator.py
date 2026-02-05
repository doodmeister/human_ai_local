from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import time
import uuid
from src.memory.metrics import metrics_registry
from src.learning.learning_law import clamp01, utility_score

@dataclass
class ConsolidationPolicy:
    salience_threshold: float = 0.55
    valence_threshold: float = 0.60
    min_rehearsals_for_promotion: int = 2
    min_age_seconds: float = 5.0
    promotion_importance_floor: float = 0.4

@dataclass
class ConsolidationEvent:
    event_id: str
    user_turn_id: str
    stored_in_stm: bool
    promoted_to_ltm: bool
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

class MemoryConsolidator:
    """Handles storing chat turns in STM and promoting to LTM with simple heuristics."""
    def __init__(self, stm: Optional[Any], ltm: Optional[Any], policy: Optional[ConsolidationPolicy] = None):
        self.stm = stm
        self.ltm = ltm
        self.policy = policy or ConsolidationPolicy()
        self._turn_stats: Dict[str, Dict[str, Any]] = {}
        self._events: list[ConsolidationEvent] = []

    def record_turn(self, turn_id: str, salience: float, valence: float, importance: float, content: str) -> ConsolidationEvent:
        now = time.time()
        stats = self._turn_stats.setdefault(turn_id, {"rehearsals": 0, "first_seen": now, "stm_id": None, "ltm_id": None, "importance": importance, "valence": valence})
        stored_in_stm = False

        # Phase 7: utility-based learning law.
        # Benefit: content salience/valence/importance (normalized to 0..1).
        # Cost: STM pressure (best-effort utilization estimate).
        benefit = clamp01((0.45 * salience) + (0.20 * abs(valence)) + (0.35 * importance))
        cost = self._estimate_stm_pressure()
        u = utility_score(benefit=benefit, cost=cost)

        # Preserve prior behavior by gating on original thresholds, but reduce to the single utility signal.
        # If thresholds indicate likely usefulness, allow utility to dominate; under pressure, utility drops.
        threshold_gate = (
            (salience >= self.policy.salience_threshold)
            or (abs(valence) >= self.policy.valence_threshold)
            or (importance >= self.policy.promotion_importance_floor)
        )
        should_store = threshold_gate and (u >= -0.15)

        if stats["stm_id"] is None and self.stm and should_store:
            mem_id = f"turn-{turn_id}"
            try:
                self.stm.store(mem_id, content, importance=importance, attention_score=salience, emotional_valence=valence)  # type: ignore[attr-defined]
                stats["stm_id"] = mem_id
                stored_in_stm = True
                metrics_registry.inc("consolidation_stm_store_total")
            except Exception:
                pass
        ev = ConsolidationEvent(
            event_id=str(uuid.uuid4()),
            user_turn_id=turn_id,
            stored_in_stm=stored_in_stm,
            promoted_to_ltm=False,
            timestamp=now,
            details={"salience": salience, "valence": valence, "importance": importance},
        )
        self._events.append(ev)
        return ev

    def mark_rehearsal(self, turn_id: str):
        stats = self._turn_stats.get(turn_id)
        if not stats:
            return
        stats["rehearsals"] += 1
        self._maybe_promote(turn_id, stats)

    def _maybe_promote(self, turn_id: str, stats: Dict[str, Any]):
        if not self.ltm or not self.stm:
            return
        if stats.get("ltm_id") is not None:
            return
        if stats.get("stm_id") is None:
            return
        age = time.time() - stats.get("first_seen", 0)
        if stats.get("rehearsals", 0) < self.policy.min_rehearsals_for_promotion:
            return
        if age < self.policy.min_age_seconds:
            return

        # Phase 7: utility-based learning law.
        # Benefit: rehearsal evidence + importance; Cost: STM pressure (promotion relieves STM but costs LTM writes).
        # We treat STM pressure as a *negative* cost to promotion (i.e., relief is beneficial), so use (1 - pressure).
        rehearsals = float(stats.get("rehearsals", 0))
        rehearsal_benefit = clamp01(rehearsals / max(1.0, float(self.policy.min_rehearsals_for_promotion) + 1.0))
        importance = float(stats.get("importance", 0.5))
        benefit = clamp01(0.55 * importance + 0.45 * rehearsal_benefit)
        # If STM is pressured, promotion is more valuable (lower cost).
        cost = clamp01(0.30 * (1.0 - self._estimate_stm_pressure()))
        u = utility_score(benefit=benefit, cost=cost)
        if u < 0.10:
            return
        importance = stats.get("importance", 0.5)
        stm_id = stats["stm_id"]
        content = None
        try:
            item = self.stm.retrieve(stm_id)  # type: ignore[attr-defined]
            if item:
                content = getattr(item, 'content', None) or getattr(item, 'text', None)
        except Exception:
            return
        if not content:
            return
        ltm_id = f"ltm-{stm_id}"
        try:
            self.ltm.store(ltm_id, content, importance=importance, emotional_valence=stats.get('valence', 0.0))  # type: ignore[attr-defined]
            stats["ltm_id"] = ltm_id
            self._events.append(ConsolidationEvent(
                event_id=str(uuid.uuid4()),
                user_turn_id=turn_id,
                stored_in_stm=False,
                promoted_to_ltm=True,
                timestamp=time.time(),
                details={"stm_id": stm_id, "ltm_id": ltm_id, "rehearsals": stats.get("rehearsals")}
            ))
            metrics_registry.inc("consolidation_ltm_promotions_total")
            metrics_registry.observe_hist("consolidation_promotion_age_seconds", age)
        except Exception:
            return

    def _estimate_stm_pressure(self) -> float:
        """Best-effort estimate of STM utilization pressure in [0,1]."""
        stm_obj = self.stm
        if stm_obj is None:
            return 0.0
        cap = getattr(stm_obj, "capacity", None)
        size = None
        try:
            if hasattr(stm_obj, "__len__"):
                size = len(stm_obj)  # type: ignore[arg-type]
        except Exception:
            size = None
        if size is None:
            size = getattr(stm_obj, "size", None)
        if isinstance(size, int) and isinstance(cap, int) and cap > 0:
            return clamp01(size / cap)
        return 0.0

    def events_tail(self, n: int = 20):
        return [e.__dict__ for e in self._events[-n:]]

    def status(self) -> Dict[str, Any]:
        promoted = sum(1 for e in self._events if e.promoted_to_ltm)
        stored = sum(1 for e in self._events if e.stored_in_stm)
        return {
            "total_events": len(self._events),
            "stored_in_stm": stored,
            "promoted_to_ltm": promoted,
            "counters": {
                "stm_store_total": metrics_registry.counters.get("consolidation_stm_store_total", 0),
                "ltm_promotions_total": metrics_registry.counters.get("consolidation_ltm_promotions_total", 0),
            },
        }
