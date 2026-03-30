from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Sequence


logger = logging.getLogger(__name__)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        maybe_mapping = value.to_dict()
        if isinstance(maybe_mapping, Mapping):
            return maybe_mapping
    return {}


class MemoryReconsolidationService:
    def __init__(
        self,
        *,
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        get_episodic: Callable[[], Any],
        get_semantic: Callable[[], Any],
        increment_operation: Callable[[str], None],
        increment_error: Callable[[str], None],
    ) -> None:
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._get_episodic = get_episodic
        self._get_semantic = get_semantic
        self._increment_operation = increment_operation
        self._increment_error = increment_error
        self._recent_events: List[Dict[str, Any]] = []

    def record_recall_outcome(
        self,
        recalled_memories: Sequence[Mapping[str, Any]] | None,
        *,
        outcome: str = "reinforce",
        response_policy: Mapping[str, Any] | Any | None = None,
        note: str | None = None,
    ) -> Dict[str, Any]:
        normalized_outcome = str(outcome or "reinforce").lower()
        if normalized_outcome not in {"reinforce", "failed", "correction"}:
            normalized_outcome = "reinforce"

        items = list(recalled_memories or [])
        scale = self._policy_scale(normalized_outcome, response_policy)
        stats: Dict[str, Any] = {
            "outcome": normalized_outcome,
            "attempted": len(items),
            "updated": 0,
            "skipped": 0,
            "errors": [],
            "scale": round(scale, 4),
        }

        for item in items:
            memory_id = item.get("id") or item.get("memory_id")
            source = str(item.get("source", "")).strip().upper()
            if not memory_id or not source:
                stats["skipped"] += 1
                continue

            try:
                updated = self._apply_to_source(
                    source=source,
                    memory_id=str(memory_id),
                    outcome=normalized_outcome,
                    scale=scale,
                    note=note,
                )
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "memory_id": str(memory_id),
                    "source": source,
                    "outcome": normalized_outcome,
                    "scale": round(scale, 4),
                    "updated": bool(updated),
                    "note": note,
                }
                self._append_event(event)
                if updated:
                    stats["updated"] += 1
                    self._increment_operation(f"reconsolidate_{normalized_outcome}")
                else:
                    stats["skipped"] += 1
            except Exception as exc:
                stats["errors"].append(f"{source}:{memory_id}:{exc}")
                self._increment_error("reconsolidate")
                logger.error("Reconsolidation failed for %s %s: %s", source, memory_id, exc)

        return stats

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._recent_events[-limit:])

    def _append_event(self, event: Dict[str, Any]) -> None:
        self._recent_events.append(dict(event))
        if len(self._recent_events) > 200:
            self._recent_events = self._recent_events[-100:]

    def _policy_scale(self, outcome: str, response_policy: Mapping[str, Any] | Any | None) -> float:
        policy_dict = _as_mapping(response_policy)
        effective = _as_mapping(policy_dict.get("effective", policy_dict))
        uncertainty = _clamp01(effective.get("uncertainty", 0.5), default=0.5)
        directness = _clamp01(effective.get("directness", 0.5), default=0.5)
        disclosure = _clamp01(effective.get("disclosure", 0.5), default=0.5)

        if outcome == "reinforce":
            raw = 0.8 + ((1.0 - uncertainty) * 0.3) + (directness * 0.15)
            return max(0.5, min(1.25, raw))

        raw = 0.9 + (directness * 0.15) + (disclosure * 0.05)
        return max(0.75, min(1.15, raw))

    def _feedback_deltas(self, outcome: str, scale: float) -> Dict[str, float]:
        if outcome == "reinforce":
            return {
                "importance_delta": 0.04 * scale,
                "confidence_delta": 0.06 * scale,
                "attention_delta": 0.05 * scale,
                "strength_increment": 0.08 * scale,
            }
        if outcome == "correction":
            return {
                "importance_delta": -0.05 * scale,
                "confidence_delta": -0.16 * scale,
                "attention_delta": -0.03 * scale,
                "strength_increment": 0.0,
            }
        return {
            "importance_delta": -0.02 * scale,
            "confidence_delta": -0.08 * scale,
            "attention_delta": -0.02 * scale,
            "strength_increment": 0.0,
        }

    def _apply_to_source(
        self,
        *,
        source: str,
        memory_id: str,
        outcome: str,
        scale: float,
        note: str | None,
    ) -> bool:
        deltas = self._feedback_deltas(outcome, scale)

        if source == "STM":
            stm = self._get_stm()
            if hasattr(stm, "apply_recall_feedback"):
                return bool(
                    stm.apply_recall_feedback(
                        memory_id,
                        importance_delta=deltas["importance_delta"],
                        attention_delta=deltas["attention_delta"],
                    )
                )
            if hasattr(stm, "retrieve"):
                return stm.retrieve(memory_id) is not None
            return False

        if source == "LTM":
            ltm = self._get_ltm()
            if hasattr(ltm, "apply_recall_feedback"):
                return bool(
                    ltm.apply_recall_feedback(
                        memory_id,
                        outcome=outcome,
                        importance_delta=deltas["importance_delta"],
                        confidence_delta=deltas["confidence_delta"],
                        note=note,
                    )
                )
            return False

        if source == "EPISODIC":
            episodic = self._get_episodic()
            if hasattr(episodic, "apply_recall_feedback"):
                return bool(
                    episodic.apply_recall_feedback(
                        memory_id,
                        outcome=outcome,
                        importance_delta=deltas["importance_delta"],
                        confidence_delta=deltas["confidence_delta"],
                        strength_increment=deltas["strength_increment"],
                        note=note,
                    )
                )
            return False

        if source == "SEMANTIC":
            semantic = self._get_semantic()
            if hasattr(semantic, "apply_recall_feedback"):
                return bool(
                    semantic.apply_recall_feedback(
                        memory_id,
                        outcome=outcome,
                        importance_delta=deltas["importance_delta"],
                        confidence_delta=deltas["confidence_delta"],
                        note=note,
                    )
                )
            return False

        return False