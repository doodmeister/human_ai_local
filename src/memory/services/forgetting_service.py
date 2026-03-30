from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping


class MemoryForgettingService:
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

    def apply_policy(
        self,
        *,
        min_importance: float = 0.3,
        min_confidence: float = 0.35,
        min_access_count: int = 1,
        min_age_days: float = 14.0,
        decay_stm: bool = True,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "stm_evicted": 0,
            "ltm_suppressed": 0,
            "episodic_suppressed": 0,
            "semantic_suppressed": 0,
            "protected": 0,
            "errors": [],
        }

        try:
            if decay_stm:
                stm = self._get_stm()
                if hasattr(stm, "decay_memories"):
                    evicted = stm.decay_memories()
                    stats["stm_evicted"] = len(evicted or [])
                    if evicted:
                        self._append_event(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "source": "STM",
                                "action": "evict",
                                "count": len(evicted),
                            }
                        )
                        self._increment_operation("forget_stm")
        except Exception as exc:
            stats["errors"].append(f"STM:{exc}")
            self._increment_error("forget_stm")

        for label, getter in (
            ("LTM", self._get_ltm),
            ("Episodic", self._get_episodic),
            ("Semantic", self._get_semantic),
        ):
            try:
                store = getter()
                if hasattr(store, "apply_forgetting_policy"):
                    result = store.apply_forgetting_policy(
                        min_importance=min_importance,
                        min_confidence=min_confidence,
                        min_access_count=min_access_count,
                        min_age_days=min_age_days,
                    )
                    suppressed = int(result.get("suppressed", 0) or 0)
                    protected = int(result.get("protected", 0) or 0)
                    stats[f"{label.lower()}_suppressed"] = suppressed
                    stats["protected"] += protected
                    if suppressed:
                        self._increment_operation(f"forget_{label.lower()}")
                    if suppressed or protected:
                        self._append_event(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "source": label.upper(),
                                "action": "suppress",
                                "suppressed": suppressed,
                                "protected": protected,
                            }
                        )
            except Exception as exc:
                stats["errors"].append(f"{label}:{exc}")
                self._increment_error(f"forget_{label.lower()}")

        return stats

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._recent_events[-limit:])

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self._recent_events.append(dict(event))
        if len(self._recent_events) > 200:
            self._recent_events = self._recent_events[-100:]