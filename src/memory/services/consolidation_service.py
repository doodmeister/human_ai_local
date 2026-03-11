from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict


logger = logging.getLogger(__name__)


class MemoryConsolidationService:
    def __init__(
        self,
        *,
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        get_config: Callable[[], Any],
        get_last_consolidation: Callable[[], datetime],
        set_last_consolidation: Callable[[datetime], None],
        get_executor: Callable[[], Any],
        get_consolidation_lock: Callable[[], Any],
    ) -> None:
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._get_config = get_config
        self._get_last_consolidation = get_last_consolidation
        self._set_last_consolidation = set_last_consolidation
        self._get_executor = get_executor
        self._get_consolidation_lock = get_consolidation_lock

    def should_consolidate(self) -> bool:
        time_elapsed = (datetime.now() - self._get_last_consolidation()).total_seconds()
        return time_elapsed >= self._get_config().consolidation_interval

    def schedule_consolidation(self, consolidate: Callable[[bool], Dict[str, Any]]) -> None:
        def consolidation_task() -> None:
            try:
                with self._get_consolidation_lock():
                    consolidate(False)
            except Exception as exc:
                logger.error("Background consolidation failed: %s", exc)

        self._get_executor().submit(consolidation_task)

    def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        if not force and not self.should_consolidate():
            return {"status": "skipped", "reason": "not due"}

        stats: Dict[str, Any] = {
            "start_time": datetime.now(),
            "consolidated_count": 0,
            "failed_count": 0,
            "errors": [],
        }

        try:
            stm = self._get_stm()
            ltm = self._get_ltm()

            if hasattr(stm, "get_all_memories"):
                stm_memories = getattr(stm, "get_all_memories")()
                stm_items = {mem.id: mem for mem in stm_memories}
            else:
                stm_items = getattr(stm, "items", {})

            for memory_id, memory_item in stm_items.items():
                try:
                    importance = getattr(memory_item, "importance", 0.0)
                    emotional_valence = getattr(memory_item, "emotional_valence", 0.0)
                    age_minutes = (datetime.now() - memory_item.encoding_time).total_seconds() / 60

                    try:
                        from src.learning.learning_law import clamp01, utility_score
                    except Exception:  # pragma: no cover
                        def clamp01(value: Any) -> float:  # type: ignore[no-redef]
                            try:
                                return float(value)
                            except (TypeError, ValueError):
                                return 0.0

                        def utility_score(*, benefit: Any, cost: Any, benefit_weight: float = 1.0, cost_weight: float = 1.0) -> float:  # type: ignore[no-redef]
                            return (benefit_weight * clamp01(benefit)) - (cost_weight * clamp01(cost))

                    benefit = clamp01(
                        max(
                            float(importance or 0.0),
                            float(abs(emotional_valence) or 0.0),
                            min(1.0, float(age_minutes) / 30.0),
                        )
                    )
                    should_consolidate = utility_score(benefit=benefit, cost=0.10) >= 0.50

                    if should_consolidate:
                        success = ltm.store(
                            memory_id=memory_id,
                            content=memory_item.content,
                            memory_type="episodic",
                            importance=importance,
                            tags=[],
                            associations=getattr(memory_item, "associations", []),
                        )

                        if success:
                            if hasattr(stm, "remove_item"):
                                getattr(stm, "remove_item")(memory_id)
                            else:
                                getattr(stm, "delete", lambda value: None)(memory_id)
                            stats["consolidated_count"] += 1
                        else:
                            stats["failed_count"] += 1
                except Exception as exc:
                    stats["errors"].append(f"Error consolidating {memory_id}: {exc}")
                    stats["failed_count"] += 1
                    logger.error("Consolidation error for %s: %s", memory_id, exc)

            self._set_last_consolidation(datetime.now())
            stats["end_time"] = datetime.now()
            stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
            logger.info("Consolidation completed: %s memories moved to LTM", stats["consolidated_count"])
        except Exception as exc:
            stats["errors"].append(f"Consolidation process error: {exc}")
            logger.error("Consolidation process error: %s", exc)

        return stats