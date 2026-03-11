from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List


logger = logging.getLogger(__name__)


class CognitiveMaintenanceService:
    def __init__(
        self,
        *,
        get_session_id: Callable[[], str],
        get_current_fatigue: Callable[[], float],
        set_current_fatigue: Callable[[float], None],
        get_attention_focus: Callable[[], List[Any]],
        get_active_goals: Callable[[], List[Any]],
        get_conversation_context: Callable[[], List[Dict[str, Any]]],
        get_memory: Callable[[], Any],
        get_attention: Callable[[], Any],
        get_sensory_processor: Callable[[], Any],
        get_dream_processor: Callable[[], Any],
    ) -> None:
        self._get_session_id = get_session_id
        self._get_current_fatigue = get_current_fatigue
        self._set_current_fatigue = set_current_fatigue
        self._get_attention_focus = get_attention_focus
        self._get_active_goals = get_active_goals
        self._get_conversation_context = get_conversation_context
        self._get_memory = get_memory
        self._get_attention = get_attention
        self._get_sensory_processor = get_sensory_processor
        self._get_dream_processor = get_dream_processor

    def get_cognitive_status(self) -> Dict[str, Any]:
        try:
            try:
                memory_status = self._get_memory().get_status()
            except Exception as exc:
                logger.error("Error getting memory status: %s", exc)
                memory_status = {"error": str(exc), "stm": {"vector_db_count": 0}, "ltm": {"vector_db_count": 0}}

            try:
                attention_status = self._get_attention().get_attention_status()
            except Exception as exc:
                logger.error("Error getting attention status: %s", exc)
                attention_status = {"error": str(exc), "available_capacity": 0.0, "cognitive_load": 0.0}

            try:
                sensory_stats = self._get_sensory_processor().get_processing_stats()
            except Exception as exc:
                logger.error("Error getting sensory stats: %s", exc)
                sensory_stats = {"error": str(exc), "total_processed": 0, "filtered_count": 0}

            current_fatigue = self._get_current_fatigue()
            attention_focus = self._get_attention_focus()
            conversation_context = self._get_conversation_context()

            return {
                "session_id": self._get_session_id(),
                "fatigue_level": current_fatigue,
                "attention_focus": attention_focus,
                "active_goals": self._get_active_goals(),
                "conversation_length": len(conversation_context),
                "last_interaction": conversation_context[-1]["timestamp"] if conversation_context else None,
                "memory_status": memory_status,
                "attention_status": attention_status,
                "sensory_processing": sensory_stats,
                "cognitive_integration": {
                    "attention_memory_sync": len(attention_focus) > 0 and memory_status.get("stm", {}).get("vector_db_count", 0) > 0,
                    "processing_capacity": attention_status.get("available_capacity", 0.0),
                    "overall_efficiency": 1.0 - current_fatigue,
                    "sensory_efficiency": 1.0 - (sensory_stats.get("filtered_count", 0) / max(1, sensory_stats.get("total_processed", 1))),
                },
            }
        except Exception as exc:
            logger.error("Critical error getting cognitive status: %s", exc)
            return {
                "session_id": self._get_session_id(),
                "fatigue_level": self._get_current_fatigue(),
                "attention_focus": [],
                "active_goals": [],
                "conversation_length": 0,
                "last_interaction": None,
                "memory_status": {"error": str(exc)},
                "attention_status": {"error": str(exc)},
                "sensory_processing": {"error": str(exc)},
                "cognitive_integration": {"error": str(exc)},
            }

    async def enter_dream_state(self, cycle_type: str = "deep") -> Dict[str, Any]:
        logger.info("Entering %s dream state for memory consolidation...", cycle_type)
        dream_results = await self._get_dream_processor().enter_dream_cycle(cycle_type)
        attention_rest = self._get_attention().rest_attention(duration_minutes=dream_results.get("actual_duration", 5))

        logger.debug("Dream state results: %s", dream_results)
        logger.debug("Attention rest results: %s", attention_rest)

        self._set_current_fatigue(self._get_attention().current_fatigue)
        logger.info("Advanced dream state processing completed")
        return dream_results

    def take_cognitive_break(self, duration_minutes: float = 1.0) -> Dict[str, Any]:
        logger.info("Taking cognitive break for %s minutes...", duration_minutes)
        rest_results = self._get_attention().rest_attention(duration_minutes)
        self._set_current_fatigue(self._get_attention().current_fatigue)

        logger.info("Cognitive break completed. Fatigue reduced by %.3f", rest_results["fatigue_reduction"])
        return {
            "break_duration": duration_minutes,
            "fatigue_before": rest_results["fatigue_reduction"] + self._get_current_fatigue(),
            "fatigue_after": self._get_current_fatigue(),
            "cognitive_load_reduction": rest_results["load_reduction"],
            "attention_items_lost": rest_results["items_lost_focus"],
            "recovery_effective": rest_results["fatigue_reduction"] > 0.05,
        }

    def force_dream_cycle(self, cycle_type: str = "deep") -> None:
        self._get_dream_processor().force_dream_cycle(cycle_type)

    def get_dream_statistics(self) -> Dict[str, Any]:
        return self._get_dream_processor().get_dream_statistics()

    def is_dreaming(self) -> bool:
        return self._get_dream_processor().is_dreaming

    async def shutdown(self) -> None:
        logger.info("Shutting down cognitive agent...")
        self._get_dream_processor().shutdown()
        logger.info("Cognitive agent shutdown complete")