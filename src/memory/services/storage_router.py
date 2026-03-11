from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List


logger = logging.getLogger(__name__)


class MemoryStorageRouter:
    def __init__(
        self,
        *,
        get_config: Callable[[], Any],
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        create_episodic_memory: Callable[..., Any],
        append_session_memory: Callable[[Dict[str, Any]], None],
        increment_operation: Callable[[str], None],
        increment_error: Callable[[str], None],
        should_consolidate: Callable[[], bool],
        schedule_consolidation: Callable[[], None],
        should_process_prospective: Callable[[], bool],
        schedule_prospective_processing: Callable[[], None],
    ) -> None:
        self._get_config = get_config
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._create_episodic_memory = create_episodic_memory
        self._append_session_memory = append_session_memory
        self._increment_operation = increment_operation
        self._increment_error = increment_error
        self._should_consolidate = should_consolidate
        self._schedule_consolidation = schedule_consolidation
        self._should_process_prospective = should_process_prospective
        self._schedule_prospective_processing = schedule_prospective_processing

    def determine_storage_system(self, importance: float, emotional_valence: float, force_ltm: bool) -> str:
        config = self._get_config()
        if (
            force_ltm
            or importance >= config.consolidation_threshold_importance
            or abs(emotional_valence) >= config.consolidation_threshold_emotional
        ):
            return "LTM"
        return "STM"

    def store_in_ltm(
        self,
        memory_id: str,
        content: Any,
        memory_type: str,
        importance: float,
        tags: List[str],
        associations: List[str],
    ) -> bool:
        try:
            return self._get_ltm().store(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                associations=associations,
            )
        except Exception as exc:
            logger.error("Failed to store %s in LTM: %s", memory_id, exc)
            return False

    def store_in_stm(
        self,
        memory_id: str,
        content: Any,
        importance: float,
        attention_score: float,
        emotional_valence: float,
    ) -> bool:
        try:
            return self._get_stm().store(
                memory_id=memory_id,
                content=content,
                importance=importance,
                attention_score=attention_score,
                emotional_valence=emotional_valence,
            )
        except Exception as exc:
            logger.error("Failed to store %s in STM: %s", memory_id, exc)
            return False

    def create_episodic_memory_for_storage(
        self,
        memory_id: str,
        content: Any,
        importance: float,
        emotional_valence: float,
        force_ltm: bool,
    ) -> Any:
        try:
            return self._create_episodic_memory(
                summary=str(content)[:128],
                detailed_content=str(content),
                importance=importance,
                emotional_valence=emotional_valence,
                stm_ids=[memory_id] if not force_ltm else [],
                ltm_ids=[memory_id] if force_ltm else [],
            )
        except Exception as exc:
            logger.warning("Failed to create episodic memory for %s: %s", memory_id, exc)
            return None

    def store_memory(
        self,
        *,
        result_factory: Callable[..., Any],
        memory_id: str,
        content: Any,
        importance: float,
        attention_score: float,
        emotional_valence: float,
        memory_type: str,
        tags: List[str],
        associations: List[str],
        force_ltm: bool,
        operation_time: datetime,
    ) -> Any:
        session_entry = {
            "memory_id": memory_id,
            "timestamp": operation_time,
            "importance": importance,
            "storage_location": None,
            "memory_type": memory_type,
        }

        if importance > 0.6:
            self.create_episodic_memory_for_storage(
                memory_id=memory_id,
                content=content,
                importance=importance,
                emotional_valence=emotional_valence,
                force_ltm=force_ltm,
            )

        storage_system = self.determine_storage_system(importance, emotional_valence, force_ltm)

        success = False
        if storage_system == "LTM":
            success = self.store_in_ltm(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                associations=associations,
            )
            session_entry["storage_location"] = "LTM"
        else:
            success = self.store_in_stm(
                memory_id=memory_id,
                content=content,
                importance=importance,
                attention_score=attention_score,
                emotional_valence=emotional_valence,
            )
            session_entry["storage_location"] = "STM"

        if success:
            self._append_session_memory(session_entry)
            self._increment_operation("store")

            if self._should_consolidate():
                self._schedule_consolidation()

            if self._get_config().auto_process_prospective and self._should_process_prospective():
                self._schedule_prospective_processing()

            return result_factory(
                success=True,
                memory_id=memory_id,
                operation_time=operation_time,
                system_used=storage_system,
            )

        self._increment_error("store")
        return result_factory(
            success=False,
            memory_id=memory_id,
            error_message=f"Failed to store in {storage_system}",
            operation_time=operation_time,
        )