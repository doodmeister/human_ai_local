from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict


logger = logging.getLogger(__name__)


class MemoryStatusService:
    def __init__(
        self,
        *,
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        get_config: Callable[[], Any],
        get_start_time: Callable[[], datetime],
        get_last_consolidation: Callable[[], datetime],
        get_session_memories_count: Callable[[], int],
        get_operation_counts: Callable[[], Dict[str, int]],
        get_error_counts: Callable[[], Dict[str, int]],
        is_active: Callable[[], bool],
        clear_session_memories: Callable[[], None],
    ) -> None:
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._get_config = get_config
        self._get_start_time = get_start_time
        self._get_last_consolidation = get_last_consolidation
        self._get_session_memories_count = get_session_memories_count
        self._get_operation_counts = get_operation_counts
        self._get_error_counts = get_error_counts
        self._is_active = is_active
        self._clear_session_memories = clear_session_memories

    def get_status(self) -> Dict[str, Any]:
        try:
            uptime = (datetime.now() - self._get_start_time()).total_seconds()
            config = self._get_config()
            stm = self._get_stm()
            ltm = self._get_ltm()
            return {
                "stm": stm.get_status() if hasattr(stm, "get_status") else {"status": "unknown"},
                "ltm": ltm.get_status() if hasattr(ltm, "get_status") else {"status": "unknown"},
                "last_consolidation": self._get_last_consolidation(),
                "session_memories_count": self._get_session_memories_count(),
                "consolidation_interval": config.consolidation_interval,
                "use_vector_stm": config.use_vector_stm,
                "use_vector_ltm": config.use_vector_ltm,
                "system_active": self._is_active(),
                "uptime_seconds": uptime,
                "operation_counts": self._get_operation_counts(),
                "error_counts": self._get_error_counts(),
                "config": {
                    "stm_capacity": config.stm_capacity,
                    "max_concurrent_operations": config.max_concurrent_operations,
                    "auto_process_prospective": config.auto_process_prospective,
                },
            }
        except Exception as exc:
            logger.error("Error getting system status: %s", exc)
            return {"error": str(exc), "system_active": False}

    def reset_session(self) -> None:
        self._clear_session_memories()
        logger.info("Memory system session reset")