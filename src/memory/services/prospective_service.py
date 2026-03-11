from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable


logger = logging.getLogger(__name__)


class MemoryProspectiveService:
    def __init__(
        self,
        *,
        get_config: Callable[[], Any],
        get_last_process: Callable[[], datetime],
        set_last_process: Callable[[datetime], None],
        get_executor: Callable[[], Any],
        get_prospective: Callable[[], Any],
        get_ltm: Callable[[], Any],
    ) -> None:
        self._get_config = get_config
        self._get_last_process = get_last_process
        self._set_last_process = set_last_process
        self._get_executor = get_executor
        self._get_prospective = get_prospective
        self._get_ltm = get_ltm

    def should_process(self) -> bool:
        time_elapsed = (datetime.now() - self._get_last_process()).total_seconds()
        return time_elapsed >= self._get_config().prospective_process_interval

    def schedule_processing(self) -> None:
        def prospective_task() -> None:
            try:
                processed = self._get_prospective().process_due_reminders(ltm_system=self._get_ltm())
                if processed > 0:
                    logger.info("Processed %s due prospective reminders", processed)
                self._set_last_process(datetime.now())
            except Exception as exc:
                logger.error("Background prospective processing failed: %s", exc)

        self._get_executor().submit(prospective_task)