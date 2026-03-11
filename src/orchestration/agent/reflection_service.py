from __future__ import annotations

from datetime import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List

import schedule


logger = logging.getLogger(__name__)


class CognitiveReflectionService:
    def __init__(self, *, get_memory: Callable[[], Any]) -> None:
        self._get_memory = get_memory
        self._reports: List[Dict[str, Any]] = []
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_running = False

    @property
    def reports(self) -> List[Dict[str, Any]]:
        return self._reports

    def reflect(self) -> Dict[str, Any]:
        timestamp = datetime.now().isoformat()
        memory = self._get_memory()
        ltm = memory.ltm
        stm = memory.stm
        ltm_metacognitive_stats = None
        ltm_health_report = None
        stm_metacognitive_stats = None

        if hasattr(ltm, "get_memory_health_report"):
            try:
                ltm_health_report = ltm.get_memory_health_report()
            except Exception:
                ltm_health_report = None

        if hasattr(stm, "get_all_memories"):
            try:
                all_memories = stm.get_all_memories()
                capacity = getattr(getattr(stm, "config", None), "capacity", None)
                capacity_value = float(capacity) if capacity else 1.0
                error_count = float(getattr(stm, "_error_count", 0.0))
                operation_count = float(getattr(stm, "_operation_count", 0.0))
                stm_metacognitive_stats = {
                    "capacity_utilization": len(all_memories) / max(1.0, capacity_value),
                    "error_rate": error_count / max(1.0, operation_count),
                    "memory_count": len(all_memories),
                    "avg_importance": sum(getattr(memory_item, "importance", 0.0) for memory_item in all_memories)
                    / max(1, len(all_memories)),
                    "recent_activity": operation_count,
                }
            except Exception:
                stm_metacognitive_stats = None

        report = {
            "timestamp": timestamp,
            "ltm_metacognitive_stats": ltm_metacognitive_stats,
            "ltm_health_report": ltm_health_report,
            "stm_metacognitive_stats": stm_metacognitive_stats,
            "ltm_status": ltm.get_status() if hasattr(ltm, "get_status") else None,
            "stm_status": stm.get_status() if hasattr(stm, "get_status") else None,
        }
        self._reports.append(report)
        return report

    def get_reports(self, n: int = 5) -> List[Dict[str, Any]]:
        return self._reports[-n:]

    def start_scheduler(self, interval_minutes: int = 10) -> None:
        if self._scheduler_running:
            logger.info("Reflection scheduler already running.")
            return

        self._scheduler_running = True
        schedule.clear("reflection")
        schedule.every(interval_minutes).minutes.do(self.reflect).tag("reflection")

        def run_scheduler() -> None:
            while self._scheduler_running:
                schedule.run_pending()
                time.sleep(1)

        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("Started metacognitive reflection scheduler (every %s min)", interval_minutes)

    def stop_scheduler(self) -> None:
        self._scheduler_running = False
        schedule.clear("reflection")
        logger.info("Stopped metacognitive reflection scheduler.")

    def get_status(self) -> Dict[str, Any]:
        status = {
            "scheduler_running": self._scheduler_running,
            "interval_minutes": None,
        }
        try:
            jobs = [job for job in schedule.get_jobs("reflection")]
            if jobs:
                status["interval_minutes"] = jobs[0].interval
        except Exception:
            pass
        return status

    def get_last_report(self) -> Dict[str, Any]:
        if self._reports:
            return self._reports[-1]
        return {}

    def clear_reports(self) -> None:
        self._reports.clear()