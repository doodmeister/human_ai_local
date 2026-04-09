from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


class FilesystemCycleTracer:
    """Persist cycle traces as JSON files for later inspection and replay."""

    def __init__(self, root_dir: str | Path = "data/cognition_traces") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def write_trace(self, trace: Mapping[str, Any]) -> str:
        trace_id = str(trace.get("cycle_id") or uuid4())
        payload = dict(trace)
        payload.setdefault("cycle_id", trace_id)
        path = self.root_dir / f"{trace_id}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=self._json_default),
            encoding="utf-8",
        )
        return trace_id

    def load_trace(self, trace_id: str) -> dict[str, Any]:
        path = self.root_dir / f"{trace_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def latest_trace(self) -> dict[str, Any] | None:
        traces = sorted(self.root_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
        if not traces:
            return None
        return json.loads(traces[-1].read_text(encoding="utf-8"))

    def latest_trace_for_session(self, session_id: str) -> dict[str, Any] | None:
        traces = self.list_traces(session_id=session_id, limit=1)
        if not traces:
            return None
        return traces[-1]

    def list_traces(self, *, session_id: str | None = None, limit: int | None = 20) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for path in sorted(self.root_dir.glob("*.json"), key=lambda item: item.stat().st_mtime):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if session_id is not None:
                stimulus = payload.get("stimulus") or {}
                if stimulus.get("session_id") != session_id:
                    continue
            traces.append(payload)
        if limit is not None and limit >= 0:
            return traces[-limit:]
        return traces

    def persist_self_model(self, session_id: str, self_model: Mapping[str, Any]) -> Path:
        directory = self.root_dir / "self_models"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._session_key(session_id)}.json"
        payload = dict(self_model)
        payload.setdefault("session_id", session_id)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=self._json_default),
            encoding="utf-8",
        )
        return path

    def load_self_model(self, session_id: str) -> dict[str, Any] | None:
        path = self.root_dir / "self_models" / f"{self._session_key(session_id)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def persist_task_queue(self, session_id: str, tasks: list[Mapping[str, Any]]) -> Path:
        directory = self.root_dir / "task_queues"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._session_key(session_id)}.json"
        path.write_text(
            json.dumps(list(tasks), indent=2, sort_keys=True, default=self._json_default),
            encoding="utf-8",
        )
        return path

    def load_task_queue(self, session_id: str) -> list[dict[str, Any]]:
        path = self.root_dir / "task_queues" / f"{self._session_key(session_id)}.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [task for task in payload if isinstance(task, dict)]

    def upsert_task_queue(self, session_id: str, tasks: list[Mapping[str, Any]]) -> Path:
        existing = {task.get("task_id"): dict(task) for task in self.load_task_queue(session_id) if task.get("task_id")}
        for task in tasks:
            task_id = task.get("task_id")
            if not task_id:
                continue
            existing[str(task_id)] = dict(task)
        ordered = sorted(
            existing.values(),
            key=lambda task: (
                float(task.get("due_at") or 0.0),
                -float(task.get("priority") or 0.0),
                str(task.get("task_id") or ""),
            ),
        )
        return self.persist_task_queue(session_id, ordered)

    def append_reflection_episode(self, session_id: str, report: Mapping[str, Any]) -> Path:
        directory = self.root_dir / "reflections" / self._session_key(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = str(report.get("timestamp") or uuid4()).replace(":", "-")
        path = directory / f"{timestamp}.json"
        payload = dict(report)
        payload.setdefault("session_id", session_id)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=self._json_default),
            encoding="utf-8",
        )
        return path

    def list_reflection_episodes(self, session_id: str, *, limit: int | None = 20) -> list[dict[str, Any]]:
        directory = self.root_dir / "reflections" / self._session_key(session_id)
        if not directory.exists():
            return []
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime)]
        if limit is not None and limit >= 0:
            return reports[-limit:]
        return reports

    def build_regression_metrics(self, *, session_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        traces = self.list_traces(session_id=session_id, limit=limit)
        if not traces:
            return {
                "trace_count": 0,
                "average_success_score": None,
                "follow_up_rate": None,
                "latest_trace_id": None,
            }
        success_scores = []
        follow_up_count = 0
        latest_trace_id = None
        for trace in traces:
            critic_report = trace.get("critic_report") or {}
            success_score = critic_report.get("success_score")
            if isinstance(success_score, (int, float)):
                success_scores.append(float(success_score))
            if critic_report.get("follow_up_recommended"):
                follow_up_count += 1
            latest_trace_id = trace.get("cycle_id") or latest_trace_id
        average_success_score = None
        if success_scores:
            average_success_score = sum(success_scores) / len(success_scores)
        return {
            "trace_count": len(traces),
            "average_success_score": average_success_score,
            "follow_up_rate": follow_up_count / len(traces),
            "latest_trace_id": latest_trace_id,
        }

    @staticmethod
    def _json_default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        return str(value)

    @staticmethod
    def _session_key(session_id: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in session_id)
