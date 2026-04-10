"""Filesystem-backed persistence for metacognition traces and related artifacts.

This module stores cycle traces, self-model snapshots, scheduled task queues,
and reflection episodes on disk with indexed lookup, atomic writes, and
best-effort locking for mutable session-scoped state.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4

from .models import AttentionStatus, AttentionUpdate, ContradictionRecord, FocusItem, GoalMetadata, MemoryStatus, MemoryUpdate, PlanMetadata, RetrievalContextItem


class FilesystemCycleTracer:
    """Persist cycle traces as JSON files for later inspection and replay."""

    def __init__(
        self,
        root_dir: str | Path = "data/cognition_traces",
        *,
        max_traces_per_session: int | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._traces_dir = self.root_dir / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._locks_dir = self.root_dir / ".locks"
        self._locks_dir.mkdir(parents=True, exist_ok=True)
        self._max_traces_per_session = max_traces_per_session
        self._lock_timeout_seconds = max(0.1, float(lock_timeout_seconds))
        self._trace_index: dict[str, dict[str, Any]] = {}
        self._rebuild_trace_index()

    def write_trace(self, trace: Mapping[str, Any]) -> str:
        payload = self._normalize(trace)
        trace_id = str(payload.get("cycle_id") or uuid4())
        payload.setdefault("cycle_id", trace_id)
        session_id = self._trace_session_id(payload)
        path = self._trace_path(trace_id, session_id)
        self._atomic_write_json(path, payload)
        self._trace_index[trace_id] = self._trace_index_entry(path=path, trace_id=trace_id, session_id=session_id)
        if self._max_traces_per_session is not None and session_id is not None:
            self._prune_traces_for_session(session_id, max_traces=self._max_traces_per_session)
        return trace_id

    def load_trace(self, trace_id: str) -> dict[str, Any] | None:
        entry = self._trace_index.get(str(trace_id))
        if entry is None:
            self._rebuild_trace_index()
            entry = self._trace_index.get(str(trace_id))
        if entry is None:
            return None
        payload = self._read_json_file(entry["path"], fallback=None)
        if not isinstance(payload, dict):
            self._trace_index.pop(str(trace_id), None)
            return None
        return payload

    def latest_trace(self) -> dict[str, Any] | None:
        for trace_id, entry in reversed(self._ordered_trace_entries()):
            payload = self._read_json_file(entry["path"], fallback=None)
            if isinstance(payload, dict):
                return payload
            self._trace_index.pop(trace_id, None)
        return None

    def latest_trace_for_session(self, session_id: str) -> dict[str, Any] | None:
        for trace_id, entry in reversed(self._ordered_trace_entries(session_id=session_id)):
            payload = self._read_json_file(entry["path"], fallback=None)
            if isinstance(payload, dict):
                return payload
            self._trace_index.pop(trace_id, None)
        return None

    def list_traces(self, *, session_id: str | None = None, limit: int | None = 20) -> list[dict[str, Any]]:
        entries = self._ordered_trace_entries(session_id=session_id)
        if limit is not None and limit >= 0:
            entries = entries[-limit:]
        traces: list[dict[str, Any]] = []
        for trace_id, entry in entries:
            payload = self._read_json_file(entry["path"], fallback=None)
            if isinstance(payload, dict):
                traces.append(payload)
                continue
            self._trace_index.pop(trace_id, None)
        return traces

    def persist_self_model(self, session_id: str, self_model: Mapping[str, Any] | Any) -> Path:
        directory = self.root_dir / "self_models"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._session_key(session_id)}.json"
        payload = self._normalize(self_model)
        payload.setdefault("session_id", session_id)
        self._locked_write_json(path, payload)
        return path

    def load_self_model(self, session_id: str) -> dict[str, Any] | None:
        path = self.root_dir / "self_models" / f"{self._session_key(session_id)}.json"
        payload = self._read_json_file(path, fallback=None)
        return payload if isinstance(payload, dict) else None

    def persist_task_queue(self, session_id: str, tasks: Sequence[Mapping[str, Any] | Any]) -> Path:
        directory = self.root_dir / "task_queues"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._session_key(session_id)}.json"
        normalized_tasks = [self._ensure_task_id(task) for task in self._normalize(list(tasks))]
        self._locked_write_json(path, normalized_tasks)
        return path

    def load_task_queue(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        path = self.root_dir / "task_queues" / f"{self._session_key(session_id)}.json"
        payload = self._read_json_file(path, fallback=[])
        if not isinstance(payload, list):
            return []
        tasks = [task for task in payload if isinstance(task, dict)]
        if limit is not None and limit >= 0:
            tasks = tasks[-limit:]
        return tasks

    def upsert_task_queue(self, session_id: str, tasks: Sequence[Mapping[str, Any] | Any]) -> Path:
        directory = self.root_dir / "task_queues"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._session_key(session_id)}.json"

        def _merge(existing_payload: Any) -> list[dict[str, Any]]:
            existing: dict[str, dict[str, Any]] = {}
            if isinstance(existing_payload, list):
                for task in existing_payload:
                    normalized_task = self._ensure_task_id(task)
                    if isinstance(normalized_task, dict):
                        existing[str(normalized_task["task_id"])] = normalized_task
            for task in tasks:
                normalized_task = self._ensure_task_id(self._normalize(task))
                if isinstance(normalized_task, dict):
                    existing[str(normalized_task["task_id"])] = normalized_task
            return sorted(
                existing.values(),
                key=lambda item: (
                    float(item.get("due_at") or 0.0),
                    -float(item.get("priority") or 0.0),
                    str(item.get("task_id") or ""),
                ),
            )

        self._locked_json_update(path, fallback=[], updater=_merge)
        return path

    def append_reflection_episode(self, session_id: str, report: Mapping[str, Any]) -> Path:
        directory = self.root_dir / "reflections" / self._session_key(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = self._session_key(str(report.get("timestamp") or uuid4()))
        path = directory / f"{timestamp}_{uuid4().hex[:6]}.json"
        payload = self._normalize(report)
        payload.setdefault("session_id", session_id)
        self._atomic_write_json(path, payload)
        return path

    def list_reflection_episodes(self, session_id: str, *, limit: int | None = 20) -> list[dict[str, Any]]:
        directory = self.root_dir / "reflections" / self._session_key(session_id)
        if not directory.exists():
            return []
        paths = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime)
        if limit is not None and limit >= 0:
            paths = paths[-limit:]
        reports: list[dict[str, Any]] = []
        for path in paths:
            payload = self._read_json_file(path, fallback=None)
            if isinstance(payload, dict):
                reports.append(payload)
        return reports

    def prune_reflection_episodes(self, session_id: str, *, max_episodes: int) -> int:
        if max_episodes < 0:
            return 0
        directory = self.root_dir / "reflections" / self._session_key(session_id)
        if not directory.exists():
            return 0
        total_pruned = 0
        for _ in range(2):
            paths = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime)
            excess = len(paths) - max_episodes
            if excess <= 0:
                break
            for path in paths[:excess]:
                path.unlink(missing_ok=True)
                total_pruned += 1
        return total_pruned

    def build_regression_metrics(
        self,
        *,
        traces: Sequence[Mapping[str, Any]] | None = None,
        session_id: str | None = None,
        limit: int | None = 20,
    ) -> dict[str, Any]:
        trace_list = list(traces) if traces is not None else self.list_traces(session_id=session_id, limit=limit)
        if not trace_list:
            return {
                "trace_count": 0,
                "average_success_score": None,
                "follow_up_rate": None,
                "latest_trace_id": None,
            }
        success_scores = []
        follow_up_count = 0
        latest_trace_id = None
        for trace in trace_list:
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
            "trace_count": len(trace_list),
            "average_success_score": average_success_score,
            "follow_up_rate": follow_up_count / len(trace_list),
            "latest_trace_id": latest_trace_id,
        }

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, (ContradictionRecord, RetrievalContextItem, MemoryStatus, AttentionStatus, GoalMetadata, PlanMetadata, MemoryUpdate, AttentionUpdate)):
            return cls._normalize(value.to_dict())
        if isinstance(value, FocusItem):
            return cls._normalize(value.to_primitive())
        if is_dataclass(value):
            return {
                field.name: cls._normalize(getattr(cast(Any, value), field.name))
                for field in fields(cast(Any, value))
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): cls._normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._normalize(item) for item in value]
        return value

    @staticmethod
    def _session_key(session_id: str) -> str:
        key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(session_id))
        key = key.strip("_")
        return (key or "unknown_session")[:200]

    def _rebuild_trace_index(self) -> None:
        self._trace_index = {}
        for path in sorted(self.root_dir.glob("*.json")):
            self._index_trace_file(path)
        if not self._traces_dir.exists():
            return
        for path in sorted(self._traces_dir.glob("*/*.json")):
            self._index_trace_file(path)

    def _index_trace_file(self, path: Path) -> None:
        payload = self._read_json_file(path, fallback=None)
        if not isinstance(payload, dict):
            return
        trace_id = str(payload.get("cycle_id") or path.stem)
        session_id = self._trace_session_id(payload)
        self._trace_index[trace_id] = self._trace_index_entry(path=path, trace_id=trace_id, session_id=session_id)

    def _trace_index_entry(self, *, path: Path, trace_id: str, session_id: str | None) -> dict[str, Any]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return {
            "trace_id": trace_id,
            "path": path,
            "mtime": mtime,
            "session_id": session_id,
        }

    def _ordered_trace_entries(self, *, session_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        entries = []
        for trace_id, entry in self._trace_index.items():
            if session_id is not None and entry.get("session_id") != session_id:
                continue
            entries.append((trace_id, entry))
        entries.sort(key=lambda item: (float(item[1].get("mtime") or 0.0), item[0]))
        return entries

    def _trace_path(self, trace_id: str, session_id: str | None) -> Path:
        session_key = self._session_key(session_id or "unknown_session")
        directory = self._traces_dir / session_key
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self._trace_filename(trace_id)}.json"

    @classmethod
    def _trace_filename(cls, trace_id: str) -> str:
        normalized = cls._session_key(trace_id)
        digest = hashlib.sha1(str(trace_id).encode("utf-8")).hexdigest()[:12]
        return f"{normalized[:120]}-{digest}"

    @staticmethod
    def _trace_session_id(payload: Mapping[str, Any]) -> str | None:
        stimulus = payload.get("stimulus") if isinstance(payload, Mapping) else None
        if isinstance(stimulus, Mapping):
            session_id = stimulus.get("session_id")
            if session_id:
                return str(session_id)
        session_id = payload.get("session_id") if isinstance(payload, Mapping) else None
        if session_id:
            return str(session_id)
        return None

    def _prune_traces_for_session(self, session_id: str, *, max_traces: int) -> int:
        if max_traces < 0:
            return 0
        entries = self._ordered_trace_entries(session_id=session_id)
        excess = len(entries) - max_traces
        if excess <= 0:
            return 0
        pruned = 0
        for trace_id, entry in entries[:excess]:
            path = entry["path"]
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            self._trace_index.pop(trace_id, None)
            pruned += 1
        return pruned

    def _locked_write_json(self, path: Path, payload: Any) -> None:
        with self._file_lock(path):
            self._atomic_write_json(path, payload)

    def _locked_json_update(self, path: Path, *, fallback: Any, updater: Callable[[Any], Any]) -> None:
        with self._file_lock(path):
            current = self._read_json_file(path, fallback=fallback)
            updated = updater(current)
            self._atomic_write_json(path, updated)

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self._normalize(payload), indent=2, sort_keys=True, default=str)
        temp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(serialized, encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @contextmanager
    def _file_lock(self, path: Path):
        relative_name = str(path.relative_to(self.root_dir)).replace("/", "_").replace("\\", "_")
        lock_name = self._session_key(relative_name)
        lock_path = self._locks_dir / f"{lock_name}.lock"
        deadline = time.monotonic() + self._lock_timeout_seconds
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for tracer lock: {lock_path.name}")
                time.sleep(0.01)
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
            yield
        finally:
            try:
                os.close(fd)
            finally:
                lock_path.unlink(missing_ok=True)

    @staticmethod
    def _read_json_file(path: Path, *, fallback: Any) -> Any:
        if not path.exists():
            return fallback
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _ensure_task_id(task: Any) -> Any:
        if not isinstance(task, dict):
            return task
        normalized_task = dict(task)
        task_id = normalized_task.get("task_id")
        if not task_id:
            normalized_task["task_id"] = f"auto-{uuid4().hex[:8]}"
        else:
            normalized_task["task_id"] = str(task_id)
        return normalized_task
