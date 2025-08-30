from __future__ import annotations
import time
from collections import defaultdict
from typing import Any, Dict, List


class ChatMetricsRegistry:
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.histograms: Dict[str, list] = {}
        self.state: Dict[str, Any] = {}
        self.events: Dict[str, List[float]] = defaultdict(list)  # timestamped events for rate calculations

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, ms: float) -> None:
        self.timings[name].append(ms)

    def observe_hist(self, name: str, value: float, max_len: int = 500):
        bucket = self.histograms.setdefault(name, [])
        bucket.append(value)
        if len(bucket) > max_len:
            # keep recent window
            del bucket[0 : len(bucket) - max_len]

    def snapshot(self) -> Dict[str, Any]:
        snap = {
            "counters": dict(self.counters),
            "timings": dict(self.timings),
            "histograms": {k: list(v) for k, v in self.histograms.items()},
            "state": dict(self.state),
        }
        # derive p95 for latency histogram if present
        if "chat_turn_latency_ms" in self.histograms and self.histograms["chat_turn_latency_ms"]:
            data = sorted(self.histograms["chat_turn_latency_ms"])
            idx = max(0, int(len(data) * 0.95) - 1)
            p95 = data[idx]
            snap["state"]["chat_turn_latency_p95_ms"] = p95
        return snap

    def snapshot_light(self, hist_tail: int = 5) -> Dict[str, Any]:
        """
        Lightweight snapshot limiting histogram arrays to last hist_tail entries.
        """
        base = self.snapshot()
        base["histograms"] = {
            k: v[-hist_tail:] for k, v in base["histograms"].items()
        }
        return base

    def get_p95(self, hist_name: str) -> float:
        data = self.histograms.get(hist_name)
        if not data:
            return 0.0
        ordered = sorted(data)
        idx = max(0, int(len(ordered) * 0.95) - 1)
        return ordered[idx]

    def percentile(self, hist_name: str, q: float) -> float:
        """
        Return q-th percentile (0-100) for a histogram; 0.0 if empty.
        """
        if not (0 <= q <= 100):
            raise ValueError("q must be in [0,100]")
        data = self.histograms.get(hist_name)
        if not data:
            return 0.0
        ordered = sorted(data)
        if not ordered:
            return 0.0
        # Nearest-rank method
        rank = int(round((q / 100) * (len(ordered) - 1)))
        return ordered[min(len(ordered) - 1, max(0, rank))]

    def mark_event(self, name: str, timestamp: float | None = None, max_len: int = 1000):
        """
        Record an instantaneous event (for throughput / rate metrics).
        """
        ts = timestamp or time.time()
        events = self.events[name]
        events.append(ts)
        if len(events) > max_len:
            del events[0 : len(events) - max_len]

    def get_rate(self, name: str, window_seconds: float = 60.0) -> float:
        """
        Events per second over the given sliding window.
        """
        now = time.time()
        events = self.events.get(name, [])
        if not events:
            return 0.0
        cutoff = now - window_seconds
        # binary search optional; simple filter for small lists
        recent = [e for e in events if e >= cutoff]
        return len(recent) / window_seconds if window_seconds > 0 else 0.0

    def reset(self):
        """
        Clear all collected metrics (useful for test isolation).
        """
        self.counters.clear()
        self.timings.clear()
        self.histograms.clear()
        self.state.clear()
        self.events.clear()

    def export_state(self) -> Dict[str, Any]:
        """
        Export current metrics (excluding raw timing lists for brevity).
        """
        snap = self.snapshot()
        return {
            "counters": snap["counters"],
            "state": snap["state"],
            "hist_summary": {k: {"count": len(v)} for k, v in self.histograms.items()},
        }

    def import_state(self, data: Dict[str, Any]):
        """
        Merge imported counters and state (does not reconstruct raw lists).
        """
        for k, v in data.get("counters", {}).items():
            self.counters[k] = int(v)
        for k, v in data.get("state", {}).items():
            self.state[k] = v

# Backward compatibility alias
MetricsRegistry = ChatMetricsRegistry


metrics_registry = ChatMetricsRegistry()


def timed(name: str):
    def _wrap(fn):
        def _inner(*a, **kw):
            start = time.time()
            try:
                return fn(*a, **kw)
            finally:
                metrics_registry.observe(name, (time.time() - start) * 1000.0)
        return _inner
    return _wrap
