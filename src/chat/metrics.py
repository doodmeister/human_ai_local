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
