"""Benchmark script for executive configurations.

Usage (manual):
    python scripts/benchmark_executive.py
(Integrate with pytest or a CI perf harness later.)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List

from src.executive.executive_core import ExecutiveController, Goal
from src.executive.decision_contextual import ContextualDecisionEngine
from src.executive.planner_hierarchical import HierarchicalPlanner

# --- Simple fakes ---
class DummyMemory:
    def __init__(self):
        self._stm = []
    async def stm_add(self, text: str, tags=None):
        self._stm.append(text)
    async def stm_recent(self, k: int = 10):
        return self._stm[-k:]
    async def ltm_similar(self, query: str, k: int = 8):
        return []
    async def ltm_add(self, text: str, tags=None):
        return "ltm::1"
    async def consolidate(self, items):
        return "consolidated:0"

class DummyLLM:
    async def complete(self, prompt: str, context, max_tokens=512):
        return ("Short answer for benchmark run.", {"tokens_used": 40, "fatigue": 0.25})

class DummyAttention:
    async def allocate(self, *, query, candidates, capacity=5):
        return []

class DummyActuator:
    async def act(self, action_text: str):
        return {"executed": False}

@dataclass
class RunResult:
    label: str
    latency_ms: float
    tokens: int

async def run_config(label: str, controller: ExecutiveController, n: int = 5) -> RunResult:
    goal = Goal(id="gbench", description="Answer user query with context", priority=0.6)
    t0 = time.time()
    tokens_total = 0
    for i in range(n):
        res = await controller.run_turn(f"query {i}", goal)
        tokens_total += res.tokens_used
    latency_ms = (time.time() - t0) * 1000.0 / n
    return RunResult(label, latency_ms, tokens_total // n)

async def main():
    mem = DummyMemory()
    att = DummyAttention()
    llm = DummyLLM()
    act = DummyActuator()
    baseline = ExecutiveController(memory=mem, attention=att, llm=llm, actuator=act)
    contextual = ExecutiveController(memory=mem, attention=att, llm=llm, actuator=act, decision_engine=ContextualDecisionEngine())
    hierarchical = ExecutiveController(memory=mem, attention=att, llm=llm, actuator=act, decision_engine=ContextualDecisionEngine(), planner=HierarchicalPlanner())
    results: List[RunResult] = []
    results.append(await run_config("baseline_weighted", baseline))
    results.append(await run_config("contextual", contextual))
    results.append(await run_config("hierarchical", hierarchical))
    print("Label,AvgLatencyMs,AvgTokens")
    for r in results:
        print(f"{r.label},{r.latency_ms:.2f},{r.tokens}")

if __name__ == "__main__":
    asyncio.run(main())
