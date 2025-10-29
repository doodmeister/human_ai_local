from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .executive_core import Goal, Task, Planner

"""Hierarchical / heuristic planner stub.

Produces a small task graph from a Goal description. This is a deterministic
placeholder; can be upgraded to LLM-backed decomposition.
"""

@dataclass(frozen=True)
class PlannedTask(Task):
    parents: List[str] = field(default_factory=list)  # type: ignore
    order: int = 0

class HierarchicalPlanner(Planner):
    def __init__(self, max_tasks: int = 4):
        self.max_tasks = max_tasks

    def expand(self, goal: Goal) -> List[Task]:  # compatible signature
        desc = goal.description.strip()
        # naive split by punctuation / 'and'
        raw_parts: List[str] = []
        for sep in [' and ', ';', '.', ',']:
            if sep in desc:
                raw_parts = [p.strip() for p in desc.split(sep) if p.strip()]
                break
        if not raw_parts:
            raw_parts = [desc]
        raw_parts = raw_parts[: self.max_tasks]

        tasks: List[Task] = []
        for i, part in enumerate(raw_parts):
            parents = [t.id for t in tasks[:-1]] if i > 0 else []
            complexity = min(1.0, 0.3 + goal.priority * 0.4 + (0.1 * i))
            urgency = min(1.0, 0.3 + goal.priority * 0.5)
            tasks.append(PlannedTask(
                id=f"htask::{goal.id}::{i}",
                goal_id=goal.id,
                description=part,
                complexity=complexity,
                urgency=urgency,
                metadata={"graph_index": i},
                parents=parents,
                order=i,
            ))
        return tasks
