from __future__ import annotations

from typing import Any

from src.executive.decision_engine import DecisionOption, DecisionResult
from src.executive.goal_manager import Goal


class ExecutiveDecisionStage:
    def __init__(self, *, config: Any, decision_engine: Any, metrics: Any) -> None:
        self._config = config
        self._decision_engine = decision_engine
        self._metrics = metrics

    def make_goal_decision(
        self,
        goal: Goal,
        *,
        procedural_matches: list[dict[str, Any]] | None = None,
    ) -> DecisionResult:
        options = [
            DecisionOption(
                name="direct_approach",
                description="Direct approach - tackle goal immediately",
                data={"approach": "direct", "risk": 0.3},
            ),
            DecisionOption(
                name="incremental_approach",
                description="Incremental approach - break into smaller steps",
                data={"approach": "incremental", "risk": 0.2},
            ),
            DecisionOption(
                name="parallel_approach",
                description="Parallel approach - work on multiple aspects simultaneously",
                data={"approach": "parallel", "risk": 0.4},
            ),
        ]

        context = {"goal_id": goal.id, "goal_priority": goal.priority.value}
        if procedural_matches:
            context["procedural_match_count"] = len(procedural_matches)
            context["procedural_match_titles"] = [
                str(proc.get("description", "") or "")
                for proc in procedural_matches[:3]
                if str(proc.get("description", "") or "").strip()
            ]
            context["procedural_top_strength"] = max(
                float(proc.get("strength", 0.0) or 0.0) for proc in procedural_matches
            )

        result = self._decision_engine.make_decision(
            options=options,
            criteria=self._decision_engine.criterion_templates.get("task_selection", []),
            strategy=self._config.decision_strategy,
            context=context,
        )
        self._metrics.inc("executive_decisions_made_total")
        return result