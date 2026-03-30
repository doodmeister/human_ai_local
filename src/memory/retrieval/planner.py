from __future__ import annotations

from enum import StrEnum
from typing import Any

from .retrieval_plan import RetrievalPlan


class RetrievalIntent(StrEnum):
    FACTUAL = "factual"
    EPISODIC = "episodic"
    CONTINUITY = "continuity"
    SOCIAL = "social"
    SELF_REFLECTIVE = "self_reflective"
    REMINDER = "reminder"
    GENERAL = "general"


class RetrievalPlanner:
    def plan(self, query: str | None, relationship_memory: Any | None = None) -> RetrievalPlan:
        query_text = (query or "").strip()
        relationship_target = self._relationship_target(relationship_memory)
        relationship_strength = self._relationship_strength(relationship_memory)
        if not query_text:
            return RetrievalPlan(
                query="",
                intent=RetrievalIntent.GENERAL.value,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                reasoning=["empty query uses default blended retrieval"],
                store_limits={"stm": 5, "ltm": 5, "episodic": 5},
            )

        lower = query_text.lower()

        if self._is_reminder_query(lower):
            return RetrievalPlan(
                query=query_text,
                intent=RetrievalIntent.REMINDER.value,
                search_stm=True,
                search_ltm=False,
                search_episodic=False,
                include_prospective=True,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                store_limits={"stm": 4, "ltm": 1, "episodic": 1},
                reasoning=["reminder-oriented query prioritizes current and prospective state"],
            )

        if self._is_continuity_query(lower):
            reasoning = ["continuity query prioritizes episodic recall and recent chapter anchors"]
            if relationship_target:
                reasoning.append("current relationship target informs autobiographical prioritization")
            return RetrievalPlan(
                query=query_text,
                intent=RetrievalIntent.CONTINUITY.value,
                search_stm=True,
                search_ltm=True,
                search_episodic=True,
                include_prospective=True,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                store_limits={"stm": 3, "ltm": 5, "episodic": 8},
                reasoning=reasoning,
            )

        if self._is_episodic_query(lower):
            return RetrievalPlan(
                query=query_text,
                intent=RetrievalIntent.EPISODIC.value,
                search_stm=True,
                search_ltm=True,
                search_episodic=True,
                include_prospective=False,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                store_limits={"stm": 4, "ltm": 4, "episodic": 8},
                reasoning=["episodic query increases episodic retrieval budget"],
            )

        if self._is_social_query(lower):
            episodic_limit = 7 if (relationship_strength or 0.0) >= 0.6 else 5
            reasoning = ["social query benefits from persistent and autobiographical context"]
            if relationship_target:
                reasoning.append("current relationship memory prioritizes interlocutor-linked recall")
            return RetrievalPlan(
                query=query_text,
                intent=RetrievalIntent.SOCIAL.value,
                search_stm=True,
                search_ltm=True,
                search_episodic=True,
                include_prospective=False,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                store_limits={"stm": 3, "ltm": 6, "episodic": episodic_limit},
                reasoning=reasoning,
            )

        if self._is_self_reflective_query(lower):
            return RetrievalPlan(
                query=query_text,
                intent=RetrievalIntent.SELF_REFLECTIVE.value,
                search_stm=True,
                search_ltm=True,
                search_episodic=False,
                include_prospective=False,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                store_limits={"stm": 4, "ltm": 5, "episodic": 1},
                reasoning=["self-reflective query deprioritizes episodic recall"],
            )

        if self._is_factual_query(lower):
            return RetrievalPlan(
                query=query_text,
                intent=RetrievalIntent.FACTUAL.value,
                search_stm=True,
                search_ltm=True,
                search_episodic=False,
                include_prospective=False,
                relationship_target=relationship_target,
                relationship_strength=relationship_strength,
                store_limits={"stm": 4, "ltm": 6, "episodic": 1},
                reasoning=["factual query prioritizes STM and LTM over episodic recall"],
            )

        return RetrievalPlan(
            query=query_text,
            intent=RetrievalIntent.GENERAL.value,
            search_stm=True,
            search_ltm=True,
            search_episodic=True,
            include_prospective=True,
            relationship_target=relationship_target,
            relationship_strength=relationship_strength,
            store_limits={"stm": 5, "ltm": 5, "episodic": 4},
            reasoning=["general query keeps blended retrieval enabled"],
        )

    def _relationship_target(self, relationship_memory: Any | None) -> str | None:
        if relationship_memory is None:
            return None
        target = str(getattr(relationship_memory, "interlocutor_id", "") or "").strip()
        return target or None

    def _relationship_strength(self, relationship_memory: Any | None) -> float | None:
        if relationship_memory is None or not hasattr(relationship_memory, "retrieval_features"):
            return None
        try:
            features = relationship_memory.retrieval_features()
            return float(features.get("relationship_strength", 0.0))
        except Exception:
            return None

    def _is_reminder_query(self, query: str) -> bool:
        return any(token in query for token in ["remind me", "reminder", "reminders", "due", "todo", "follow up"])

    def _is_episodic_query(self, query: str) -> bool:
        return any(
            token in query
            for token in [
                "when did",
                "last time",
                "recently",
                "earlier",
                "happened",
                "discuss",
                "talk about",
                "mentioned",
            ]
        )

    def _is_social_query(self, query: str) -> bool:
        return any(
            token in query
            for token in ["between us", "our relationship", "do you know me", "trust", "rapport", "how we"]
        )

    def _is_continuity_query(self, query: str) -> bool:
        return any(
            token in query
            for token in ["what changed", "changed lately", "lately", "since then", "how have things changed"]
        )

    def _is_self_reflective_query(self, query: str) -> bool:
        return any(
            token in query
            for token in ["how are you feeling", "how are you", "about yourself", "who are you", "your mood"]
        )

    def _is_factual_query(self, query: str) -> bool:
        return any(
            token in query
            for token in ["what do you know", "remember about", "prefer", "preference", "favorite", "know about"]
        )