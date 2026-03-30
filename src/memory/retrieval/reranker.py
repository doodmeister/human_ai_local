from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.memory.autobiographical import AutobiographicalGraph
from src.memory.schema import CanonicalMemoryItem

from .retrieval_plan import RetrievalPlan


@dataclass(slots=True)
class RerankedMemoryItem:
    item: CanonicalMemoryItem
    score: float
    factors: dict[str, float]


class MemoryReranker:
    def rerank(
        self,
        items: list[CanonicalMemoryItem],
        *,
        relationship_memory: Any | None = None,
        retrieval_plan: RetrievalPlan | None = None,
        autobiographical_graph: AutobiographicalGraph | None = None,
    ) -> list[CanonicalMemoryItem]:
        reranked: list[RerankedMemoryItem] = []
        seen_content_keys: set[str] = set()

        for item in items:
            content_key = self._content_key(item.content)
            if content_key in seen_content_keys:
                continue
            seen_content_keys.add(content_key)

            signals = self._score_signals(
                item,
                relationship_memory=relationship_memory,
                retrieval_plan=retrieval_plan,
                autobiographical_graph=autobiographical_graph,
            )
            factors = self._score_factors(signals)
            score = sum(factors.values())
            item.metadata["rerank_signals"] = signals
            item.metadata["rerank_score"] = score
            item.metadata["rerank_factors"] = factors
            reranked.append(RerankedMemoryItem(item=item, score=score, factors=factors))

        reranked.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.item.memory_kind.value,
                candidate.item.memory_id,
            )
        )
        return [candidate.item for candidate in reranked]

    def _score_signals(
        self,
        item: CanonicalMemoryItem,
        *,
        relationship_memory: Any | None,
        retrieval_plan: RetrievalPlan | None,
        autobiographical_graph: AutobiographicalGraph | None,
    ) -> dict[str, float]:
        return {
            "similarity": float(item.metadata.get("similarity", 0.0) or 0.0),
            "recency": float(item.metadata.get("recency", 0.0) or 0.0),
            "activation": float(item.metadata.get("activation", 0.0) or 0.0),
            "salience": float(item.metadata.get("salience", 0.0) or 0.0),
            "importance": float(item.importance or 0.0),
            "confidence": float(item.confidence or 0.0),
            "relationship": self._relationship_signal(item, relationship_memory, retrieval_plan),
            "continuity": self._continuity_signal(item, retrieval_plan, autobiographical_graph),
        }

    def _score_factors(self, signals: dict[str, float]) -> dict[str, float]:
        similarity = signals["similarity"] * 0.35
        recency = signals["recency"] * 0.2
        activation = signals["activation"] * 0.2
        salience = signals["salience"] * 0.1
        importance = signals["importance"] * 0.1
        confidence = signals["confidence"] * 0.05
        relationship = signals["relationship"] * 0.18
        continuity = signals["continuity"] * 0.12
        return {
            "similarity": similarity,
            "recency": recency,
            "activation": activation,
            "salience": salience,
            "importance": importance,
            "confidence": confidence,
            "relationship": relationship,
            "continuity": continuity,
        }

    def _relationship_signal(
        self,
        item: CanonicalMemoryItem,
        relationship_memory: Any | None,
        retrieval_plan: RetrievalPlan | None,
    ) -> float:
        target_id = None
        if retrieval_plan is not None and retrieval_plan.relationship_target:
            target_id = retrieval_plan.relationship_target
        elif relationship_memory is not None:
            target_id = str(getattr(relationship_memory, "interlocutor_id", "") or "") or None
        if not target_id:
            return 0.0

        participants = [str(value) for value in item.metadata.get("participants", []) or []]
        entities = [str(value) for value in item.entities]
        relationship_match = 0.0
        if item.relationship_target == target_id:
            relationship_match = 1.0
        elif target_id in participants:
            relationship_match = 0.9
        elif target_id in entities:
            relationship_match = 0.75

        if retrieval_plan is not None and retrieval_plan.intent == "social":
            return relationship_match
        return relationship_match * 0.65

    def _continuity_signal(
        self,
        item: CanonicalMemoryItem,
        retrieval_plan: RetrievalPlan | None,
        autobiographical_graph: AutobiographicalGraph | None,
    ) -> float:
        if retrieval_plan is None or autobiographical_graph is None:
            return 0.0
        if retrieval_plan.intent not in {"continuity", "episodic", "social"}:
            return 0.0

        signal = 0.0
        defining_moments = {
            event_id
            for chapter in autobiographical_graph.chapters
            for event_id in chapter.defining_moment_ids
        }
        if item.memory_id in defining_moments:
            signal += 0.7 if retrieval_plan.intent == "continuity" else 0.4

        recent_chapter = None
        if autobiographical_graph.chapters:
            recent_chapter = max(
                autobiographical_graph.chapters,
                key=lambda chapter: chapter.end_time or chapter.start_time,
            )
        if recent_chapter is not None and item.narrative_role == recent_chapter.life_period:
            signal += 0.6 if retrieval_plan.intent == "continuity" else 0.35

        return min(1.0, signal)

    def _content_key(self, content: str) -> str:
        return " ".join(content.lower().split())