from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from src.memory.schema import CanonicalMemoryItem, MemoryKind


def _dedupe(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


@dataclass(slots=True)
class AutobiographicalLink:
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AutobiographicalChapter:
    chapter_id: str
    title: str
    life_period: str
    summary: str
    event_ids: list[str] = field(default_factory=list)
    goal_ids: list[str] = field(default_factory=list)
    participant_ids: list[str] = field(default_factory=list)
    defining_moment_ids: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "life_period": self.life_period,
            "summary": self.summary,
            "event_ids": list(self.event_ids),
            "goal_ids": list(self.goal_ids),
            "participant_ids": list(self.participant_ids),
            "defining_moment_ids": list(self.defining_moment_ids),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass(slots=True)
class AutobiographicalGraph:
    links: list[AutobiographicalLink] = field(default_factory=list)
    chapters: list[AutobiographicalChapter] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "links": [link.to_dict() for link in self.links],
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }


class AutobiographicalGraphBuilder:
    def build(self, items: Iterable[CanonicalMemoryItem]) -> AutobiographicalGraph:
        episodic_items = [item for item in items if item.memory_kind == MemoryKind.EPISODIC]
        links: list[AutobiographicalLink] = []
        chapters_by_period: dict[str, list[CanonicalMemoryItem]] = {}

        for item in episodic_items:
            participants = _dedupe(item.metadata.get("participants", []))
            if item.relationship_target:
                participants = _dedupe([item.relationship_target, *participants])
            for participant_id in participants:
                links.append(
                    AutobiographicalLink(
                        source_id=item.memory_id,
                        target_id=f"person:{participant_id}",
                        relationship_type="participant",
                    )
                )
            for goal_id in item.goal_ids:
                links.append(
                    AutobiographicalLink(
                        source_id=item.memory_id,
                        target_id=f"goal:{goal_id}",
                        relationship_type="goal",
                    )
                )
            for source_event_id in item.source_event_ids:
                links.append(
                    AutobiographicalLink(
                        source_id=item.memory_id,
                        target_id=f"event:{source_event_id}",
                        relationship_type="source_event",
                    )
                )
            for related_episode_id in item.metadata.get("related_episodes", []):
                links.append(
                    AutobiographicalLink(
                        source_id=item.memory_id,
                        target_id=f"episode:{related_episode_id}",
                        relationship_type="related_episode",
                    )
                )

            life_period = item.narrative_role or str(item.metadata.get("life_period") or "general")
            chapters_by_period.setdefault(life_period, []).append(item)

        chapters = [self._build_chapter(life_period, chapter_items) for life_period, chapter_items in sorted(chapters_by_period.items())]
        return AutobiographicalGraph(links=links, chapters=chapters)

    def _build_chapter(self, life_period: str, items: list[CanonicalMemoryItem]) -> AutobiographicalChapter:
        ordered_items = sorted(
            items,
            key=lambda item: item.encoding_time or datetime.min,
        )
        event_ids = [item.memory_id for item in ordered_items]
        goal_ids = _dedupe(goal_id for item in ordered_items for goal_id in item.goal_ids)
        participant_ids = _dedupe(
            participant_id
            for item in ordered_items
            for participant_id in [item.relationship_target, *list(item.metadata.get("participants", []))]
            if participant_id
        )
        defining_moment_ids = [
            item.memory_id
            for item in ordered_items
            if item.metadata.get("defining_moment")
            or item.importance >= 0.75
            or abs(item.emotional_valence) >= 0.6
        ]
        summary_parts = [item.summary or item.content for item in ordered_items[:3]]
        title = life_period.replace("_", " ").title()
        summary = f"{title}: " + "; ".join(summary_parts)
        return AutobiographicalChapter(
            chapter_id=f"chapter:{life_period}",
            title=title,
            life_period=life_period,
            summary=summary,
            event_ids=event_ids,
            goal_ids=goal_ids,
            participant_ids=participant_ids,
            defining_moment_ids=defining_moment_ids,
            start_time=ordered_items[0].encoding_time if ordered_items else None,
            end_time=ordered_items[-1].encoding_time if ordered_items else None,
        )