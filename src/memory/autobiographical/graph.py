from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
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


def _humanize_identifier(value: str) -> str:
    text = re.sub(r"[_\s]+", " ", str(value or "").strip())
    return text.strip()


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutobiographicalLink":
        return cls(
            source_id=str(data.get("source_id") or ""),
            target_id=str(data.get("target_id") or ""),
            relationship_type=str(data.get("relationship_type") or ""),
            weight=float(data.get("weight", 1.0) or 1.0),
            metadata=dict(data.get("metadata", {})),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutobiographicalChapter":
        return cls(
            chapter_id=str(data.get("chapter_id") or ""),
            title=str(data.get("title") or ""),
            life_period=str(data.get("life_period") or "general"),
            summary=str(data.get("summary") or ""),
            event_ids=_dedupe(data.get("event_ids", [])),
            goal_ids=_dedupe(data.get("goal_ids", [])),
            participant_ids=_dedupe(data.get("participant_ids", [])),
            defining_moment_ids=_dedupe(data.get("defining_moment_ids", [])),
            start_time=_coerce_datetime(data.get("start_time")),
            end_time=_coerce_datetime(data.get("end_time")),
        )


@dataclass(slots=True)
class AutobiographicalGraph:
    links: list[AutobiographicalLink] = field(default_factory=list)
    chapters: list[AutobiographicalChapter] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "links": [link.to_dict() for link in self.links],
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutobiographicalGraph":
        return cls(
            links=[
                AutobiographicalLink.from_dict(dict(link))
                for link in data.get("links", [])
                if isinstance(link, dict)
            ],
            chapters=[
                AutobiographicalChapter.from_dict(dict(chapter))
                for chapter in data.get("chapters", [])
                if isinstance(chapter, dict)
            ],
        )

    def merged_with(self, other: "AutobiographicalGraph" | None) -> "AutobiographicalGraph":
        if other is None:
            return AutobiographicalGraph(
                links=list(self.links),
                chapters=list(self.chapters),
            )

        links: list[AutobiographicalLink] = []
        seen_links: set[tuple[str, str, str]] = set()
        for link in [*self.links, *other.links]:
            key = (link.source_id, link.target_id, link.relationship_type)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append(link)

        chapters_by_id: dict[str, AutobiographicalChapter] = {}
        for chapter in self.chapters:
            chapters_by_id[chapter.chapter_id] = chapter
        for chapter in other.chapters:
            existing = chapters_by_id.get(chapter.chapter_id)
            chapters_by_id[chapter.chapter_id] = chapter if existing is None else _merge_chapters(existing, chapter)

        chapters = sorted(
            chapters_by_id.values(),
            key=lambda chapter: (
                chapter.start_time or datetime.min,
                chapter.chapter_id,
            ),
        )
        return AutobiographicalGraph(links=links, chapters=chapters)


def _merge_chapters(
    first: AutobiographicalChapter,
    second: AutobiographicalChapter,
) -> AutobiographicalChapter:
    summary_source = first if len(first.event_ids) >= len(second.event_ids) else second
    start_candidates = [value for value in (first.start_time, second.start_time) if value is not None]
    end_candidates = [value for value in (first.end_time, second.end_time) if value is not None]
    return AutobiographicalChapter(
        chapter_id=first.chapter_id or second.chapter_id,
        title=first.title or second.title,
        life_period=first.life_period or second.life_period,
        summary=summary_source.summary,
        event_ids=_dedupe([*first.event_ids, *second.event_ids]),
        goal_ids=_dedupe([*first.goal_ids, *second.goal_ids]),
        participant_ids=_dedupe([*first.participant_ids, *second.participant_ids]),
        defining_moment_ids=_dedupe([*first.defining_moment_ids, *second.defining_moment_ids]),
        start_time=min(start_candidates) if start_candidates else None,
        end_time=max(end_candidates) if end_candidates else None,
    )


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
        if goal_ids:
            summary_parts.append(f"Goals: {', '.join(_humanize_identifier(goal_id) for goal_id in goal_ids[:2])}")
        if participant_ids:
            summary_parts.append(f"People: {', '.join(_humanize_identifier(participant_id) for participant_id in participant_ids[:2])}")
        if defining_moment_ids:
            summary_parts.append(f"Defining moments: {len(defining_moment_ids)}")
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