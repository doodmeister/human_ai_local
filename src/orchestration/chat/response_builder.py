from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

from .constants import PREVIEW_MAX_CONTENT_CHARS, PREVIEW_MAX_ITEMS


class ChatResponseBuilder:
    """Assemble and serialize chat-oriented response payloads."""

    def __init__(
        self,
        *,
        section_headers: Dict[str, str],
        get_config: Callable[[], Dict[str, Any]],
    ) -> None:
        self._section_headers = section_headers
        self._get_config = get_config

    def merge_intent_sections(
        self,
        sections: list[Dict[str, Any]],
        base_response: str | None,
    ) -> str:
        blocks: list[str] = []
        for section in sections:
            header = self._section_headers.get(
                section["intent"],
                section["intent"].replace("_", " ").title(),
            )
            content = (section.get("content") or "").strip()
            if content:
                blocks.append(f"{header}:\n{content}")
        if base_response:
            stripped = base_response.strip()
            if stripped:
                blocks.append(stripped)
        return "\n\n".join(blocks).strip()

    def summarize_context_items(self, items: Iterable[Any]) -> list[Dict[str, Any]]:
        cfg = self._get_config()
        max_items = cfg.get("preview_max_items", PREVIEW_MAX_ITEMS)
        max_chars = cfg.get("preview_max_content_chars", PREVIEW_MAX_CONTENT_CHARS)
        sorted_items = sorted(
            enumerate(items),
            key=lambda pair: (
                pair[1].rank if pair[1].rank is not None else 1_000_000,
                pair[1].source_system or "",
                pair[1].content or "",
                pair[0],
            ),
        )
        out: list[Dict[str, Any]] = []
        for _idx, context_item in sorted_items[:max_items]:
            content = context_item.content
            if len(content) > max_chars:
                content = content[: max_chars - 3] + "..."
            out.append(
                {
                    "source": context_item.source_system,
                    "reason": context_item.reason,
                    "rank": context_item.rank,
                    "composite": context_item.scores.get("composite"),
                    "content": content,
                }
            )
        return out

    @staticmethod
    def trace_to_dict(built: Any) -> Dict[str, Any]:
        return {
            "stages": [
                {
                    "name": stage.name,
                    "candidates_in": stage.candidates_in,
                    "candidates_out": stage.candidates_out,
                    "latency_ms": round(stage.latency_ms, 2),
                    "added": stage.added,
                }
                for stage in built.trace.pipeline_stages
            ],
            "notes": built.trace.notes,
            "degraded": built.trace.degraded_mode,
        }