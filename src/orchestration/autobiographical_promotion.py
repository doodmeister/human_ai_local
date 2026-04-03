from __future__ import annotations

from typing import Any, Iterable, Sequence

from src.memory.encoding import EventEncoder
from src.memory.autobiographical import AutobiographicalGraphBuilder


def derive_autobiographical_life_period(
    *,
    intent_type: str | None = None,
    tick: Any = None,
    default_prefix: str = "chat",
) -> str:
    if tick is not None:
        try:
            narrative = tick.state.get("narrative")
        except Exception:
            narrative = None
        if narrative is not None and hasattr(narrative, "to_dict"):
            try:
                narrative_data = narrative.to_dict()
            except Exception:
                narrative_data = {}
            active_themes = [
                str(theme).strip().lower().replace(" ", "_")
                for theme in narrative_data.get("active_themes", [])
            ]
            for theme in active_themes:
                if theme:
                    return f"theme_{theme}"

    normalized_intent = str(intent_type or "").strip().lower()
    if normalized_intent:
        return f"{default_prefix}_{normalized_intent}"
    return f"{default_prefix}_general"


def promote_interaction_to_autobiographical_memory(
    *,
    memory: Any,
    autobiographical_store: Any,
    session: Any,
    user_content: str,
    assistant_content: str,
    importance: float,
    emotional_valence: float,
    source_event_ids: Sequence[str] | None = None,
    intent_type: str | None = None,
    tick: Any = None,
    goal_ids: Iterable[str] | None = None,
    default_prefix: str = "chat",
) -> str | None:
    if memory is None or not hasattr(memory, "create_episodic_memory"):
        return None
    if autobiographical_store is None or not hasattr(autobiographical_store, "merge"):
        return None
    session_id = str(getattr(session, "session_id", "") or "").strip()
    if not session_id:
        return None

    participants: list[str] = []
    relationship = getattr(session, "_relationship_memory_snapshot", None)
    relationship_target = str(getattr(relationship, "interlocutor_id", "") or "").strip()
    if relationship_target:
        participants.append(relationship_target)
    if session_id not in participants:
        participants.append(session_id)

    life_period = derive_autobiographical_life_period(
        intent_type=intent_type,
        tick=tick,
        default_prefix=default_prefix,
    )
    label = str(intent_type or "conversation").replace("_", " ")
    summary_line = f"Conversation about {label}."
    detailed_content = "\n".join(
        [
            summary_line,
            f"User: {user_content}",
            f"Assistant: {assistant_content}",
        ]
    )

    try:
        episode_id = memory.create_episodic_memory(
            summary=summary_line,
            detailed_content=detailed_content,
            participants=participants,
            importance=max(float(importance or 0.0), 0.55),
            emotional_valence=float(emotional_valence or 0.0),
            life_period=life_period,
        )
    except Exception:
        return None

    if not episode_id:
        return None

    setattr(session, "_last_autobiographical_episode_id", str(episode_id))
    episode_payload = _load_episode_payload(memory, str(episode_id))
    if episode_payload is None:
        return str(episode_id)

    episode_payload.setdefault("id", str(episode_id))
    episode_payload.setdefault("context", {"participants": participants})
    context = episode_payload.get("context")
    if isinstance(context, dict):
        merged_context = dict(context)
        existing_participants = list(merged_context.get("participants", []))
        merged_context["participants"] = [
            *existing_participants,
            *[participant for participant in participants if participant not in existing_participants],
        ]
        episode_payload["context"] = merged_context
    episode_payload["source_memory_ids"] = list(source_event_ids or [])
    episode_payload["life_period"] = str(episode_payload.get("life_period") or life_period)

    canonical_item = EventEncoder().encode_episode(
        episode_payload,
        goal_ids=sorted(str(goal_id) for goal_id in (goal_ids or []) if str(goal_id).strip()),
        relationship_target=relationship_target or session_id,
    )
    graph = autobiographical_store.merge(
        session_id,
        AutobiographicalGraphBuilder().build([canonical_item]),
    )
    setattr(session, "_autobiographical_graph_snapshot", graph)
    semantic_fact_ids = _store_semantic_preference_facts(
        memory=memory,
        session=session,
        importance=max(float(importance or 0.0), 0.55),
        source_event_ids=source_event_ids,
        user_content=user_content,
        tick=tick,
        goal_ids=goal_ids,
    )
    if semantic_fact_ids:
        setattr(session, "_last_semantic_fact_ids", [item["fact_id"] for item in semantic_fact_ids])
        setattr(session, "_last_semantic_products", semantic_fact_ids)
    return str(episode_id)


def _load_episode_payload(memory: Any, episode_id: str) -> dict[str, Any] | None:
    episodic = getattr(memory, "episodic", None)
    if episodic is None:
        return None

    if hasattr(episodic, "retrieve_memory"):
        try:
            episode = episodic.retrieve_memory(episode_id)
            if episode is not None and hasattr(episode, "to_dict"):
                return dict(episode.to_dict())
            if isinstance(episode, dict):
                return dict(episode)
        except Exception:
            pass

    if hasattr(episodic, "retrieve"):
        try:
            episode = episodic.retrieve(episode_id)
            if isinstance(episode, dict):
                return dict(episode)
        except Exception:
            pass

    return None


def _store_semantic_preference_facts(
    *,
    memory: Any,
    session: Any,
    importance: float,
    source_event_ids: Sequence[str] | None,
    user_content: str,
    tick: Any,
    goal_ids: Iterable[str] | None,
) -> list[dict[str, str]]:
    if not hasattr(memory, "store_fact"):
        return []

    relationship = getattr(session, "_relationship_memory_snapshot", None)
    norms = list(getattr(relationship, "recurring_norms", []) or [])
    semantic_candidates = _semantic_turn_facts(
        norms=norms,
        tick=tick,
        goal_ids=goal_ids,
    )
    if not semantic_candidates:
        return []

    relationship_target = str(
        getattr(relationship, "interlocutor_id", "") or getattr(session, "session_id", "")
    ).strip()
    interaction_count = int(getattr(relationship, "interaction_count", 0) or 0)
    semantic_source = _derive_semantic_fact_source(user_content)
    confidence = min(0.9, 0.62 + (min(interaction_count, 10) * 0.02) + (float(importance or 0.0) * 0.18))
    fact_records: list[dict[str, str]] = []
    for candidate in semantic_candidates:
        subject = candidate["subject"]
        predicate = candidate["predicate"]
        object_value = candidate["object"]
        try:
            fact_id = memory.store_fact(
                subject,
                predicate,
                object_value,
                content=_semantic_fact_text(subject, predicate, object_value),
                metadata={
                    "source": semantic_source,
                    "confidence": confidence,
                    "importance": max(0.6, float(importance or 0.0)),
                    "relationship_target": relationship_target,
                    "source_event_ids": list(source_event_ids or []),
                    "derived_from": candidate["derived_from"],
                },
            )
        except Exception:
            continue
        if fact_id:
            fact_records.append(
                {
                    "fact_id": str(fact_id),
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_value,
                    "derived_from": candidate["derived_from"],
                }
            )
    return fact_records


def _derive_semantic_fact_source(user_content: str) -> str:
    lowered = str(user_content or "").strip().lower()
    correction_markers = (
        "actually",
        "instead",
        "rather",
        "correcting",
        "correction",
        "not anymore",
        "switch to",
    )
    if any(marker in lowered for marker in correction_markers):
        return "explicit_user_correction"
    return "user_assertion"


def _semantic_facts_from_norms(norms: Sequence[str]) -> list[tuple[str, str, str]]:
    facts: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for norm in norms:
        normalized = str(norm).strip()
        lowered = normalized.lower()
        fact: tuple[str, str, str] | None = None
        if lowered.startswith("prefers "):
            fact = ("user", "prefers", normalized[8:].strip())
        elif lowered.startswith("likes "):
            fact = ("user", "prefers", normalized[6:].strip())
        elif lowered.startswith("asks for "):
            fact = ("user", "asks_for", normalized[9:].strip())
        if fact is None or not fact[2]:
            continue
        dedupe_key = (fact[0].lower(), fact[1].lower(), fact[2].lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        facts.append(fact)
    return facts


def _semantic_turn_facts(
    *,
    norms: Sequence[str],
    tick: Any,
    goal_ids: Iterable[str] | None,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for subject, predicate, object_value in _semantic_facts_from_norms(norms):
        dedupe_key = (subject.lower(), predicate.lower(), object_value.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_value,
                "derived_from": "relationship_memory.recurring_norms",
            }
        )

    for theme in _semantic_focus_values(tick=tick, goal_ids=goal_ids):
        dedupe_key = ("user", "focuses_on", theme.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            {
                "subject": "user",
                "predicate": "focuses_on",
                "object": theme,
                "derived_from": "narrative.active_themes",
            }
        )

    return candidates


def _semantic_focus_values(*, tick: Any, goal_ids: Iterable[str] | None) -> list[str]:
    focus_values: list[str] = []
    for theme in _narrative_active_themes(tick):
        normalized = _normalize_semantic_value(theme)
        if normalized and normalized not in focus_values:
            focus_values.append(normalized)

    for goal_id in goal_ids or []:
        normalized = _normalize_semantic_value(str(goal_id))
        if normalized and normalized not in focus_values and _looks_like_readable_goal(normalized):
            focus_values.append(normalized)

    return focus_values


def _narrative_active_themes(tick: Any) -> list[str]:
    if tick is None:
        return []
    try:
        narrative = tick.state.get("narrative")
    except Exception:
        narrative = None
    if narrative is None or not hasattr(narrative, "to_dict"):
        return []
    try:
        data = narrative.to_dict()
    except Exception:
        return []
    return [str(theme) for theme in data.get("active_themes", []) if str(theme).strip()]


def _looks_like_readable_goal(value: str) -> bool:
    lowered = value.lower()
    if len(lowered) < 4:
        return False
    if lowered.count("-") >= 3 and " " not in lowered:
        return False
    return any(ch.isalpha() for ch in lowered)


def _normalize_semantic_value(value: str) -> str:
    normalized = " ".join(str(value or "").strip().replace("_", " ").split())
    return normalized


def _semantic_fact_text(subject: str, predicate: str, object_value: str) -> str:
    if predicate == "prefers":
        return f"The {subject} prefers {object_value}."
    if predicate == "asks_for":
        return f"The {subject} asks for {object_value}."
    if predicate == "focuses_on":
        return f"The {subject} is focused on {object_value}."
    return f"The {subject} {predicate} {object_value}."