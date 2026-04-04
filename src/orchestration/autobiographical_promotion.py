from __future__ import annotations

from datetime import datetime, timedelta
import re
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
    if _is_defining_moment_interaction(
        user_content=user_content,
        assistant_content=assistant_content,
        intent_type=intent_type,
        importance=importance,
        emotional_valence=emotional_valence,
    ):
        episode_payload["defining_moment"] = True

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
    prospective_reminder_ids = _store_prospective_follow_up_reminders(
        memory=memory,
        session=session,
        user_content=user_content,
        importance=max(float(importance or 0.0), 0.55),
        source_event_ids=source_event_ids,
        goal_ids=goal_ids,
        intent_type=intent_type,
    )
    if prospective_reminder_ids:
        setattr(session, "_last_prospective_reminder_ids", [item["reminder_id"] for item in prospective_reminder_ids])
        setattr(session, "_last_prospective_products", prospective_reminder_ids)
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
        user_content=user_content,
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


def _store_prospective_follow_up_reminders(
    *,
    memory: Any,
    session: Any,
    user_content: str,
    importance: float,
    source_event_ids: Sequence[str] | None,
    goal_ids: Iterable[str] | None,
    intent_type: str | None,
) -> list[dict[str, str]]:
    normalized_intent = str(intent_type or "").strip().lower()
    if normalized_intent == "reminder_request":
        return []

    prospective = _prospective_memory_backend(memory)
    if prospective is None or not hasattr(prospective, "add_reminder"):
        return []

    relationship = getattr(session, "_relationship_memory_snapshot", None)
    relationship_target = str(
        getattr(relationship, "interlocutor_id", "") or getattr(session, "session_id", "")
    ).strip()
    reminder_candidates = _prospective_turn_reminders(user_content)
    if not reminder_candidates:
        return []

    reminder_records: list[dict[str, str]] = []
    goal_tags = [f"goal:{goal_id}" for goal_id in sorted(str(goal_id) for goal_id in (goal_ids or []) if str(goal_id).strip())]
    for candidate in reminder_candidates:
        metadata = {
            "source": "autobiographical_promotion",
            "source_event_ids": list(source_event_ids or []),
            "relationship_target": relationship_target,
            "importance": max(0.6, float(importance or 0.0)),
            "derived_from": candidate["derived_from"],
            "goal_ids": [tag.removeprefix("goal:") for tag in goal_tags],
        }
        if candidate["due_hint"]:
            metadata["due_hint"] = candidate["due_hint"]
        try:
            reminder = prospective.add_reminder(
                candidate["content"],
                due_time=candidate["due_time"],
                tags=["auto-generated", "promoted-turn", *goal_tags, *candidate["tags"]],
                metadata=metadata,
            )
        except Exception:
            continue
        reminder_id = str(getattr(reminder, "id", "") or "").strip()
        if not reminder_id:
            continue
        reminder_records.append(
            {
                "reminder_id": reminder_id,
                "content": candidate["content"],
                "derived_from": candidate["derived_from"],
            }
        )
    return reminder_records


def _prospective_memory_backend(memory: Any) -> Any:
    try:
        prospective = getattr(memory, "prospective", None)
    except Exception:
        prospective = None
    if prospective is not None:
        return prospective
    return getattr(memory, "_prospective", None)


def _prospective_turn_reminders(user_content: str) -> list[dict[str, Any]]:
    text = str(user_content or "").strip()
    if not text:
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, formatter, derived_from, tags in (
        (
            re.compile(r"\bremind me to (?P<task>.+)", re.IGNORECASE),
            lambda task: task,
            "user_content.reminder_phrase",
            ["reminder"],
        ),
        (
            re.compile(r"\bremember to (?P<task>.+)", re.IGNORECASE),
            lambda task: task,
            "user_content.reminder_phrase",
            ["reminder"],
        ),
        (
            re.compile(r"\bfollow up on (?P<task>.+)", re.IGNORECASE),
            lambda task: f"Follow up on {task}",
            "user_content.follow_up_phrase",
            ["follow-up"],
        ),
        (
            re.compile(r"\bcheck back on (?P<task>.+)", re.IGNORECASE),
            lambda task: f"Check back on {task}",
            "user_content.follow_up_phrase",
            ["follow-up"],
        ),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        task_text = match.group("task")
        task_body, due_time, due_hint = _extract_due_time(task_text)
        task_body = _normalize_reminder_text(task_body)
        if not _is_meaningful_reminder_text(task_body):
            continue
        content = _normalize_reminder_text(formatter(task_body))
        dedupe_key = content.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            {
                "content": content,
                "due_time": due_time,
                "due_hint": due_hint,
                "derived_from": derived_from,
                "tags": list(tags),
            }
        )
    return candidates


def _extract_due_time(task_text: str) -> tuple[str, datetime | None, str | None]:
    cleaned = str(task_text or "")
    now = datetime.now()

    relative_match = re.search(
        r"\bin\s+(?P<count>\d+)\s+(?P<unit>minute|minutes|hour|hours|day|days|week|weeks)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if relative_match is not None:
        count = int(relative_match.group("count"))
        unit = relative_match.group("unit").lower()
        if unit.startswith("minute"):
            due_time = now + timedelta(minutes=count)
        elif unit.startswith("hour"):
            due_time = now + timedelta(hours=count)
        elif unit.startswith("day"):
            due_time = now + timedelta(days=count)
        else:
            due_time = now + timedelta(weeks=count)
        cleaned = (cleaned[: relative_match.start()] + cleaned[relative_match.end() :]).strip()
        return cleaned, due_time, relative_match.group(0).strip()

    for phrase, delta in (
        ("later today", timedelta(hours=4)),
        ("tomorrow", timedelta(days=1)),
        ("next week", timedelta(days=7)),
    ):
        phrase_match = re.search(rf"\b{re.escape(phrase)}\b", cleaned, flags=re.IGNORECASE)
        if phrase_match is None:
            continue
        cleaned = (cleaned[: phrase_match.start()] + cleaned[phrase_match.end() :]).strip()
        return cleaned, now + delta, phrase

    return cleaned, None, None


def _normalize_reminder_text(value: str) -> str:
    return " ".join(str(value or "").strip(" .,!?:;").split())


def _is_meaningful_reminder_text(value: str) -> bool:
    lowered = value.lower()
    if len(lowered) < 4:
        return False
    if lowered in {"it", "this", "that", "them", "something"}:
        return False
    return any(char.isalpha() for char in lowered)


def _is_defining_moment_interaction(
    *,
    user_content: str,
    assistant_content: str,
    intent_type: str | None,
    importance: float,
    emotional_valence: float,
) -> bool:
    if float(importance or 0.0) >= 0.8 or abs(float(emotional_valence or 0.0)) >= 0.7:
        return True

    normalized_intent = str(intent_type or "").strip().lower()
    if normalized_intent in {"milestone_capture", "retrospective", "turning_point"}:
        return True

    combined = " ".join(
        part.strip().lower()
        for part in (str(user_content or ""), str(assistant_content or ""))
        if str(part or "").strip()
    )
    if not combined:
        return False

    return any(
        marker in combined
        for marker in (
            "major pivot",
            "turning point",
            "major milestone",
            "key milestone",
            "breakthrough",
            "defining moment",
            "reoriented the roadmap",
            "changed the project direction",
        )
    )


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
    user_content: str = "",
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

    for subject, predicate, object_value, derived_from in _semantic_content_facts(user_content):
        dedupe_key = (subject.lower(), predicate.lower(), object_value.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_value,
                "derived_from": derived_from,
            }
        )

    return candidates


def _semantic_content_facts(user_content: str) -> list[tuple[str, str, str, str]]:
    text = str(user_content or "").strip()
    if not text:
        return []

    candidates: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pattern, derived_from in (
        (re.compile(r"\bi value\s+(?P<values>.+)", re.IGNORECASE), "user_content.stated_values_phrase"),
        (re.compile(r"\bi care about\s+(?P<values>.+)", re.IGNORECASE), "user_content.stated_values_phrase"),
        (re.compile(r"\bwhat matters to me is\s+(?P<values>.+)", re.IGNORECASE), "user_content.stated_values_phrase"),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        for value in _split_semantic_value_list(match.group("values")):
            dedupe_key = ("user", "values", value.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(("user", "values", value, derived_from))
    return candidates


def _split_semantic_value_list(raw: str) -> list[str]:
    cleaned = str(raw or "")
    cleaned = re.split(r"[.!?]", cleaned, maxsplit=1)[0]
    parts = re.split(r"\b(?:and|or)\b|,|/", cleaned)
    values: list[str] = []
    for part in parts:
        normalized = _normalize_semantic_value(part)
        normalized = re.sub(r"^(about|around|for)\s+", "", normalized, flags=re.IGNORECASE)
        if not _looks_like_meaningful_semantic_value(normalized):
            continue
        if normalized not in values:
            values.append(normalized)
    return values


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


def _looks_like_meaningful_semantic_value(value: str) -> bool:
    lowered = value.lower().strip()
    if len(lowered) < 3:
        return False
    if lowered in {"it", "this", "that", "things", "something", "anything", "everything"}:
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
    if predicate == "values":
        return f"The {subject} values {object_value}."
    return f"The {subject} {predicate} {object_value}."