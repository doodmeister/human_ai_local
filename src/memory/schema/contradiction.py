from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
import json
import re
from typing import Any, Mapping, Sequence


_SOURCE_WEIGHTS = {
    "explicit_user_correction": 1.0,
    "user_assertion": 0.92,
    "chat_capture": 0.8,
    "api": 0.75,
    "semantic": 0.7,
    "imported": 0.65,
    "inferred": 0.5,
    "unknown": 0.5,
}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _normalize_component(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def build_contradiction_set_id(
    subject: str | None,
    predicate: str | None,
    relationship_target: str | None = None,
) -> str:
    subject_part = _normalize_component(subject) or "unknown-subject"
    predicate_part = _normalize_component(predicate) or "unknown-predicate"
    relationship_part = _normalize_component(relationship_target)
    digest_source = "|".join(part for part in (subject_part, predicate_part, relationship_part) if part)
    digest = sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", f"{subject_part}-{predicate_part}").strip("-")
    return f"belief-{slug[:40]}-{digest}"


def source_weight(source: str | None) -> float:
    key = _normalize_component(source) or "unknown"
    return _SOURCE_WEIGHTS.get(key, _SOURCE_WEIGHTS["unknown"])


def weighted_belief_score(confidence: float | None, source: str | None) -> float:
    return _clamp01(confidence, default=0.55) * source_weight(source)


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [str(item) for item in decoded if str(item).strip()]
    return []


def merge_source_history(existing: Any, new_source: str | None) -> list[str]:
    merged: list[str] = []
    for source_name in [*parse_json_list(existing), *( [str(new_source)] if new_source else [] )]:
        normalized = str(source_name).strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    return merged


@dataclass(slots=True)
class BeliefRevisionDecision:
    contradiction_set_id: str
    candidate_fact_id: str
    candidate_status: str
    candidate_confidence: float
    source_weight: float
    rationale: str
    merge_into_fact_id: str | None = None
    winning_fact_id: str | None = None
    conflicting_fact_ids: list[str] = field(default_factory=list)
    fact_updates: dict[str, dict[str, Any]] = field(default_factory=dict)


def evaluate_belief_revision(
    *,
    subject: str,
    predicate: str,
    object_value: Any,
    candidate_fact_id: str,
    candidate_source: str | None,
    candidate_confidence: float | None,
    existing_facts: Sequence[Mapping[str, Any]],
    relationship_target: str | None = None,
) -> BeliefRevisionDecision:
    contradiction_set_id = next(
        (
            str(fact.get("contradiction_set_id"))
            for fact in existing_facts
            if fact.get("contradiction_set_id")
        ),
        build_contradiction_set_id(subject, predicate, relationship_target),
    )
    candidate_object = _normalize_component(object_value)
    candidate_confidence_value = _clamp01(candidate_confidence, default=0.65)
    candidate_weight = source_weight(candidate_source)
    candidate_score = candidate_confidence_value * candidate_weight

    relevant_facts = [
        fact
        for fact in existing_facts
        if _normalize_component(fact.get("subject")) == _normalize_component(subject)
        and _normalize_component(fact.get("predicate")) == _normalize_component(predicate)
    ]
    same_object = [fact for fact in relevant_facts if _normalize_component(fact.get("object")) == candidate_object]
    conflicting = [fact for fact in relevant_facts if _normalize_component(fact.get("object")) != candidate_object]

    if same_object:
        winner = max(
            same_object,
            key=lambda fact: weighted_belief_score(fact.get("confidence"), fact.get("source")),
        )
        winner_id = str(winner.get("fact_id") or winner.get("id") or "")
        merged_confidence = _clamp01(
            max(candidate_confidence_value, _clamp01(winner.get("confidence"), default=0.55))
            + (0.04 * candidate_weight),
            default=candidate_confidence_value,
        )
        updates: dict[str, dict[str, Any]] = {}
        if winner_id:
            updates[winner_id] = {
                "belief_status": "active",
                "confidence": merged_confidence,
                "contradiction_set_id": contradiction_set_id,
                "support_count": int(winner.get("support_count") or 1) + 1,
                "source_weight": max(candidate_weight, source_weight(winner.get("source"))),
            }
        return BeliefRevisionDecision(
            contradiction_set_id=contradiction_set_id,
            candidate_fact_id=candidate_fact_id,
            candidate_status="merged",
            candidate_confidence=merged_confidence,
            source_weight=candidate_weight,
            rationale="reinforced_existing_belief",
            merge_into_fact_id=winner_id or None,
            winning_fact_id=winner_id or None,
            conflicting_fact_ids=[str(fact.get("fact_id") or fact.get("id")) for fact in conflicting if fact.get("fact_id") or fact.get("id")],
            fact_updates=updates,
        )

    if not conflicting:
        return BeliefRevisionDecision(
            contradiction_set_id=contradiction_set_id,
            candidate_fact_id=candidate_fact_id,
            candidate_status="active",
            candidate_confidence=candidate_confidence_value,
            source_weight=candidate_weight,
            rationale="new_belief",
        )

    best_existing = max(
        conflicting,
        key=lambda fact: weighted_belief_score(fact.get("confidence"), fact.get("source")),
    )
    best_existing_id = str(best_existing.get("fact_id") or best_existing.get("id") or "")
    best_existing_score = weighted_belief_score(best_existing.get("confidence"), best_existing.get("source"))
    best_existing_weight = source_weight(best_existing.get("source"))
    best_existing_confidence = _clamp01(best_existing.get("confidence"), default=0.55)

    explicit_correction = _normalize_component(candidate_source) == "explicit_user_correction"
    promote_candidate = candidate_score >= (best_existing_score + 0.05)
    if not promote_candidate and explicit_correction:
        promote_candidate = candidate_score >= (best_existing_score - 0.05)
    if not promote_candidate and candidate_weight > best_existing_weight:
        promote_candidate = candidate_confidence_value >= (best_existing_confidence - 0.02)

    updates = {
        str(fact.get("fact_id") or fact.get("id")): {"contradiction_set_id": contradiction_set_id}
        for fact in conflicting
        if fact.get("fact_id") or fact.get("id")
    }

    if promote_candidate:
        for fact in conflicting:
            fact_id = str(fact.get("fact_id") or fact.get("id") or "")
            if not fact_id:
                continue
            updates[fact_id].update(
                {
                    "belief_status": "quarantined",
                    "superseded_by": candidate_fact_id,
                    "quarantine_reason": "superseded_by_higher_weighted_belief",
                }
            )
        return BeliefRevisionDecision(
            contradiction_set_id=contradiction_set_id,
            candidate_fact_id=candidate_fact_id,
            candidate_status="active",
            candidate_confidence=candidate_confidence_value,
            source_weight=candidate_weight,
            rationale="supersedes_conflicting_belief",
            winning_fact_id=candidate_fact_id,
            conflicting_fact_ids=[fact_id for fact_id in updates if fact_id],
            fact_updates=updates,
        )

    quarantined_confidence = min(candidate_confidence_value, max(0.25, best_existing_score - 0.05))
    return BeliefRevisionDecision(
        contradiction_set_id=contradiction_set_id,
        candidate_fact_id=candidate_fact_id,
        candidate_status="quarantined",
        candidate_confidence=quarantined_confidence,
        source_weight=candidate_weight,
        rationale="lower_weighted_confidence_than_existing_belief",
        winning_fact_id=best_existing_id or None,
        conflicting_fact_ids=[fact_id for fact_id in updates if fact_id],
        fact_updates=updates,
    )