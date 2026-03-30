from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _normalize_ids(ids: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in ids:
        item_id = str(value)
        if item_id in seen:
            continue
        seen.add(item_id)
        normalized.append(item_id)
    return tuple(normalized)


def precision_at_k(retrieved_ids: Iterable[str], expected_ids: Iterable[str]) -> float:
    retrieved = _normalize_ids(retrieved_ids)
    expected = set(_normalize_ids(expected_ids))
    if not retrieved:
        return 0.0
    relevant = sum(1 for item_id in retrieved if item_id in expected)
    return relevant / len(retrieved)


def irrelevant_context_rate(retrieved_ids: Iterable[str], expected_ids: Iterable[str]) -> float:
    retrieved = _normalize_ids(retrieved_ids)
    expected = set(_normalize_ids(expected_ids))
    if not retrieved:
        return 0.0
    irrelevant = sum(1 for item_id in retrieved if item_id not in expected)
    return irrelevant / len(retrieved)


def expected_coverage(retrieved_ids: Iterable[str], expected_ids: Iterable[str]) -> float:
    retrieved = set(_normalize_ids(retrieved_ids))
    expected = _normalize_ids(expected_ids)
    if not expected:
        return 1.0
    matched = sum(1 for item_id in expected if item_id in retrieved)
    return matched / len(expected)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    precision: float
    irrelevant_context_rate: float
    expected_coverage: float
    retrieved_count: int
    relevant_count: int
    irrelevant_count: int


def score_retrieval(retrieved_ids: Iterable[str], expected_ids: Iterable[str]) -> RetrievalMetrics:
    retrieved = _normalize_ids(retrieved_ids)
    expected = set(_normalize_ids(expected_ids))
    relevant_count = sum(1 for item_id in retrieved if item_id in expected)
    irrelevant_count = len(retrieved) - relevant_count
    return RetrievalMetrics(
        precision=precision_at_k(retrieved, expected),
        irrelevant_context_rate=irrelevant_context_rate(retrieved, expected),
        expected_coverage=expected_coverage(retrieved, expected),
        retrieved_count=len(retrieved),
        relevant_count=relevant_count,
        irrelevant_count=irrelevant_count,
    )