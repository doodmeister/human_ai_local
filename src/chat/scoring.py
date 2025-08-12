from __future__ import annotations
from typing import List, Dict, Tuple
from .models import ContextItem
from .constants import SCORING_PROFILE_VERSION

WEIGHTS: Dict[str, float] = {
    "recency": 0.25,
    "similarity": 0.30,
    "activation": 0.25,
    "salience": 0.15,
    "importance": 0.05,
    "overlap": 0.20,   # only applies to fallback items
    "focus": 0.40,     # attention focus forces high composite
    "state": 0.00,     # executive mode informational
}

def composite_score(ci: ContextItem) -> float:
    s = 0.0
    for k, w in WEIGHTS.items():
        if k in ci.scores:
            s += ci.scores[k] * w
    if ci.forced:
        s += 1.0  # strong bump for forced (attention focus)
    return s

def score_and_rank(items: List[ContextItem]) -> List[ContextItem]:
    for ci in items:
        ci.scores["composite"] = composite_score(ci)
        # Derive reason refinement
        if ci.reason == "semantic_match":
            ci.reason = f"semantic_match (c={ci.scores['composite']:.2f})"
        elif ci.reason == "recent_turn":
            ci.reason = f"recent_turn (c={ci.scores['composite']:.2f})"
        elif ci.reason == "fallback_word_overlap":
            ci.reason = f"fallback_overlap (c={ci.scores['composite']:.2f})"
        elif ci.reason == "attention_focus":
            ci.reason = f"attention_focus (c={ci.scores['composite']:.2f})"
    # Stable deterministic ordering: composite desc, then source_system, then content
    ordered = sorted(
        items,
        key=lambda x: (-x.scores.get("composite", 0.0), x.source_system, x.content),
    )
    for idx, ci in enumerate(ordered, start=1):
        ci.rank = idx
    return ordered

def summarize(items: List[ContextItem]) -> List[Tuple[int, str, float]]:
    return [(ci.rank, ci.source_system, ci.scores.get("composite", 0.0)) for ci in items]

def factor_breakdown(ci: ContextItem, include_forced: bool = True):
    """
    Returns list of dicts: factor, value, weight, contribution.
    Includes 'forced_bonus' if ci.forced.
    """
    rows = []
    for k, w in WEIGHTS.items():
        if k in ci.scores:
            val = ci.scores[k]
            rows.append({
                "factor": k,
                "value": val,
                "weight": w,
                "contribution": val * w
            })
    if include_forced and ci.forced:
        rows.append({
            "factor": "forced_bonus",
            "value": 1.0,
            "weight": 1.0,
            "contribution": 1.0
        })
    # Sort by contribution descending
    rows.sort(key=lambda r: -r["contribution"])
    return rows

def get_scoring_profile_version() -> str:
    return SCORING_PROFILE_VERSION
