from __future__ import annotations
from typing import List, Dict, Any
from .models import ContextItem
from .scoring import factor_breakdown

def build_item_provenance(items: List[ContextItem], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Produce structured provenance details per context item.
    """
    out: List[Dict[str, Any]] = []
    for ci in items:
        factors = factor_breakdown(ci)
        composite = ci.scores.get("composite", 0.0)
        partial_sum = sum(f["contribution"] for f in factors)
        out.append({
            "source_id": ci.source_id,
            "source_system": ci.source_system,
            "reason": ci.reason,
            "composite": composite,
            "factors": [
                {
                    **f,
                    "category": (
                        "attention" if f["factor"] in ("focus", "forced_bonus")
                        else "executive" if f["factor"] in ("mode_confidence", "state")
                        else "retrieval"
                    ),
                }
                for f in factors[:top_n]
            ],
            "composite_vs_factor_sum_delta": round(composite - partial_sum, 6)
        })
    return out
