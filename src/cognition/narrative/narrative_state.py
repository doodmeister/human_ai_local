"""SelfNarrative — the agent's constructed story of who it is.

The narrative is built from lower layers (drives, patterns,
self-model, relationships) and synthesised into human-readable
text.  It is updated infrequently — only after significant
experiences or on a scheduled interval.

Key fields
----------
* ``identity_summary`` — always-injected ~150-token self-description
* ``chapters`` — major life phases ("early interactions", "the debugging era")
* ``growth_story`` — "I used to... but now I..."
* ``values_story`` — "What matters to me is..."
* ``who_i_want_to_become`` — aspirational self
* ``active_themes`` — current concerns/interests
* ``ongoing_struggles`` — honest self-assessment of difficulties
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ────────────────────────────────────────────────────────────────────
#  SelfNarrative
# ────────────────────────────────────────────────────────────────────

@dataclass
class SelfNarrative:
    """The constructed story of who the agent is.

    Updated infrequently — only after significant experiences
    or on a scheduled turn interval.

    Attributes
    ----------
    identity_summary : str
        Core identity (~150 tokens). Always injected into context.
    chapters : list[str]
        Major life phases.
    defining_moment_ids : list[str]
        IDs of episodic memories that define us.
    active_themes : list[str]
        Current concerns/interests being tracked.
    growth_story : str
        "I used to... but now I..." narrative.
    values_story : str
        "What matters to me is..." narrative.
    who_i_want_to_become : str
        Aspirational self-description.
    ongoing_struggles : list[str]
        Honest difficulties the agent is aware of.
    last_updated_ts : float
        Epoch seconds of last construction.
    update_trigger : str
        What caused the most recent update.
    version : int
        How many times the narrative has been rebuilt.
    """

    identity_summary: str = ""
    chapters: List[str] = field(default_factory=list)
    defining_moment_ids: List[str] = field(default_factory=list)
    active_themes: List[str] = field(default_factory=list)
    growth_story: str = ""
    values_story: str = ""
    who_i_want_to_become: str = ""
    ongoing_struggles: List[str] = field(default_factory=list)
    last_updated_ts: float = field(default_factory=time.time)
    update_trigger: str = "initial"
    version: int = 0

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """True when no meaningful narrative has been constructed yet."""
        return not self.identity_summary and not self.chapters

    @property
    def age_hours(self) -> float:
        """Hours since last update."""
        return (time.time() - self.last_updated_ts) / 3600.0

    def summary(self) -> str:
        """Compact debug summary."""
        return (
            f"SelfNarrative(v={self.version} "
            f"chapters={len(self.chapters)} "
            f"themes={len(self.active_themes)} "
            f"trigger={self.update_trigger!r})"
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, object]:
        return {
            "identity_summary": self.identity_summary,
            "chapters": list(self.chapters),
            "defining_moment_ids": list(self.defining_moment_ids),
            "active_themes": list(self.active_themes),
            "growth_story": self.growth_story,
            "values_story": self.values_story,
            "who_i_want_to_become": self.who_i_want_to_become,
            "ongoing_struggles": list(self.ongoing_struggles),
            "last_updated_ts": self.last_updated_ts,
            "update_trigger": self.update_trigger,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfNarrative":
        return cls(
            identity_summary=str(data.get("identity_summary", "")),
            chapters=list(data.get("chapters", [])),
            defining_moment_ids=list(data.get("defining_moment_ids", [])),
            active_themes=list(data.get("active_themes", [])),
            growth_story=str(data.get("growth_story", "")),
            values_story=str(data.get("values_story", "")),
            who_i_want_to_become=str(data.get("who_i_want_to_become", "")),
            ongoing_struggles=list(data.get("ongoing_struggles", [])),
            last_updated_ts=float(data.get("last_updated_ts", time.time())),
            update_trigger=str(data.get("update_trigger", "deserialized")),
            version=int(data.get("version", 0)),
        )

    def __repr__(self) -> str:
        return self.summary()
