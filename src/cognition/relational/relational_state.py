"""Data structures for Layer 2: Relational Field.

``RelationalModel`` captures the deep, felt quality of a single
relationship.  ``RelationalField`` is the container that holds all
tracked relationships and provides query helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Relationship status constants ───────────────────────────────────
RELATIONSHIP_STATUSES = ("active", "dormant", "strained", "growing")


def _default_drive_effects() -> Dict[str, float]:
    return {
        "connection": 0.0,
        "competence": 0.0,
        "autonomy": 0.0,
        "understanding": 0.0,
        "meaning": 0.0,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ────────────────────────────────────────────────────────────────────
#  RelationalModel
# ────────────────────────────────────────────────────────────────────

@dataclass
class RelationalModel:
    """Model of one significant relationship.

    Attributes
    ----------
    person_id : str
        Unique identifier for the person (typically session_id or username).
    person_name : str
        Display name.
    felt_quality : float
        -1 (draining) to +1 (nourishing).  Updated each turn via the
        ``RelationalProcessor``.
    attachment_strength : float
        0 to 1.  Grows with interaction count, decays with idle time.
    interaction_count : int
        Total turns exchanged.
    recurring_patterns : list[str]
        Short natural-language observations about recurring dynamics.
    gifts : list[str]
        What this relationship has given us (learned patience, new
        perspective, etc.).
    drive_effects : dict[str, float]
        Smoothed per-drive effect of interactions with this person.
        Negative values mean the drive is *satisfied* by the relationship.
    significant_moment_ids : list[str]
        Memory IDs of especially important turns.
    first_interaction_ts : float
        Epoch timestamp of the first interaction.
    last_interaction_ts : float
        Epoch timestamp of the most recent interaction.
    current_status : str
        One of ``RELATIONSHIP_STATUSES``.
    """

    person_id: str
    person_name: str = ""
    felt_quality: float = 0.0
    attachment_strength: float = 0.0
    interaction_count: int = 0

    recurring_patterns: List[str] = field(default_factory=list)
    gifts: List[str] = field(default_factory=list)
    drive_effects: Dict[str, float] = field(default_factory=_default_drive_effects)
    significant_moment_ids: List[str] = field(default_factory=list)

    first_interaction_ts: float = field(default_factory=time.time)
    last_interaction_ts: float = field(default_factory=time.time)
    last_attachment_decay_ts: float = 0.0
    current_status: str = "active"

    # ── Helpers ──────────────────────────────────────────────────────

    def is_significant(self, threshold: int = 5) -> bool:
        """Has the relationship crossed the significance threshold?"""
        return self.interaction_count >= threshold

    def hours_since_last_interaction(self) -> float:
        """Hours elapsed since the last recorded interaction."""
        if self.last_interaction_ts <= 0.0:
            return 0.0
        return (time.time() - self.last_interaction_ts) / 3600.0

    def time_known_hours(self) -> float:
        """Hours since the first interaction."""
        if self.first_interaction_ts <= 0.0:
            return 0.0
        return (time.time() - self.first_interaction_ts) / 3600.0

    def summary(self) -> str:
        """One-line human-readable summary."""
        name = self.person_name or self.person_id
        quality_word = (
            "nourishing" if self.felt_quality > 0.3
            else "strained" if self.felt_quality < -0.3
            else "neutral"
        )
        return (
            f"{name}: {quality_word} (quality={self.felt_quality:+.2f}, "
            f"attachment={self.attachment_strength:.2f}, "
            f"turns={self.interaction_count}, status={self.current_status})"
        )

    def describe_felt_quality(self) -> str:
        """Natural-language description of the felt quality."""
        q = self.felt_quality
        if q > 0.6:
            return f"Interactions with {self.person_name or self.person_id} feel warm and nourishing."
        if q > 0.2:
            return f"Interactions with {self.person_name or self.person_id} feel generally positive."
        if q > -0.2:
            return f"Interactions with {self.person_name or self.person_id} feel neutral."
        if q > -0.6:
            return f"Interactions with {self.person_name or self.person_id} feel somewhat draining."
        return f"Interactions with {self.person_name or self.person_id} feel difficult and tense."

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "felt_quality": round(self.felt_quality, 4),
            "attachment_strength": round(self.attachment_strength, 4),
            "interaction_count": self.interaction_count,
            "recurring_patterns": list(self.recurring_patterns),
            "gifts": list(self.gifts),
            "drive_effects": {k: round(v, 4) for k, v in self.drive_effects.items()},
            "significant_moment_ids": list(self.significant_moment_ids),
            "first_interaction_ts": self.first_interaction_ts,
            "last_interaction_ts": self.last_interaction_ts,
            "last_attachment_decay_ts": self.last_attachment_decay_ts,
            "current_status": self.current_status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationalModel":
        drive_effects = _default_drive_effects()
        drive_effects.update({
            key: float(value)
            for key, value in dict(d.get("drive_effects", {})).items()
        })
        return cls(
            person_id=d["person_id"],
            person_name=d.get("person_name", ""),
            felt_quality=float(d.get("felt_quality", 0.0)),
            attachment_strength=float(d.get("attachment_strength", 0.0)),
            interaction_count=int(d.get("interaction_count", 0)),
            recurring_patterns=list(d.get("recurring_patterns", [])),
            gifts=list(d.get("gifts", [])),
            drive_effects=drive_effects,
            significant_moment_ids=list(d.get("significant_moment_ids", [])),
            first_interaction_ts=float(d.get("first_interaction_ts", 0.0)),
            last_interaction_ts=float(d.get("last_interaction_ts", 0.0)),
            last_attachment_decay_ts=float(d.get("last_attachment_decay_ts", 0.0)),
            current_status=d.get("current_status", "active"),
        )


# ────────────────────────────────────────────────────────────────────
#  RelationalField
# ────────────────────────────────────────────────────────────────────

@dataclass
class RelationalField:
    """Container of all tracked relationships.

    The field provides query helpers and tracks the *current interlocutor*
    so that downstream consumers know whose relational context to inject.
    """

    relationships: Dict[str, RelationalModel] = field(default_factory=dict)
    current_interlocutor: Optional[str] = None  # person_id
    max_relationships: Optional[int] = None

    # ── Query helpers ────────────────────────────────────────────────

    def get(self, person_id: str) -> Optional[RelationalModel]:
        """Return the model for *person_id*, or ``None``."""
        return self.relationships.get(person_id)

    def get_or_create(self, person_id: str, person_name: str = "") -> RelationalModel:
        """Return existing model or create a fresh one."""
        if person_id not in self.relationships:
            self._evict_if_needed()
            self.relationships[person_id] = RelationalModel(
                person_id=person_id,
                person_name=person_name or person_id,
            )
        return self.relationships[person_id]

    def _evict_if_needed(self) -> None:
        if self.max_relationships is None or self.max_relationships <= 0:
            return
        if len(self.relationships) < self.max_relationships:
            return

        evictable = [
            (person_id, rel)
            for person_id, rel in self.relationships.items()
            if person_id != self.current_interlocutor
        ]
        if not evictable:
            evictable = list(self.relationships.items())
        if not evictable:
            return

        person_id, _rel = min(
            evictable,
            key=lambda item: (
                item[1].current_status != "dormant",
                item[1].attachment_strength,
                item[1].interaction_count,
                item[1].last_interaction_ts if item[1].last_interaction_ts > 0.0 else -1.0,
                item[0],
            ),
        )
        del self.relationships[person_id]

    def set_interlocutor(self, person_id: Optional[str]) -> None:
        """Set the current interlocutor for context injection."""
        self.current_interlocutor = person_id

    def current_relationship(self) -> Optional[RelationalModel]:
        """Return the model for the current interlocutor, if set."""
        if self.current_interlocutor is None:
            return None
        return self.relationships.get(self.current_interlocutor)

    def significant_relationships(self, threshold: int = 5) -> List[RelationalModel]:
        """All relationships that have crossed the significance threshold."""
        return [r for r in self.relationships.values() if r.is_significant(threshold)]

    def most_nourishing(self, n: int = 3) -> List[RelationalModel]:
        """Top-N relationships by felt quality."""
        return sorted(
            self.relationships.values(),
            key=lambda r: r.felt_quality,
            reverse=True,
        )[:n]

    def most_attached(self, n: int = 3) -> List[RelationalModel]:
        """Top-N relationships by attachment strength."""
        return sorted(
            self.relationships.values(),
            key=lambda r: r.attachment_strength,
            reverse=True,
        )[:n]

    def count(self) -> int:
        return len(self.relationships)

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_interlocutor": self.current_interlocutor,
            "max_relationships": self.max_relationships,
            "relationships": {
                pid: rm.to_dict()
                for pid, rm in self.relationships.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationalField":
        rels = {}
        for pid, rdict in d.get("relationships", {}).items():
            rels[pid] = RelationalModel.from_dict(rdict)
        return cls(
            relationships=rels,
            current_interlocutor=d.get("current_interlocutor"),
            max_relationships=d.get("max_relationships"),
        )
