from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Pattern
import re
import time

@dataclass
class CapturedMemory:
    memory_type: str
    content: str
    subject: Optional[str]
    predicate: Optional[str]
    obj: Optional[str]
    raw_text: str

class MemoryCaptureModule:
    """Phase 1 heuristic memory extractor.

    Extracts simple human-like memory units from user utterances.
    Categories: identity_fact, preference, goal_intent, task_directive, feedback_evaluation, emotional_state
    """
    def __init__(self) -> None:
        self.patterns: List[tuple[str, Pattern[str]]] = [
            ("identity_fact", re.compile(r"^(?P<subject>[A-Z][\w '\-]{1,60}) is (?:your |my |the |a )?(?P<pred>[^.?!]+)", re.IGNORECASE)),
            ("identity_fact", re.compile(r"^(?:my name is|i am|i'm) (?P<self>[^.?!]+)", re.IGNORECASE)),
            ("preference", re.compile(r"i (?:really |pretty )?(?:like|love|enjoy|prefer) (?P<obj>[^.?!]+)", re.IGNORECASE)),
            ("goal_intent", re.compile(r"i (?:need|want|plan|intend) to (?P<obj>[^.?!]+)", re.IGNORECASE)),
            ("task_directive", re.compile(r"remind me to (?P<obj>[^.?!]+)", re.IGNORECASE)),
            ("feedback_evaluation", re.compile(r"that (?:was|is) (?P<obj>helpful|great|awesome|wrong|incorrect|confusing)", re.IGNORECASE)),
            ("emotional_state", re.compile(r"i (?:feel|am|i'm feeling) (?P<obj>[^.?!]+)", re.IGNORECASE)),
        ]

    def extract(self, text: str) -> List[CapturedMemory]:
        out: List[CapturedMemory] = []
        norm = text.strip()
        for mtype, pattern in self.patterns:
            m = pattern.search(norm)
            if not m:
                continue
            subject = None
            predicate = None
            obj = None
            content = None
            if mtype == "identity_fact":
                if "subject" in m.groupdict():
                    subject = m.group("subject").strip()
                    predicate = "is"
                    obj = m.group("pred").strip() if m.groupdict().get("pred") else None
                    content = f"{subject} is {obj}" if obj else norm
                elif "self" in m.groupdict():
                    subject = "user"
                    predicate = "is"
                    obj = m.group("self").strip()
                    content = f"User identity: {obj}"
            elif mtype == "preference":
                subject = "user"
                predicate = "likes"
                obj = m.group("obj").strip()
                content = f"User likes {obj}"
            elif mtype == "goal_intent":
                subject = "user"
                predicate = "intends"
                obj = m.group("obj").strip()
                content = f"User intends to {obj}"
            elif mtype == "task_directive":
                subject = "user"
                predicate = "remind"
                obj = m.group("obj").strip()
                content = f"Reminder: {obj}"
            elif mtype == "feedback_evaluation":
                subject = "user"
                predicate = "feedback"
                obj = m.group("obj").strip()
                content = f"Feedback: {obj}"
            elif mtype == "emotional_state":
                subject = "user"
                predicate = "feels"
                obj = m.group("obj").strip()
                content = f"User feels {obj}"
            if content:
                out.append(CapturedMemory(mtype, content, subject, predicate, obj, raw_text=norm))
        return out

class MemoryCaptureCache:
    """Tracks frequency/first/last seen for captured items."""
    def __init__(self) -> None:
        self._index: Dict[str, Dict[str, Any]] = {}

    def key(self, cm: CapturedMemory) -> str:
        return f"{cm.memory_type}|{(cm.subject or '').lower()}|{(cm.predicate or '').lower()}|{(cm.obj or '').lower()}"[:256]

    def update(self, cm: CapturedMemory) -> Dict[str, Any]:
        k = self.key(cm)
        now = time.time()
        entry = self._index.get(k)
        if entry is None:
            entry = {
                "frequency": 0,
                "first_seen_ts": now,
                "last_seen_ts": now,
                "memory_type": cm.memory_type,
                "subject": cm.subject,
                "predicate": cm.predicate,
                "object": cm.obj,
            }
            self._index[k] = entry
        entry["frequency"] += 1
        entry["last_seen_ts"] = now
        return entry

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._index.get(key)

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self._index.values())
