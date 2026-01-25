from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
import time
import uuid
import threading

from .models import TurnRecord


@dataclass
class ConversationSession:
    """Holds turn history & metadata for a single chat session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    turns: Deque[TurnRecord] = field(default_factory=lambda: deque(maxlen=250))
    last_access: float = field(default_factory=time.time)

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.last_access = time.time()

    def recent_turns(self, limit: int = 8) -> List[TurnRecord]:
        return list(self.turns)[-limit:]


class SessionManager:
    """LRU session manager for conversation sessions."""
    def __init__(self, max_sessions: int = 100):
        self._max_sessions = max_sessions
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.RLock()

    def create_or_get(self, session_id: Optional[str] = None) -> ConversationSession:
        with self._lock:
            if session_id and session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.last_access = time.time()
                return sess
            new_id = session_id or str(uuid.uuid4())
            sess = ConversationSession(session_id=new_id)
            self._sessions[new_id] = sess
            self._evict_if_needed()
            return sess

    def add_turn(self, session_id: str, turn: TurnRecord) -> None:
        sess = self.create_or_get(session_id)
        sess.add_turn(turn)

    def _evict_if_needed(self) -> None:
        if len(self._sessions) <= self._max_sessions:
            return
        # Evict oldest last_access
        sorted_ids = sorted(self._sessions.items(), key=lambda kv: kv[1].last_access)
        while len(self._sessions) > self._max_sessions:
            sid, _ = sorted_ids.pop(0)
            del self._sessions[sid]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "turns_total": sum(len(s.turns) for s in self._sessions.values()),
            }

    def list_session_ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())
