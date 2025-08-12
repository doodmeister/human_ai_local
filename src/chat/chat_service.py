from __future__ import annotations

from typing import Dict, Any, Optional
import time
import asyncio

from .conversation_session import SessionManager
from .models import TurnRecord
from .context_builder import ContextBuilder
from .emotion_salience import estimate_salience_and_valence
from .metrics import metrics_registry
from .provenance import build_item_provenance
from .scoring import get_scoring_profile_version
from .constants import PREVIEW_MAX_ITEMS, PREVIEW_MAX_CONTENT_CHARS


class ChatService:
    """
    Orchestrates chat turn processing:
    - Adds user turn with salience/valence tagging
    - Builds context (via ContextBuilder)
    - Generates placeholder assistant response (to be replaced with LLM call)
    - Applies consolidation decision heuristic
    - Returns structured payload
    """

    def __init__(
        self,
        session_manager: SessionManager,
        context_builder: ContextBuilder,
    ):
        self.sessions = session_manager
        self.context_builder = context_builder
        self.consolidation_log: list = []  # (turn_id, status, salience, valence)

    def get_context_preview(
        self,
        message: str,
        session_id: Optional[str] = None,
        flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight deterministic context preview (no full trace, no model response).
        """
        flags = flags or {}
        sess = self.sessions.create_or_get(session_id or "default")
        # Add a transient turn (not persisted) for preview retrieval basis
        builder = self.context_builder
        built = builder.build(
            sess,
            query=message,
            include_attention=not flags.get("disable_attention", False),
            include_memory=not flags.get("disable_memory", False),
            include_trace=False,
        )
        items_summary = self._summarize_context_items(built.items)
        metrics_registry.inc("context_preview_calls_total")
        return {
            "session_id": sess.session_id,
            "scoring_version": get_scoring_profile_version(),
            "item_count": len(items_summary),
            "items": items_summary,
        }

    def process_user_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        t_start = time.time()
        flags = flags or {}
        sess = self.sessions.create_or_get(session_id)
        salience, valence = estimate_salience_and_valence(message)
        importance = self._estimate_importance(message, salience, valence)
        user_turn = TurnRecord(
            role="user",
            content=message,
            salience=salience,
            emotional_valence=valence,
            importance=importance,
        )
        sess.add_turn(user_turn)

        # Build context (use user message as query)
        built = self.context_builder.build(
            session=sess,
            query=message,
            include_attention=flags.get("include_attention", True),
            include_memory=flags.get("include_memory", True),
            include_trace=flags.get("include_trace", True),
        )

        # Placeholder assistant response (replace with LLM integration)
        response_text = self._generate_placeholder_response(message)

        assistant_turn = TurnRecord(
            role="assistant",
            content=response_text,
            salience=salience * 0.5,
            emotional_valence=valence * 0.3,
            importance=importance * 0.5,
        )
        sess.add_turn(assistant_turn)

        # Consolidation decision (placeholder)
        t_cons = time.time()
        stored = self._maybe_consolidate(user_turn, assistant_turn)
        built.metrics.consolidation_time_ms = (time.time() - t_cons) * 1000.0
        built.metrics.consolidated_user_turn = stored
        self.consolidation_log.append(
            {
                "user_turn_id": user_turn.turn_id,
                "status": user_turn.consolidation_status,
                "salience": user_turn.salience,
                "valence": user_turn.valence,
                "timestamp": time.time(),
            }
        )
        # Build payload
        payload = {
            "session_id": sess.session_id,
            "user_turn_id": user_turn.turn_id,
            "assistant_turn_id": assistant_turn.turn_id,
            "response": assistant_turn.content,
            "metrics": {
                "turn_latency_ms": built.metrics.turn_latency_ms,
                "retrieval_time_ms": built.metrics.retrieval_time_ms,
                "stm_hits": built.metrics.stm_hits,
                "ltm_hits": built.metrics.ltm_hits,
                "episodic_hits": built.metrics.episodic_hits,
                "fallback_used": built.metrics.fallback_used,
                "consolidation_time_ms": built.metrics.consolidation_time_ms,
                "consolidated_user_turn": built.metrics.consolidated_user_turn,
            },
            "context_items": [
                {
                    "source_id": ci.source_id,
                    "source_system": ci.source_system,
                    "reason": ci.reason,
                    "rank": ci.rank,
                    "content": ci.content,
                    "scores": ci.scores,
                }
                for ci in built.items
            ],
        }
        # Add trace if requested
        if flags.get("include_trace"):
            payload["trace"] = {
                "pipeline": [s.__dict__ for s in built.trace.pipeline_stages],
                "notes": list(built.trace.notes),
            }
            payload["trace"]["provenance_details"] = build_item_provenance(built.items)
            payload["trace"]["scoring_version"] = get_scoring_profile_version()
            if any(ci.source_system == "attention" for ci in built.items):
                payload["trace"].setdefault("notes", []).append("attention_focus items included")
            if any(ci.source_system == "executive" for ci in built.items):
                payload["trace"].setdefault("notes", []).append("executive_mode state included")
            payload["trace"]["consolidation_log_tail"] = self.consolidation_log[-10:]
        # Turn latency + histogram observe
        total_lat = (time.time() - t_start) * 1000.0
        payload["metrics"]["turn_latency_ms"] = total_lat
        metrics_registry.observe_hist("chat_turn_latency_ms", total_lat)
        snap = metrics_registry.snapshot()
        if "chat_turn_latency_p95_ms" in snap["state"]:
            p95 = snap["state"]["chat_turn_latency_p95_ms"]
            payload["metrics"]["latency_p95_ms"] = p95
            # Performance degradation flag (compare to configured target if available)
            target = self.context_builder.cfg.get("performance_target_p95_ms")
            if isinstance(target, (int, float)) and target > 0:
                degraded = p95 > target
                metrics_registry.state["performance_degraded"] = degraded
                payload["metrics"]["performance_degraded"] = degraded
        return payload

    async def process_user_message_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        flags: Optional[Dict[str, bool]] = None,
        token_delay_ms: int = 5,
    ):
        """
        Async streaming placeholder.
        Yields dict chunks: first metadata, then token chunks, final summary.
        """
        flags = flags or {}
        base = self.process_user_message(message, session_id=session_id, flags=flags)
        full_text = base["response"]
        yield {"type": "meta", "session_id": base["session_id"], "user_turn_id": base["user_turn_id"]}
        for token in full_text.split():
            await asyncio.sleep(token_delay_ms / 1000.0)
            yield {"type": "token", "t": token}
        yield {"type": "final", "data": base}

    # --- Helpers ---

    def _estimate_importance(self, content: str, salience: float, valence: float) -> float:
        length_factor = min(1.0, len(content.split()) / 30.0)
        emotional_weight = min(1.0, abs(valence))
        return max(salience * 0.5 + length_factor * 0.3 + emotional_weight * 0.2, salience * 0.6)

    def _generate_placeholder_response(self, user_text: str) -> str:
        # Deterministic placeholder for early integration testing.
        snippet = user_text[:60].strip()
        return f"Acknowledged: {snippet}"

    def _maybe_consolidate(self, user_turn: TurnRecord, assistant_turn: TurnRecord) -> bool:
        # existing heuristic stub: high salience OR strong valence magnitude
        sal = user_turn.salience
        val_mag = abs(user_turn.valence)
        if sal >= 0.55 or val_mag >= 0.6:
            user_turn.consolidation_status = "stored"
            return True
        user_turn.consolidation_status = "skipped"
        return False

    def _summarize_context_items(self, items):
        max_items = self.context_builder.cfg.get("preview_max_items", PREVIEW_MAX_ITEMS)
        max_chars = self.context_builder.cfg.get("preview_max_content_chars", PREVIEW_MAX_CONTENT_CHARS)
        out = []
        for ci in items[:max_items]:
            content = ci.content
            if len(content) > max_chars:
                content = content[: max_chars - 3] + "..."
            out.append(
                {
                    "source": ci.source_system,
                    "reason": ci.reason,
                    "rank": ci.rank,
                    "composite": ci.scores.get("composite"),
                    "content": content,
                }
            )
        return out

    def _trace_to_dict(self, built) -> Dict[str, Any]:
        return {
            "stages": [
                {
                    "name": s.name,
                    "candidates_in": s.candidates_in,
                    "candidates_out": s.candidates_out,
                    "latency_ms": round(s.latency_ms, 2),
                    "added": s.added,
                }
                for s in built.trace.pipeline_stages
            ],
            "notes": built.trace.notes,
            "degraded": built.trace.degraded_mode,
        }

    def performance_status(self) -> Dict[str, Any]:
        """
        Lightweight performance status (p95 + degradation flag if computed).
        """
        p95 = metrics_registry.get_p95("chat_turn_latency_ms")
        target = self.context_builder.cfg.get("performance_target_p95_ms")
        degraded = False
        if isinstance(target, (int, float)) and target > 0:
            degraded = p95 > target
        return {
            "latency_p95_ms": p95,
            "target_p95_ms": target,
            "performance_degraded": degraded,
        }
