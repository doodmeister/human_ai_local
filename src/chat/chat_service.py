from __future__ import annotations

from typing import Dict, Any, Optional
import time
import asyncio
from datetime import datetime
from collections import deque
from pydantic import BaseModel, Field
try:  # Pydantic v2
    from pydantic import ConfigDict
except Exception:  # fallback for older versions
    ConfigDict = None  # type: ignore

from .conversation_session import SessionManager
from .models import TurnRecord
from .context_builder import ContextBuilder
from .emotion_salience import estimate_salience_and_valence
from .metrics import metrics_registry
from .provenance import build_item_provenance
from .scoring import get_scoring_profile_version
from .constants import PREVIEW_MAX_ITEMS, PREVIEW_MAX_CONTENT_CHARS
from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from .memory_capture import MemoryCaptureModule, MemoryCaptureCache, CapturedMemory  # added import near top
import re  # added for fact question pattern


class ChatService:
    """
    Orchestrates chat turn processing:
    - Adds user turn with salience/valence tagging
    - Builds context (via ContextBuilder)
    - Generates placeholder assistant response (to be replaced with LLM call)
    - Applies consolidation decision heuristic
    - Returns structured payload
    """

    def __init__(self,
                 session_manager: SessionManager,
                 context_builder: ContextBuilder,
                 consolidator: Optional[Any] = None,
                 agent: Optional[Any] = None) -> None:
        self.sessions = session_manager
        self.context_builder = context_builder
        self.consolidator = consolidator
        self.agent = agent
        # Memory capture modules (Phase 1)
        self._capture = MemoryCaptureModule()
        self._capture_cache = MemoryCaptureCache()
        # Consolidation event log (simple list for recent trace enrichment)
        self.consolidation_log = []  # entries: dict(user_turn_id,status,salience,valence,timestamp)
        # Metacognitive tracking fields
        self._turn_counter = 0
        self._last_metacog_snapshot = None
        self._metacog_interval = int(self.context_builder.cfg.get("metacog_turn_interval", 5))
        # Snapshot history ring buffer
        history_size = int(self.context_builder.cfg.get("metacog_snapshot_history_size", 50))
        self._metacog_history = deque(maxlen=history_size)
    # Metacog metrics counters (initialized lazily via metrics_registry)
    # Counter names:
    #  - metacog_snapshots_total
    #  - metacog_advisory_items_total
    #  - metacog_stm_high_util_events_total
    #  - metacog_performance_degraded_events_total

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
        # --- Phase 1 memory capture ---
        try:
            captures = self._capture.extract(message)
        except Exception:
            captures = []
        stored_captures: list[dict] = []
        for cm in captures:
            # Gather prior objects for contradiction check BEFORE updating frequency
            prior_objs = set()
            if cm.memory_type == "identity_fact" and cm.subject:
                try:
                    subj_l = (cm.subject or "").lower()
                    for rec in self._capture_cache.as_list():
                        if rec.get("memory_type") == "identity_fact" and (rec.get("subject") or "").lower() == subj_l:
                            pobj = (rec.get("object") or "").lower()
                            if pobj:
                                prior_objs.add(pobj)
                except Exception:
                    prior_objs = set()
            stats = self._capture_cache.update(cm)
            meta = {
                "memory_type": cm.memory_type,
                "subject": (cm.subject or "").lower(),
                "predicate": (cm.predicate or "").lower(),
                "object": (cm.obj or "").lower(),
                "raw_text": cm.raw_text,
                "frequency": stats["frequency"],
                "first_seen_ts": stats["first_seen_ts"],
                "last_seen_ts": stats["last_seen_ts"],
                "extracted_from_turn_id": user_turn.turn_id,
            }
            # Contradiction detection for identity facts (same subject different object values over time)
            if cm.memory_type == "identity_fact" and meta["subject"] and meta["object"]:
                try:
                    obj_l = meta["object"]
                    if prior_objs and obj_l not in prior_objs:
                        meta["contradiction"] = True
                        meta["contradicted_prior"] = list(prior_objs)[:5]
                        metrics_registry.inc("captured_memory_contradictions_total")
                except Exception:
                    pass
            # Frequency reinforcement: if frequency passes thresholds, mark reinforcement and slightly boost importance
            try:
                freq = stats.get("frequency", 0)
                if freq in (2, 3, 5, 8):  # key reinforcement milestones
                    meta["reinforced"] = True
                    # Boost importance proportionally (capped)
                    boost = 0.05 if freq == 2 else 0.07 if freq == 3 else 0.10 if freq == 5 else 0.12
                    user_turn.importance = min(1.0, (user_turn.importance or 0.0) + boost)
                    metrics_registry.inc("captured_memory_reinforcements_total")
                    # Attempt STM refresh (rehearsal) if stm supports method
                    stm_obj = getattr(self.consolidator, "stm", None) if self.consolidator is not None else None
                    if stm_obj is not None:
                        # If STM has method to update activation; else re-add item to refresh recency
                        if hasattr(stm_obj, "refresh_item_activation"):
                            try:
                                key = f"{cm.memory_type}:{meta['subject']}:{meta['object']}"
                                stm_obj.refresh_item_activation(key)  # type: ignore
                            except Exception:
                                pass
                        elif hasattr(stm_obj, "add_item"):
                            try:
                                stm_obj.add_item(content=cm.content, metadata={**meta, "rehearsal": True})  # type: ignore
                            except Exception:
                                pass
            except Exception:
                pass
            # Attempt to store in STM if accessible via consolidator
            try:
                stm_obj = None
                if self.consolidator is not None:
                    stm_obj = getattr(self.consolidator, "stm", None)
                if stm_obj is not None and hasattr(stm_obj, "add_item"):
                    stm_obj.add_item(content=cm.content, metadata=meta)  # type: ignore
                # Fallback: if no add_item but store_memory exists
                elif stm_obj is not None and hasattr(stm_obj, "store_memory"):
                    stm_obj.store_memory(memory_id=f"cap-{user_turn.turn_id[:8]}-{meta['frequency']}", content=cm.content, importance=0.5)  # type: ignore
            except Exception:
                pass
            # Semantic promotion (identity/preference/goal) when frequency threshold hit
            try:
                freq = meta.get("frequency", 0)
                if cm.memory_type in ("identity_fact", "preference", "goal_intent") and freq in (3, 5):
                    # Access semantic memory through context_builder if available
                    sem = getattr(self.context_builder, "semantic", None)
                    if sem is not None and hasattr(sem, "add_item"):
                        promo_meta = {k: meta[k] for k in ("memory_type", "subject", "predicate", "object", "frequency") if k in meta}
                        promo_meta.update({
                            "promotion_reason": "frequency_threshold",
                            "promotion_freq": freq,
                            "source": "chat_capture",
                        })
                        try:
                            sem.add_item(content=cm.content, metadata=promo_meta)  # type: ignore
                            meta["promoted_semantic"] = True
                            metrics_registry.inc("captured_memory_semantic_promotions_total")
                        except Exception:
                            pass
            except Exception:
                pass
            stored_captures.append({"content": cm.content, **meta})
        # Optionally attach capture summary to metrics registry state for introspection
        if stored_captures:
            try:
                metrics_registry.state["last_captured_memories"] = stored_captures[-5:]
                metrics_registry.inc("captured_memory_units_total", len(stored_captures))
            except Exception:
                pass

        # --- Adaptive retrieval limits (temporary adjustment of max_context_items) ---
        cfg = self.context_builder.cfg
        original_max_ctx = cfg.get("max_context_items")
        adaptive_retrieval_applied = False
        try:
            degraded_flag = metrics_registry.state.get("performance_degraded")
            stm_util_ratio = None
            # Estimate STM utilization cheaply via consolidator if available
            cons = getattr(self, "consolidator", None)
            if cons is not None:
                stm_obj = getattr(cons, "stm", None)
                if stm_obj is not None:
                    cap = getattr(stm_obj, "capacity", None)
                    size = None
                    if hasattr(stm_obj, "__len__"):
                        try:
                            size = len(stm_obj)  # type: ignore
                        except Exception:
                            size = None
                    if size is None:
                        size = getattr(stm_obj, "size", None)
                    if isinstance(size, int) and isinstance(cap, int) and cap:
                        stm_util_ratio = min(1.0, size / cap)
            # Heuristic: if degraded or high STM utilization, reduce context size 25-50%
            if original_max_ctx and isinstance(original_max_ctx, int) and original_max_ctx > 4:
                reduce_factor = 1.0
                if degraded_flag:
                    reduce_factor *= 0.75
                if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                    reduce_factor *= 0.75  # compounding -> potentially 0.56
                if reduce_factor < 0.999:
                    new_limit = max(4, int(original_max_ctx * reduce_factor))
                    if new_limit < original_max_ctx:
                        cfg["max_context_items"] = new_limit
                        adaptive_retrieval_applied = True
                        metrics_registry.inc("adaptive_retrieval_applied_total")
        except Exception:
            pass

        built = self.context_builder.build(
            session=sess,
            query=message,
            include_attention=flags.get("include_attention", True),
            include_memory=flags.get("include_memory", True),
            include_trace=flags.get("include_trace", True),
        )
        # Restore original context limit after build
        if adaptive_retrieval_applied:
            try:
                cfg["max_context_items"] = original_max_ctx
            except Exception:
                pass

        # Prospective memory injection (non-invasive: append dicts only)
        extra_due: list[Dict[str, Any]] = []
        try:
            pm = get_inmemory_prospective_memory()
            # Track injected reminder ids inside session state to avoid double counting
            injected_key = "_prospective_injected_ids"
            injected_ids = getattr(sess, injected_key, set())
            if not isinstance(injected_ids, set):  # safety
                injected_ids = set()
            for r in pm.check_due():
                extra_due.append({
                    "source_id": f"reminder-{r.id}",
                    "source_system": "prospective",
                    "reason": "due_reminder",
                    "rank": 0,
                    "content": f"REMINDER: {r.content}",
                    "scores": {"reminder": 1.0, "composite": 1.0},
                })
                if r.id not in injected_ids:
                    metrics_registry.inc("prospective_reminders_injected_total")
                    injected_ids.add(r.id)
            setattr(sess, injected_key, injected_ids)
        except Exception:
            extra_due = []

        assistant_content = self._invoke_agent_response(message, built)
        # Before forming assistant response, check retrieval for simple questions
        try:
            answer = self._attempt_fact_answer(message)
            if answer:
                assistant_content = answer
        except Exception:
            pass
        if not assistant_content:
            assistant_content = self._generate_placeholder_response(message)
        assistant_turn = TurnRecord(
            role="assistant",
            content=assistant_content,
            salience=salience * 0.5,
            emotional_valence=valence * 0.3,
            importance=importance * 0.5,
        )
        sess.add_turn(assistant_turn)

        t_cons = time.time()
        # Adaptive threshold tweak: if high STM utilization or performance degraded, raise salience threshold slightly.
        adaptive_cfg = None
        original_sal_thr = None
        original_val_thr = None
        try:
            adaptive_cfg = getattr(self, "context_builder").cfg
            original_sal_thr = adaptive_cfg.get("consolidation_salience_threshold")
            original_val_thr = adaptive_cfg.get("consolidation_valence_threshold")
            # Estimate STM utilization (best-effort)
            stm_util = None
            cons = getattr(self, "consolidator", None)
            if cons is not None:
                stm_obj = getattr(cons, "stm", None)
                if stm_obj is not None:
                    cap = getattr(stm_obj, "capacity", None)
                    size = None
                    if hasattr(stm_obj, "__len__"):
                        try:
                            size = len(stm_obj)  # type: ignore
                        except Exception:
                            size = None
                    if size is None:
                        size = getattr(stm_obj, "size", None)
                    if isinstance(size, int) and isinstance(cap, int) and cap > 0:
                        stm_util = size / cap
            degraded = metrics_registry.state.get("performance_degraded")
            if adaptive_cfg is not None:
                # Base thresholds
                base_sal = adaptive_cfg.get("consolidation_salience_threshold", 0.55)
                base_val = adaptive_cfg.get("consolidation_valence_threshold", 0.60)
                sal_adj = base_sal
                val_adj = base_val
                if stm_util is not None and stm_util >= 0.85:
                    sal_adj += 0.05  # slightly more selective
                if degraded:
                    sal_adj += 0.05  # further tighten under performance pressure
                # Cap adjustments
                sal_adj = min(sal_adj, 0.85)
                adaptive_cfg["consolidation_salience_threshold"] = sal_adj
                adaptive_cfg["consolidation_valence_threshold"] = val_adj
        except Exception:
            pass
        stored = self._maybe_consolidate(user_turn, assistant_turn)
        # Restore original thresholds to avoid permanent drift
        try:
            if adaptive_cfg is not None:
                if original_sal_thr is not None:
                    adaptive_cfg["consolidation_salience_threshold"] = original_sal_thr
                if original_val_thr is not None:
                    adaptive_cfg["consolidation_valence_threshold"] = original_val_thr
        except Exception:
            pass
        built.metrics.consolidation_time_ms = (time.time() - t_cons) * 1000.0
        built.metrics.consolidated_user_turn = stored
        self.consolidation_log.append({
            "user_turn_id": user_turn.turn_id,
            "status": user_turn.consolidation_status,
            "salience": user_turn.salience,
            "valence": user_turn.emotional_valence,
            "timestamp": time.time(),
        })
        # Metacog turn counter / periodic snapshot
        self._turn_counter += 1
        attach_metacog = False
        if self._metacog_interval > 0 and (self._turn_counter % self._metacog_interval == 0):
            try:
                self._last_metacog_snapshot = self._metacog_snapshot()
                metrics_registry.inc("metacog_snapshots_total")
                attach_metacog = True
            except Exception:
                pass

        payload = {
            "session_id": sess.session_id,
            "user_turn_id": user_turn.turn_id,
            "assistant_turn_id": assistant_turn.turn_id,
            "response": assistant_turn.content,
            # Newly exposed: most recent captured memory units extracted from this user message
            "captured_memories": stored_captures,
            "metrics": {
                "turn_latency_ms": built.metrics.turn_latency_ms,
                "retrieval_time_ms": built.metrics.retrieval_time_ms,
                "stm_hits": built.metrics.stm_hits,
                "ltm_hits": built.metrics.ltm_hits,
                "episodic_hits": built.metrics.episodic_hits,
                "fallback_used": built.metrics.fallback_used,
                "consolidation_time_ms": built.metrics.consolidation_time_ms,
                "consolidated_user_turn": built.metrics.consolidated_user_turn,
                # Debug: add consolidation decision info
                "user_salience": user_turn.salience,
                "user_valence": user_turn.emotional_valence,
                "user_importance": user_turn.importance,
                "consolidation_status": user_turn.consolidation_status,
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
            ] + extra_due,
        }
        if attach_metacog and self._last_metacog_snapshot:
            payload["metacog"] = self._last_metacog_snapshot
            # Persist snapshot into LTM (best-effort) if LTM available via context builder
            try:
                ltm = getattr(self.context_builder, "ltm", None)
                snap = dict(self._last_metacog_snapshot)
                snap["type"] = "meta_reflection"
                # Add to history first
                try:
                    self._metacog_history.append(snap)
                except Exception:
                    pass
                if ltm is not None and hasattr(ltm, "add_item"):
                    # Expect signature add_item(content: str, metadata: dict | None)
                    content = (
                        f"Metacog snapshot turn={snap.get('turn_counter')} "
                        f"perf_p95={snap.get('performance', {}).get('latency_p95_ms')} "
                        f"stm_util={snap.get('stm_utilization')}"
                    )
                    try:
                        ltm.add_item(content=content, metadata=snap)  # type: ignore
                    except Exception:
                        pass
            except Exception:
                pass
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

        total_lat = (time.time() - t_start) * 1000.0
        payload["metrics"]["turn_latency_ms"] = total_lat
        metrics_registry.observe_hist("chat_turn_latency_ms", total_lat)
        ema_alpha = 0.2
        prev_ema = metrics_registry.state.get("ema_turn_latency_ms")
        ema = total_lat if prev_ema is None else prev_ema + ema_alpha * (total_lat - prev_ema)
        metrics_registry.state["ema_turn_latency_ms"] = ema
        payload["metrics"]["ema_turn_latency_ms"] = ema
        metrics_registry.mark_event("chat_turn")
        window = self.context_builder.cfg.get("throughput_window_seconds", 60)
        tps = metrics_registry.get_rate("chat_turn", window_seconds=float(window))
        metrics_registry.state["chat_turns_per_sec"] = tps
        payload["metrics"]["chat_turns_per_sec"] = tps
        snap = metrics_registry.snapshot()
        if "chat_turn_latency_p95_ms" in snap.get("state", {}):
            p95 = snap["state"]["chat_turn_latency_p95_ms"]
            payload["metrics"]["latency_p95_ms"] = p95
            target = self.context_builder.cfg.get("performance_target_p95_ms")
            if isinstance(target, (int, float)) and target > 0:
                degraded = p95 > target
                metrics_registry.state["performance_degraded"] = degraded
                payload["metrics"]["performance_degraded"] = degraded
        # --- Dynamic metacog interval adjustment ---
        try:
            # Use a simple stability heuristic: if last 5 turns not degraded and stm util < 70%, relax interval (+1 up to 10)
            # If degraded or high util (>=85%) tighten (-1 down to min 2)
            history_util = []
            # Estimate current stm util again (cheap best-effort)
            stm_util = None
            cons = getattr(self, "consolidator", None)
            if cons is not None:
                stm_obj = getattr(cons, "stm", None)
                if stm_obj is not None:
                    cap = getattr(stm_obj, "capacity", None)
                    size = None
                    if hasattr(stm_obj, "__len__"):
                        try:
                            size = len(stm_obj)  # type: ignore
                        except Exception:
                            size = None
                    if size is None:
                        size = getattr(stm_obj, "size", None)
                    if isinstance(size, int) and isinstance(cap, int) and cap:
                        stm_util = min(1.0, size / cap)
            degraded_flag = metrics_registry.state.get("performance_degraded")
            current_interval = self._metacog_interval
            new_interval = current_interval
            if degraded_flag or (stm_util is not None and stm_util >= 0.85):
                if current_interval > 2:
                    new_interval = current_interval - 1
            else:
                if (stm_util is None or stm_util < 0.70) and not degraded_flag:
                    if current_interval < 10:
                        new_interval = current_interval + 1
            if new_interval != current_interval:
                self._metacog_interval = new_interval
                metrics_registry.state["metacog_interval"] = new_interval
        except Exception:
            pass
        # --- Adaptive STM activation weight modulation (meta-driven) ---
        try:
            # Accessible stm via consolidator if available
            stm_obj = None
            if getattr(self, 'consolidator', None) is not None:
                stm_obj = getattr(self.consolidator, 'stm', None)
            # Only proceed if stm exposes set_activation_weights
            if stm_obj is not None and hasattr(stm_obj, 'set_activation_weights') and hasattr(stm_obj, 'get_activation_weights'):
                weights = stm_obj.get_activation_weights()
                degraded_flag = metrics_registry.state.get('performance_degraded')
                # Recompute util (reuse above if still in scope)
                stm_util_ratio = None
                try:
                    cap = getattr(stm_obj, 'capacity', None)
                    size = None
                    if hasattr(stm_obj, '__len__'):
                        size = len(stm_obj)  # type: ignore
                    if isinstance(size, int) and isinstance(cap, int) and cap:
                        stm_util_ratio = min(1.0, size / cap)
                except Exception:
                    pass
                rec = weights.get('recency', 0.4)
                freq = weights.get('frequency', 0.3)
                sal = weights.get('salience', 0.3)
                adjusted = False
                # If degraded: bias toward recency (fresh context more reliable)
                if degraded_flag:
                    rec += 0.05; sal -= 0.025; freq -= 0.025; adjusted = True
                # If high utilization: bias toward salience (select stronger memories)
                if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                    sal += 0.05; rec -= 0.025; freq -= 0.025; adjusted = True
                # Clamp non-negative
                if adjusted:
                    rec = max(0.01, rec)
                    freq = max(0.01, freq)
                    sal = max(0.01, sal)
                    stm_obj.set_activation_weights(recency=rec, frequency=freq, salience=sal)
                    metrics_registry.inc('metacog_activation_weight_adjustments_total')
        except Exception:
            pass
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
        base = await asyncio.to_thread(
            self.process_user_message,
            message,
            session_id,
            flags,
        )
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

    def _invoke_agent_response(self, message: str, built: Any) -> str:
        """Invoke the configured agent to generate a response with context."""
        if not self.agent or not hasattr(self.agent, "_generate_response"):
            return "[LLM unavailable - agent not configured with ChatService]"

        # Build a lightweight memory context payload for the agent
        memory_context = []
        try:
            for item in list(getattr(built, "items", [])[:5]):
                scores = getattr(item, "scores", {}) or {}
                relevance = scores.get("composite") or scores.get("similarity", 0.0)
                metadata = getattr(item, "metadata", {}) or {}
                memory_context.append({
                    "id": getattr(item, "source_id", ""),
                    "content": getattr(item, "content", ""),
                    "source": getattr(item, "source_system", "unknown"),
                    "relevance": relevance,
                    "timestamp": metadata.get("timestamp") or getattr(item, "timestamp", None),
                })
        except Exception:
            memory_context = []

        try:
            result = self.agent._generate_response(
                processed_input={"raw_input": message, "processed_at": datetime.now()},
                memory_context=memory_context,
                attention_scores={},
            )
            if asyncio.iscoroutine(result):
                return str(self._run_coroutine_sync(result))
            return str(result)
        except Exception as exc:
            if "result" in locals() and asyncio.iscoroutine(result):
                try:
                    result.close()
                except Exception:
                    pass
            return f"Error generating response: {exc}"

    def _run_coroutine_sync(self, coro: Any) -> Any:
        """Execute a coroutine to completion from a synchronous context."""
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Running inside an active loop; caller should avoid this path.
            coro.close()
            raise

    def _generate_placeholder_response(self, user_text: str) -> str:
        # Deterministic placeholder for early integration testing.
        snippet = user_text[:60].strip()
        return f"Acknowledged: {snippet}"

    # --- Metacognition ---

    class SnapshotModel(BaseModel):
        ts: float = Field(..., description="Timestamp (epoch seconds)")
        turn_counter: int
        performance: Optional[Dict[str, Any]] = None
        recent_consolidation_selectivity: Optional[float] = None
        promotion_age_p95_seconds: Optional[float] = None
        stm_utilization: Optional[float] = None
        stm_capacity: Optional[int] = None
        last_user_turn_status: Optional[str] = None
        # Allow extra fields (pydantic v2 style if available, else legacy Config)
        if ConfigDict is not None:  # pydantic v2
            model_config = ConfigDict(extra="allow")  # type: ignore
        else:  # pragma: no cover - legacy compatibility
            class Config:  # type: ignore
                extra = "allow"

    def _metacog_snapshot(self) -> Dict[str, Any]:
        """Generate a lightweight metacognitive snapshot.

        Returns a dict with:
        - timestamp
        - turn_counter
        - stm_utilization (0..1) if consolidator present
        - stm_capacity
        - recent_consolidation_selectivity (ltm_promotions / stm_store) if counters present
        - performance (latency_p95_ms, degraded flag) if available
        - last_user_turn_status (status of most recent consolidation attempt)
        Safe – all failures swallowed returning partial data.
        """
        snap: Dict[str, Any] = {
            "ts": time.time(),
            "turn_counter": self._turn_counter,
        }
        try:
            # Performance state
            perf_p95 = metrics_registry.get_p95("chat_turn_latency_ms")
            degraded = metrics_registry.state.get("performance_degraded")
            snap["performance"] = {"latency_p95_ms": perf_p95, "degraded": degraded}
            # Consolidation selectivity
            stm_total = metrics_registry.counters.get("consolidation_stm_store_total")
            ltm_promos = metrics_registry.counters.get("consolidation_ltm_promotions_total")
            if isinstance(stm_total, (int, float)) and stm_total > 0 and isinstance(ltm_promos, (int, float)):
                snap["recent_consolidation_selectivity"] = ltm_promos / max(stm_total, 1)
            # Promotion age p95 if recorded
            if "consolidation_promotion_age_seconds" in metrics_registry.histograms:
                try:
                    age_p95 = metrics_registry.percentile("consolidation_promotion_age_seconds", 95)
                    snap["promotion_age_p95_seconds"] = age_p95
                except Exception:
                    pass
            # STM utilization via consolidator if it exposes method / attrs
            util = None
            capacity = None
            if self.consolidator is not None:
                # Expect consolidator.stm for vector STM with capacity attr or method
                stm_obj = getattr(self.consolidator, "stm", None)
                if stm_obj is not None:
                    try:
                        capacity = getattr(stm_obj, "capacity", None)
                        size = None
                        if hasattr(stm_obj, "__len__"):
                            try:
                                size = len(stm_obj)  # type: ignore
                            except Exception:
                                size = None
                        if size is None:
                            size = getattr(stm_obj, "size", None)
                        if isinstance(size, int) and isinstance(capacity, int) and capacity > 0:
                            util = min(1.0, size / capacity)
                    except Exception:
                        pass
            if util is not None:
                snap["stm_utilization"] = util
            if capacity is not None:
                snap["stm_capacity"] = capacity
            # Last user turn consolidation status
            if self.consolidation_log:
                snap["last_user_turn_status"] = self.consolidation_log[-1]["status"]
        except Exception:
            pass
        try:
            # Validate and coerce via model (drops invalid fields if any)
            model = self.SnapshotModel(**snap)
            # Prefer model_dump in v2; fallback to dict()
            if hasattr(model, "model_dump"):
                return model.model_dump()
            return model.dict()
        except Exception:
            return snap

    def _maybe_consolidate(self, user_turn: TurnRecord, assistant_turn: TurnRecord) -> bool:
        # Prefer new consolidator if available
        if self.consolidator:
            try:
                # Get current thresholds from config (may be overridden per-request)
                cfg = getattr(self, "context_builder").cfg if hasattr(self, "context_builder") else {}
                sal_thr = cfg.get("consolidation_salience_threshold", 0.55)
                val_thr = cfg.get("consolidation_valence_threshold", 0.60)
                
                # Temporarily override policy thresholds to match request-level config
                original_sal = self.consolidator.policy.salience_threshold
                original_val = self.consolidator.policy.valence_threshold
                self.consolidator.policy.salience_threshold = sal_thr
                self.consolidator.policy.valence_threshold = val_thr
                
                ev = self.consolidator.record_turn(
                    turn_id=user_turn.turn_id,
                    salience=user_turn.salience or 0.0,
                    valence=user_turn.emotional_valence or 0.0,
                    importance=user_turn.importance or 0.0,
                    content=user_turn.content,
                )
                
                # Restore original policy thresholds
                self.consolidator.policy.salience_threshold = original_sal
                self.consolidator.policy.valence_threshold = original_val
                
                # Mark rehearsal opportunity (assistant referencing user content)
                self.consolidator.mark_rehearsal(user_turn.turn_id)
                if ev.stored_in_stm:
                    user_turn.consolidation_status = "stored"
                    metrics_registry.inc("consolidated_stored_total")
                    return True
                else:
                    user_turn.consolidation_status = "skipped"
                    metrics_registry.inc("consolidated_skipped_total")
                    return False
            except Exception:
                pass  # fall back to legacy heuristic
        # Legacy heuristic fallback
        cfg = getattr(self, "context_builder").cfg if hasattr(self, "context_builder") else {}
        sal_thr = cfg.get("consolidation_salience_threshold", 0.55)
        val_thr = cfg.get("consolidation_valence_threshold", 0.60)
        sal = user_turn.salience
        val_mag = abs(user_turn.emotional_valence)
        if (sal is not None and sal >= sal_thr) or (val_mag is not None and val_mag >= val_thr):
            user_turn.consolidation_status = "stored"
            metrics_registry.inc("consolidated_stored_total")
            return True
        user_turn.consolidation_status = "skipped"
        metrics_registry.inc("consolidated_skipped_total")
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
        tps = metrics_registry.get_rate("chat_turn", window_seconds=float(self.context_builder.cfg.get("throughput_window_seconds", 60)))
        ema = metrics_registry.state.get("ema_turn_latency_ms", 0.0)
        # Consolidation metrics enrichment (if consolidator active / metrics present)
        cons_counters: Dict[str, Any] = {}
        promotion_age_p95 = 0.0
        # Counters emitted by MemoryConsolidator
        if metrics_registry.counters.get("consolidation_stm_store_total") is not None:
            cons_counters["stm_store_total"] = metrics_registry.counters.get("consolidation_stm_store_total", 0)
        if metrics_registry.counters.get("consolidation_ltm_promotions_total") is not None:
            cons_counters["ltm_promotions_total"] = metrics_registry.counters.get("consolidation_ltm_promotions_total", 0)
        # Promotion age histogram p95 (seconds)
        if "consolidation_promotion_age_seconds" in metrics_registry.histograms:
            promotion_age_p95 = metrics_registry.percentile("consolidation_promotion_age_seconds", 95)
        base = {
            "latency_p95_ms": p95,
            "target_p95_ms": target,
            "performance_degraded": degraded,
            "ema_turn_latency_ms": ema,
            "chat_turns_per_sec": tps,
        }
        # Inject metacog counters summary if present
        try:
            mc = {}
            for k in ("metacog_snapshots_total", "metacog_advisory_items_total", "metacog_stm_high_util_events_total", "metacog_performance_degraded_events_total", "adaptive_retrieval_applied_total"):
                if k in metrics_registry.counters:
                    mc[k] = metrics_registry.counters.get(k, 0)
            if mc:
                base["metacog"] = {"counters": mc, "interval": self._metacog_interval}
        except Exception:
            pass
        if cons_counters:
            stm_total = float(cons_counters.get("stm_store_total", 0))
            ltm_total = float(cons_counters.get("ltm_promotions_total", 0))
            selectivity = (ltm_total / stm_total) if stm_total > 0 else 0.0
            # Recent window stats (last N promotion ages) for volatility insight
            ages_hist = metrics_registry.histograms.get("consolidation_promotion_age_seconds", [])
            recent_window = ages_hist[-5:] if ages_hist else []
            recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0.0
            # Alerting: configurable threshold for promotion age p95
            alert_threshold = self.context_builder.cfg.get("consolidation_promotion_age_p95_alert_seconds")
            age_alert = False
            if isinstance(alert_threshold, (int, float)) and alert_threshold > 0 and promotion_age_p95 > alert_threshold:
                age_alert = True
                metrics_registry.state["consolidation_age_alert"] = True
            base["consolidation"] = {
                "counters": cons_counters,
                "promotion_age_p95_seconds": promotion_age_p95,
                "selectivity_ratio": selectivity,
                "recent_promotion_age_seconds": {
                    "count": len(recent_window),
                    "avg": recent_avg,
                    "values": recent_window,
                },
                "promotion_age_alert": age_alert,
                "promotion_age_alert_threshold": alert_threshold,
            }
        return base

    def _attempt_fact_answer(self, message: str) -> Optional[str]:
        msg = message.strip().lower()
        # Supported patterns:
        #  - who is X
        #  - what is X
        #  - tell me about X
        #  - what does X do
        subj = None
        m = re.match(r"^(who|what) is ([^?]+)\??$", msg)
        if m:
            subj = m.group(2).strip()
        if subj is None:
            m2 = re.match(r"^tell me about ([^?]+)\??$", msg)
            if m2:
                subj = m2.group(1).strip()
        if subj is None:
            m3 = re.match(r"^what does ([^?]+) do\??$", msg)
            if m3:
                subj = m3.group(1).strip()
        if subj is None:
            return None
        # Search capture cache (reverse for recency bias)
        subj_l = subj.lower()
        for rec in reversed(self._capture_cache.as_list()):
            rsubj = rec.get("subject") or ""
            if rsubj and (subj_l == rsubj or subj_l in rsubj):
                mtype = rec.get("memory_type")
                obj = rec.get("object")
                if mtype == "identity_fact" and obj:
                    return f"{subj.title()} is {obj}."
                if mtype == "preference" and obj:
                    return f"{subj.title()} likes {obj}."
                if mtype == "goal_intent" and obj:
                    return f"{subj.title()} intends to {obj}."
        return None
