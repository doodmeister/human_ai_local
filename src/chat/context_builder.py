from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional, Callable, Any, Dict
import threading

from .models import (
    ContextItem,
    ProvenanceTrace,
    PipelineStageResult,
    ChatMetrics,
    BuiltContext,
)
from .conversation_session import ConversationSession
from .scoring import score_and_rank
from .constants import (
    RANK_BASE_ATTENTION,
    RANK_BASE_FALLBACK_RECENT,
    RANK_BASE_FALLBACK_TOKEN,
    RANK_BASE_EXECUTIVE,
)
from .metrics import metrics_registry


class ContextBuilder:
    """Staged context construction pipeline (extended scaffold with basic fallback)."""

    def __init__(
        self,
        chat_config: Optional[Dict[str, Any]] = None,
        stm: Any = None,
        ltm: Any = None,
        episodic: Any = None,
        attention: Any = None,
        executive: Any = None,
        prospective: Any = None,
    ):
        # Start with global defaults, then overlay any provided config
        from src.core.config import get_chat_config
        self.cfg = get_chat_config().to_dict()
        if chat_config:
            self.cfg.update(chat_config)
        # Backward compatibility: ensure timeouts structure
        if "timeouts" not in self.cfg:
            self.cfg["timeouts"] = {"retrieval_ms": self.cfg.get("retrieval_timeout_ms", 400)}
        self.stm = stm
        self.ltm = ltm
        self.episodic = episodic
        self.attention = attention
        self.executive = executive
        self.prospective = prospective
        self._last_stage_stats: Dict[str, Any] = {}

    # Public API
    def build(
        self,
        session: ConversationSession,
        query: Optional[str] = None,
        include_attention: bool = True,
        include_memory: bool = True,
        include_trace: bool = True,
    ) -> BuiltContext:
        self._current_session = session  # track for fallback enrichment
        start = time.time()
        trace = ProvenanceTrace()
        metrics = ChatMetrics()
        items: List[ContextItem] = []
        used_turn_ids: List[str] = []
        fallback_used = False

        # 1. Recent turns
        self._run_stage(
            name="recent_turns",
            func=lambda: self._select_recent_turns(session),
            sink=items,
            trace=trace,
            used_turns=used_turn_ids,
        )

        # 2. Memory stages
        if include_memory:
            fallback_used |= self._stage_with_fallback(
                name="stm_semantic",
                retrieval=lambda: self._retrieve_stm(query),
                sink=items,
                trace=trace,
            )
            fallback_used |= self._stage_with_fallback(
                name="ltm_semantic",
                retrieval=lambda: self._retrieve_ltm(query),
                sink=items,
                trace=trace,
            )
            fallback_used |= self._stage_with_fallback(
                name="episodic_anchors",
                retrieval=lambda: self._retrieve_episodic(query),
                sink=items,
                trace=trace,
            )

        # 3. Attention & executive
        if include_attention:
            self._run_stage(
                name="attention_focus",
                func=self._inject_attention_focus,
                sink=items,
                trace=trace,
            )
        self._run_stage(
            name="executive_mode",
            func=self._inject_executive_mode,
            sink=items,
            trace=trace,
        )
        self._run_stage(
            name="prospective_reminders",
            func=self._inject_upcoming_reminders,
            sink=items,
            trace=trace,
        )

        # Composite scoring & deterministic ranking
        items = score_and_rank(items)

        # Truncate after ranking
        items = items[: self.cfg["max_context_items"]]

        # Metacognitive context injection (lightweight):
        # Add advisory items if performance degraded or STM utilization high.
        try:
            meta_items: List[ContextItem] = []
            degraded = metrics_registry.state.get("performance_degraded")
            stm_util = None
            stm_capacity = None
            if self.stm is not None:
                cap = getattr(self.stm, "capacity", None)
                size = None
                if hasattr(self.stm, "__len__"):
                    try:
                        size = len(self.stm)  # type: ignore
                    except Exception:
                        size = None
                if size is None:
                    size = getattr(self.stm, "size", None)
                if isinstance(size, int) and isinstance(cap, int) and cap > 0:
                    stm_capacity = cap
                    stm_util = min(1.0, size / cap)
            # High utilization threshold (fixed 0.85 for now)
            if stm_util is not None and stm_util >= 0.85:
                meta_items.append(
                    ContextItem(
                        source_id="metacog-stm-pressure",
                        source_system="metacog",
                        reason="stm_high_utilization",
                        content=f"STM utilization high ({stm_util:.2%}) capacity={stm_capacity}",
                        rank=0,
                        scores={"composite": 0.0},
                    )
                )
                try:
                    metrics_registry.inc("metacog_stm_high_util_events_total")
                except Exception:
                    pass
            if degraded:
                meta_items.append(
                    ContextItem(
                        source_id="metacog-performance",
                        source_system="metacog",
                        reason="performance_degraded",
                        content="System performance degraded (p95 latency above target)",
                        rank=0,
                        scores={"composite": 0.0},
                    )
                )
                try:
                    metrics_registry.inc("metacog_performance_degraded_events_total")
                except Exception:
                    pass
            if meta_items:
                # Append without disturbing prior ranking (metacog rank=0 will be re-scored earlier if needed)
                items.extend(meta_items)
                try:
                    metrics_registry.inc("metacog_advisory_items_total", value=len(meta_items))
                except Exception:
                    pass
        except Exception:
            pass

        # Metrics
        metrics.stm_hits = sum(1 for c in items if c.source_system == "stm")
        metrics.ltm_hits = sum(1 for c in items if c.source_system == "ltm")
        metrics.episodic_hits = sum(1 for c in items if c.source_system == "episodic")
        metrics.fallback_used = fallback_used
        metrics.turn_latency_ms = (time.time() - start) * 1000.0
        metrics.retrieval_time_ms = sum(
            s.latency_ms for s in trace.pipeline_stages if "semantic" in s.name or "anchors" in s.name
        )

        if fallback_used:
            trace.degraded_mode = True
            trace.notes.append("Degraded mode: fallback retrieval used")

        return BuiltContext(
            session_id=session.session_id,
            items=items,
            trace=trace if include_trace else ProvenanceTrace(),
            metrics=metrics,
            used_turn_ids=used_turn_ids,
        )

    # --- New helper wrapping stage with fallback awareness ---
    def _stage_with_fallback(
        self,
        name: str,
        retrieval: Callable[[], List[ContextItem]],
        sink: List[ContextItem],
        trace: ProvenanceTrace,
    ) -> bool:
        before = len(sink)
        t0 = time.time()
        fallback_flag = False
        self._last_stage_stats = {}
        try:
            produced = retrieval() or []
            # Detect fallback by reason tag
            if any("fallback" in ci.reason for ci in produced):
                fallback_flag = True
        except Exception as e:  # pragma: no cover
            produced = []
            fallback_flag = True
            self._last_stage_stats = {"error": str(e), "kept": 0, "filtered": 0}
            trace.notes.append(f"stage {name} exception -> degraded: {e}")
            metrics_registry.inc(f"{name}_errors_total")
        for ci in produced:
            if ci.rank == 0:
                ci.rank = len(sink) + 1
        sink.extend(produced)
        base_rationale = "fallback" if fallback_flag else "semantic_retrieval"
        stats_part = ""
        if self._last_stage_stats:
            kept = self._last_stage_stats.get("kept")
            filtered = self._last_stage_stats.get("filtered")
            threshold = self._last_stage_stats.get("threshold")
            stats_part = f"; kept={kept} filtered={filtered} threshold={threshold}"
        stage = PipelineStageResult(
            name=name,
            candidates_in=before,
            candidates_out=len(sink),
            latency_ms=(time.time() - t0) * 1000.0,
            rationale=base_rationale + stats_part,
            added=len(sink) - before,
        )
        trace.add_stage(stage)
        return fallback_flag

    # --- Retrieval helpers with threshold filtering ---

    def _retrieve_stm(self, query: Optional[str]) -> List[ContextItem]:
        if not self.stm or not query:
            return []
        try:
            raw = self._timed_search(self.stm, query, "stm")
            items = self._wrap_memory_results(raw, "stm", "activation")
            threshold = self.cfg.get("stm_activation_min", 0.15)
            kept, filtered = [], 0
            for ci in items:
                act = ci.scores.get("activation", 0.0)
                if act >= threshold:
                    kept.append(ci)
                else:
                    filtered += 1
            self._last_stage_stats = {"kept": len(kept), "filtered": filtered, "threshold": threshold}
            return kept
        except Exception as e:  # pragma: no cover
            metrics_registry.inc("stm_semantic_errors_total")
            return self._fallback_word_overlap(query, source="stm_fallback", note=str(e))

    def _retrieve_ltm(self, query: Optional[str]) -> List[ContextItem]:
        if not self.ltm or not query:
            return []
        try:
            raw = self._timed_search(self.ltm, query, "ltm")
            items = self._wrap_memory_results(raw, "ltm", "strength")
            threshold = self.cfg.get("ltm_similarity_threshold", 0.62)
            kept, filtered = [], 0
            for ci in items:
                sim = ci.scores.get("similarity", 0.0)
                if sim >= threshold:
                    kept.append(ci)
                else:
                    filtered += 1
            self._last_stage_stats = {"kept": len(kept), "filtered": filtered, "threshold": threshold}
            return kept
        except Exception as e:  # pragma: no cover
            metrics_registry.inc("ltm_semantic_errors_total")
            return self._fallback_word_overlap(query, source="ltm_fallback", note=str(e))

    def _retrieve_episodic(self, query: Optional[str]) -> List[ContextItem]:
        if not self.episodic or not query:
            return []
        try:
            raw = self._timed_search(self.episodic, query, "episodic")
            items = self._wrap_memory_results(raw, "episodic", "importance")
            # no threshold filtering yet; record stats
            self._last_stage_stats = {"kept": len(items), "filtered": 0, "threshold": None}
            return items
        except Exception as e:  # pragma: no cover
            metrics_registry.inc("episodic_anchors_errors_total")
            return self._fallback_word_overlap(query, source="episodic_fallback", note=str(e))

    def _timed_search(self, memory_obj: Any, query: str, label: str):
        result_container = {}
        exc_container = {}
        def _run():
            try:
                # Expect memory_obj.search(query=...) returns list[dict]
                result = memory_obj.search(query=query)  # type: ignore
                result_container["res"] = self._normalize_memory_results(result)
            except Exception as e:  # pragma: no cover
                exc_container["exc"] = e
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(self.cfg["timeouts"]["retrieval_ms"] / 1000.0)
        if th.is_alive():
            raise TimeoutError(f"{label} search timeout")
        if "exc" in exc_container:
            raise exc_container["exc"]
        return result_container.get("res", []) or []

    # --- Normalization helper to adapt various memory search return formats ---
    def _normalize_memory_results(self, raw: Any) -> List[Dict[str, Any]]:
        """Normalize diverse memory search outputs into list[dict].

        Supported input shapes:
          - list[dict] (already in target shape)
          - list[Tuple[MemoryItem, float]]
          - list[VectorMemoryResult]
          - None / other -> []
        Dict schema produced (subset may be used downstream):
          {id, content, activation, similarity, recency, salience}
        """
        if not raw:
            return []
        now = datetime.now()
        if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], dict)):
            # Map generic dict results (LTM/Episodic) to unified schema
            out: List[Dict[str, Any]] = []
            for i, d in enumerate(raw):  # type: ignore
                if not isinstance(d, dict):
                    continue
                content = d.get('content') or d.get('text') or d.get('summary') or ''
                if not content:
                    continue
                # Similarity mapping
                similarity = d.get('similarity')
                if similarity is None:
                    similarity = d.get('similarity_score')
                try:
                    similarity = float(similarity) if similarity is not None else 0.0
                except Exception:
                    similarity = 0.0
                # Activation candidates
                act = d.get('activation')
                if act is None:
                    imp = d.get('importance')
                    conf = d.get('confidence')
                    try:
                        if imp is not None or conf is not None:
                            imp_v = float(imp) if imp is not None else 0.5
                            conf_v = float(conf) if conf is not None else 0.5
                            act = max(0.0, min(1.0, imp_v * 0.6 + conf_v * 0.4))
                        else:
                            act = 0.0
                    except Exception:
                        act = 0.0
                # Recency estimation
                recency = 0.0
                ts_raw = d.get('last_access') or d.get('encoding_time') or d.get('timestamp')
                if isinstance(ts_raw, str):
                    try:
                        dt = datetime.fromisoformat(ts_raw)
                        age_hours = (now - dt).total_seconds() / 3600.0
                        recency = max(0.0, 1.0 - age_hours / 24.0)  # simple 24h decay
                    except Exception:
                        recency = 0.0
                salience = 0.0
                for k_sal in ('salience','vividness','emotional_valence'):
                    if k_sal in d:
                        try:
                            sv = abs(float(d[k_sal]))
                            salience = max(salience, min(1.0, sv))
                        except Exception:
                            pass
                out.append({
                    'id': d.get('id', f'dict-{i}'),
                    'content': content,
                    'activation': float(act) if act is not None else 0.0,
                    'similarity': similarity,
                    'recency': recency,
                    'salience': salience,
                })
            return out

        normalized: List[Dict[str, Any]] = []

        def compute_activation(item: Any) -> Dict[str, float]:
            try:
                enc = getattr(item, 'encoding_time', None)
                if not isinstance(enc, datetime):
                    enc = now
                age_hours = (now - enc).total_seconds() / 3600.0
                decay_rate = float(getattr(item, 'decay_rate', 0.1) or 0.1)
                recency = max(0.0, 1.0 - age_hours * decay_rate)
                access_count = int(getattr(item, 'access_count', 0) or 0)
                frequency = min(1.0, access_count / 10.0)
                importance = float(getattr(item, 'importance', 0.5) or 0.5)
                attention_score = float(getattr(item, 'attention_score', 0.0) or 0.0)
                salience = (importance + attention_score) / 2.0
                activation = (recency * 0.4) + (frequency * 0.3) + (salience * 0.3)
                return {'activation': max(0.0, min(1.0, activation)), 'recency': recency, 'salience': salience}
            except Exception:
                return {'activation': 0.0, 'recency': 0.0, 'salience': 0.0}

        # VectorMemoryResult pattern
        if isinstance(raw, list) and raw and hasattr(raw[0], 'item') and hasattr(raw[0], 'similarity_score'):
            for r in raw:  # type: ignore
                try:
                    item = getattr(r, 'item', None)
                    if not item:
                        continue
                    meta = compute_activation(item)
                    normalized.append({
                        'id': getattr(item, 'id', ''),
                        'content': getattr(item, 'content', ''),
                        'activation': meta['activation'],
                        'similarity': float(getattr(r, 'similarity_score', 0.0) or 0.0),
                        'recency': meta['recency'],
                        'salience': meta['salience'],
                    })
                except Exception:
                    continue
            return normalized

        # Tuple (MemoryItem, score)
        if isinstance(raw, list) and raw and isinstance(raw[0], tuple):
            for tup in raw:
                try:
                    if len(tup) < 1:
                        continue
                    item = tup[0]
                    similarity = float(tup[1]) if len(tup) > 1 else 0.0
                    meta = compute_activation(item)
                    normalized.append({
                        'id': getattr(item, 'id', ''),
                        'content': getattr(item, 'content', ''),
                        'activation': meta['activation'],
                        'similarity': similarity,
                        'recency': meta['recency'],
                        'salience': meta['salience'],
                    })
                except Exception:
                    continue
            return normalized

        return []

    def _wrap_memory_results(
        self,
        raw: List[Dict[str, Any]],
        source_system: str,
        key_activation: str,
    ) -> List[ContextItem]:
        items: List[ContextItem] = []
        for idx, r in enumerate(raw):
            content = r.get("content") or r.get("text") or r.get("summary") or ""
            if not content:
                continue
            activation = float(r.get(key_activation, 0.0))
            similarity = float(r.get("similarity", 0.0))
            scores = {
                "activation": activation,
                "similarity": similarity,
                "recency": r.get("recency", 0.0),
                "salience": r.get("salience", 0.0),
            }
            rid = str(r.get("id", idx))
            if source_system == "ltm" and (rid.startswith("ltm-turn-") or r.get("original_stm_id")):
                scores["promoted_from_stm"] = 1.0
            items.append(
                ContextItem(
                    source_id=rid,
                    source_system=source_system,
                    content=content,
                    rank=100 + idx,  # after recent turns
                    reason="semantic_match",
                    scores=scores,
                    metadata={"raw": r},
                )
            )
        return items

    # --- Fallback word overlap (enhanced multi-tier) ---

    def _fallback_word_overlap(self, query: str, source: str, note: str = "") -> List[ContextItem]:
        words = {w for w in query.lower().split() if w}
        if not words:
            return []
        items: List[ContextItem] = []
        # Tier 1: recent turns overlap
        recent_turns = []
        try:
            recent_turns = list(self._current_session.recent_turns(5))  # type: ignore[attr-defined]
        except Exception:
            recent_turns = []
        tier_recent_count = 0
        for rt in recent_turns:
            rt_words = set(rt.content.lower().split())
            if not rt_words:
                continue
            overlap = len(words & rt_words) / max(1, len(words | rt_words))
            if overlap <= 0:
                continue
            items.append(
                ContextItem(
                    source_id=f"{source}:recent:{rt.turn_id}",
                    source_system=source,
                    content=f"(fallback recent) {rt.content}",
                    rank=RANK_BASE_FALLBACK_RECENT + len(items),
                    reason="fallback_recent_overlap",
                    scores={"overlap": overlap},
                    metadata={"note": note, "turn_id": rt.turn_id},
                )
            )
            tier_recent_count += 1
        # Tier 2 token
        tier_token_count = 0
        for w in sorted(words, key=len, reverse=True):
            overlap = 1.0 / max(1, len(words))
            if overlap < self.cfg["fallback_min_overlap"]:
                continue
            items.append(
                ContextItem(
                    source_id=f"{source}:token:{w}",
                    source_system=source,
                    content=f"(fallback token) {w}",
                    rank=RANK_BASE_FALLBACK_TOKEN + len(items),
                    reason="fallback_word_overlap",
                    scores={"overlap": overlap},
                    metadata={"note": note},
                )
            )
            tier_token_count += 1
        if tier_recent_count:
            metrics_registry.inc("fallback_recent_items_total", tier_recent_count)
        if tier_token_count:
            metrics_registry.inc("fallback_token_items_total", tier_token_count)
        metrics_registry.inc("fallback_tiers_used_total", int(tier_recent_count > 0) + int(tier_token_count > 0))
        return items

    # --- Attention & Executive Injectors (placeholders) ---

    def _inject_attention_focus(self) -> List[ContextItem]:
        if not self.attention:
            return []
        try:
            focus_items = getattr(self.attention, "current_focus", [])  # list[str] or list[dict]
        except Exception:  # pragma: no cover
            return []
        out: List[ContextItem] = []
        for idx, fi in enumerate(focus_items):
            # derive novelty/intensity if present
            focus_score = 1.0
            if isinstance(fi, dict):
                focus_score = float(fi.get("attention_score", 1.0))
                content = fi.get("content", str(fi))
                source_id = str(fi.get("id", idx))
            else:
                content = str(fi)
                source_id = str(idx)
            out.append(
                ContextItem(
                    source_id=source_id,
                    source_system="attention",
                    content=content,
                    rank=RANK_BASE_ATTENTION + idx,
                    reason="attention_focus",
                    forced=True,
                    scores={"focus": focus_score},
                )
            )
        return out

    def _inject_executive_mode(self) -> List[ContextItem]:
        """Inject executive state including active goals into context."""
        items: List[ContextItem] = []
        
        if not self.executive:
            return items
            
        # Basic mode info
        try:
            mode = getattr(self.executive, "mode", "UNKNOWN")
        except Exception:  # pragma: no cover
            mode = "UNKNOWN"
        try:
            perf = getattr(self.executive, "performance_state", None)
        except Exception:
            perf = None
        
        items.append(
            ContextItem(
                source_id="executive_mode",
                source_system="executive",
                content=f"ExecutiveMode: {mode}",
                rank=RANK_BASE_EXECUTIVE,
                reason="executive_state",
                scores={"state": 1.0, "mode_confidence": getattr(perf, "confidence", 0.0) if perf else 0.0},
            )
        )
        
        # Inject active goals if goal_manager available
        try:
            goal_manager = getattr(self.executive, "goal_manager", None)
            if goal_manager is not None:
                active_goals = []
                # Try to get active goals
                if hasattr(goal_manager, "get_active_goals"):
                    active_goals = goal_manager.get_active_goals()
                elif hasattr(goal_manager, "goals"):
                    all_goals = goal_manager.goals
                    active_goals = [g for g in all_goals.values() if g.status in ("pending", "in_progress", "active")]
                
                if active_goals:
                    # Create a concise summary of active goals for LLM context
                    goal_summaries = []
                    for goal in active_goals[:5]:  # Limit to top 5 goals
                        title = getattr(goal, "title", getattr(goal, "description", str(goal)))
                        priority = getattr(goal, "priority", 0.5)
                        status = getattr(goal, "status", "pending")
                        deadline = getattr(goal, "deadline", None)
                        
                        summary = f"- {title} (priority: {priority:.0%}, status: {status})"
                        if deadline:
                            summary += f" [deadline: {deadline}]"
                        goal_summaries.append(summary)
                    
                    goals_content = "USER'S ACTIVE GOALS:\n" + "\n".join(goal_summaries)
                    if len(active_goals) > 5:
                        goals_content += f"\n... and {len(active_goals) - 5} more goals"
                    
                    items.append(
                        ContextItem(
                            source_id="active_goals",
                            source_system="executive",
                            content=goals_content,
                            rank=RANK_BASE_EXECUTIVE - 1,  # Slightly higher priority than mode
                            reason="active_goals_context",
                            scores={"relevance": 0.9, "goal_count": len(active_goals)},
                            metadata={"goal_count": len(active_goals)},
                        )
                    )
        except Exception:
            # Don't fail if goal injection doesn't work
            pass
        
        return items

    def _inject_upcoming_reminders(self) -> List[ContextItem]:
        """
        Inject upcoming reminders (especially plan steps) into LLM context.
        
        This enables the LLM to proactively suggest next actions based on
        the user's scheduled tasks and auto-generated plan reminders.
        """
        from datetime import timedelta
        
        items: List[ContextItem] = []
        
        if not self.prospective:
            return items
        
        try:
            # Get reminders due within next 2 hours
            upcoming = self.prospective.get_upcoming(within=timedelta(hours=2))
            
            if not upcoming:
                return items
            
            # Separate plan steps from regular reminders
            plan_reminders = []
            regular_reminders = []
            
            for r in upcoming:
                if r.metadata.get("auto_generated") or "plan-step" in r.tags:
                    plan_reminders.append(r)
                else:
                    regular_reminders.append(r)
            
            # Build plan steps content (higher priority)
            if plan_reminders:
                plan_lines = []
                for r in plan_reminders[:5]:  # Limit to 5
                    due_in = ""
                    if r.due_time:
                        delta = r.due_time - datetime.now()
                        mins = int(delta.total_seconds() / 60)
                        if mins < 60:
                            due_in = f" (due in {mins}m)"
                        else:
                            due_in = f" (due in {mins // 60}h {mins % 60}m)"
                    plan_lines.append(f"• {r.content}{due_in}")
                
                plan_content = "UPCOMING PLAN STEPS:\n" + "\n".join(plan_lines)
                if len(plan_reminders) > 5:
                    plan_content += f"\n... and {len(plan_reminders) - 5} more steps"
                
                items.append(
                    ContextItem(
                        source_id="plan_steps",
                        source_system="prospective",
                        content=plan_content,
                        rank=RANK_BASE_EXECUTIVE - 2,  # High priority, before goals
                        reason="upcoming_plan_steps",
                        scores={"urgency": 0.95, "count": len(plan_reminders)},
                        metadata={"reminder_type": "plan_step", "count": len(plan_reminders)},
                    )
                )
            
            # Build regular reminders content
            if regular_reminders:
                reminder_lines = []
                for r in regular_reminders[:3]:  # Limit to 3
                    due_in = ""
                    if r.due_time:
                        delta = r.due_time - datetime.now()
                        mins = int(delta.total_seconds() / 60)
                        if mins < 60:
                            due_in = f" (in {mins}m)"
                        else:
                            due_in = f" (in {mins // 60}h)"
                    reminder_lines.append(f"• {r.content}{due_in}")
                
                reminders_content = "UPCOMING REMINDERS:\n" + "\n".join(reminder_lines)
                
                items.append(
                    ContextItem(
                        source_id="reminders",
                        source_system="prospective",
                        content=reminders_content,
                        rank=RANK_BASE_EXECUTIVE,
                        reason="upcoming_reminders",
                        scores={"urgency": 0.8, "count": len(regular_reminders)},
                        metadata={"reminder_type": "regular", "count": len(regular_reminders)},
                    )
                )
            
        except Exception:
            # Don't fail if reminder injection doesn't work
            pass
        
        return items

    # --- Existing stage runner & recent turns selection (unchanged logic with minor adjustments) ---

    def _run_stage(
        self,
        name: str,
        func: Callable[[], List[ContextItem]],
        sink: List[ContextItem],
        trace: ProvenanceTrace,
        used_turns: Optional[List[str]] = None,
    ) -> None:
        t0 = time.time()
        before = len(sink)
        try:
            produced = func() or []
        except Exception as e:  # pragma: no cover
            produced = []
            trace.notes.append(f"stage {name} error: {e}")
        for ci in produced:
            if ci.rank == 0:
                ci.rank = len(sink) + 1
        sink.extend(produced)
        latency_ms = (time.time() - t0) * 1000.0
        stage = PipelineStageResult(
            name=name,
            candidates_in=before,
            candidates_out=len(sink),
            latency_ms=latency_ms,
            rationale="scaffold",
            added=len(sink) - before,
        )
        trace.add_stage(stage)
        if used_turns and name == "recent_turns":
            for ci in produced:
                tid = ci.metadata.get("turn_id")
                if tid:
                    used_turns.append(tid)

    def _select_recent_turns(self, session: ConversationSession) -> List[ContextItem]:
        turns = session.recent_turns(self.cfg["max_recent_turns"])
        items: List[ContextItem] = []
        for t in turns:
            # Skip the most recent assistant reply (we only want prior context + user turns)
            if t.role not in ("user", "system"):
                continue
            scores = {
                "recency": 1.0 - (time.time() - t.timestamp) / 3600.0,  # crude hour decay
            }
            # Inject emotional salience factors if present on the TurnRecord
            sal = getattr(t, "salience", None)
            if isinstance(sal, (int, float)):
                scores["salience"] = max(0.0, min(1.0, float(sal)))
            val = getattr(t, "valence", None)
            if isinstance(val, (int, float)):
                scores["importance"] = max(scores.get("importance", 0.0), min(1.0, abs(val)))  # reuse importance weight
            items.append(
                ContextItem(
                    source_id=t.turn_id,
                    source_system="recent",
                    content=t.content,
                    rank=0,
                    reason="recent_turn",
                    scores=scores,
                )
            )
        return items
