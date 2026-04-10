from __future__ import annotations

import re
import time
from datetime import datetime, timezone
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
from src.memory.autobiographical import AutobiographicalGraph, AutobiographicalGraphBuilder
from src.memory.schema import CanonicalMemoryItem, canonical_item_to_context_payload, normalize_memory_results
from src.memory.retrieval import MemoryReranker, RetrievalPlan, RetrievalPlanner


class ContextBuilder:
    """Staged context construction pipeline (extended scaffold with basic fallback)."""

    def __init__(
        self,
        chat_config: Optional[Dict[str, Any]] = None,
        stm: Any = None,
        ltm: Any = None,
        episodic: Any = None,
        semantic: Any = None,
        attention: Any = None,
        executive: Any = None,
        prospective: Any = None,
        autobiographical_store: Any = None,
    ):
        # Start with global defaults, then overlay any provided config
        from src.core.config import get_chat_config
        self.cfg = get_chat_config().to_dict()
        if chat_config:
            self.cfg.update(chat_config)
        # Backward compatibility: ensure timeouts structure
        if not self.cfg.get("timeouts"):
            self.cfg["timeouts"] = {"retrieval_ms": self.cfg.get("retrieval_timeout_ms", 400)}
        self.stm = stm
        self.ltm = ltm
        self.episodic = episodic
        self.semantic = semantic
        self.attention = attention
        self.executive = executive
        self.prospective = prospective
        self._autobiographical_store = autobiographical_store
        self._retrieval_planner = RetrievalPlanner()
        self._memory_reranker = MemoryReranker()
        self._autobiographical_graph_builder = AutobiographicalGraphBuilder()
        self._last_stage_stats: Dict[str, Any] = {}

    # Public API
    def build(
        self,
        session: ConversationSession,
        query: Optional[str] = None,
        include_attention: bool = True,
        include_memory: bool = True,
        include_trace: bool = True,
        read_only_retrieval: bool = False,
    ) -> BuiltContext:
        self._current_session = session  # track for fallback enrichment
        self._read_only_retrieval = read_only_retrieval
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
        retrieval_plan: Optional[RetrievalPlan] = None
        relationship_memory = self._current_relationship_memory()
        if include_memory:
            retrieval_plan = self._retrieval_planner.plan(query, relationship_memory=relationship_memory)
            trace.add_stage(
                PipelineStageResult(
                    name="retrieval_plan",
                    candidates_in=len(items),
                    candidates_out=len(items),
                    latency_ms=0.0,
                    rationale=retrieval_plan.intent,
                    metadata=retrieval_plan.to_dict(),
                )
            )
            if retrieval_plan.search_stm:
                fallback_used |= self._stage_with_fallback(
                    name="stm_semantic",
                    retrieval=lambda: self._retrieve_stm(query, retrieval_plan),
                    sink=items,
                    trace=trace,
                )
            if retrieval_plan.search_ltm:
                fallback_used |= self._stage_with_fallback(
                    name="ltm_semantic",
                    retrieval=lambda: self._retrieve_ltm(query, retrieval_plan),
                    sink=items,
                    trace=trace,
                )
            if retrieval_plan.search_episodic:
                fallback_used |= self._stage_with_fallback(
                    name="episodic_anchors",
                    retrieval=lambda: self._retrieve_episodic(query, retrieval_plan),
                    sink=items,
                    trace=trace,
                )
            self._rerank_memory_context_items(
                items,
                retrieval_plan=retrieval_plan,
                relationship_memory=relationship_memory,
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
        self._run_stage(
            name="autobiographical_chapter",
            func=lambda: self._inject_autobiographical_chapter_context(retrieval_plan),
            sink=items,
            trace=trace,
        )

        # Phase 2, Layer 0: Drive context injection
        self._run_stage(
            name="drive_context",
            func=self._inject_drive_context,
            sink=items,
            trace=trace,
        )

        # Phase 2, Layer 1: Felt-sense / mood context injection
        self._run_stage(
            name="felt_sense_context",
            func=self._inject_felt_sense_context,
            sink=items,
            trace=trace,
        )

        # Phase 2, Layer 2: Relational context injection
        self._run_stage(
            name="relational_context",
            func=self._inject_relational_context,
            sink=items,
            trace=trace,
        )

        # Phase 2, Layer 3: Emergent patterns context injection
        self._run_stage(
            name="pattern_context",
            func=self._inject_pattern_context,
            sink=items,
            trace=trace,
        )

        # Phase 2, Layer 4: Self-model context injection
        self._run_stage(
            name="selfmodel_context",
            func=self._inject_selfmodel_context,
            sink=items,
            trace=trace,
        )

        # Phase 2, Layer 5: Narrative context injection
        self._run_stage(
            name="narrative_context",
            func=self._inject_narrative_context,
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
            metadata=dict(self._last_stage_stats),
        )
        trace.add_stage(stage)
        return fallback_flag

    # --- Retrieval helpers with threshold filtering ---

    def _retrieve_stm(self, query: Optional[str], retrieval_plan: Optional[RetrievalPlan] = None) -> List[ContextItem]:
        if not self.stm or not query:
            return []
        try:
            raw = self._timed_search(
                self.stm,
                query,
                "stm",
                max_results=retrieval_plan.limit_for("stm", 5) if retrieval_plan else None,
            )
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

    def _retrieve_ltm(self, query: Optional[str], retrieval_plan: Optional[RetrievalPlan] = None) -> List[ContextItem]:
        if not self.ltm or not query:
            return []
        try:
            raw = self._timed_search(
                self.ltm,
                query,
                "ltm",
                max_results=retrieval_plan.limit_for("ltm", 5) if retrieval_plan else None,
            )
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

    def _retrieve_episodic(self, query: Optional[str], retrieval_plan: Optional[RetrievalPlan] = None) -> List[ContextItem]:
        if not self.episodic or not query:
            return []
        try:
            raw = self._timed_search(
                self.episodic,
                query,
                "episodic",
                max_results=retrieval_plan.limit_for("episodic", 5) if retrieval_plan else None,
            )
            items = self._wrap_memory_results(raw, "episodic", "importance")
            items, chapter_backfill, chapter_id = self._prefer_persisted_chapter_episodic_items(
                items,
                retrieval_plan=retrieval_plan,
            )
            self._last_stage_stats = {
                "kept": len(items),
                "filtered": 0,
                "threshold": None,
                "chapter_backfill": chapter_backfill,
                "chapter_id": chapter_id,
            }
            return items
        except Exception as e:  # pragma: no cover
            metrics_registry.inc("episodic_anchors_errors_total")
            return self._fallback_word_overlap(query, source="episodic_fallback", note=str(e))

    def _timed_search(self, memory_obj: Any, query: str, label: str, max_results: Optional[int] = None):
        result_container = {}
        exc_container = {}
        read_only_retrieval = bool(getattr(self, "_read_only_retrieval", False))
        def _run():
            try:
                # Search signatures vary across memory stores; try richer kwargs first.
                result = None
                if max_results is not None:
                    attempts = []
                    if read_only_retrieval:
                        attempts.extend(
                            [
                                {"query": query, "max_results": max_results, "limit": max_results, "update_access": False},
                                {"query": query, "limit": max_results, "update_access": False},
                                {"query": query, "max_results": max_results, "update_access": False},
                            ]
                        )
                    attempts.extend(
                        [
                            {"query": query, "max_results": max_results, "limit": max_results},
                            {"query": query, "limit": max_results},
                            {"query": query, "max_results": max_results},
                        ]
                    )
                    for kwargs in attempts:
                        try:
                            result = memory_obj.search(**kwargs)  # type: ignore[arg-type]
                            break
                        except TypeError:
                            continue
                if result is None:
                    if read_only_retrieval:
                        try:
                            result = memory_obj.search(query=query, update_access=False)  # type: ignore[arg-type]
                        except TypeError:
                            result = None
                    if result is None:
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
        """Normalize diverse memory search outputs through canonical memory adapters."""
        canonical_items = normalize_memory_results(raw)
        canonical_items = self._memory_reranker.rerank(canonical_items)
        return [canonical_item_to_context_payload(item) for item in canonical_items if item.content]

    def _current_relationship_memory(self) -> Any | None:
        session = getattr(self, "_current_session", None)
        if session is None:
            return None
        return getattr(session, "_relationship_memory_snapshot", None)

    def _current_autobiographical_graph(self) -> AutobiographicalGraph | None:
        session = getattr(self, "_current_session", None)
        if session is None:
            return None

        snapshot = getattr(session, "autobiographical_graph_snapshot", None)
        if snapshot is None:
            snapshot = getattr(session, "_autobiographical_graph_snapshot", None)
        if isinstance(snapshot, AutobiographicalGraph):
            return snapshot
        if isinstance(snapshot, dict):
            try:
                graph = AutobiographicalGraph.from_dict(snapshot)
            except Exception:
                graph = None
            else:
                session.autobiographical_graph_snapshot = graph
                return graph

        store = self._autobiographical_store
        if store is None or not hasattr(store, "get"):
            return None
        try:
            graph = store.get(session.session_id)
        except Exception:
            return None
        if graph is not None:
            session.autobiographical_graph_snapshot = graph
        return graph

    def _selected_persisted_chapter(self, retrieval_plan: RetrievalPlan | None) -> Any | None:
        if retrieval_plan is None or retrieval_plan.intent not in {"continuity", "episodic", "social"}:
            return None

        graph = self._current_autobiographical_graph()
        if graph is None or not graph.chapters:
            return None
        return self._select_autobiographical_chapter(graph, retrieval_plan.query)

    def _prefer_persisted_chapter_episodic_items(
        self,
        items: List[ContextItem],
        *,
        retrieval_plan: RetrievalPlan | None,
    ) -> tuple[List[ContextItem], int, str | None]:
        chapter = self._selected_persisted_chapter(retrieval_plan)
        if chapter is None or not hasattr(self.episodic, "retrieve_memory"):
            return items, 0, None

        defining_moment_ids = set(getattr(chapter, "defining_moment_ids", []) or [])
        ordered_episode_ids: list[str] = []
        for episode_id in [
            *list(getattr(chapter, "defining_moment_ids", []) or []),
            *list(getattr(chapter, "event_ids", []) or []),
        ]:
            episode_key = str(episode_id or "").strip()
            if episode_key and episode_key not in ordered_episode_ids:
                ordered_episode_ids.append(episode_key)
        if not ordered_episode_ids:
            return items, 0, str(getattr(chapter, "chapter_id", "") or "") or None

        existing_by_id = {item.source_id: item for item in items}
        backfilled: list[ContextItem] = []
        chapter_backfill = 0
        for episode_id in ordered_episode_ids:
            continuity_boost = 1.0 if episode_id in defining_moment_ids else 0.72
            existing_item = existing_by_id.get(episode_id)
            if existing_item is not None:
                existing_item.scores["continuity"] = max(
                    float(existing_item.scores.get("continuity", 0.0) or 0.0),
                    continuity_boost,
                )
                existing_item.metadata["persisted_chapter_id"] = getattr(chapter, "chapter_id", None)
                existing_item.metadata["from_persisted_chapter"] = True
                continue

            try:
                episode = self.episodic.retrieve_memory(episode_id)  # type: ignore[call-arg]
            except Exception:
                continue
            if episode is None or getattr(episode, "suppressed", False):
                continue

            payloads = self._normalize_memory_results([episode.to_dict()])
            chapter_items = self._wrap_memory_results(payloads, "episodic", "importance")
            if not chapter_items:
                continue
            for chapter_item in chapter_items:
                chapter_item.reason = "persisted_chapter_anchor"
                chapter_item.scores["continuity"] = continuity_boost
                chapter_item.metadata["persisted_chapter_id"] = getattr(chapter, "chapter_id", None)
                chapter_item.metadata["from_persisted_chapter"] = True
            backfilled.extend(chapter_items)
            chapter_backfill += len(chapter_items)

        if not backfilled:
            return items, 0, str(getattr(chapter, "chapter_id", "") or "") or None

        combined: list[ContextItem] = []
        seen_ids: set[str] = set()
        for candidate in [*backfilled, *items]:
            if candidate.source_id in seen_ids:
                continue
            seen_ids.add(candidate.source_id)
            combined.append(candidate)

        limit = retrieval_plan.limit_for("episodic", 5) if retrieval_plan is not None else len(combined)
        return combined[:limit], chapter_backfill, str(getattr(chapter, "chapter_id", "") or "") or None

    def _inject_autobiographical_chapter_context(
        self,
        retrieval_plan: RetrievalPlan | None,
    ) -> List[ContextItem]:
        items: List[ContextItem] = []
        chapter = self._selected_persisted_chapter(retrieval_plan)
        if chapter is None or not chapter.summary:
            return items

        recency = self._chapter_recency_score(chapter)
        content_lines = [f"CURRENT CHAPTER: {chapter.summary}"]
        if chapter.goal_ids:
            content_lines.append("Goals in play: " + ", ".join(chapter.goal_ids[:3]))
        if chapter.participant_ids:
            content_lines.append("People in chapter: " + ", ".join(chapter.participant_ids[:3]))

        items.append(
            ContextItem(
                source_id=chapter.chapter_id,
                source_system="autobiographical",
                content="\n".join(content_lines),
                rank=RANK_BASE_EXECUTIVE - 1,
                reason="persisted_chapter_context",
                scores={
                    "continuity": 1.0,
                    "recency": recency,
                    "importance": 0.85,
                },
                metadata={
                    "life_period": chapter.life_period,
                    "goal_ids": list(chapter.goal_ids),
                    "participant_ids": list(chapter.participant_ids),
                    "defining_moment_ids": list(chapter.defining_moment_ids),
                },
            )
        )
        return items

    def _select_autobiographical_chapter(
        self,
        graph: AutobiographicalGraph,
        query: str | None,
    ) -> Any:
        if not graph.chapters:
            return None

        query_tokens = self._autobiographical_query_tokens(query)
        ranked = sorted(
            graph.chapters,
            key=lambda chapter: (
                -self._chapter_query_overlap(chapter, query_tokens),
                -(chapter.end_time.timestamp() if chapter.end_time is not None else float("-inf")),
                -(chapter.start_time.timestamp() if chapter.start_time is not None else float("-inf")),
                chapter.chapter_id,
            ),
        )
        return ranked[0]

    def _chapter_query_overlap(self, chapter: Any, query_tokens: set[str]) -> int:
        if not query_tokens:
            return 0
        chapter_tokens = self._autobiographical_query_tokens(
            " ".join(
                [
                    str(getattr(chapter, "title", "") or ""),
                    str(getattr(chapter, "summary", "") or ""),
                    str(getattr(chapter, "life_period", "") or ""),
                    *[str(goal_id) for goal_id in getattr(chapter, "goal_ids", [])],
                    *[str(participant_id) for participant_id in getattr(chapter, "participant_ids", [])],
                ]
            )
        )
        return len(query_tokens & chapter_tokens)

    def _autobiographical_query_tokens(self, value: str | None) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9-]+", str(value or "").lower()) if len(token) > 1}

    def _chapter_recency_score(self, chapter: Any) -> float:
        timestamp = getattr(chapter, "end_time", None) or getattr(chapter, "start_time", None)
        if not isinstance(timestamp, datetime):
            return 0.4
        age_hours = max(0.0, (datetime.now() - timestamp).total_seconds() / 3600.0)
        return 1.0 / (1.0 + (age_hours / 24.0))

    def _rerank_memory_context_items(
        self,
        items: List[ContextItem],
        *,
        retrieval_plan: RetrievalPlan,
        relationship_memory: Any | None,
        trace: ProvenanceTrace,
    ) -> None:
        memory_sources = {"stm", "ltm", "episodic"}
        prefix_items = [item for item in items if item.source_system not in memory_sources]
        memory_items = [item for item in items if item.source_system in memory_sources]
        if len(memory_items) < 2:
            return

        t0 = time.time()
        canonical_items: list[CanonicalMemoryItem] = []
        lookup: dict[str, ContextItem] = {}
        for context_item in memory_items:
            raw_payload = context_item.metadata.get("raw", {})
            canonical_payload = raw_payload.get("canonical") if isinstance(raw_payload, dict) else None
            if not isinstance(canonical_payload, dict):
                continue
            try:
                canonical_item = CanonicalMemoryItem.from_dict(canonical_payload)
            except Exception:
                continue
            canonical_items.append(canonical_item)
            lookup[canonical_item.memory_id] = context_item

        if len(canonical_items) < 2:
            return

        current_graph = self._autobiographical_graph_builder.build(canonical_items)
        persisted_graph = self._current_autobiographical_graph()
        autobiographical_graph = current_graph if persisted_graph is None else persisted_graph.merged_with(current_graph)
        reranked = self._memory_reranker.rerank(
            canonical_items,
            relationship_memory=relationship_memory,
            retrieval_plan=retrieval_plan,
            autobiographical_graph=autobiographical_graph,
        )

        reordered: list[ContextItem] = []
        for canonical_item in reranked:
            context_item = lookup.get(canonical_item.memory_id)
            if context_item is None:
                continue
            signals = dict(canonical_item.metadata.get("rerank_signals", {}))
            context_item.scores["importance"] = float(canonical_item.importance or 0.0)
            if signals.get("relationship", 0.0) > 0.0:
                context_item.scores["relationship"] = float(signals["relationship"])
            if signals.get("continuity", 0.0) > 0.0:
                context_item.scores["continuity"] = float(signals["continuity"])
            context_item.metadata["rerank_factors"] = dict(canonical_item.metadata.get("rerank_factors", {}))
            context_item.metadata["rerank_signals"] = signals
            context_item.metadata["rerank_score"] = canonical_item.metadata.get("rerank_score")
            reordered.append(context_item)

        if len(reordered) != len(memory_items):
            return

        items[:] = prefix_items + reordered
        trace.add_stage(
            PipelineStageResult(
                name="memory_rerank",
                candidates_in=len(memory_items),
                candidates_out=len(reordered),
                latency_ms=(time.time() - t0) * 1000.0,
                rationale="relationship_aware_cross_store_rerank",
                metadata={
                    "intent": retrieval_plan.intent,
                    "relationship_target": retrieval_plan.relationship_target,
                    "current_chapter_count": len(current_graph.chapters),
                    "persisted_chapter_count": len(persisted_graph.chapters) if persisted_graph is not None else 0,
                    "chapter_count": len(autobiographical_graph.chapters),
                },
            )
        )

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
            now = datetime.now(timezone.utc)
            
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
                        delta = r.due_time - now
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
                        delta = r.due_time - now
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

    def _inject_drive_context(self) -> List[ContextItem]:
        """Phase 2, Layer 0: Inject drive state and conflicts into LLM context.

        The drive system is owned by ChatService; we access it via the
        current session's most recent CognitiveTick state bag (threaded through
        by ChatService before calling build()).  As a fallback, we attempt to
        lazy-import and read the global drive state.
        """
        items: List[ContextItem] = []

        try:
            from src.cognition.drives import DriveState, DriveProcessor

            # Attempt to get drive state from session state (set by ChatService)
            drive_state: DriveState | None = None
            drive_processor: DriveProcessor | None = None

            # Access via the attached session if ChatService stored it
            session = getattr(self, "_current_session", None)
            if session is not None:
                drive_state = getattr(session, "_drive_state_snapshot", None)

            # If not found via session, try to read from a module-level singleton
            if drive_state is None:
                return items

            # Build drive processor for context generation helpers
            drive_processor = DriveProcessor()

            # Main drive context item
            drive_summary = drive_processor.drive_context_summary(drive_state)
            items.append(
                ContextItem(
                    source_id="drive_state",
                    source_system="drives",
                    content=drive_summary,
                    rank=RANK_BASE_EXECUTIVE + 1,
                    reason="drive_pressure",
                    scores={"drive_pressure": drive_state.total_pressure()},
                    metadata=drive_state.get_pressure(),
                )
            )

            # Conflict context (only if conscious conflicts exist)
            conflicts = drive_processor.detect_conflicts(drive_state)
            conflict_summary = drive_processor.conflict_context_summary(conflicts)
            if conflict_summary:
                items.append(
                    ContextItem(
                        source_id="drive_conflict",
                        source_system="drives",
                        content=conflict_summary,
                        rank=RANK_BASE_EXECUTIVE + 2,
                        reason="internal_conflict",
                        scores={"tension": max(c.tension for c in conflicts if c.conscious)},
                    )
                )

        except ImportError:
            pass  # Drive system not installed yet — graceful degradation
        except Exception:
            pass

        return items

    def _inject_felt_sense_context(self) -> List[ContextItem]:
        """Phase 2, Layer 1: Inject felt-sense and mood into LLM context.

        Like drives, felt-sense state is owned by ChatService and passed
        to ContextBuilder via the session snapshot.
        """
        items: List[ContextItem] = []

        try:
            from src.cognition.felt_sense import MoodLabeler

            session = getattr(self, "_current_session", None)
            mood = None
            if session is not None:
                mood = getattr(session, "_mood_snapshot", None)

            if mood is None:
                return items

            # Build context string
            summary = MoodLabeler.mood_context_summary(mood)
            items.append(
                ContextItem(
                    source_id="felt_sense_mood",
                    source_system="felt_sense",
                    content=summary,
                    rank=RANK_BASE_EXECUTIVE + 3,
                    reason="mood_state",
                    scores={"mood_intensity": mood.arousal, "mood_valence": mood.valence},
                    metadata={"mood_label": mood.label, "confidence": mood.confidence},
                )
            )

        except ImportError:
            pass  # Felt-sense system not installed — graceful degradation
        except Exception:
            pass

        return items

    def _inject_relational_context(self) -> List[ContextItem]:
        """Phase 2, Layer 2: Inject relationship context into LLM prompt.

        The relational model for the current interlocutor is stashed on
        the session by ChatService.  Only injected when the relationship
        is significant (enough interactions to be meaningful).
        """
        items: List[ContextItem] = []

        try:
            from src.cognition.relational import RelationalModel, RelationalProcessor

            session = getattr(self, "_current_session", None)
            rel_model: RelationalModel | None = None
            if session is not None:
                rel_model = getattr(session, "_relational_model_snapshot", None)

            if rel_model is None:
                return items

            # Only inject for significant relationships
            if not rel_model.is_significant():
                return items

            summary = RelationalProcessor.relational_context_summary(rel_model)
            items.append(
                ContextItem(
                    source_id="relational_context",
                    source_system="relational",
                    content=summary,
                    rank=RANK_BASE_EXECUTIVE + 4,
                    reason="relationship_context",
                    scores={
                        "felt_quality": rel_model.felt_quality,
                        "attachment": rel_model.attachment_strength,
                    },
                    metadata={
                        "person_id": rel_model.person_id,
                        "status": rel_model.current_status,
                    },
                )
            )

        except ImportError:
            pass  # Relational system not installed — graceful degradation
        except Exception:
            pass

        return items

    def _inject_pattern_context(self) -> List[ContextItem]:
        """Phase 2, Layer 3: Inject emergent-pattern context into LLM prompt.

        The pattern field is stashed on the session by ChatService.  Only
        injected when patterns have been detected (count > 0).
        """
        items: List[ContextItem] = []

        try:
            from src.cognition.patterns import PatternField, PatternDetector

            session = getattr(self, "_current_session", None)
            pf: PatternField | None = None
            if session is not None:
                pf = getattr(session, "_pattern_field_snapshot", None)

            if pf is None or pf.count() == 0:
                return items

            # Only inject if there are established patterns
            active = pf.active_patterns(min_strength=0.05)
            if not active:
                return items

            summary = PatternDetector.pattern_context_summary(pf)
            items.append(
                ContextItem(
                    source_id="pattern_context",
                    source_system="patterns",
                    content=summary,
                    rank=RANK_BASE_EXECUTIVE + 5,
                    reason="emergent_patterns",
                    scores={
                        "pattern_count": float(len(active)),
                        "top_strength": active[0].strength if active else 0.0,
                    },
                    metadata={
                        "dominant": [p.name for p in pf.dominant_patterns()],
                    },
                )
            )

        except ImportError:
            pass  # Pattern system not installed — graceful degradation
        except Exception:
            pass

        return items

    def _inject_selfmodel_context(self) -> List[ContextItem]:
        """Phase 2, Layer 4: Inject self-model context into LLM prompt.

        The self-model is stashed on the session by ChatService.  Only
        injected when a self-model has been built.  Blind spots are
        NOT included — the context represents what the agent believes
        about itself.
        """
        items: List[ContextItem] = []

        try:
            from src.cognition.selfmodel import SelfModel, SelfModelBuilder

            session = getattr(self, "_current_session", None)
            sm: SelfModel | None = None
            if session is not None:
                sm = getattr(session, "_self_model_snapshot", None)

            if sm is None:
                return items

            # Only inject if there's meaningful self-knowledge
            if not sm.perceived_patterns:
                return items

            summary = SelfModelBuilder.self_model_context_summary(sm)
            if not summary:
                return items

            items.append(
                ContextItem(
                    source_id="selfmodel_context",
                    source_system="selfmodel",
                    content=summary,
                    rank=RANK_BASE_EXECUTIVE + 6,
                    reason="self_model",
                    scores={
                        "self_regard": sm.self_regard,
                        "identity_stability": sm.identity_stability,
                    },
                    metadata={
                        "perceived_count": len(sm.perceived_patterns),
                        "blind_spot_count": sm.blind_spot_count,
                        "has_discoveries": len(sm.recent_discoveries) > 0,
                    },
                )
            )

        except ImportError:
            pass  # Self-model system not installed — graceful degradation
        except Exception:
            pass

        return items

    def _inject_narrative_context(self) -> List[ContextItem]:
        """Phase 2, Layer 5: Inject narrative context into LLM prompt.

        The narrative is stashed on the session by ChatService.  Only
        injected when a narrative has been constructed and is non-empty.
        The identity_summary is the primary content, supplemented by
        growth story and current themes.
        """
        items: List[ContextItem] = []

        try:
            from src.cognition.narrative import SelfNarrative, NarrativeConstructor

            session = getattr(self, "_current_session", None)
            narr: SelfNarrative | None = None
            if session is not None:
                narr = getattr(session, "_narrative_snapshot", None)

            if narr is None or narr.is_empty:
                return items

            summary = NarrativeConstructor.narrative_context_summary(narr)
            if not summary:
                return items

            items.append(
                ContextItem(
                    source_id="narrative_context",
                    source_system="narrative",
                    content=summary,
                    rank=RANK_BASE_EXECUTIVE + 7,
                    reason="narrative",
                    scores={
                        "version": float(narr.version),
                    },
                    metadata={
                        "chapter_count": len(narr.chapters),
                        "theme_count": len(narr.active_themes),
                        "has_growth_story": bool(narr.growth_story),
                        "trigger": narr.update_trigger,
                    },
                )
            )

        except ImportError:
            pass  # Narrative system not installed — graceful degradation
        except Exception:
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
        now_ts = int(time.time())
        for t in turns:
            # Skip the most recent assistant reply (we only want prior context + user turns)
            if t.role not in ("user", "system"):
                continue
            scores = {
                "recency": round(max(0.0, 1.0 - (now_ts - t.timestamp) / 3600.0), 4),
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
