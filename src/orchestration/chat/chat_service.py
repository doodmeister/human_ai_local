from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional, Set, Tuple, List, Callable
from functools import partial
import asyncio
import logging
from datetime import datetime
from collections import deque
from .conversation_session import SessionManager
from .models import TurnRecord
from .context_builder import ContextBuilder
from .metrics import metrics_registry
from .scoring import get_scoring_profile_version
from .memory_capture import MemoryCaptureModule, MemoryCaptureCache  # added import near top
from .intent_classifier_v2 import (
    IntentClassifierV2,
    IntentV2,
    ConversationContext,
    create_intent_classifier_v2,
)
from .goal_handlers import GoalIntentHandler
from .metacog_manager import MetacogManager
from .memory_query_parser import create_memory_query_parser  # Production Phase 1 - Task 8
from .intent_router import ChatIntentRouter
from .intent_actions import ChatIntentActions
from .response_builder import ChatResponseBuilder
from .status_service import ChatStatusService
from .turn_pipeline import ChatTurnPipeline
from .turn_support import ChatTurnSupport
from .turn_context import ChatTurnContextBuilder
from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from src.orchestration.autobiographical_promotion import (
    PromotionResult,
    derive_autobiographical_life_period,
    promote_interaction_to_autobiographical_memory,
)
from src.orchestration.cognitive_layers import ChatCognitiveLayerRuntime

logger = logging.getLogger(__name__)


class ChatService:
    """
    Orchestrates chat turn processing:
    - Adds user turn with salience/valence tagging
    - Builds context (via ContextBuilder)
    - Generates assistant response via configured agent when needed
    - Applies consolidation decision heuristic
    - Returns structured payload
    """

    _INTENT_HANDLER_PRIORITY = (
        "goal_update",
        "goal_query",
        "memory_query",
        "reminder_request",
        "performance_query",
        "system_status",
    )
    _INTENT_SECTION_HEADERS = {
        "goal_update": "Goal update",
        "goal_query": "Goal update",
        "memory_query": "Memory lookup",
        "performance_query": "Performance metrics",
        "system_status": "System status",
        "reminder_request": "Reminders",
    }

    def __init__(self,
                 session_manager: SessionManager,
                 context_builder: ContextBuilder,
                 consolidator: Optional[Any] = None,
                 agent: Optional[Any] = None,
                 metacognitive_controller: Optional[Any] = None) -> None:
        self.sessions = session_manager
        self.context_builder = context_builder
        self.consolidator = consolidator
        self.agent = agent
        self._metacognitive_controller = metacognitive_controller
        self._last_metacognitive_cycle = None
        self._capture = MemoryCaptureModule()
        self._capture_cache = MemoryCaptureCache()
        # Production Phase 1: Intent classification and goal detection
        self._intent_classifiers: Dict[str, IntentClassifierV2] = {}
        self._session_goal_index: Dict[str, Set[str]] = {}
        self._goal_detector = None  # Lazy init to avoid circular dependency with executive system
        self._orchestrator = None  # Lazy init for background goal execution (Task 6)
        self._goal_handler: Optional[GoalIntentHandler] = None
        # Production Phase 1 - Task 8: Memory query handling
        self._memory_query_parser = create_memory_query_parser()
        self._memory_query_interface = None  # Lazy init with memory systems
        self._intent_router = ChatIntentRouter(handler_priority=self._INTENT_HANDLER_PRIORITY)
        self._intent_actions = ChatIntentActions(
            memory_query_parser=self._memory_query_parser,
            get_consolidator=lambda: self.consolidator,
            get_prospective_memory=self._get_prospective_memory,
            format_due_phrase=self._format_due_phrase,
            resolve_reminder_due_time=self._resolve_reminder_due_time,
        )
        self._response_builder = ChatResponseBuilder(
            section_headers=self._INTENT_SECTION_HEADERS,
            get_config=lambda: self.context_builder.cfg,
        )
        self._status_service = ChatStatusService(
            get_config=lambda: self.context_builder.cfg,
            get_stm_usage_snapshot=self._get_stm_usage_snapshot,
            get_active_goal_count=lambda session_id: len(self._session_goal_index.get(session_id, set())),
            get_metacog_interval=lambda: self._metacog_interval,
            get_consolidation_event_count=lambda: len(self.consolidation_log),
        )
        self._turn_pipeline = ChatTurnPipeline()
        self._turn_support = ChatTurnSupport(
            get_agent=lambda: self.agent,
            get_capture_records=lambda: self._capture_cache.as_list(),
            get_consolidator=lambda: self.consolidator,
            get_config=lambda: self.context_builder.cfg,
        )
        self._turn_context_builder = ChatTurnContextBuilder(
            get_active_goal_ids=lambda session_id: self._session_goal_index.get(session_id, set()),
        )
        # Consolidation event log (simple list for recent trace enrichment)
        self.consolidation_log = []  # entries: dict(user_turn_id,status,salience,valence,timestamp)
        # Metacognitive tracking fields
        self._turn_counter = 0
        self._last_metacog_snapshot = None
        self._metacog_interval = int(self.context_builder.cfg.get("metacog_turn_interval", 5))
        self._metacog_manager = MetacogManager(metrics_registry)
        # Snapshot history ring buffer
        history_size = int(self.context_builder.cfg.get("metacog_snapshot_history_size", 50))
        self._metacog_history = deque(maxlen=history_size)
        self._cognitive_layers = ChatCognitiveLayerRuntime()
        self._context_preview_cache: Dict[
            Tuple[str, str, Tuple[Tuple[str, bool], ...]],
            Tuple[Tuple[Any, ...], Dict[str, Any]],
        ] = {}
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
        cache_key = self._context_preview_cache_key(sess, message, flags)
        state_signature = self._context_preview_state_signature(sess)
        cached = self._context_preview_cache.get(cache_key)
        if cached is not None and cached[0] == state_signature:
            return copy.deepcopy(cached[1])

        # Add a transient turn (not persisted) for preview retrieval basis
        builder = self.context_builder
        built = builder.build(
            sess,
            query=message,
            include_attention=not flags.get("disable_attention", False),
            include_memory=not flags.get("disable_memory", False),
            include_trace=False,
            read_only_retrieval=True,
        )
        items_summary = self._summarize_context_items(built.items)
        metrics_registry.inc("context_preview_calls_total")
        preview = {
            "session_id": sess.session_id,
            "scoring_version": get_scoring_profile_version(),
            "item_count": len(items_summary),
            "items": items_summary,
        }
        self._context_preview_cache[cache_key] = (state_signature, copy.deepcopy(preview))
        return preview

    def process_user_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        payload = self._turn_pipeline.process_user_message(
            service=self,
            message=message,
            session_id=session_id,
            flags=flags,
        )
        cycle = self._run_metacognitive_cycle(
            message=message,
            session_id=payload.get("session_id") or session_id,
            flags=flags,
        )
        if cycle is not None:
            payload["metacognitive_cycle"] = self._summarize_metacognitive_cycle(cycle)
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

    def _get_intent_classifier(self, session_id: str) -> IntentClassifierV2:
        """Return or initialize the session-scoped intent classifier."""
        classifier = self._intent_classifiers.get(session_id)
        if classifier is None:
            classifier = create_intent_classifier_v2(context=ConversationContext())
            self._intent_classifiers[session_id] = classifier
        # Refresh active goal context to keep boosts accurate
        classifier.context.active_goals = set(self._session_goal_index.get(session_id, set()))
        return classifier

    def _get_goal_handler(self) -> GoalIntentHandler:
        """Return a goal intent handler bound to the current orchestrator."""
        if self._goal_handler is None or self._goal_handler.orchestrator is not self._orchestrator:
            self._goal_handler = GoalIntentHandler(self._orchestrator)
        return self._goal_handler

    def _format_goal_confirmation(self, detected_goal: Any) -> str:
        return self._get_goal_handler().format_goal_confirmation(detected_goal)

    def _handle_goal_query(self, intent: IntentV2, session_id: str) -> Optional[str]:
        return self._get_goal_handler().handle_goal_query(intent, session_id)

    def _handle_goal_update(self, intent: IntentV2, session_id: str) -> Optional[str]:
        return self._get_goal_handler().handle_goal_update(intent, session_id)

    def _record_session_goal(self, session_id: str, goal_id: str) -> None:
        """Track goal IDs per session for context-aware intent boosts."""
        goals = self._session_goal_index.setdefault(session_id, set())
        goals.add(goal_id)

    def _fallback_intent(self, message: str) -> IntentV2:
        """Create a safe general_chat intent if classification fails."""
        return IntentV2(
            intent_type="general_chat",
            confidence=1.0,
            entities={},
            original_message=message,
            matched_patterns=[],
        )

    def _serialize_intent(self, intent: IntentV2) -> Dict[str, Any]:
        """Serialize IntentV2 for API payloads."""
        return {
            "type": intent.intent_type,
            "confidence": intent.confidence,
            "entities": intent.entities,
            "matched_patterns": intent.matched_patterns,
            "secondary_intents": intent.secondary_intents,
            "is_ambiguous": intent.is_ambiguous,
            "ambiguity_score": intent.ambiguity_score,
            "context_boost": intent.context_boost,
            "conversation_context": intent.conversation_context,
        }

    def _plan_intent_execution(self, intent: IntentV2) -> list[str]:
        """Return ordered list of intent handlers to execute for this turn."""
        return self._intent_router.plan_intent_execution(intent)

    def _run_intent_handlers(
        self,
        intent: IntentV2,
        message: str,
        session_id: str,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """Execute all applicable intent handlers and return sections + execution log."""
        return self._intent_router.run_intent_handlers(
            intent=intent,
            message=message,
            session_id=session_id,
            resolve_handler=self._resolve_intent_handler,
        )

    def _merge_intent_sections(
        self,
        sections: list[Dict[str, Any]],
        base_response: Optional[str],
    ) -> str:
        """Merge intent-specific sections with the base assistant response."""
        return self._response_builder.merge_intent_sections(sections, base_response)

    def _get_intent_confidence(self, intent: IntentV2, intent_name: str) -> float:
        """Return the classifier confidence for the given intent name."""
        return self._intent_router.get_intent_confidence(intent, intent_name)

    def _resolve_intent_handler(
        self,
        handler_name: str,
        intent: IntentV2,
        message: str,
        session_id: str,
    ) -> tuple[Callable[[], Any] | None, str | None]:
        if handler_name == "goal_update":
            if self._orchestrator is None:
                return None, "orchestrator_unavailable"
            return partial(self._handle_goal_update, intent, session_id), None
        if handler_name == "goal_query":
            if self._orchestrator is None:
                return None, "orchestrator_unavailable"
            return partial(self._handle_goal_query, intent, session_id), None
        if handler_name == "memory_query":
            return partial(self._handle_memory_query, message, session_id), None
        if handler_name == "reminder_request":
            return partial(self._handle_reminder_request, intent, session_id), None
        if handler_name == "performance_query":
            return partial(self._handle_performance_query, intent, session_id), None
        if handler_name == "system_status":
            return partial(self._handle_system_status, intent, session_id), None
        return None, "handler_not_found"

    def _build_session_context(
        self,
        session_id: str,
        intent: IntentV2,
        detected_goal: Optional[Any],
        stored_captures: list[Dict[str, Any]],
        extra_due: list[Dict[str, Any]],
        upcoming_reminders: List[Any],
    ) -> Dict[str, Any]:
        """Construct a lightweight session context snapshot for UI consumers."""
        return self._turn_context_builder.build_session_context(
            session_id=session_id,
            intent=intent,
            detected_goal=detected_goal,
            stored_captures=stored_captures,
            extra_due=extra_due,
            upcoming_reminders=upcoming_reminders,
        )

    # --- Helpers ---

    def _get_stm_usage_snapshot(self) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """Return STM size, capacity, and utilization ratio if available."""
        if self.consolidator is None:
            return None, None, None
        stm_obj = getattr(self.consolidator, "stm", None)
        if stm_obj is None:
            return None, None, None
        size = None
        try:
            if hasattr(stm_obj, "__len__"):
                size = len(stm_obj)  # type: ignore[arg-type]
        except Exception:
            size = None
        if size is None:
            fallback_size = getattr(stm_obj, "size", None)
            if isinstance(fallback_size, int):
                size = fallback_size
        capacity = getattr(stm_obj, "capacity", None)
        if not isinstance(capacity, int):
            capacity = None
        utilization = None
        if isinstance(size, int) and isinstance(capacity, int) and capacity > 0:
            utilization = min(1.0, max(0.0, size / capacity))
        return size, capacity, utilization

    def _format_latency_ms(self, value: Optional[Any]) -> str:
        """Format latency or duration values for chat output."""
        return self._status_service.format_latency_ms(value)

    def _format_percentage(self, value: Optional[Any]) -> str:
        """Format ratio as percentage string."""
        return self._status_service.format_percentage(value)

    def _resolve_reminder_due_time(self, intent: IntentV2) -> Optional[datetime]:
        """Convert intent entities into an absolute reminder due time."""
        return self._status_service.resolve_reminder_due_time(intent)

    def _format_due_phrase(self, due_time: Optional[datetime]) -> str:
        """Return a friendly phrase for reminder due times."""
        return self._status_service.format_due_phrase(due_time)

    def _get_prospective_memory(self) -> Any:
        """Compatibility seam for reminder flows and tests."""
        return get_inmemory_prospective_memory()

    def _format_proactive_reminder_summary(
        self,
        due_reminders: List[Any],
        upcoming_reminders: List[Any],
    ) -> str:
        return self._status_service.format_proactive_reminder_summary(due_reminders, upcoming_reminders)

    def _serialize_reminder_brief(self, reminder: Any) -> Dict[str, Any]:
        return self._status_service.serialize_reminder_brief(reminder)

    def _get_stat_value(self, stats: Any, key: str, default: Any) -> float:
        """Safely extract a numeric statistic value with graceful fallback."""
        return self._status_service.get_stat_value(stats, key, default)

    def _estimate_importance(self, content: str, salience: float, valence: float) -> float:
        return self._turn_support.estimate_importance(content, salience, valence)

    # ------------------------------------------------------------------
    # Phase 2, Layer 0: Drive system helpers
    # ------------------------------------------------------------------

    def _get_drive_system(self):
        """Lazy-init and return (DriveState, DriveProcessor) pair."""
        return self._cognitive_layers.get_drive_system()

    def get_drive_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current drive state (for API/telemetry)."""
        return self._cognitive_layers.get_drive_state()

    # ------------------------------------------------------------------
    # Phase 2, Layer 1: Felt-sense / mood helpers
    # ------------------------------------------------------------------

    def _get_felt_sense_system(self):
        """Lazy-init and return (FeltSenseGenerator, FeltSenseHistory, MoodLabeler)."""
        return self._cognitive_layers.get_felt_sense_system()

    def get_mood_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current mood (for API/telemetry)."""
        return self._cognitive_layers.get_mood_state()

    # ------------------------------------------------------------------
    # Phase 2, Layer 2: Relational field helpers
    # ------------------------------------------------------------------

    def _get_relational_system(self):
        """Lazy-init and return (RelationalField, RelationalProcessor) pair."""
        return self._cognitive_layers.get_relational_system()

    def get_relational_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current relational field (for API/telemetry)."""
        return self._cognitive_layers.get_relational_state()

    # ------------------------------------------------------------------
    # Phase 2, Layer 3: Emergent patterns helpers
    # ------------------------------------------------------------------

    def _get_pattern_system(self):
        """Lazy-init and return (PatternField, PatternDetector) pair."""
        return self._cognitive_layers.get_pattern_system()

    def get_pattern_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for emergent patterns (for API/telemetry)."""
        return self._cognitive_layers.get_pattern_state()

    # ------------------------------------------------------------------
    # Phase 2, Layer 4: Self-model helpers
    # ------------------------------------------------------------------

    def _get_self_model_system(self):
        """Lazy-init and return SelfModelBuilder."""
        return self._cognitive_layers.get_self_model_system()

    def get_self_model_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for self-model (for API/telemetry).

        Blind spots are excluded from the public-facing dict.
        """
        return self._cognitive_layers.get_self_model_state()

    # ------------------------------------------------------------------
    # Phase 2, Layer 5: Narrative helpers
    # ------------------------------------------------------------------

    def _get_narrative_system(self):
        """Lazy-init and return NarrativeConstructor."""
        return self._cognitive_layers.get_narrative_system()

    def get_narrative_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for narrative (for API/telemetry)."""
        return self._cognitive_layers.get_narrative_state()

    def get_response_policy_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current response policy (for API/telemetry)."""
        return self._cognitive_layers.get_response_policy_state()

    def _handle_memory_query(self, message: str, session_id: str) -> Optional[str]:
        return self._intent_actions.handle_memory_query(message, session_id)

    def _handle_reminder_request(self, intent: IntentV2, session_id: str) -> Optional[str]:
        return self._intent_actions.handle_reminder_request(intent, session_id)

    def _handle_performance_query(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """Return a formatted performance snapshot for performance_query intents."""
        try:
            return self._status_service.handle_performance_query(intent, session_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Performance status lookup failed: %s", exc, exc_info=True)
            return None

    def _handle_system_status(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """Summarize current cognitive system health for system_status intents."""
        try:
            return self._status_service.handle_system_status(intent, session_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("System status lookup failed: %s", exc, exc_info=True)
            return None

    def _invoke_agent_response(self, message: str, built: Any) -> str:
        """Invoke the configured agent to generate a response with context."""
        return self._turn_support.invoke_agent_response(message, built)

    def _run_coroutine_sync(self, coro: Any) -> Any:
        """Execute a coroutine to completion from a synchronous context."""
        return self._turn_support._run_coroutine_sync(coro)

    def _maybe_consolidate(self, user_turn: TurnRecord, assistant_turn: TurnRecord) -> bool:
        return self._turn_support.maybe_consolidate(user_turn, assistant_turn)

    def _get_memory_system(self) -> Any:
        agent = self.agent
        if agent is None:
            return None
        return getattr(agent, "memory", None)

    def _derive_autobiographical_life_period(self, intent: Any = None, tick: Any = None) -> str:
        return derive_autobiographical_life_period(
            intent_type=str(getattr(intent, "intent_type", "") or ""),
            tick=tick,
            default_prefix="chat",
        )

    def _promote_autobiographical_turn(
        self,
        *,
        session: Any,
        user_turn: TurnRecord,
        assistant_turn: TurnRecord,
        intent: Any = None,
        tick: Any = None,
    ) -> str | None:
        try:
            result = promote_interaction_to_autobiographical_memory(
                memory=self._get_memory_system(),
                autobiographical_store=getattr(self.context_builder, "_autobiographical_store", None),
                session=session,
                user_content=user_turn.content,
                assistant_content=assistant_turn.content,
                importance=float(user_turn.importance or 0.0),
                emotional_valence=float(user_turn.emotional_valence or 0.0),
                source_event_ids=[user_turn.turn_id, assistant_turn.turn_id],
                intent_type=str(getattr(intent, "intent_type", "") or "conversation"),
                tick=tick,
                goal_ids=sorted(self._session_goal_index.get(session.session_id, set())),
                default_prefix="chat",
            )
            self._apply_promotion_result(session, result)
            return result.episode_id
        except Exception as exc:
            logger.debug("Autobiographical episodic promotion skipped: %s", exc)
            return None

    @staticmethod
    def _apply_promotion_result(session: Any, result: PromotionResult) -> None:
        if result.episode_id:
            setattr(session, "_last_autobiographical_episode_id", result.episode_id)
        if result.graph_snapshot is not None:
            setattr(session, "_autobiographical_graph_snapshot", result.graph_snapshot)
        if result.semantic_fact_ids:
            setattr(session, "_last_semantic_fact_ids", list(result.semantic_fact_ids))
        if result.semantic_products:
            setattr(session, "_last_semantic_products", [dict(item) for item in result.semantic_products])
        if result.prospective_reminder_ids:
            setattr(session, "_last_prospective_reminder_ids", list(result.prospective_reminder_ids))
        if result.prospective_products:
            setattr(session, "_last_prospective_products", [dict(item) for item in result.prospective_products])

    def _summarize_context_items(self, items):
        return self._response_builder.summarize_context_items(items)

    def _context_preview_cache_key(
        self,
        session: Any,
        message: str,
        flags: Dict[str, bool],
    ) -> Tuple[str, str, Tuple[Tuple[str, bool], ...]]:
        normalized_flags = tuple(sorted((str(key), bool(value)) for key, value in flags.items()))
        return session.session_id, message, normalized_flags

    def _context_preview_state_signature(self, session: Any) -> Tuple[Any, ...]:
        turn_ids = tuple(getattr(turn, "turn_id", None) for turn in getattr(session, "turns", ()))
        active_goals = tuple(sorted(self._session_goal_index.get(session.session_id, set())))
        performance_degraded = bool(metrics_registry.state.get("performance_degraded"))
        return turn_ids, active_goals, performance_degraded

    def _trace_to_dict(self, built) -> Dict[str, Any]:
        return self._response_builder.trace_to_dict(built)

    def performance_status(self) -> Dict[str, Any]:
        return self._status_service.build_performance_status()

    def get_metacog_status(self, history_limit: int = 10) -> Dict[str, Any]:
        snapshot = self._last_metacog_snapshot
        if not snapshot:
            if self._last_metacognitive_cycle is None:
                return {"available": False}
            return {
                "available": True,
                "last_cycle_summary": self._summarize_metacognitive_cycle(self._last_metacognitive_cycle),
            }

        try:
            history_tail = list(self._metacog_history)[-max(0, history_limit):]
        except Exception:
            history_tail = []
        status = {
            "available": True,
            "snapshot": snapshot,
            "history_tail": history_tail,
        }
        if self._last_metacognitive_cycle is not None:
            status["last_cycle_summary"] = self._summarize_metacognitive_cycle(self._last_metacognitive_cycle)
        return status

    def set_metacognitive_controller(self, controller: Any) -> None:
        self._metacognitive_controller = controller

    def get_last_cycle_summary(self) -> Dict[str, Any]:
        cycle = self._last_metacognitive_cycle
        if cycle is None:
            return {"available": False}
        return self._summarize_metacognitive_cycle(cycle)

    def _run_metacognitive_cycle(
        self,
        *,
        message: str,
        session_id: Optional[str],
        flags: Optional[Dict[str, bool]],
    ) -> Any:
        controller = self._metacognitive_controller
        if controller is None:
            return None
        from src.orchestration.metacognition import Stimulus

        try:
            cycle = controller.run_cycle(
                Stimulus(
                    session_id=session_id or "default",
                    user_input=message,
                    input_type="text",
                    turn_index=self._turn_counter,
                    metadata={"flags": dict(flags or {})},
                )
            )
        except Exception as exc:
            logger.debug("Metacognitive cycle skipped in ChatService: %s", exc)
            return None
        self._last_metacognitive_cycle = cycle
        return cycle

    def _summarize_metacognitive_cycle(self, cycle: Any) -> Dict[str, Any]:
        plan = cycle.plan
        execution_result = cycle.execution_result
        critic_report = cycle.critic_report
        selected_goal = plan.selected_goal if plan is not None else None
        acts = plan.acts if plan is not None else ()
        return {
            "available": True,
            "cycle_id": cycle.cycle_id,
            "trace_id": cycle.trace_id,
            "selected_goal_id": selected_goal.goal_id if selected_goal is not None else None,
            "selected_goal_kind": selected_goal.kind.value if selected_goal is not None else None,
            "act_types": [act.act_type.value for act in acts],
            "success_score": critic_report.success_score if critic_report is not None else None,
            "follow_up_recommended": critic_report.follow_up_recommended if critic_report is not None else None,
            "response_text": execution_result.response_text if execution_result is not None else None,
            "scheduled_tasks": [self._serialize_metacognitive_value(task) for task in cycle.scheduled_tasks],
        }

    @staticmethod
    def _serialize_metacognitive_value(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "to_dict"):
            maybe_mapping = value.to_dict()
            if isinstance(maybe_mapping, dict):
                return dict(maybe_mapping)
        return {"value": value}

    def _attempt_fact_answer(self, message: str) -> Optional[str]:
        return self._turn_support.attempt_fact_answer(message)
