from __future__ import annotations

from typing import Dict, Any, Optional, Set, Tuple, List
from functools import partial
import time
import asyncio
import logging
from datetime import datetime, timedelta
from collections import deque
from .conversation_session import SessionManager
from .models import TurnRecord
from .context_builder import ContextBuilder
from .emotion_salience import estimate_salience_and_valence
from .metrics import metrics_registry
from .provenance import build_item_provenance
from .scoring import get_scoring_profile_version
from .constants import PREVIEW_MAX_ITEMS, PREVIEW_MAX_CONTENT_CHARS
from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from .memory_capture import MemoryCaptureModule, MemoryCaptureCache  # added import near top
from .intent_classifier_v2 import (
    IntentClassifierV2,
    IntentV2,
    ConversationContext,
    create_intent_classifier_v2,
)
from .goal_handlers import GoalIntentHandler
from .metacog_manager import MetacogManager
from .goal_detector import GoalDetector  # Production Phase 1
from .executive_orchestrator import ExecutiveOrchestrator  # Production Phase 1 - Task 6
from .memory_query_parser import create_memory_query_parser  # Production Phase 1 - Task 8
from .memory_query_interface import create_memory_query_interface  # Production Phase 1 - Task 8
import re  # added for fact question pattern
from src.orchestration import CognitiveStep, CognitiveTick
from src.learning.learning_law import clamp01, utility_score

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
                 agent: Optional[Any] = None) -> None:
        self.sessions = session_manager
        self.context_builder = context_builder
        self.consolidator = consolidator
        self.agent = agent
        # Memory capture modules (Phase 1)
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
        # Phase 2, Layer 0: Drive system (lazy init)
        self._drive_state = None
        self._drive_processor = None
        self._drive_last_turn_time: Optional[float] = None
        self._drive_impact_history: list = []  # recent DriveImpact list for sensitivity adaptation
        # Phase 2, Layer 1: Felt-sense / mood system (lazy init)
        self._felt_sense_generator = None
        self._felt_sense_history = None
        self._mood_labeler = None
        self._current_mood = None
        self._recent_valences: list = []  # rolling valence window for felt-sense generation
        # Phase 2, Layer 2: Relational field (lazy init)
        self._relational_field = None
        self._relational_processor = None
        # Phase 2, Layer 3: Emergent patterns (lazy init)
        self._pattern_field = None
        self._pattern_detector = None
        self._pattern_turn_counter = 0
        # Phase 2, Layer 4: Self-model (lazy init)
        self._self_model = None
        self._self_model_builder = None
        self._self_model_turn_counter = 0
        # Phase 2, Layer 5: Narrative (lazy init)
        self._narrative = None
        self._narrative_constructor = None
        self._narrative_turn_counter = 0
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

        tick = CognitiveTick(owner="chat_service", kind="chat_turn")
        tick.state["session_id"] = sess.session_id

        # Perceive
        tick.assert_step(CognitiveStep.PERCEIVE)
        
        # Production Phase 1: Classify user intent
        intent: IntentV2
        classifier = self._get_intent_classifier(sess.session_id)
        try:
            intent = classifier.classify(message, use_context=True)
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc, exc_info=True)
            intent = self._fallback_intent(message)
        detected_goal = None
        intent_sections: list[Dict[str, Any]] = []
        intent_execution_log: list[Dict[str, Any]] = []
        
        # Detect and create goals automatically
        if intent.intent_type == 'goal_creation':
            try:
                # Lazy init goal detector
                if self._goal_detector is None:
                    from src.executive.integration import ExecutiveSystem
                    executive_system = ExecutiveSystem()
                    self._goal_detector = GoalDetector(executive_system)
                
                detected_goal = self._goal_detector.detect_goal(
                    message,
                    sess.session_id,
                    intent=intent,
                )
                if detected_goal:
                    metrics_registry.inc("goals_auto_detected_total")
                    self._record_session_goal(sess.session_id, detected_goal.goal_id)
                    # Update classifier context with fresh goal count for future turns
                    classifier.context.active_goals = set(self._session_goal_index.get(sess.session_id, set()))
                    
                    # Start background execution
                    if self._orchestrator is None:
                        from src.executive.integration import ExecutiveSystem
                        executive_system = ExecutiveSystem()
                        self._orchestrator = ExecutiveOrchestrator(executive_system)
                    try:
                        asyncio.create_task(self._orchestrator.execute_goal_async(detected_goal.goal_id))
                    except RuntimeError:
                        # No event loop running, skip async execution for now
                        pass
            except Exception as e:
                # Don't fail the conversation if goal detection fails
                print(f"Goal detection error: {e}")
                detected_goal = None
        
        salience, valence = estimate_salience_and_valence(message)
        importance = self._estimate_importance(message, salience, valence)

        # Phase 2, Layer 0: Update drive state from this turn
        drive_impact = None
        drive_conflicts = []
        try:
            ds, dp = self._get_drive_system()
            now_ts = time.time()
            elapsed_min = (now_ts - self._drive_last_turn_time) / 60.0 if self._drive_last_turn_time else 0.0
            self._drive_last_turn_time = now_ts
            ds, drive_impact = dp.process_turn(
                ds, message,
                salience=salience, valence=valence, elapsed_minutes=elapsed_min,
            )
            drive_conflicts = dp.detect_conflicts(ds)
            # Stash in tick state for downstream stages
            tick.state["drive_state"] = ds
            tick.state["drive_impact"] = drive_impact
            tick.state["drive_conflicts"] = drive_conflicts
            # Make drive state accessible to ContextBuilder via session
            setattr(sess, "_drive_state_snapshot", ds)
            # Track impacts for long-term sensitivity adaptation
            self._drive_impact_history.append(drive_impact)
            if len(self._drive_impact_history) > 100:
                self._drive_impact_history = self._drive_impact_history[-50:]
            # Periodically adapt baselines (every 20 turns)
            if self._turn_counter % 20 == 0 and self._turn_counter > 0:
                dp.adapt_baselines(ds)
                dp.adapt_sensitivities(ds, self._drive_impact_history[-20:])
        except Exception as exc:
            logger.debug("Drive processing skipped: %s", exc)

        # Phase 2, Layer 1: Generate felt sense and derive mood
        try:
            self._recent_valences.append(valence)
            if len(self._recent_valences) > 10:
                self._recent_valences = self._recent_valences[-10:]

            ds_for_felt = tick.state.get("drive_state") or self._drive_state
            if ds_for_felt is not None:
                fsg, fsh, ml = self._get_felt_sense_system()
                felt = fsg.generate(ds_for_felt, recent_valences=self._recent_valences)
                fsh.update(felt)
                mood = ml.label_mood(felt)
                self._current_mood = mood

                tick.state["felt_sense"] = felt
                tick.state["mood"] = mood
                tick.state["felt_sense_trend"] = fsh.trend()
                setattr(sess, "_felt_sense_snapshot", felt)
                setattr(sess, "_mood_snapshot", mood)
        except Exception as exc:
            logger.debug("Felt-sense processing skipped: %s", exc)

        # Phase 2, Layer 2: Update relational field for current interlocutor
        try:
            rf, rp = self._get_relational_system()
            person_id = sess.session_id or "default"
            rf.set_interlocutor(person_id)
            drive_impact_obj = tick.state.get("drive_impact")
            rel_model = rp.process_turn(
                rf, person_id, message,
                valence=valence, salience=salience,
                drive_impact=drive_impact_obj,
            )
            tick.state["relational_model"] = rel_model
            setattr(sess, "_relational_model_snapshot", rel_model)
        except Exception as exc:
            logger.debug("Relational processing skipped: %s", exc)

        # Phase 2, Layer 3: Periodic emergent pattern detection
        try:
            self._pattern_turn_counter += 1
            pf, pd = self._get_pattern_system()
            if self._pattern_turn_counter % pd.config.detection_interval == 0:
                pd.detect_patterns(
                    pf,
                    drive_state=tick.state.get("drive_state"),
                    drive_impacts=self._drive_impact_history[-20:] or None,
                    felt_sense_history=self._felt_sense_history,
                    relational_field=self._relational_field,
                    conflicts=tick.state.get("drive_conflicts") or None,
                )
            tick.state["pattern_field"] = pf
            setattr(sess, "_pattern_field_snapshot", pf)
        except Exception as exc:
            logger.debug("Pattern detection skipped: %s", exc)

        # Phase 2, Layer 4: Periodic self-model rebuild
        try:
            self._self_model_turn_counter += 1
            smb = self._get_self_model_system()
            if self._self_model_turn_counter % smb.config.update_interval == 0:
                self._self_model = smb.build_self_model(
                    pattern_field=self._pattern_field,
                    drive_state=tick.state.get("drive_state"),
                    felt_sense=tick.state.get("felt_sense"),
                    mood=tick.state.get("mood"),
                    existing_self_model=self._self_model,
                )
            if self._self_model is not None:
                tick.state["self_model"] = self._self_model
                setattr(sess, "_self_model_snapshot", self._self_model)
        except Exception as exc:
            logger.debug("Self-model update skipped: %s", exc)

        # Phase 2, Layer 5: Narrative construction
        try:
            self._narrative_turn_counter += 1
            nc = self._get_narrative_system()
            should, trigger = nc.should_update(
                self_model=self._self_model,
                previous_narrative=self._narrative,
                turn_counter=self._narrative_turn_counter,
            )
            if should:
                self._narrative = nc.construct_narrative(
                    self_model=self._self_model,
                    pattern_field=self._pattern_field,
                    drive_state=tick.state.get("drive_state"),
                    relational_field=self._relational_field,
                    mood=tick.state.get("mood"),
                    previous_narrative=self._narrative,
                    trigger=trigger,
                )
            if self._narrative is not None:
                tick.state["narrative"] = self._narrative
                setattr(sess, "_narrative_snapshot", self._narrative)
        except Exception as exc:
            logger.debug("Narrative construction skipped: %s", exc)

        user_turn = TurnRecord(
            role="user",
            content=message,
            salience=salience,
            emotional_valence=valence,
            importance=importance,
        )
        sess.add_turn(user_turn)

        tick.advance(CognitiveStep.PERCEIVE)

        # Update STM
        tick.assert_step(CognitiveStep.UPDATE_STM)
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
            # Use correct method name or implement fallback
            try:
                update_fn = getattr(self._capture_cache, "update", None)
                if callable(update_fn):
                    stats = update_fn(cm)
                else:
                    update_capture_fn = getattr(self._capture_cache, "update_capture", None)
                    if callable(update_capture_fn):
                        stats = update_capture_fn(cm)
                    else:
                        # Fallback: create stats manually
                        stats = {
                            "frequency": 1,
                            "first_seen_ts": time.time(),
                            "last_seen_ts": time.time(),
                        }
            except Exception:
                stats = {
                    "frequency": 1,
                    "first_seen_ts": time.time(),
                    "last_seen_ts": time.time(),
                }
            default_ts = time.time()
            frequency = self._get_stat_value(stats, "frequency", 1)
            first_seen_ts = self._get_stat_value(stats, "first_seen_ts", default_ts)
            last_seen_ts = self._get_stat_value(stats, "last_seen_ts", default_ts)
            meta = {
                "memory_type": cm.memory_type,
                "subject": (cm.subject or "").lower(),
                "predicate": (cm.predicate or "").lower(),
                "object": (cm.obj or "").lower(),
                "raw_text": cm.raw_text,
                "frequency": frequency,
                "first_seen_ts": first_seen_ts,
                "last_seen_ts": last_seen_ts,
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
                freq = meta.get("frequency", 0)
                if freq in (2, 3, 5, 8):  # key reinforcement milestones
                    # Phase 7: reinforcement is a utility-driven learning effect.
                    # Benefit: milestone-scaled frequency signal.
                    # Cost: small update cost + STM pressure (reinforcement increases downstream retrieval likelihood).
                    stm_pressure = 0.0
                    stm_obj = None
                    try:
                        stm_obj = getattr(self.consolidator, "stm", None) if self.consolidator is not None else None
                        if stm_obj is not None:
                            cap = getattr(stm_obj, "capacity", None)
                            size = None
                            if hasattr(stm_obj, "get_all_memories"):
                                try:
                                    size = len(stm_obj.get_all_memories())  # type: ignore
                                except Exception:
                                    size = None
                            if size is None:
                                size = len(getattr(stm_obj, "items", {}) or {})
                            if cap:
                                stm_pressure = clamp01(float(size) / max(1.0, float(cap)))
                    except Exception:
                        stm_pressure = 0.0

                    milestone_benefit = 0.40 if freq == 2 else 0.50 if freq == 3 else 0.70 if freq == 5 else 0.80
                    u = utility_score(
                        benefit=clamp01(milestone_benefit),
                        cost=clamp01(0.05 + 0.25 * stm_pressure),
                    )
                    u01 = clamp01(u)
                    if u01 > 0.0:
                        meta["reinforced"] = True
                        # Prior boosts preserved as upper-bounds; scaled by utility.
                        max_boost = 0.05 if freq == 2 else 0.07 if freq == 3 else 0.10 if freq == 5 else 0.12
                        boost = max_boost * u01
                        user_turn.importance = min(1.0, (user_turn.importance or 0.0) + boost)
                        metrics_registry.inc("captured_memory_reinforcements_total")

                        # Attempt STM refresh (rehearsal) if stm supports method.
                        if stm_obj is not None:
                            # If STM has method to update activation; else re-store to refresh recency
                            if hasattr(stm_obj, "refresh_item_activation"):
                                try:
                                    key = f"{cm.memory_type}:{meta['subject']}:{meta['object']}"
                                    stm_obj.refresh_item_activation(key)  # type: ignore
                                except Exception as e:
                                    logger.debug("STM refresh_item_activation failed: %s", e)
                            elif hasattr(stm_obj, "store"):
                                try:
                                    key = f"{cm.memory_type}:{meta['subject']}:{meta['object']}"
                                    stm_obj.store(
                                        memory_id=f"rehears-{user_turn.turn_id[:8]}-{key}",
                                        content=cm.content,
                                        importance=user_turn.importance or 0.5,
                                        attention_score=salience,
                                        emotional_valence=valence,
                                    )
                                except Exception as e:
                                    logger.debug("STM rehearsal store failed: %s", e)
            except Exception:
                pass
            # Attempt to store in STM if accessible via consolidator
            try:
                stm_obj = None
                if self.consolidator is not None:
                    stm_obj = getattr(self.consolidator, "stm", None)
                if stm_obj is not None and hasattr(stm_obj, "store"):
                    mem_key = f"{cm.memory_type}:{meta['subject']}:{meta['object']}"
                    stored = stm_obj.store(
                        memory_id=f"cap-{user_turn.turn_id[:8]}-{mem_key}",
                        content=cm.content,
                        importance=importance,
                        attention_score=salience,
                        emotional_valence=valence,
                    )
                    if stored:
                        logger.info("Captured memory stored to STM: %s [%s]", mem_key, cm.memory_type)
                        metrics_registry.inc("stm_capture_stores_total")
                    else:
                        logger.warning("STM store returned False for: %s", mem_key)
                elif stm_obj is not None:
                    logger.warning("STM object has no store() method: %s", type(stm_obj).__name__)
            except Exception as e:
                logger.error("STM capture store failed: %s", e, exc_info=True)
            # Semantic promotion (identity/preference/goal) when frequency threshold hit
            try:
                freq = meta.get("frequency", 0)
                if cm.memory_type in ("identity_fact", "preference", "goal_intent") and freq in (3, 5):
                    # Phase 7: semantic promotion is a learning effect driven by utility.
                    # Benefit: repeated mention (frequency) of high-value memory types.
                    # Cost: write cost + STM pressure.
                    stm_pressure = 0.0
                    try:
                        stm_obj = getattr(self.consolidator, "stm", None) if self.consolidator is not None else None
                        if stm_obj is not None:
                            cap = getattr(stm_obj, "capacity", None)
                            size = None
                            if hasattr(stm_obj, "get_all_memories"):
                                try:
                                    size = len(stm_obj.get_all_memories())  # type: ignore
                                except Exception:
                                    size = None
                            if size is None:
                                size = len(getattr(stm_obj, "items", {}) or {})
                            if cap:
                                stm_pressure = clamp01(float(size) / max(1.0, float(cap)))
                    except Exception:
                        stm_pressure = 0.0

                    benefit = 0.65 if freq == 3 else 0.80
                    # Prefer promoting identity/preference/goal slightly more.
                    if cm.memory_type in ("identity_fact", "preference"):
                        benefit = min(1.0, benefit + 0.05)
                    u01 = clamp01(utility_score(benefit=clamp01(benefit), cost=clamp01(0.15 + 0.20 * stm_pressure)))
                    if u01 <= 0.0:
                        raise RuntimeError("semantic promotion utility <= 0")

                    # Access semantic memory through context_builder if available
                    sem = getattr(self.context_builder, "semantic", None)
                    if sem is not None and hasattr(sem, "store_fact"):
                        promo_meta = {k: meta[k] for k in ("memory_type", "subject", "predicate", "object", "frequency") if k in meta}
                        promo_meta.update({
                            "promotion_reason": "frequency_threshold",
                            "promotion_freq": freq,
                            "promotion_utility": float(u01),
                            "source": "chat_capture",
                        })
                        try:
                            sem.store_fact(content=cm.content, metadata=promo_meta)  # type: ignore
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

            # Phase 7: utility-based learning law.
            # Benefit estimate: rolling retrieval benefit (EMA), default neutral 0.5.
            benefit_est = clamp01(metrics_registry.state.get("retrieval_benefit_ema", 0.5))
            # Cost pressure: latency degradation + STM pressure.
            cost = 0.0
            if degraded_flag:
                cost += 0.6
            if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                cost += 0.6
            cost = clamp01(cost)
            u = utility_score(benefit=benefit_est, cost=cost)

            # Reduce context size when utility indicates cost outweighs benefit.
            if original_max_ctx and isinstance(original_max_ctx, int) and original_max_ctx > 4:
                if u < 0.0:
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

        tick.advance(CognitiveStep.UPDATE_STM)

        # Retrieve
        tick.assert_step(CognitiveStep.RETRIEVE)

        built = self.context_builder.build(
            session=sess,
            query=message,
            include_attention=flags.get("include_attention", True),
            include_memory=flags.get("include_memory", True),
            include_trace=flags.get("include_trace", True),
        )

        # Phase 7: update rolling retrieval benefit signal (EMA) for next turn.
        try:
            hits = 0.0
            try:
                hits += float(getattr(built.metrics, "stm_hits", 0) or 0)
                hits += float(getattr(built.metrics, "ltm_hits", 0) or 0)
                hits += float(getattr(built.metrics, "episodic_hits", 0) or 0)
            except Exception:
                hits = 0.0
            denom = 1.0
            if isinstance(original_max_ctx, int) and original_max_ctx > 0:
                denom = float(original_max_ctx)
            elif hasattr(built, "items") and built.items:
                denom = float(max(1, len(built.items)))
            benefit_now = clamp01(hits / max(1.0, denom))
            prev = metrics_registry.state.get("retrieval_benefit_ema")
            alpha = 0.2
            metrics_registry.state["retrieval_benefit_ema"] = benefit_now if prev is None else (float(prev) + alpha * (benefit_now - float(prev)))
        except Exception:
            pass
        # Restore original context limit after build
        if adaptive_retrieval_applied:
            try:
                cfg["max_context_items"] = original_max_ctx
            except Exception:
                pass

        # Prospective memory injection (non-invasive: append dicts only)
        extra_due: list[Dict[str, Any]] = []
        due_reminders: list[Any] = []
        upcoming_reminders: list[Any] = []
        proactive_due_surface: list[Any] = []
        proactive_upcoming_surface: list[Any] = []
        proactive_summary: Optional[str] = None
        try:
            pm = get_inmemory_prospective_memory()
            # Track injected reminder ids inside session state to avoid double counting
            injected_key = "_prospective_injected_ids"
            injected_ids = getattr(sess, injected_key, set())
            if not isinstance(injected_ids, set):  # safety
                injected_ids = set()
            due_reminders = list(pm.check_due())
            for r in due_reminders:
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

            upcoming_window = float(cfg.get("prospective_upcoming_window_seconds", 1800.0))
            upcoming_limit = int(cfg.get("prospective_upcoming_limit", 3))
            now_dt = datetime.now()
            if upcoming_window > 0:
                raw_upcoming = pm.get_upcoming(within=timedelta(seconds=upcoming_window))
                due_ids = {r.id for r in due_reminders}
                for reminder in raw_upcoming:
                    due_time = getattr(reminder, "due_time", None)
                    if reminder.id in due_ids:
                        continue
                    if due_time is not None and due_time <= now_dt:
                        continue
                    upcoming_reminders.append(reminder)
                if upcoming_limit > 0:
                    upcoming_reminders = upcoming_reminders[:upcoming_limit]

            proactive_ack = getattr(sess, "_proactive_reminder_ack", {})
            if not isinstance(proactive_ack, dict):
                proactive_ack = {}
            ack_cooldown = float(cfg.get("prospective_ack_cooldown_seconds", 600.0))
            now_ts = time.time()
            due_ids = {r.id for r in due_reminders}
            for reminder in due_reminders + upcoming_reminders:
                if not getattr(reminder, "id", None):
                    continue
                last_ack = proactive_ack.get(reminder.id)
                if last_ack is not None and now_ts - last_ack < ack_cooldown:
                    continue
                proactive_ack[reminder.id] = now_ts
                if reminder.id in due_ids:
                    proactive_due_surface.append(reminder)
                else:
                    proactive_upcoming_surface.append(reminder)
            setattr(sess, "_proactive_reminder_ack", proactive_ack)

            if proactive_due_surface or proactive_upcoming_surface:
                proactive_summary = self._format_proactive_reminder_summary(
                    proactive_due_surface,
                    proactive_upcoming_surface,
                )
        except Exception:
            extra_due = []
            due_reminders = []
            upcoming_reminders = []
            proactive_due_surface = []
            proactive_upcoming_surface = []
            proactive_summary = None

        tick.advance(CognitiveStep.RETRIEVE)

        # Decide (single decision owner per tick)
        tick.assert_step(CognitiveStep.DECIDE)

        assistant_content: Optional[str] = None
        # Before forming assistant response, check retrieval for simple questions
        try:
            answer = self._attempt_fact_answer(message)
            if answer:
                assistant_content = answer
        except Exception:
            pass
        
        # Production Phase 1: Enrich response with goal confirmation if detected
        if detected_goal and detected_goal.confirmation_needed:
            goal_confirmation = self._get_goal_handler().format_goal_confirmation(detected_goal)
            if assistant_content:
                assistant_content = assistant_content + "\n\n" + goal_confirmation
            else:
                assistant_content = goal_confirmation
        
        # Production Phase 1 - Multi-intent fan-out (goal + memory queries, etc.)
        intent_sections, intent_execution_log = self._run_intent_handlers(
            intent=intent,
            message=message,
            session_id=sess.session_id,
        )
        assistant_content = self._merge_intent_sections(intent_sections, assistant_content)
        if proactive_summary:
            assistant_content = (proactive_summary + "\n\n" + (assistant_content or "")).strip()

        session_context = self._build_session_context(
            sess.session_id,
            intent,
            detected_goal,
            stored_captures,
            extra_due,
            upcoming_reminders,
        )
        setattr(sess, "_session_context_snapshot", session_context)

        tick.mark_decided({
            "intent": getattr(intent, "intent_type", None) if intent else None,
            "goal_detected": bool(detected_goal),
        })
        tick.advance(CognitiveStep.DECIDE)

        # Act
        tick.assert_step(CognitiveStep.ACT)

        if not assistant_content:
            assistant_content = self._invoke_agent_response(message, built)
        assistant_turn = TurnRecord(
            role="assistant",
            content=assistant_content,
            salience=salience * 0.5,
            emotional_valence=valence * 0.3,
            importance=importance * 0.5,
        )
        sess.add_turn(assistant_turn)

        tick.advance(CognitiveStep.ACT)

        # Reflect
        tick.assert_step(CognitiveStep.REFLECT)

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
                # Phase 7: utility-based learning law.
                # Under high cost pressure, tighten consolidation selectivity (increase salience threshold).
                cost = 0.0
                if degraded:
                    cost += 0.6
                if stm_util is not None and stm_util >= 0.85:
                    cost += 0.6
                u = utility_score(benefit=clamp01(metrics_registry.state.get("retrieval_benefit_ema", 0.5)), cost=clamp01(cost))
                if u < 0.0:
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

        tick.advance(CognitiveStep.REFLECT)

        # Consolidate
        tick.assert_step(CognitiveStep.CONSOLIDATE)
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
                self._last_metacog_snapshot = self._metacog_manager.snapshot(
                    turn_counter=self._turn_counter,
                    consolidation_log=self.consolidation_log,
                    consolidator=self.consolidator,
                )
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
            # Production Phase 1: Intent and goal detection
            "intent": self._serialize_intent(intent) if intent else None,
            "intent_sections": intent_sections,
            "intent_results": intent_execution_log,
            "session_context": session_context,
            "proactive_reminders": {
                "summary": proactive_summary,
                "due": [self._serialize_reminder_brief(r) for r in proactive_due_surface],
                "upcoming": [self._serialize_reminder_brief(r) for r in proactive_upcoming_surface],
            },
            "detected_goal": {
                "goal_id": detected_goal.goal_id,
                "title": detected_goal.title,
                "description": detected_goal.description,
                "deadline": detected_goal.deadline.isoformat() if detected_goal.deadline else None,
                "priority": detected_goal.priority.value if detected_goal.priority else None,
                "estimated_duration_minutes": int(detected_goal.estimated_duration.total_seconds() / 60) if detected_goal.estimated_duration else None,
            } if detected_goal else None,
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
        # Phase 2, Layer 0: Attach drive state to payload
        try:
            ds = tick.state.get("drive_state")
            if ds is not None:
                payload["drives"] = ds.to_dict()
                di = tick.state.get("drive_impact")
                if di is not None:
                    payload["drives"]["impact"] = di.summary()
                dc = tick.state.get("drive_conflicts")
                if dc:
                    payload["drives"]["conflicts"] = [c.describe() for c in dc]
        except Exception:
            pass
        # Phase 2, Layer 1: Attach felt sense and mood to payload
        try:
            mood = tick.state.get("mood")
            if mood is not None:
                payload["mood"] = mood.to_dict()
                payload["mood"]["trend"] = tick.state.get("felt_sense_trend", "stable")
        except Exception:
            pass
        # Phase 2, Layer 2: Attach relational context to payload
        try:
            rel_model = tick.state.get("relational_model")
            if rel_model is not None and rel_model.is_significant(
                self._get_relational_system()[1].config.significant_interaction_threshold
            ):
                payload["relationship"] = rel_model.to_dict()
        except Exception:
            pass
        # Phase 2, Layer 3: Attach emergent patterns to payload
        try:
            pf = tick.state.get("pattern_field")
            if pf is not None and pf.count() > 0:
                payload["patterns"] = pf.to_dict()
        except Exception:
            pass
        # Phase 2, Layer 4: Attach self-model to payload
        try:
            sm = tick.state.get("self_model")
            if sm is not None:
                # Exclude blind spots from user-facing payload
                sm_dict = sm.to_dict()
                sm_dict.pop("_blind_spots", None)
                payload["self_model"] = sm_dict
        except Exception:
            pass
        # Phase 2, Layer 5: Attach narrative to payload
        try:
            narr = tick.state.get("narrative")
            if narr is not None and not narr.is_empty:
                payload["narrative"] = narr.to_dict()
        except Exception:
            pass
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
            self._metacog_interval = self._metacog_manager.adjust_interval(
                current_interval=self._metacog_interval,
                consolidator=self.consolidator,
            )
        except Exception:
            pass
        # --- Adaptive STM activation weight modulation (meta-driven) ---
        try:
            self._metacog_manager.adjust_activation_weights(consolidator=self.consolidator)
        except Exception:
            pass
        tick.finish()
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
        allowed = set(self._INTENT_HANDLER_PRIORITY)
        ordered_candidates: list[str] = []
        if intent.intent_type in allowed:
            ordered_candidates.append(intent.intent_type)
        for intent_name, _conf in intent.secondary_intents:
            if intent_name in allowed and intent_name not in ordered_candidates:
                ordered_candidates.append(intent_name)
        plan = sorted(
            ordered_candidates,
            key=lambda name: (
                self._INTENT_HANDLER_PRIORITY.index(name),
                ordered_candidates.index(name),
            ),
        )
        return plan

    def _run_intent_handlers(
        self,
        intent: IntentV2,
        message: str,
        session_id: str,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """Execute all applicable intent handlers and return sections + execution log."""
        sections: list[Dict[str, Any]] = []
        execution_log: list[Dict[str, Any]] = []
        plan = self._plan_intent_execution(intent)
        for handler_name in plan:
            confidence = self._get_intent_confidence(intent, handler_name)
            handler_callable = None
            if handler_name == "goal_update":
                if self._orchestrator is None:
                    execution_log.append({
                        "intent": handler_name,
                        "confidence": confidence,
                        "handled": False,
                        "response": None,
                        "error": "orchestrator_unavailable",
                        "duration_ms": 0.0,
                    })
                    continue
                handler_callable = partial(self._handle_goal_update, intent, session_id)
            elif handler_name == "goal_query":
                if self._orchestrator is None:
                    execution_log.append({
                        "intent": handler_name,
                        "confidence": confidence,
                        "handled": False,
                        "response": None,
                        "error": "orchestrator_unavailable",
                        "duration_ms": 0.0,
                    })
                    continue
                handler_callable = partial(self._handle_goal_query, intent, session_id)
            elif handler_name == "memory_query":
                handler_callable = partial(self._handle_memory_query, message, session_id)
            elif handler_name == "reminder_request":
                handler_callable = partial(self._handle_reminder_request, intent, session_id)
            elif handler_name == "performance_query":
                handler_callable = partial(self._handle_performance_query, intent, session_id)
            elif handler_name == "system_status":
                handler_callable = partial(self._handle_system_status, intent, session_id)
            if handler_callable is None:
                continue
            metrics_registry.inc(f"intent_handler_{handler_name}_attempts_total")
            start = time.time()
            response_text = None
            error_text = None
            try:
                response_text = handler_callable()
                if response_text:
                    sections.append({
                        "intent": handler_name,
                        "confidence": confidence,
                        "content": response_text,
                    })
                    metrics_registry.inc(f"intent_handler_{handler_name}_handled_total")
            except Exception as exc:  # pragma: no cover - defensive guard
                error_text = str(exc)
                response_text = None
            duration_ms = (time.time() - start) * 1000.0
            execution_log.append({
                "intent": handler_name,
                "confidence": confidence,
                "handled": bool(response_text),
                "response": response_text,
                "error": error_text,
                "duration_ms": duration_ms,
            })
        return sections, execution_log

    def _merge_intent_sections(
        self,
        sections: list[Dict[str, Any]],
        base_response: Optional[str],
    ) -> str:
        """Merge intent-specific sections with the base assistant response."""
        blocks: list[str] = []
        for section in sections:
            header = self._INTENT_SECTION_HEADERS.get(
                section["intent"],
                section["intent"].replace("_", " ").title(),
            )
            content = section.get("content") or ""
            content = content.strip()
            if content:
                blocks.append(f"{header}:\n{content}")
        if base_response:
            stripped = base_response.strip()
            if stripped:
                blocks.append(stripped)
        merged = "\n\n".join(blocks).strip()
        return merged

    def _get_intent_confidence(self, intent: IntentV2, intent_name: str) -> float:
        """Return the classifier confidence for the given intent name."""
        if intent_name == intent.intent_type:
            return intent.confidence
        for secondary_name, secondary_conf in intent.secondary_intents:
            if secondary_name == intent_name:
                return secondary_conf
        return 0.0

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
        active_goals = sorted(self._session_goal_index.get(session_id, set()))
        context = {
            "session_id": session_id,
            "active_goal_ids": active_goals,
            "captured_memory_count": len(stored_captures),
            "prospective_due_count": len(extra_due),
            "prospective_upcoming_count": len(upcoming_reminders),
            "last_intent": intent.intent_type,
            "classifier_context": intent.conversation_context,
        }
        if upcoming_reminders:
            next_due = next((rem for rem in upcoming_reminders if getattr(rem, "due_time", None)), None)
            if next_due is not None:
                due_time = getattr(next_due, "due_time", None)
                context["next_upcoming_reminder"] = {
                    "content": getattr(next_due, "content", ""),
                    "due_time": due_time.isoformat() if due_time else None,
                }
        if detected_goal is not None:
            context["last_detected_goal_id"] = detected_goal.goal_id
        return context

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
        try:
            if isinstance(value, (int, float)):
                return f"{value:.0f} ms"
        except Exception:
            pass
        return "unknown"

    def _format_percentage(self, value: Optional[Any]) -> str:
        """Format ratio as percentage string."""
        try:
            if isinstance(value, (int, float)):
                return f"{value * 100:.0f}%"
        except Exception:
            pass
        return "unknown"

    def _resolve_reminder_due_time(self, intent: IntentV2) -> Optional[datetime]:
        """Convert intent entities into an absolute reminder due time."""
        offset = intent.entities.get("reminder_offset_seconds")
        if offset is not None:
            try:
                seconds = float(offset)
                if seconds > 0:
                    return datetime.now() + timedelta(seconds=seconds)
            except (TypeError, ValueError):
                pass
        due_text = intent.entities.get("reminder_due_time")
        if isinstance(due_text, str):
            try:
                return datetime.fromisoformat(due_text)
            except Exception:
                pass
        return None

    def _format_due_phrase(self, due_time: Optional[datetime]) -> str:
        """Return a friendly phrase for reminder due times."""
        if due_time is None:
            return "no specific time"
        now = datetime.now()
        delta = due_time - now
        total_seconds = delta.total_seconds()
        if total_seconds < -60:
            return "past due"
        if total_seconds < 60:
            return "due now"
        minutes = total_seconds / 60
        if minutes < 60:
            return f"due in {int(round(minutes))} min"
        hours = minutes / 60
        if hours < 24:
            return f"due in {hours:.1f} hr"
        days = hours / 24
        return f"due in {days:.1f} days"

    def _format_proactive_reminder_summary(
        self,
        due_reminders: List[Any],
        upcoming_reminders: List[Any],
    ) -> str:
        lines: List[str] = []
        if due_reminders:
            if len(due_reminders) == 1:
                lines.append(f"🔔 Reminder now due: {getattr(due_reminders[0], 'content', '')}")
            else:
                lines.append("🔔 Reminders now due:")
                for reminder in due_reminders[:3]:
                    lines.append(f" • {getattr(reminder, 'content', '')}")
                if len(due_reminders) > 3:
                    lines.append(f"   (+{len(due_reminders) - 3} more)")
        if upcoming_reminders:
            lines.append("⏰ Coming up soon:")
            for reminder in upcoming_reminders[:3]:
                lines.append(
                    f" • {getattr(reminder, 'content', '')} ({self._format_due_phrase(getattr(reminder, 'due_time', None))})"
                )
            if len(upcoming_reminders) > 3:
                lines.append(f"   (+{len(upcoming_reminders) - 3} more)")
        return "\n".join(lines).strip()

    def _serialize_reminder_brief(self, reminder: Any) -> Dict[str, Any]:
        due_time = getattr(reminder, "due_time", None)
        return {
            "id": getattr(reminder, "id", None),
            "content": getattr(reminder, "content", ""),
            "due_time": due_time.isoformat() if isinstance(due_time, datetime) else None,
            "due_phrase": self._format_due_phrase(due_time) if due_time else "no specific time",
        }

    def _get_stat_value(self, stats: Any, key: str, default: Any) -> float:
        """Safely extract a numeric statistic value with graceful fallback."""
        value: Any = default
        try:
            if isinstance(stats, dict):
                value = stats.get(key, default)
            else:
                value = getattr(stats, key, default)
        except Exception:
            value = default

        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(default)
            except (TypeError, ValueError):
                return 0.0

    def _estimate_importance(self, content: str, salience: float, valence: float) -> float:
        length_factor = min(1.0, len(content.split()) / 30.0)
        emotional_weight = min(1.0, abs(valence))
        return max(salience * 0.5 + length_factor * 0.3 + emotional_weight * 0.2, salience * 0.6)

    # ------------------------------------------------------------------
    # Phase 2, Layer 0: Drive system helpers
    # ------------------------------------------------------------------

    def _get_drive_system(self):
        """Lazy-init and return (DriveState, DriveProcessor) pair."""
        if self._drive_state is None:
            from src.cognition.drives import DriveState, DriveProcessor, DriveConfig
            from src.core.config import get_global_config
            cfg = get_global_config()
            drive_cfg = cfg.drives if cfg.drives is not None else DriveConfig()
            self._drive_processor = DriveProcessor(config=drive_cfg)
            self._drive_state = DriveState()
            self._drive_last_turn_time = time.time()
            logger.info("Drive system initialized: %s", self._drive_state.summary())
        return self._drive_state, self._drive_processor

    def get_drive_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current drive state (for API/telemetry)."""
        if self._drive_state is None:
            return None
        return self._drive_state.to_dict()

    # ------------------------------------------------------------------
    # Phase 2, Layer 1: Felt-sense / mood helpers
    # ------------------------------------------------------------------

    def _get_felt_sense_system(self):
        """Lazy-init and return (FeltSenseGenerator, FeltSenseHistory, MoodLabeler)."""
        if self._felt_sense_generator is None:
            from src.cognition.felt_sense import (
                FeltSenseGenerator, FeltSenseHistory, MoodLabeler, FeltSenseConfig,
            )
            from src.core.config import get_global_config
            cfg = get_global_config()
            fs_cfg = cfg.felt_sense if cfg.felt_sense is not None else FeltSenseConfig()
            self._felt_sense_generator = FeltSenseGenerator(config=fs_cfg)
            self._felt_sense_history = FeltSenseHistory(max_size=fs_cfg.history_size)
            self._mood_labeler = MoodLabeler(config=fs_cfg)
            logger.info("Felt-sense system initialized")
        return self._felt_sense_generator, self._felt_sense_history, self._mood_labeler

    def get_mood_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current mood (for API/telemetry)."""
        if self._current_mood is None:
            return None
        return self._current_mood.to_dict()

    # ------------------------------------------------------------------
    # Phase 2, Layer 2: Relational field helpers
    # ------------------------------------------------------------------

    def _get_relational_system(self):
        """Lazy-init and return (RelationalField, RelationalProcessor) pair."""
        if self._relational_field is None:
            from src.cognition.relational import (
                RelationalField, RelationalProcessor, RelationalConfig,
            )
            from src.core.config import get_global_config
            cfg = get_global_config()
            rel_cfg = cfg.relational if cfg.relational is not None else RelationalConfig()
            self._relational_field = RelationalField()
            self._relational_processor = RelationalProcessor(config=rel_cfg)
            logger.info("Relational field initialized")
        return self._relational_field, self._relational_processor

    def get_relational_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for current relational field (for API/telemetry)."""
        if self._relational_field is None:
            return None
        return self._relational_field.to_dict()

    # ------------------------------------------------------------------
    # Phase 2, Layer 3: Emergent patterns helpers
    # ------------------------------------------------------------------

    def _get_pattern_system(self):
        """Lazy-init and return (PatternField, PatternDetector) pair."""
        if self._pattern_field is None:
            from src.cognition.patterns import (
                PatternField, PatternDetector, PatternConfig,
            )
            from src.core.config import get_global_config
            cfg = get_global_config()
            pat_cfg = cfg.patterns if cfg.patterns is not None else PatternConfig()
            self._pattern_field = PatternField(max_patterns=pat_cfg.max_patterns)
            self._pattern_detector = PatternDetector(config=pat_cfg)
            logger.info("Pattern system initialized")
        return self._pattern_field, self._pattern_detector

    def get_pattern_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for emergent patterns (for API/telemetry)."""
        if self._pattern_field is None:
            return None
        return self._pattern_field.to_dict()

    # ------------------------------------------------------------------
    # Phase 2, Layer 4: Self-model helpers
    # ------------------------------------------------------------------

    def _get_self_model_system(self):
        """Lazy-init and return SelfModelBuilder."""
        if self._self_model_builder is None:
            from src.cognition.selfmodel import (
                SelfModelBuilder, SelfModelConfig,
            )
            from src.core.config import get_global_config
            cfg = get_global_config()
            sm_cfg = cfg.selfmodel if cfg.selfmodel is not None else SelfModelConfig()
            self._self_model_builder = SelfModelBuilder(config=sm_cfg)
            logger.info("Self-model system initialized")
        return self._self_model_builder

    def get_self_model_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for self-model (for API/telemetry).

        Blind spots are excluded from the public-facing dict.
        """
        if self._self_model is None:
            return None
        d = self._self_model.to_dict()
        d.pop("_blind_spots", None)
        return d

    # ------------------------------------------------------------------
    # Phase 2, Layer 5: Narrative helpers
    # ------------------------------------------------------------------

    def _get_narrative_system(self):
        """Lazy-init and return NarrativeConstructor."""
        if self._narrative_constructor is None:
            from src.cognition.narrative import (
                NarrativeConstructor, NarrativeConfig,
            )
            from src.core.config import get_global_config
            cfg = get_global_config()
            narr_cfg = cfg.narrative if cfg.narrative is not None else NarrativeConfig()
            self._narrative_constructor = NarrativeConstructor(config=narr_cfg)
            logger.info("Narrative system initialized")
        return self._narrative_constructor

    def get_narrative_state(self) -> Optional[Dict[str, Any]]:
        """Public accessor for narrative (for API/telemetry)."""
        if self._narrative is None:
            return None
        return self._narrative.to_dict()

    def _handle_memory_query(self, message: str, session_id: str) -> Optional[str]:
        """
        Handle memory query intent using MemoryQueryParser and Interface.
        
        Args:
            message: User's query message
            session_id: Current session ID
            
        Returns:
            Formatted memory response or None
        """
        try:
            # Parse the query
            query_result = self._memory_query_parser.parse_query(message)
            
            # Lazy init memory query interface with actual memory systems
            if self._memory_query_interface is None:
                # Get memory systems from consolidator
                stm = None
                ltm = None
                episodic = None
                
                if self.consolidator:
                    stm = getattr(self.consolidator, 'stm', None)
                    ltm = getattr(self.consolidator, 'ltm', None)
                    episodic = getattr(self.consolidator, 'episodic', None)
                
                self._memory_query_interface = create_memory_query_interface(
                    stm=stm,
                    ltm=ltm,
                    episodic=episodic
                )
            
            # Execute query
            response = self._memory_query_interface.execute_query(query_result)
            
            # Format for chat
            formatted = self._memory_query_interface.format_response(response)
            
            return formatted
        
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error handling memory query: {e}", exc_info=True)
            return None

    def _handle_reminder_request(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """Create or list reminders using the prospective memory subsystem."""
        try:
            pm = get_inmemory_prospective_memory()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Prospective memory unavailable: %s", exc, exc_info=True)
            return None

        action = (intent.entities.get("reminder_action") or "create").lower()

        if action in ("list", "show"):
            reminders = pm.list_reminders(include_completed=False)
            if not reminders:
                return "You don't have any active reminders."
            lines = ["Active reminders:"]
            for reminder in reminders[:5]:
                lines.append(f"- {reminder.content} ({self._format_due_phrase(getattr(reminder, 'due_time', None))})")
            if len(reminders) > 5:
                lines.append(f"…and {len(reminders) - 5} more")
            return "\n".join(lines)

        if action in ("due", "upcoming"):
            window = intent.entities.get("reminder_upcoming_window_seconds")
            reminders = []
            if action == "due":
                reminders = pm.get_due_reminders()
            elif window:
                try:
                    seconds = float(window)
                    reminders = pm.get_upcoming(within=timedelta(seconds=seconds))
                except Exception:
                    reminders = pm.get_upcoming(within=timedelta(hours=1))
            else:
                reminders = pm.get_upcoming(within=timedelta(hours=1))
            if not reminders:
                return "No reminders are due right now."
            lines = ["Reminders due soon:"]
            for reminder in reminders[:5]:
                lines.append(f"- {reminder.content} ({self._format_due_phrase(getattr(reminder, 'due_time', None))})")
            return "\n".join(lines)

        # Default: create a new reminder
        reminder_text = intent.entities.get("reminder_text") or intent.entities.get("reminder_description")
        if not reminder_text:
            return "Tell me what you'd like me to remind you about."

        due_time = self._resolve_reminder_due_time(intent)
        reminder = pm.add_reminder(reminder_text, due_time=due_time)
        due_phrase = self._format_due_phrase(getattr(reminder, "due_time", None))
        return f"Reminder set: '{reminder.content}' ({due_phrase})."

    def _handle_performance_query(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """Return a formatted performance snapshot for performance_query intents."""
        try:
            status = self.performance_status()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Performance status lookup failed: %s", exc, exc_info=True)
            return None

        if not isinstance(status, dict):
            return None

        metric_focus = intent.entities.get("metric_type", "general")
        latency_text = self._format_latency_ms(status.get("latency_p95_ms"))
        target_text = self._format_latency_ms(status.get("target_p95_ms"))
        ema_text = self._format_latency_ms(status.get("ema_turn_latency_ms"))
        throughput = status.get("chat_turns_per_sec")
        if throughput is None:
            throughput_text = "unknown throughput"
        else:
            try:
                throughput_text = f"{float(throughput):.2f} turns/sec"
            except (TypeError, ValueError):
                throughput_text = "unknown throughput"
        degraded = bool(status.get("performance_degraded"))
        health_text = "⚠️ Running slower than target" if degraded else "✅ Meeting latency target"

        lines = [
            f"- Latency p95: {latency_text} (target {target_text})",
            f"- EMA latency: {ema_text}",
            f"- Throughput: {throughput_text}",
            f"- Status: {health_text}",
        ]
        if metric_focus and metric_focus != "general":
            focus_text = metric_focus.replace("_", " ")
            lines.append(f"- Metric focus: watching {focus_text} trends")
        return "\n".join(lines)

    def _handle_system_status(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """Summarize current cognitive system health for system_status intents."""
        detail_level = intent.entities.get("detail_level", "normal")
        component_focus = intent.entities.get("component", "general")
        active_goals = len(self._session_goal_index.get(session_id, set()))
        size, capacity, util = self._get_stm_usage_snapshot()
        if size is None and capacity is None:
            stm_line = "STM load: unavailable (not configured)"
        elif capacity:
            stm_line = f"STM load: {size}/{capacity} items ({self._format_percentage(util)})"
        else:
            stm_line = f"STM load: {size or 0} items"

        perf_snapshot = None
        try:
            perf_snapshot = self.performance_status()
        except Exception:
            perf_snapshot = None

        latency_line = None
        if isinstance(perf_snapshot, dict):
            latency_line = f"Latency: {self._format_latency_ms(perf_snapshot.get('latency_p95_ms'))}"
            if perf_snapshot.get("performance_degraded"):
                latency_line += " (⚠️ above target)"
            else:
                latency_line += " (✅ stable)"

        due_count = 0
        upcoming_hour = 0
        try:
            pm = get_inmemory_prospective_memory()
            reminders = pm.list_reminders(include_completed=False)
            now = datetime.now()
            for rem in reminders:
                due_time = getattr(rem, "due_time", None)
                if not due_time:
                    continue
                delta = (due_time - now).total_seconds()
                if delta <= 0:
                    due_count += 1
                elif delta <= 3600:
                    upcoming_hour += 1
        except Exception:
            pass

        lines = [
            f"- Active goals: {active_goals}",
            f"- {stm_line}",
            f"- Prospective reminders: {due_count} due, {upcoming_hour} within an hour",
            f"- Metacog cadence: every {max(1, self._metacog_interval)} turns",
        ]

        if latency_line and component_focus in ("general", "performance", "cognitive"):
            lines.append(f"- {latency_line}")

        if detail_level == "detailed":
            ema_text = self._format_latency_ms(perf_snapshot.get("ema_turn_latency_ms") if isinstance(perf_snapshot, dict) else None)
            lines.append(f"- EMA latency: {ema_text}")
            lines.append(f"- Consolidation history: {len(self.consolidation_log)} recent events")

        return "\n".join(lines)

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

        result = None
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
            if result is not None and asyncio.iscoroutine(result):
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
        sorted_items = sorted(
            enumerate(items),
            key=lambda pair: (
                pair[1].rank if pair[1].rank is not None else 1_000_000,
                pair[1].source_system or "",
                pair[1].content or "",
                pair[0],
            ),
        )
        out = []
        for _idx, ci in sorted_items[:max_items]:
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
