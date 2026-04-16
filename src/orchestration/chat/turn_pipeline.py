from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.learning.learning_law import clamp01, utility_score
from src.orchestration import CognitiveStep, CognitiveTick

from .emotion_salience import estimate_salience_and_valence
from .executive_orchestrator import ExecutiveOrchestrator
from .goal_detector import GoalDetector
from .metrics import metrics_registry
from .models import TurnRecord
from .provenance import build_item_provenance
from .scoring import get_scoring_profile_version

if TYPE_CHECKING:
    from .chat_service import ChatService


logger = logging.getLogger(__name__)


class ChatTurnPipeline:
    def process_user_message(
        self,
        service: ChatService,
        message: str,
        session_id: Optional[str] = None,
        flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        t_start = time.time()
        flags = flags or {}
        sess = service.sessions.create_or_get(session_id)

        tick = CognitiveTick(owner="chat_service", kind="chat_turn")
        tick.state["session_id"] = sess.session_id

        tick.assert_step(CognitiveStep.PERCEIVE)

        classifier = service._get_intent_classifier(sess.session_id)
        try:
            intent = classifier.classify(message, use_context=True)
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc, exc_info=True)
            intent = service._fallback_intent(message)
        detected_goal = None
        intent_sections: list[Dict[str, Any]] = []
        intent_execution_log: list[Dict[str, Any]] = []

        if intent.intent_type == "goal_creation":
            try:
                if service._goal_detector is None:
                    from src.executive.integration import ExecutiveSystem

                    executive_system = ExecutiveSystem()
                    service._goal_detector = GoalDetector(executive_system)

                detected_goal = service._goal_detector.detect_goal(
                    message,
                    sess.session_id,
                    intent=intent,
                )
                if detected_goal:
                    metrics_registry.inc("goals_auto_detected_total")
                    service._record_session_goal(sess.session_id, detected_goal.goal_id)
                    classifier.context.active_goals = set(service._session_goal_index.get(sess.session_id, set()))

                    if service._orchestrator is None:
                        executive_system = getattr(service._goal_detector, "executive", None)
                        if executive_system is None:
                            from src.executive.integration import ExecutiveSystem

                            executive_system = ExecutiveSystem()
                        service._orchestrator = ExecutiveOrchestrator(executive_system)
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None:
                        loop.create_task(service._orchestrator.execute_goal_async(detected_goal.goal_id))
            except Exception as exc:
                print(f"Goal detection error: {exc}")
                detected_goal = None

        salience, valence = estimate_salience_and_valence(message)
        importance = service._estimate_importance(message, salience, valence)

        service._cognitive_layers.process_turn(
            session=sess,
            tick=tick,
            message=message,
            salience=salience,
            valence=valence,
            global_turn_counter=service._turn_counter,
        )

        user_turn = TurnRecord(
            role="user",
            content=message,
            salience=salience,
            emotional_valence=valence,
            importance=importance,
        )
        sess.add_turn(user_turn)

        tick.advance(CognitiveStep.PERCEIVE)

        tick.assert_step(CognitiveStep.UPDATE_STM)
        try:
            captures = service._capture.extract(message)
        except Exception:
            captures = []
        stored_captures: list[dict] = []
        for cm in captures:
            prior_objs = set()
            if cm.memory_type == "identity_fact" and cm.subject:
                try:
                    subj_l = (cm.subject or "").lower()
                    for rec in service._capture_cache.as_list():
                        if rec.get("memory_type") == "identity_fact" and (rec.get("subject") or "").lower() == subj_l:
                            pobj = (rec.get("object") or "").lower()
                            if pobj:
                                prior_objs.add(pobj)
                except Exception:
                    prior_objs = set()
            try:
                update_fn = getattr(service._capture_cache, "update", None)
                if callable(update_fn):
                    stats = update_fn(cm)
                else:
                    update_capture_fn = getattr(service._capture_cache, "update_capture", None)
                    if callable(update_capture_fn):
                        stats = update_capture_fn(cm)
                    else:
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
            frequency = service._get_stat_value(stats, "frequency", 1)
            first_seen_ts = service._get_stat_value(stats, "first_seen_ts", default_ts)
            last_seen_ts = service._get_stat_value(stats, "last_seen_ts", default_ts)
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
            if cm.memory_type == "identity_fact" and meta["subject"] and meta["object"]:
                try:
                    obj_l = meta["object"]
                    if prior_objs and obj_l not in prior_objs:
                        meta["contradiction"] = True
                        meta["contradicted_prior"] = list(prior_objs)[:5]
                        metrics_registry.inc("captured_memory_contradictions_total")
                except Exception:
                    pass
            try:
                freq = meta.get("frequency", 0)
                if freq in (2, 3, 5, 8):
                    stm_pressure = 0.0
                    stm_obj = None
                    try:
                        stm_obj = getattr(service.consolidator, "stm", None) if service.consolidator is not None else None
                        if stm_obj is not None:
                            cap = getattr(stm_obj, "capacity", None)
                            size = None
                            if hasattr(stm_obj, "get_all_memories"):
                                try:
                                    size = len(stm_obj.get_all_memories())
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
                        max_boost = 0.05 if freq == 2 else 0.07 if freq == 3 else 0.10 if freq == 5 else 0.12
                        boost = max_boost * u01
                        user_turn.importance = min(1.0, (user_turn.importance or 0.0) + boost)
                        metrics_registry.inc("captured_memory_reinforcements_total")

                        if stm_obj is not None:
                            if hasattr(stm_obj, "refresh_item_activation"):
                                try:
                                    key = f"{cm.memory_type}:{meta['subject']}:{meta['object']}"
                                    stm_obj.refresh_item_activation(key)
                                except Exception as exc:
                                    logger.debug("STM refresh_item_activation failed: %s", exc)
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
                                except Exception as exc:
                                    logger.debug("STM rehearsal store failed: %s", exc)
            except Exception:
                pass
            try:
                stm_obj = None
                if service.consolidator is not None:
                    stm_obj = getattr(service.consolidator, "stm", None)
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
            except Exception as exc:
                logger.error("STM capture store failed: %s", exc, exc_info=True)
            try:
                freq = meta.get("frequency", 0)
                if cm.memory_type in ("identity_fact", "preference", "goal_intent") and freq in (3, 5):
                    stm_pressure = 0.0
                    try:
                        stm_obj = getattr(service.consolidator, "stm", None) if service.consolidator is not None else None
                        if stm_obj is not None:
                            cap = getattr(stm_obj, "capacity", None)
                            size = None
                            if hasattr(stm_obj, "get_all_memories"):
                                try:
                                    size = len(stm_obj.get_all_memories())
                                except Exception:
                                    size = None
                            if size is None:
                                size = len(getattr(stm_obj, "items", {}) or {})
                            if cap:
                                stm_pressure = clamp01(float(size) / max(1.0, float(cap)))
                    except Exception:
                        stm_pressure = 0.0

                    benefit = 0.65 if freq == 3 else 0.80
                    if cm.memory_type in ("identity_fact", "preference"):
                        benefit = min(1.0, benefit + 0.05)
                    u01 = clamp01(utility_score(benefit=clamp01(benefit), cost=clamp01(0.15 + 0.20 * stm_pressure)))
                    if u01 <= 0.0:
                        raise RuntimeError("semantic promotion utility <= 0")

                    sem = getattr(service.context_builder, "semantic", None)
                    if sem is not None and hasattr(sem, "store_fact") and cm.subject and cm.predicate:
                        promo_meta = {k: meta[k] for k in ("memory_type", "subject", "predicate", "object", "frequency") if k in meta}
                        promo_meta.update(
                            {
                                "promotion_reason": "frequency_threshold",
                                "promotion_freq": freq,
                                "promotion_utility": float(u01),
                                "source": "explicit_user_correction" if meta.get("contradiction") else "user_assertion",
                                "confidence": min(0.95, 0.68 + (0.06 * freq) + (0.10 * u01)),
                                "importance": max(0.55, min(1.0, importance)),
                            }
                        )
                        try:
                            fact_id = sem.store_fact(
                                subject=cm.subject,
                                predicate=cm.predicate,
                                object_val=cm.obj,
                                content=cm.content,
                                metadata=promo_meta,
                            )
                            meta["promoted_semantic"] = bool(fact_id)
                            if fact_id:
                                meta["semantic_fact_id"] = fact_id
                            get_revision_event = getattr(sem, "get_last_revision_event", None)
                            if callable(get_revision_event):
                                revision_event = get_revision_event()
                                if revision_event:
                                    meta["semantic_revision"] = revision_event
                                    if revision_event.get("revision_reason") == "supersedes_conflicting_belief":
                                        metrics_registry.inc("captured_memory_correction_revisions_total")
                            metrics_registry.inc("captured_memory_semantic_promotions_total")
                        except Exception:
                            pass
            except Exception:
                pass
            stored_captures.append({"content": cm.content, **meta})
        if stored_captures:
            try:
                metrics_registry.state["last_captured_memories"] = stored_captures[-5:]
                metrics_registry.inc("captured_memory_units_total", len(stored_captures))
            except Exception:
                pass

        cfg = service.context_builder.cfg
        original_max_ctx = cfg.get("max_context_items")
        adaptive_retrieval_applied = False
        try:
            degraded_flag = metrics_registry.state.get("performance_degraded")
            stm_util_ratio = None
            cons = getattr(service, "consolidator", None)
            if cons is not None:
                stm_obj = getattr(cons, "stm", None)
                if stm_obj is not None:
                    cap = getattr(stm_obj, "capacity", None)
                    size = None
                    if hasattr(stm_obj, "__len__"):
                        try:
                            size = len(stm_obj)
                        except Exception:
                            size = None
                    if size is None:
                        size = getattr(stm_obj, "size", None)
                    if isinstance(size, int) and isinstance(cap, int) and cap:
                        stm_util_ratio = min(1.0, size / cap)

            benefit_est = clamp01(metrics_registry.state.get("retrieval_benefit_ema", 0.5))
            cost = 0.0
            if degraded_flag:
                cost += 0.6
            if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                cost += 0.6
            cost = clamp01(cost)
            u = utility_score(benefit=benefit_est, cost=cost)

            if original_max_ctx and isinstance(original_max_ctx, int) and original_max_ctx > 4:
                if u < 0.0:
                    reduce_factor = 1.0
                    if degraded_flag:
                        reduce_factor *= 0.75
                    if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                        reduce_factor *= 0.75
                    if reduce_factor < 0.999:
                        new_limit = max(4, int(original_max_ctx * reduce_factor))
                        if new_limit < original_max_ctx:
                            cfg["max_context_items"] = new_limit
                            adaptive_retrieval_applied = True
                            metrics_registry.inc("adaptive_retrieval_applied_total")
        except Exception:
            pass

        tick.advance(CognitiveStep.UPDATE_STM)

        tick.assert_step(CognitiveStep.RETRIEVE)

        built = service.context_builder.build(
            session=sess,
            query=message,
            include_attention=flags.get("include_attention", True),
            include_memory=flags.get("include_memory", True),
            include_trace=flags.get("include_trace", True),
        )

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
        if adaptive_retrieval_applied:
            try:
                cfg["max_context_items"] = original_max_ctx
            except Exception:
                pass

        extra_due: list[Dict[str, Any]] = []
        due_reminders: list[Any] = []
        upcoming_reminders: list[Any] = []
        proactive_due_surface: list[Any] = []
        proactive_upcoming_surface: list[Any] = []
        proactive_summary: Optional[str] = None
        try:
            pm = service._get_prospective_memory()
            injected_key = "_prospective_injected_ids"
            injected_ids = getattr(sess, injected_key, set())
            if not isinstance(injected_ids, set):
                injected_ids = set()
            due_reminders = list(pm.check_due())
            for reminder in due_reminders:
                extra_due.append(
                    {
                        "source_id": f"reminder-{reminder.id}",
                        "source_system": "prospective",
                        "reason": "due_reminder",
                        "rank": 0,
                        "content": f"REMINDER: {reminder.content}",
                        "scores": {"reminder": 1.0, "composite": 1.0},
                    }
                )
                if reminder.id not in injected_ids:
                    metrics_registry.inc("prospective_reminders_injected_total")
                    injected_ids.add(reminder.id)
            setattr(sess, injected_key, injected_ids)

            upcoming_window = float(cfg.get("prospective_upcoming_window_seconds", 1800.0))
            upcoming_limit = int(cfg.get("prospective_upcoming_limit", 3))
            now_dt = datetime.now(timezone.utc)
            if upcoming_window > 0:
                raw_upcoming = pm.get_upcoming(within=timedelta(seconds=upcoming_window))
                due_ids = {reminder.id for reminder in due_reminders}
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
            due_ids = {reminder.id for reminder in due_reminders}
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
                proactive_summary = service._format_proactive_reminder_summary(
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

        tick.assert_step(CognitiveStep.DECIDE)

        assistant_content: Optional[str] = None
        try:
            answer = service._attempt_fact_answer(message)
            if answer:
                assistant_content = answer
        except Exception:
            pass

        if detected_goal and detected_goal.confirmation_needed:
            goal_confirmation = service._get_goal_handler().format_goal_confirmation(detected_goal)
            if assistant_content:
                assistant_content = assistant_content + "\n\n" + goal_confirmation
            else:
                assistant_content = goal_confirmation

        intent_sections, intent_execution_log = service._run_intent_handlers(
            intent=intent,
            message=message,
            session_id=sess.session_id,
        )
        assistant_content = service._merge_intent_sections(intent_sections, assistant_content)
        if proactive_summary:
            assistant_content = (proactive_summary + "\n\n" + (assistant_content or "")).strip()

        session_context = service._build_session_context(
            sess.session_id,
            intent,
            detected_goal,
            stored_captures,
            extra_due,
            upcoming_reminders,
        )
        setattr(sess, "_session_context_snapshot", session_context)

        tick.mark_decided(
            {
                "intent": getattr(intent, "intent_type", None) if intent else None,
                "goal_detected": bool(detected_goal),
            }
        )
        tick.advance(CognitiveStep.DECIDE)

        tick.assert_step(CognitiveStep.ACT)

        if not assistant_content:
            assistant_content = service._invoke_agent_response(message, built)
        assistant_turn = TurnRecord(
            role="assistant",
            content=assistant_content,
            salience=salience * 0.5,
            emotional_valence=valence * 0.3,
            importance=importance * 0.5,
        )
        sess.add_turn(assistant_turn)

        tick.advance(CognitiveStep.ACT)

        tick.assert_step(CognitiveStep.REFLECT)

        t_cons = time.time()
        adaptive_cfg = None
        original_sal_thr = None
        original_val_thr = None
        try:
            adaptive_cfg = getattr(service, "context_builder").cfg
            original_sal_thr = adaptive_cfg.get("consolidation_salience_threshold")
            original_val_thr = adaptive_cfg.get("consolidation_valence_threshold")
            stm_util = None
            cons = getattr(service, "consolidator", None)
            if cons is not None:
                stm_obj = getattr(cons, "stm", None)
                if stm_obj is not None:
                    cap = getattr(stm_obj, "capacity", None)
                    size = None
                    if hasattr(stm_obj, "__len__"):
                        try:
                            size = len(stm_obj)
                        except Exception:
                            size = None
                    if size is None:
                        size = getattr(stm_obj, "size", None)
                    if isinstance(size, int) and isinstance(cap, int) and cap > 0:
                        stm_util = size / cap
            degraded = metrics_registry.state.get("performance_degraded")
            if adaptive_cfg is not None:
                base_sal = adaptive_cfg.get("consolidation_salience_threshold", 0.55)
                base_val = adaptive_cfg.get("consolidation_valence_threshold", 0.60)
                sal_adj = base_sal
                val_adj = base_val
                cost = 0.0
                if degraded:
                    cost += 0.6
                if stm_util is not None and stm_util >= 0.85:
                    cost += 0.6
                u = utility_score(
                    benefit=clamp01(metrics_registry.state.get("retrieval_benefit_ema", 0.5)),
                    cost=clamp01(cost),
                )
                if u < 0.0:
                    if stm_util is not None and stm_util >= 0.85:
                        sal_adj += 0.05
                    if degraded:
                        sal_adj += 0.05
                sal_adj = min(sal_adj, 0.85)
                adaptive_cfg["consolidation_salience_threshold"] = sal_adj
                adaptive_cfg["consolidation_valence_threshold"] = val_adj
        except Exception:
            pass

        tick.advance(CognitiveStep.REFLECT)

        tick.assert_step(CognitiveStep.CONSOLIDATE)
        stored = service._maybe_consolidate(user_turn, assistant_turn)
        autobiographical_episode_id = None
        if stored:
            try:
                autobiographical_episode_id = service._promote_autobiographical_turn(
                    session=sess,
                    user_turn=user_turn,
                    assistant_turn=assistant_turn,
                    intent=intent,
                    tick=tick,
                )
            except Exception:
                autobiographical_episode_id = None
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
        service.consolidation_log.append(
            {
                "user_turn_id": user_turn.turn_id,
                "status": user_turn.consolidation_status,
                "salience": user_turn.salience,
                "valence": user_turn.emotional_valence,
                "timestamp": time.time(),
            }
        )
        service._turn_counter += 1
        attach_metacog = False
        if service._metacog_interval > 0 and (service._turn_counter % service._metacog_interval == 0):
            try:
                service._last_metacog_snapshot = service._metacog_manager.snapshot(
                    turn_counter=service._turn_counter,
                    consolidation_log=service.consolidation_log,
                    consolidator=service.consolidator,
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
            "captured_memories": stored_captures,
            "intent": service._serialize_intent(intent) if intent else None,
            "intent_sections": intent_sections,
            "intent_results": intent_execution_log,
            "session_context": session_context,
            "proactive_reminders": {
                "summary": proactive_summary,
                "due": [service._serialize_reminder_brief(reminder) for reminder in proactive_due_surface],
                "upcoming": [service._serialize_reminder_brief(reminder) for reminder in proactive_upcoming_surface],
            },
            "detected_goal": {
                "goal_id": detected_goal.goal_id,
                "title": detected_goal.title,
                "description": detected_goal.description,
                "deadline": detected_goal.deadline.isoformat() if detected_goal.deadline else None,
                "priority": detected_goal.priority.value if detected_goal.priority else None,
                "estimated_duration_minutes": int(detected_goal.estimated_duration.total_seconds() / 60) if detected_goal.estimated_duration else None,
            }
            if detected_goal
            else None,
            "metrics": {
                "turn_latency_ms": built.metrics.turn_latency_ms,
                "retrieval_time_ms": built.metrics.retrieval_time_ms,
                "stm_hits": built.metrics.stm_hits,
                "ltm_hits": built.metrics.ltm_hits,
                "episodic_hits": built.metrics.episodic_hits,
                "fallback_used": built.metrics.fallback_used,
                "consolidation_time_ms": built.metrics.consolidation_time_ms,
                "consolidated_user_turn": built.metrics.consolidated_user_turn,
                "user_salience": user_turn.salience,
                "user_valence": user_turn.emotional_valence,
                "user_importance": user_turn.importance,
                "consolidation_status": user_turn.consolidation_status,
                "autobiographical_episode_id": autobiographical_episode_id,
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
            ]
            + extra_due,
        }
        try:
            ds = tick.state.get("drive_state")
            if ds is not None:
                payload["drives"] = ds.to_dict()
                di = tick.state.get("drive_impact")
                if di is not None:
                    payload["drives"]["impact"] = di.summary()
                dc = tick.state.get("drive_conflicts")
                if dc:
                    payload["drives"]["conflicts"] = [conflict.describe() for conflict in dc]
        except Exception:
            pass
        try:
            mood = tick.state.get("mood")
            if mood is not None:
                payload["mood"] = mood.to_dict()
                payload["mood"]["trend"] = tick.state.get("felt_sense_trend", "stable")
        except Exception:
            pass
        try:
            rel_model = tick.state.get("relational_model")
            if rel_model is not None and rel_model.is_significant(
                service._get_relational_system()[1].config.significant_interaction_threshold
            ):
                payload["relationship"] = rel_model.to_dict()
        except Exception:
            pass
        try:
            pf = tick.state.get("pattern_field")
            if pf is not None and pf.count() > 0:
                payload["patterns"] = pf.to_dict()
        except Exception:
            pass
        try:
            sm = tick.state.get("self_model")
            if sm is not None:
                sm_dict = sm.to_dict()
                sm_dict.pop("_blind_spots", None)
                payload["self_model"] = sm_dict
        except Exception:
            pass
        try:
            narr = tick.state.get("narrative")
            if narr is not None and not narr.is_empty:
                payload["narrative"] = narr.to_dict()
        except Exception:
            pass
        try:
            response_policy = tick.state.get("response_policy")
            if response_policy is not None:
                payload["response_policy"] = response_policy.to_dict()
        except Exception:
            pass
        if attach_metacog and service._last_metacog_snapshot:
            payload["metacog"] = service._last_metacog_snapshot
            try:
                ltm = getattr(service.context_builder, "ltm", None)
                snap = dict(service._last_metacog_snapshot)
                snap["type"] = "meta_reflection"
                try:
                    service._metacog_history.append(snap)
                except Exception:
                    pass
                if ltm is not None and hasattr(ltm, "add_item"):
                    content = (
                        f"Metacog snapshot turn={snap.get('turn_counter')} "
                        f"perf_p95={snap.get('performance', {}).get('latency_p95_ms')} "
                        f"stm_util={snap.get('stm_utilization')}"
                    )
                    try:
                        ltm.add_item(content=content, metadata=snap)
                    except Exception:
                        pass
            except Exception:
                pass
        if flags.get("include_trace"):
            payload["trace"] = {
                "pipeline": [stage.__dict__ for stage in built.trace.pipeline_stages],
                "notes": list(built.trace.notes),
            }
            payload["trace"]["provenance_details"] = build_item_provenance(built.items)
            payload["trace"]["scoring_version"] = get_scoring_profile_version()
            if any(ci.source_system == "attention" for ci in built.items):
                payload["trace"].setdefault("notes", []).append("attention_focus items included")
            if any(ci.source_system == "executive" for ci in built.items):
                payload["trace"].setdefault("notes", []).append("executive_mode state included")
            payload["trace"]["consolidation_log_tail"] = service.consolidation_log[-10:]

        total_lat = (time.time() - t_start) * 1000.0
        payload["metrics"]["turn_latency_ms"] = total_lat
        metrics_registry.observe_hist("chat_turn_latency_ms", total_lat)
        ema_alpha = 0.2
        prev_ema = metrics_registry.state.get("ema_turn_latency_ms")
        ema = total_lat if prev_ema is None else prev_ema + ema_alpha * (total_lat - prev_ema)
        metrics_registry.state["ema_turn_latency_ms"] = ema
        payload["metrics"]["ema_turn_latency_ms"] = ema
        metrics_registry.mark_event("chat_turn")
        window = service.context_builder.cfg.get("throughput_window_seconds", 60)
        tps = metrics_registry.get_rate("chat_turn", window_seconds=float(window))
        metrics_registry.state["chat_turns_per_sec"] = tps
        payload["metrics"]["chat_turns_per_sec"] = tps
        snap = metrics_registry.snapshot()
        if "chat_turn_latency_p95_ms" in snap.get("state", {}):
            p95 = snap["state"]["chat_turn_latency_p95_ms"]
            payload["metrics"]["latency_p95_ms"] = p95
            target = service.context_builder.cfg.get("performance_target_p95_ms")
            if isinstance(target, (int, float)) and target > 0:
                degraded = p95 > target
                metrics_registry.state["performance_degraded"] = degraded
                payload["metrics"]["performance_degraded"] = degraded
        try:
            service._metacog_interval = service._metacog_manager.adjust_interval(
                current_interval=service._metacog_interval,
                consolidator=service.consolidator,
            )
        except Exception:
            pass
        try:
            service._metacog_manager.adjust_activation_weights(consolidator=service.consolidator)
        except Exception:
            pass
        tick.finish()
        return payload