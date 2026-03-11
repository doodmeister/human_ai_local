from __future__ import annotations

import asyncio
from datetime import datetime
import re
from typing import Any, Callable, Dict, Optional

from .metrics import metrics_registry


class ChatTurnSupport:
    def __init__(
        self,
        get_agent: Callable[[], Any],
        get_capture_records: Callable[[], list[dict]],
        get_consolidator: Callable[[], Any],
        get_config: Callable[[], Dict[str, Any]],
    ) -> None:
        self._get_agent = get_agent
        self._get_capture_records = get_capture_records
        self._get_consolidator = get_consolidator
        self._get_config = get_config

    def estimate_importance(self, content: str, salience: float, valence: float) -> float:
        length_factor = min(1.0, len(content.split()) / 30.0)
        emotional_weight = min(1.0, abs(valence))
        return max(salience * 0.5 + length_factor * 0.3 + emotional_weight * 0.2, salience * 0.6)

    def invoke_agent_response(self, message: str, built: Any) -> str:
        agent = self._get_agent()
        if not agent or not hasattr(agent, "_generate_response"):
            return "[LLM unavailable - agent not configured with ChatService]"

        memory_context = []
        try:
            for item in list(getattr(built, "items", [])[:5]):
                scores = getattr(item, "scores", {}) or {}
                relevance = scores.get("composite") or scores.get("similarity", 0.0)
                metadata = getattr(item, "metadata", {}) or {}
                memory_context.append(
                    {
                        "id": getattr(item, "source_id", ""),
                        "content": getattr(item, "content", ""),
                        "source": getattr(item, "source_system", "unknown"),
                        "relevance": relevance,
                        "timestamp": metadata.get("timestamp") or getattr(item, "timestamp", None),
                    }
                )
        except Exception:
            memory_context = []

        result = None
        try:
            result = agent._generate_response(
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

    def maybe_consolidate(self, user_turn: Any, assistant_turn: Any) -> bool:
        consolidator = self._get_consolidator()
        cfg = self._get_config()
        if consolidator:
            try:
                sal_thr = cfg.get("consolidation_salience_threshold", 0.55)
                val_thr = cfg.get("consolidation_valence_threshold", 0.60)

                original_sal = consolidator.policy.salience_threshold
                original_val = consolidator.policy.valence_threshold
                consolidator.policy.salience_threshold = sal_thr
                consolidator.policy.valence_threshold = val_thr

                ev = consolidator.record_turn(
                    turn_id=user_turn.turn_id,
                    salience=user_turn.salience or 0.0,
                    valence=user_turn.emotional_valence or 0.0,
                    importance=user_turn.importance or 0.0,
                    content=user_turn.content,
                )

                consolidator.policy.salience_threshold = original_sal
                consolidator.policy.valence_threshold = original_val

                consolidator.mark_rehearsal(user_turn.turn_id)
                if ev.stored_in_stm:
                    user_turn.consolidation_status = "stored"
                    metrics_registry.inc("consolidated_stored_total")
                    return True

                user_turn.consolidation_status = "skipped"
                metrics_registry.inc("consolidated_skipped_total")
                return False
            except Exception:
                pass

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

    def attempt_fact_answer(self, message: str) -> Optional[str]:
        msg = message.strip().lower()
        subject = None
        match = re.match(r"^(who|what) is ([^?]+)\??$", msg)
        if match:
            subject = match.group(2).strip()
        if subject is None:
            match = re.match(r"^tell me about ([^?]+)\??$", msg)
            if match:
                subject = match.group(1).strip()
        if subject is None:
            match = re.match(r"^what does ([^?]+) do\??$", msg)
            if match:
                subject = match.group(1).strip()
        if subject is None:
            return None

        subject_lower = subject.lower()
        for record in reversed(self._get_capture_records()):
            record_subject = record.get("subject") or ""
            if record_subject and (subject_lower == record_subject or subject_lower in record_subject):
                memory_type = record.get("memory_type")
                obj = record.get("object")
                if memory_type == "identity_fact" and obj:
                    return f"{subject.title()} is {obj}."
                if memory_type == "preference" and obj:
                    return f"{subject.title()} likes {obj}."
                if memory_type == "goal_intent" and obj:
                    return f"{subject.title()} intends to {obj}."
        return None

    def _run_coroutine_sync(self, coro: Any) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            coro.close()
            raise