from __future__ import annotations

from typing import Any, Dict, Iterable


class ChatTurnContextBuilder:
    """Build lightweight session context snapshots for chat payloads."""

    def __init__(self, *, get_active_goal_ids: callable) -> None:
        self._get_active_goal_ids = get_active_goal_ids

    def build_session_context(
        self,
        *,
        session_id: str,
        intent: Any,
        detected_goal: Any | None,
        stored_captures: list[Dict[str, Any]],
        extra_due: list[Dict[str, Any]],
        upcoming_reminders: Iterable[Any],
    ) -> Dict[str, Any]:
        active_goals = sorted(self._get_active_goal_ids(session_id))
        reminder_list = list(upcoming_reminders)
        context = {
            "session_id": session_id,
            "active_goal_ids": active_goals,
            "captured_memory_count": len(stored_captures),
            "prospective_due_count": len(extra_due),
            "prospective_upcoming_count": len(reminder_list),
            "last_intent": intent.intent_type,
            "classifier_context": intent.conversation_context,
        }
        if reminder_list:
            next_due = next((rem for rem in reminder_list if getattr(rem, "due_time", None)), None)
            if next_due is not None:
                due_time = getattr(next_due, "due_time", None)
                context["next_upcoming_reminder"] = {
                    "content": getattr(next_due, "content", ""),
                    "due_time": due_time.isoformat() if due_time else None,
                }
        if detected_goal is not None:
            context["last_detected_goal_id"] = detected_goal.goal_id
        return context