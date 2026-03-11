from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any, Callable, Optional

from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory

from .memory_query_interface import create_memory_query_interface


logger = logging.getLogger(__name__)


class ChatIntentActions:
    def __init__(
        self,
        memory_query_parser: Any,
        get_consolidator: Callable[[], Any],
        format_due_phrase: Callable[[Any], str],
        resolve_reminder_due_time: Callable[[Any], Any],
    ) -> None:
        self._memory_query_parser = memory_query_parser
        self._get_consolidator = get_consolidator
        self._format_due_phrase = format_due_phrase
        self._resolve_reminder_due_time = resolve_reminder_due_time
        self._memory_query_interface = None

    def handle_memory_query(self, message: str, session_id: str) -> Optional[str]:
        del session_id
        try:
            query_result = self._memory_query_parser.parse_query(message)

            if self._memory_query_interface is None:
                stm = None
                ltm = None
                episodic = None
                consolidator = self._get_consolidator()
                if consolidator:
                    stm = getattr(consolidator, "stm", None)
                    ltm = getattr(consolidator, "ltm", None)
                    episodic = getattr(consolidator, "episodic", None)

                self._memory_query_interface = create_memory_query_interface(
                    stm=stm,
                    ltm=ltm,
                    episodic=episodic,
                )

            response = self._memory_query_interface.execute_query(query_result)
            return self._memory_query_interface.format_response(response)
        except Exception as exc:
            logger.error("Error handling memory query: %s", exc, exc_info=True)
            return None

    def handle_reminder_request(self, intent: Any, session_id: str) -> Optional[str]:
        del session_id
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

        reminder_text = intent.entities.get("reminder_text") or intent.entities.get("reminder_description")
        if not reminder_text:
            return "Tell me what you'd like me to remind you about."

        due_time = self._resolve_reminder_due_time(intent)
        reminder = pm.add_reminder(reminder_text, due_time=due_time)
        due_phrase = self._format_due_phrase(getattr(reminder, "due_time", None))
        return f"Reminder set: '{reminder.content}' ({due_phrase})."