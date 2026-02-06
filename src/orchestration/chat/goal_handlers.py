from __future__ import annotations

from typing import Optional, Any
from datetime import datetime, timedelta

from .intent_classifier_v2 import IntentV2


class GoalIntentHandler:
    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator

    @property
    def orchestrator(self) -> Any:
        return self._orchestrator

    def format_goal_confirmation(self, detected_goal: Any) -> str:
        """
        Format a natural language goal confirmation for the user.

        Production Phase 1: Automatic goal detection integration.
        """
        confirmation = f"\n\n\U0001F3AF **Goal Created**: {detected_goal.title}"

        if detected_goal.deadline:
            deadline_date = detected_goal.deadline
            today = datetime.now().date()
            if deadline_date.date() == today:
                deadline_str = "today"
            elif deadline_date.date() == (today + timedelta(days=1)):
                deadline_str = "tomorrow"
            else:
                days_until = (deadline_date.date() - today).days
                if days_until <= 7:
                    deadline_str = deadline_date.strftime("%A")
                else:
                    deadline_str = deadline_date.strftime("%B %d")

            confirmation += f" (Due: {deadline_str})"

        if hasattr(detected_goal, "priority") and detected_goal.priority:
            try:
                from src.executive.goal_manager import GoalPriority
                if detected_goal.priority == GoalPriority.HIGH:
                    confirmation += " \u26a1"
            except Exception:
                pass

        if detected_goal.estimated_duration:
            minutes = int(detected_goal.estimated_duration.total_seconds() / 60)
            if minutes < 60:
                confirmation += f"\n\U0001F4CA Estimated time: {minutes} minutes"
            else:
                hours = minutes / 60
                confirmation += f"\n\U0001F4CA Estimated time: {hours:.1f} hours"

        confirmation += (
            f"\n\nI'll track this goal for you. "
            f"Ask me 'How's my {detected_goal.title[:20]}...' to check progress anytime!"
        )

        return confirmation

    def handle_goal_query(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """
        Handle goal status queries (Task 6).

        Args:
            intent: Classified intent with goal_query type
            session_id: Current session ID

        Returns:
            Formatted status text or None if no goals found
        """
        if not self._orchestrator:
            return None

        query_subtype = intent.entities.get("query_subtype", "general")

        if query_subtype == "list_all":
            return self._format_all_goals_for_chat()

        if query_subtype == "execution_progress":
            return self._format_execution_progress_for_chat()

        goal_identifier = (
            intent.entities.get("goal_identifier")
            or intent.entities.get("goal_reference")
        )
        if not goal_identifier:
            return self._format_all_goals_for_chat()

        try:
            goal_manager = self._get_goal_manager()
            if goal_manager is None:
                active_executions = self._orchestrator.list_active_executions()
                if not active_executions:
                    return "You don't have any active goals at the moment."
                return self._orchestrator.format_status_for_chat(active_executions[0].goal_id)

            matching_goals = []
            for goal_id, goal in goal_manager.goals.items():
                if goal_identifier.lower() in goal.title.lower():
                    matching_goals.append(goal)

            if not matching_goals:
                return f"I couldn't find a goal matching '{goal_identifier}'."

            goal = matching_goals[0]
            return self._orchestrator.format_status_for_chat(goal.goal_id)
        except Exception:
            active_executions = self._orchestrator.list_active_executions()
            if not active_executions:
                return "You don't have any active goals at the moment."
            return self._orchestrator.format_status_for_chat(active_executions[0].goal_id)

    def handle_goal_update(self, intent: IntentV2, session_id: str) -> Optional[str]:
        """
        Handle goal update intent (complete, cancel, change priority/deadline).

        Args:
            intent: Classified intent with goal_update entities
            session_id: Current session ID

        Returns:
            Confirmation message or error
        """
        try:
            if not self._orchestrator:
                return "Goal management is not available right now."

            goal_manager = self._get_goal_manager()
            if not goal_manager:
                return "Goal management is not available right now."

            entities = intent.entities if hasattr(intent, "entities") else {}
            goal_reference = entities.get("goal_reference", "")
            update_action = entities.get("update_action", "")
            new_priority = entities.get("new_priority")
            new_deadline = entities.get("new_deadline")

            if not goal_reference:
                return (
                    "I couldn't identify which goal you want to update. "
                    "Try being more specific, like 'mark the report goal as complete'."
                )

            if not update_action:
                return (
                    "I'm not sure what you want to do with that goal. "
                    "You can complete it, cancel it, or change its priority or deadline."
                )

            all_goals = goal_manager.goals if hasattr(goal_manager, "goals") else {}
            matching_goal = None
            matching_goal_id = None
            goal_reference_lower = goal_reference.lower()

            for goal_id, goal in all_goals.items():
                goal_title = (
                    getattr(goal, "title", str(goal)).lower()
                    if hasattr(goal, "title")
                    else str(goal).lower()
                )
                if goal_reference_lower in goal_title or goal_title in goal_reference_lower:
                    matching_goal = goal
                    matching_goal_id = goal_id
                    break

            if not matching_goal:
                for goal_id, goal in all_goals.items():
                    goal_title = (
                        getattr(goal, "title", str(goal)).lower()
                        if hasattr(goal, "title")
                        else str(goal).lower()
                    )
                    ref_words = set(goal_reference_lower.split())
                    title_words = set(goal_title.split())
                    if ref_words & title_words:
                        matching_goal = goal
                        matching_goal_id = goal_id
                        break

            if not matching_goal:
                return (
                    f"I couldn't find a goal matching '{goal_reference}'. "
                    "Try saying 'check my goals' to see your current goals."
                )

            goal_title = getattr(matching_goal, "title", str(matching_goal))

            if update_action == "complete":
                if hasattr(goal_manager, "complete_goal"):
                    goal_manager.complete_goal(matching_goal_id)
                elif hasattr(matching_goal, "status"):
                    matching_goal.status = "completed"
                return f"\u2705 Marked '{goal_title}' as complete. Nice work!"

            if update_action == "cancel":
                if hasattr(goal_manager, "cancel_goal"):
                    goal_manager.cancel_goal(matching_goal_id)
                elif hasattr(goal_manager, "remove_goal"):
                    goal_manager.remove_goal(matching_goal_id)
                elif hasattr(matching_goal, "status"):
                    matching_goal.status = "cancelled"
                return f"\U0001F6AB Cancelled '{goal_title}'."

            if update_action == "priority_change":
                if new_priority and hasattr(goal_manager, "update_goal_priority"):
                    goal_manager.update_goal_priority(matching_goal_id, new_priority)
                    return f"\U0001F4CA Updated priority of '{goal_title}' to {new_priority}."
                if new_priority and hasattr(matching_goal, "priority"):
                    matching_goal.priority = new_priority
                    return f"\U0001F4CA Updated priority of '{goal_title}' to {new_priority}."
                return f"I couldn't update the priority. Try saying 'set {goal_title} to high priority'."

            if update_action == "deadline_change":
                if new_deadline and hasattr(goal_manager, "update_goal_deadline"):
                    goal_manager.update_goal_deadline(matching_goal_id, new_deadline)
                    return f"\U0001F4C5 Updated deadline of '{goal_title}' to {new_deadline}."
                if new_deadline and hasattr(matching_goal, "deadline"):
                    matching_goal.deadline = new_deadline
                    return f"\U0001F4C5 Updated deadline of '{goal_title}' to {new_deadline}."
                return f"I couldn't update the deadline. Try saying 'change {goal_title} deadline to Friday'."

            return (
                f"I'm not sure how to '{update_action}' a goal. "
                "You can complete, cancel, or change the priority or deadline."
            )

        except Exception as e:
            return f"I had trouble updating the goal: {str(e)}"

    def _get_goal_manager(self) -> Any:
        if not self._orchestrator:
            return None
        if hasattr(self._orchestrator, "executive_system"):
            return self._orchestrator.executive_system.goal_manager
        if hasattr(self._orchestrator, "_executive_system"):
            return getattr(self._orchestrator, "_executive_system").goal_manager
        return None

    def _format_all_goals_for_chat(self) -> str:
        """
        Format all goals for chat display.

        Returns:
            Formatted string with all goals
        """
        try:
            goal_manager = self._get_goal_manager()

            if not goal_manager or not goal_manager.goals:
                return (
                    "You don't have any goals at the moment. "
                    "Try saying something like 'I need to finish the report by Friday' to create one!"
                )

            active_goals = []
            completed_goals = []
            failed_goals = []

            for goal_id, goal in goal_manager.goals.items():
                status = getattr(goal, "status", "unknown")
                if hasattr(status, "value") and not isinstance(status, str):
                    status = status.value
                status_lower = str(status).lower()

                goal_info = {
                    "id": goal_id,
                    "title": goal.title,
                    "priority": getattr(goal, "priority", "medium"),
                    "deadline": getattr(goal, "deadline", None),
                    "status": status_lower,
                }

                if status_lower in ["completed", "done"]:
                    completed_goals.append(goal_info)
                elif status_lower in ["failed", "cancelled"]:
                    failed_goals.append(goal_info)
                else:
                    active_goals.append(goal_info)

            lines = ["\U0001F4CB **Your Goals**\n"]

            if active_goals:
                lines.append("**Active:**")
                for g in active_goals:
                    priority_emoji = {"high": "\U0001F534", "medium": "\U0001F7E1", "low": "\U0001F7E2"}.get(
                        str(getattr(g["priority"], "value", g["priority"]))
                        .lower(),
                        "\u26aa",
                    )
                    deadline_str = ""
                    if g["deadline"]:
                        if hasattr(g["deadline"], "strftime"):
                            deadline_str = f" (due: {g['deadline'].strftime('%b %d')})"
                        else:
                            deadline_str = f" (due: {str(g['deadline'])[:10]})"
                    lines.append(f"  {priority_emoji} {g['title']}{deadline_str}")

            if completed_goals:
                lines.append("\n**Completed:** \u2705")
                for g in completed_goals[:3]:
                    lines.append(f"  \u2713 {g['title']}")
                if len(completed_goals) > 3:
                    lines.append(f"  \u2026and {len(completed_goals) - 3} more")

            if not active_goals and not completed_goals:
                return (
                    "You don't have any goals yet. "
                    "Try saying something like 'I need to finish the report by Friday' to create one!"
                )

            lines.append(f"\n*{len(active_goals)} active, {len(completed_goals)} completed*")
            lines.append("\nAsk me about a specific goal for details, or say 'I need to...' to create a new one.")

            return "\n".join(lines)

        except Exception as e:
            return f"I had trouble getting your goals. Error: {str(e)}"

    def _format_execution_progress_for_chat(self) -> str:
        """
        Format all active goal executions for chat display.

        Returns:
            Formatted string with execution progress
        """
        try:
            if not self._orchestrator:
                return "Goal execution is not available right now."

            active_executions = self._orchestrator.list_active_executions()

            if not active_executions:
                return (
                    "No goals are currently being executed. "
                    "All your goals are either waiting to start or already complete."
                )

            lines = ["\u2699\ufe0f **Currently Executing:**\n"]

            for progress in active_executions:
                phase_emoji = {
                    "queued": "\u23f3",
                    "deciding": "\U0001F914",
                    "planning": "\U0001F4DD",
                    "scheduling": "\U0001F4C5",
                    "executing": "\u25b6\ufe0f",
                    "completed": "\u2705",
                    "failed": "\u274c",
                    "cancelled": "\U0001F6AB",
                }.get(progress.phase.value, "\u2699\ufe0f")

                lines.append(f"{phase_emoji} **{progress.goal_title}**")
                lines.append(f"   Phase: {progress.phase.value.title()}")

                if progress.total_actions > 0:
                    lines.append(
                        f"   Progress: {progress.progress_percent:.0f}% "
                        f"({progress.actions_completed}/{progress.total_actions} steps)"
                    )

                if progress.current_action:
                    lines.append(f"   Current: {progress.current_action}")

                if hasattr(progress, "start_time") and progress.start_time:
                    elapsed = datetime.now() - progress.start_time
                    if elapsed.total_seconds() < 60:
                        lines.append(f"   Running: {int(elapsed.total_seconds())}s")
                    else:
                        lines.append(f"   Running: {int(elapsed.total_seconds() / 60)}m")

                lines.append("")

            lines.append(f"*{len(active_executions)} goal(s) in progress*")
            lines.append("\nSay 'check my goals' to see all goals, or ask about a specific one for details.")

            return "\n".join(lines)

        except Exception as e:
            return f"I had trouble getting execution progress: {str(e)}"
