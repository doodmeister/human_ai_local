from __future__ import annotations

from datetime import datetime, timedelta
import logging

from src.executive.planning.goap_planner import Plan
from src.executive.scheduling import Schedule


logger = logging.getLogger(__name__)


class ExecutiveReminderStage:
    def create_reminders_from_schedule(self, goal_id: str, schedule: Schedule, plan: Plan) -> int:
        try:
            from src.memory.prospective.prospective_memory import create_prospective_memory, get_prospective_memory

            try:
                prospective = get_prospective_memory()
            except Exception:
                prospective = create_prospective_memory(use_vector=False)

            if prospective is None:
                logger.warning("Prospective memory not available - skipping reminder creation")
                return 0

            action_descriptions = {}
            for index, step in enumerate(plan.steps):
                action_name = getattr(step, "name", str(step))
                action_descriptions[action_name] = {
                    "description": getattr(step, "description", action_name),
                    "sequence": index + 1,
                    "total": len(plan.steps),
                }

            reminders_created = 0
            base_time = datetime.now()
            for task in schedule.tasks:
                if task.scheduled_start is not None:
                    due_time = base_time + timedelta(minutes=task.scheduled_start)
                else:
                    due_time = base_time + timedelta(hours=reminders_created + 1)

                action_info = action_descriptions.get(task.id, {})
                seq = action_info.get("sequence", reminders_created + 1)
                total = action_info.get("total", len(schedule.tasks))
                task_name = getattr(task, "name", task.id)
                content = f"[Step {seq}/{total}] {task_name}"

                reminder = prospective.add_reminder(
                    content=content,
                    due_time=due_time,
                    tags=["auto-generated", "plan-step", f"goal:{goal_id}"],
                    metadata={
                        "goal_id": goal_id,
                        "task_id": task.id,
                        "sequence": seq,
                        "total_steps": total,
                        "auto_generated": True,
                        "source": "executive_system",
                    },
                )

                if reminder:
                    reminders_created += 1
                    logger.debug("Created reminder: %s due at %s", content, due_time)

            logger.info("Created %s reminders for goal %s", reminders_created, goal_id)
            return reminders_created
        except Exception as exc:
            logger.warning("Failed to create reminders from schedule: %s", exc)
            return 0