"""
Proactive Agency Module

Enables George to proactively suggest actions based on:
- Due/overdue reminders
- Upcoming plan steps
- Goal deadlines
- Inactivity periods

This module bridges the executive system with user engagement,
making the system feel more like a proactive assistant.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of proactive suggestions."""
    DUE_REMINDER = "due_reminder"
    UPCOMING_STEP = "upcoming_step"
    GOAL_DEADLINE = "goal_deadline"
    INACTIVITY_CHECK = "inactivity_check"
    GOAL_PROGRESS = "goal_progress"


@dataclass
class ProactiveSuggestion:
    """A proactive suggestion for the user."""
    type: SuggestionType
    message: str
    priority: float  # 0.0 to 1.0
    source_id: Optional[str] = None  # reminder_id, goal_id, etc.
    action_hint: Optional[str] = None  # Suggested action
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "message": self.message,
            "priority": self.priority,
            "source_id": self.source_id,
            "action_hint": self.action_hint,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


class ProactiveAgencySystem:
    """
    Monitors user state and generates proactive suggestions.
    
    Designed to make George feel more like a proactive assistant
    that anticipates user needs rather than waiting passively.
    """
    
    def __init__(
        self,
        prospective: Any = None,
        executive: Any = None,
        inactivity_threshold_minutes: int = 30,
    ):
        self.prospective = prospective
        self.executive = executive
        self.inactivity_threshold = timedelta(minutes=inactivity_threshold_minutes)
        self.last_user_activity: Optional[datetime] = None
        self._suggestions_cache: List[ProactiveSuggestion] = []
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=30)  # Refresh every 30s
        
    def record_user_activity(self) -> None:
        """Record that the user interacted with the system."""
        self.last_user_activity = datetime.now()
    
    def get_suggestions(self, force_refresh: bool = False) -> List[ProactiveSuggestion]:
        """
        Get current proactive suggestions, sorted by priority.
        
        Args:
            force_refresh: Force recalculation even if cache is valid
            
        Returns:
            List of suggestions sorted by priority (highest first)
        """
        now = datetime.now()
        
        # Check cache validity
        if not force_refresh and self._cache_time:
            if now - self._cache_time < self._cache_ttl:
                return self._suggestions_cache
        
        suggestions: List[ProactiveSuggestion] = []
        
        # Check due reminders
        suggestions.extend(self._check_due_reminders())
        
        # Check upcoming plan steps
        suggestions.extend(self._check_upcoming_steps())
        
        # Check goal deadlines
        suggestions.extend(self._check_goal_deadlines())
        
        # Check for inactivity
        if self._check_inactivity():
            suggestions.append(self._create_inactivity_suggestion())
        
        # Sort by priority (highest first)
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        
        # Update cache
        self._suggestions_cache = suggestions
        self._cache_time = now
        
        return suggestions
    
    def get_urgent_suggestions(self) -> List[ProactiveSuggestion]:
        """Get only high-priority suggestions (priority >= 0.8)."""
        return [s for s in self.get_suggestions() if s.priority >= 0.8]
    
    def format_suggestions_for_llm(self, max_items: int = 3) -> str:
        """
        Format suggestions as a context string for the LLM.
        
        Args:
            max_items: Maximum number of suggestions to include
            
        Returns:
            Formatted string for LLM context injection
        """
        suggestions = self.get_suggestions()[:max_items]
        
        if not suggestions:
            return ""
        
        lines = ["PROACTIVE SUGGESTIONS:"]
        for s in suggestions:
            priority_icon = "🔴" if s.priority >= 0.9 else "🟡" if s.priority >= 0.7 else "🟢"
            lines.append(f"{priority_icon} {s.message}")
            if s.action_hint:
                lines.append(f"   → Suggested: {s.action_hint}")
        
        return "\n".join(lines)
    
    def _check_due_reminders(self) -> List[ProactiveSuggestion]:
        """Check for due and overdue reminders."""
        suggestions = []
        
        if not self.prospective:
            return suggestions
        
        try:
            due_reminders = self.prospective.get_due_reminders()
            
            for reminder in due_reminders[:5]:  # Limit to 5
                # Calculate how overdue
                overdue_minutes = 0
                if reminder.due_time:
                    overdue_minutes = (datetime.now() - reminder.due_time).total_seconds() / 60
                
                # Higher priority for more overdue items
                priority = min(1.0, 0.8 + (overdue_minutes / 60) * 0.2)
                
                # Check if it's a plan step
                is_plan_step = reminder.metadata.get("auto_generated") or "plan-step" in reminder.tags
                
                if is_plan_step:
                    message = f"Plan step due: {reminder.content}"
                    action_hint = "Would you like help completing this step?"
                else:
                    message = f"Reminder due: {reminder.content}"
                    action_hint = "Shall I help you with this?"
                
                if overdue_minutes > 5:
                    message = f"OVERDUE ({int(overdue_minutes)}min): {reminder.content}"
                    priority = min(1.0, priority + 0.1)
                
                suggestions.append(ProactiveSuggestion(
                    type=SuggestionType.DUE_REMINDER,
                    message=message,
                    priority=priority,
                    source_id=reminder.id,
                    action_hint=action_hint,
                    metadata={
                        "reminder_id": reminder.id,
                        "is_plan_step": is_plan_step,
                        "overdue_minutes": overdue_minutes
                    }
                ))
                
        except Exception as e:
            logger.debug(f"Error checking due reminders: {e}")
        
        return suggestions
    
    def _check_upcoming_steps(self) -> List[ProactiveSuggestion]:
        """Check for upcoming plan steps that need preparation."""
        suggestions = []
        
        if not self.prospective:
            return suggestions
        
        try:
            upcoming = self.prospective.get_upcoming(within=timedelta(minutes=30))
            
            # Only include plan steps that are coming up soon
            for reminder in upcoming:
                if not (reminder.metadata.get("auto_generated") or "plan-step" in reminder.tags):
                    continue
                
                if reminder.due_time:
                    minutes_until = (reminder.due_time - datetime.now()).total_seconds() / 60
                    
                    if minutes_until <= 15:  # Within 15 minutes
                        suggestions.append(ProactiveSuggestion(
                            type=SuggestionType.UPCOMING_STEP,
                            message=f"Coming up in {int(minutes_until)}min: {reminder.content}",
                            priority=0.7,
                            source_id=reminder.id,
                            action_hint="Start preparing for this step?",
                            metadata={
                                "reminder_id": reminder.id,
                                "minutes_until": minutes_until
                            }
                        ))
                        
        except Exception as e:
            logger.debug(f"Error checking upcoming steps: {e}")
        
        return suggestions
    
    def _check_goal_deadlines(self) -> List[ProactiveSuggestion]:
        """Check for goals with approaching deadlines."""
        suggestions = []
        
        if not self.executive:
            return suggestions
        
        try:
            goal_manager = getattr(self.executive, "goal_manager", None)
            if not goal_manager:
                return suggestions
            
            # Get active goals
            active_goals = []
            if hasattr(goal_manager, "get_active_goals"):
                active_goals = goal_manager.get_active_goals()
            elif hasattr(goal_manager, "goals"):
                active_goals = [
                    g for g in goal_manager.goals.values() 
                    if g.status in ("pending", "in_progress", "active")
                ]
            
            for goal in active_goals:
                deadline = getattr(goal, "deadline", None)
                if not deadline:
                    continue
                
                # Parse deadline if string
                if isinstance(deadline, str):
                    try:
                        deadline = datetime.fromisoformat(deadline)
                    except ValueError:
                        continue
                
                time_until = deadline - datetime.now()
                hours_until = time_until.total_seconds() / 3600
                
                if hours_until < 0:
                    # Overdue
                    suggestions.append(ProactiveSuggestion(
                        type=SuggestionType.GOAL_DEADLINE,
                        message=f"OVERDUE GOAL: {goal.title}",
                        priority=1.0,
                        source_id=goal.id,
                        action_hint="This goal is past its deadline. What should we do?",
                        metadata={"goal_id": goal.id, "hours_overdue": abs(hours_until)}
                    ))
                elif hours_until <= 4:
                    # Within 4 hours
                    suggestions.append(ProactiveSuggestion(
                        type=SuggestionType.GOAL_DEADLINE,
                        message=f"Goal deadline approaching ({int(hours_until)}h): {goal.title}",
                        priority=0.85,
                        source_id=goal.id,
                        action_hint="Focus on this goal to meet the deadline?",
                        metadata={"goal_id": goal.id, "hours_until": hours_until}
                    ))
                    
        except Exception as e:
            logger.debug(f"Error checking goal deadlines: {e}")
        
        return suggestions
    
    def _check_inactivity(self) -> bool:
        """Check if user has been inactive too long."""
        if not self.last_user_activity:
            return False
        
        return datetime.now() - self.last_user_activity > self.inactivity_threshold
    
    def _create_inactivity_suggestion(self) -> ProactiveSuggestion:
        """Create suggestion for inactive user."""
        return ProactiveSuggestion(
            type=SuggestionType.INACTIVITY_CHECK,
            message="You've been quiet for a while. Need any help with your goals?",
            priority=0.5,
            action_hint="I can summarize your pending tasks or suggest next steps.",
            metadata={"minutes_inactive": int((datetime.now() - self.last_user_activity).total_seconds() / 60)}
        )


# Singleton instance
_proactive_system: Optional[ProactiveAgencySystem] = None


def get_proactive_system() -> ProactiveAgencySystem:
    """Get or create the singleton proactive agency system."""
    global _proactive_system
    if _proactive_system is None:
        # Try to get prospective and executive from existing singletons
        prospective = None
        executive = None
        
        try:
            from src.memory.prospective.prospective_memory import get_prospective_memory
            prospective = get_prospective_memory()
        except Exception:
            pass
        
        try:
            from src.executive.executive_agent import ExecutiveAgent
            executive = ExecutiveAgent()
        except Exception:
            pass
        
        _proactive_system = ProactiveAgencySystem(
            prospective=prospective,
            executive=executive
        )
    
    return _proactive_system


def create_proactive_system(
    prospective: Any = None,
    executive: Any = None,
    **kwargs
) -> ProactiveAgencySystem:
    """Create a new proactive agency system with custom components."""
    return ProactiveAgencySystem(
        prospective=prospective,
        executive=executive,
        **kwargs
    )
