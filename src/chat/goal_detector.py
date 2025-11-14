"""
Goal Detection Engine for Chat-First Integration

Automatically detects and creates goals from natural language conversation.
Integrates with Executive System for seamless goal management.
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta, date

from src.chat.intent_classifier_v2 import (
    IntentClassifierV2,
    IntentV2,
    ConversationContext,
    create_intent_classifier_v2,
)
from src.executive.integration import ExecutiveSystem
from src.executive.goal_manager import GoalPriority


@dataclass
class DetectedGoal:
    """Represents a goal detected from conversation"""
    goal_id: str
    title: str
    description: str
    deadline: Optional[datetime]
    priority: GoalPriority
    success_criteria: List[str]
    estimated_duration: Optional[timedelta]
    detected: bool = True
    confirmation_needed: bool = True


class GoalDetector:
    """Detects and extracts goals from natural language"""
    
    def __init__(self, executive_system: Optional[ExecutiveSystem] = None):
        """
        Initialize goal detector.
        
        Args:
            executive_system: Executive system for goal creation (optional, for testing)
        """
        self.executive = executive_system or ExecutiveSystem()
        self._intent_classifiers: Dict[str, IntentClassifierV2] = {}
    
    def detect_goal(
        self,
        message: str,
        session_id: str,
        user_context: Optional[Dict] = None,
        intent: Optional[IntentV2] = None,
    ) -> Optional[DetectedGoal]:
        """
        Analyzes message and creates goal if detected.
        
        Args:
            message: User's natural language message
            session_id: Current session ID
            user_context: Additional context (current goals, preferences, etc.)
            
        Returns:
            DetectedGoal if goal detected and created, None otherwise
        """
        # Classify intent
        if intent is None:
            intent_classifier = self._get_intent_classifier(session_id)
            intent = intent_classifier.classify(message)
        
        # Only process goal creation intents
        if intent.intent_type != 'goal_creation':
            return None
        
        # Extract goal details from entities
        entities = intent.entities
        
        # Generate title from description
        title = self._generate_title(entities.get('goal_description', message))
        
        # Parse deadline
        deadline = self._parse_deadline(entities.get('deadline'))
        
        # Infer priority
        priority = self._infer_priority(entities, deadline)
        
        # Generate success criteria
        success_criteria = self._generate_success_criteria(entities, message)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(entities, title)
        
        # Create goal in executive system
        try:
            goal_ref = self.executive.goal_manager.create_goal(
                title=title,
                description=entities.get('goal_description', message),
                priority=priority,
                deadline=deadline
            )
            goal_id: str
            if isinstance(goal_ref, str):
                goal_id = goal_ref
            elif hasattr(goal_ref, "goal_id"):
                goal_id = getattr(goal_ref, "goal_id")
            elif hasattr(goal_ref, "id"):
                goal_id = getattr(goal_ref, "id")
            else:
                goal_id = str(goal_ref)
            
            return DetectedGoal(
                goal_id=goal_id,
                title=title,
                description=entities.get('goal_description', message),
                deadline=deadline,
                priority=priority,
                success_criteria=success_criteria,
                estimated_duration=estimated_duration,
                detected=True,
                confirmation_needed=True
            )
        
        except Exception as e:
            # Log error but don't fail the conversation
            print(f"Error creating goal: {e}")
            return None
    
    def _generate_title(self, description: str) -> str:
        """
        Generate concise goal title from description.
        
        Args:
            description: Full goal description
            
        Returns:
            Concise title (max 60 chars)
        """
        if not description:
            return "Untitled Goal"
        
        # Clean up the description
        description = description.strip()
        
        # Extract key action verbs and nouns
        # Remove filler words
        filler_words = {'the', 'a', 'an', 'to', 'and', 'or', 'but', 'for', 'with', 'on', 'in', 'at'}
        words = description.split()
        important_words = [w for w in words if w.lower() not in filler_words]
        
        # Use first 8 important words or truncate at 60 chars
        title = ' '.join(important_words[:8])
        if len(title) > 60:
            title = title[:57] + "..."
        
        # Capitalize first letter
        return title[0].upper() + title[1:] if title else "Untitled Goal"

    def _get_intent_classifier(self, session_id: str) -> IntentClassifierV2:
        classifier = self._intent_classifiers.get(session_id)
        if classifier is None:
            classifier = create_intent_classifier_v2(context=ConversationContext())
            self._intent_classifiers[session_id] = classifier
        return classifier
    
    def _parse_deadline(self, deadline_str: Optional[str]) -> Optional[datetime]:
        """
        Parse natural language deadline into datetime.
        
        Supports: today, tomorrow, Monday-Sunday, next week, etc.
        
        Args:
            deadline_str: Natural language deadline
            
        Returns:
            Datetime object or None
        """
        if not deadline_str:
            return None
        
        deadline_lower = deadline_str.lower().strip()
        now = datetime.now()
        
        # Today (end of day)
        if deadline_lower == 'today':
            return datetime.combine(now.date(), datetime.max.time())
        
        # Tomorrow
        if deadline_lower == 'tomorrow':
            tomorrow = now + timedelta(days=1)
            return datetime.combine(tomorrow.date(), datetime.max.time())
        
        # Days of week
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in weekdays.items():
            if day_name in deadline_lower:
                # Find next occurrence of this weekday
                current_weekday = now.weekday()
                days_ahead = day_num - current_weekday
                if days_ahead <= 0:  # Target day already passed this week
                    days_ahead += 7
                next_day = now + timedelta(days=days_ahead)
                return datetime.combine(next_day.date(), datetime.max.time())
        
        # Next week (7 days from now)
        if 'next week' in deadline_lower:
            next_week = now + timedelta(days=7)
            return datetime.combine(next_week.date(), datetime.max.time())
        
        # End of week (next Friday)
        if 'end of week' in deadline_lower or 'eow' in deadline_lower:
            current_weekday = now.weekday()
            days_to_friday = 4 - current_weekday
            if days_to_friday <= 0:
                days_to_friday += 7
            friday = now + timedelta(days=days_to_friday)
            return datetime.combine(friday.date(), datetime.max.time())
        
        # End of month
        if 'end of month' in deadline_lower or 'eom' in deadline_lower:
            # Last day of current month
            next_month = now.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            return datetime.combine(last_day.date(), datetime.max.time())
        
        # If we can't parse it, return None
        return None
    
    def _infer_priority(self, entities: Dict, deadline: Optional[datetime]) -> GoalPriority:
        """
        Infer goal priority from context.
        
        Args:
            entities: Extracted entities from intent classification
            deadline: Parsed deadline
            
        Returns:
            GoalPriority enum value
        """
        # Check explicit priority from entities
        explicit_priority = entities.get('priority', 'medium')
        if explicit_priority == 'high':
            return GoalPriority.HIGH
        elif explicit_priority == 'low':
            return GoalPriority.LOW
        
        # Check urgency indicators
        if entities.get('urgent', False):
            return GoalPriority.HIGH
        
        # Check deadline proximity
        if deadline:
            time_until_deadline = deadline - datetime.now()
            if time_until_deadline < timedelta(days=1):
                return GoalPriority.HIGH
            elif time_until_deadline < timedelta(days=3):
                return GoalPriority.MEDIUM
        
        return GoalPriority.MEDIUM
    
    def _generate_success_criteria(self, entities: Dict, message: str) -> List[str]:
        """
        Generate success criteria for the goal.
        
        Args:
            entities: Extracted entities
            message: Original message
            
        Returns:
            List of success criteria strings
        """
        criteria = []
        
        # Default criterion: task completion
        description = entities.get('goal_description', '')
        if description:
            # Extract the main action
            criteria.append(f"Task completed: {description}")
        
        # Add deadline criterion if present
        if entities.get('deadline'):
            criteria.append(f"Completed by deadline: {entities['deadline']}")
        
        # Add quality/review criterion for certain keywords
        quality_keywords = ['review', 'analyze', 'prepare', 'create', 'write']
        if any(keyword in message.lower() for keyword in quality_keywords):
            criteria.append("Quality standards met")
        
        return criteria if criteria else ["Task completed successfully"]
    
    def _estimate_duration(self, entities: Dict, title: str) -> Optional[timedelta]:
        """
        Estimate task duration based on task type.
        
        Args:
            entities: Extracted entities
            title: Goal title
            
        Returns:
            Estimated duration or None
        """
        # Simple heuristic-based estimation
        title_lower = title.lower()
        description = entities.get('goal_description', '').lower()
        combined = title_lower + ' ' + description
        
        # Quick tasks (< 30 min)
        quick_keywords = ['send', 'email', 'message', 'call', 'check', 'review']
        if any(keyword in combined for keyword in quick_keywords):
            return timedelta(minutes=15)
        
        # Medium tasks (30 min - 2 hours)
        medium_keywords = ['prepare', 'create', 'write', 'update', 'fix']
        if any(keyword in combined for keyword in medium_keywords):
            return timedelta(hours=1)
        
        # Large tasks (2+ hours)
        large_keywords = ['analyze', 'develop', 'implement', 'research', 'comprehensive', 'quarterly']
        if any(keyword in combined for keyword in large_keywords):
            return timedelta(hours=4)
        
        # Default: 1 hour
        return timedelta(hours=1)


def create_goal_detector(executive_system: Optional[ExecutiveSystem] = None) -> GoalDetector:
    """Factory function to create goal detector"""
    return GoalDetector(executive_system)
