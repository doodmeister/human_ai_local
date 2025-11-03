"""
Dynamic Goal Priority Calculator

Calculates goal priorities based on multiple factors:
- Urgency: Deadline proximity, time sensitivity
- Importance: User weight, system weight, strategic value
- Dependencies: Blocking other goals, blocked by others
- Resources: Availability, cost, contention

Combines factors using weighted scoring to produce dynamic priorities
that adapt to changing conditions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import math


class PriorityFactor(Enum):
    """Factors that influence goal priority."""
    URGENCY = "urgency"
    IMPORTANCE = "importance"
    DEPENDENCIES = "dependencies"
    RESOURCES = "resources"


@dataclass
class PriorityWeights:
    """Weights for different priority factors."""
    urgency: float = 0.3
    importance: float = 0.4
    dependencies: float = 0.2
    resources: float = 0.1
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.urgency + self.importance + self.dependencies + self.resources
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"Priority weights must sum to 1.0, got {total}")


@dataclass
class GoalContext:
    """Context information for priority calculation."""
    goal_id: str
    deadline: Optional[datetime] = None
    user_importance: float = 5.0  # 1-10 scale
    system_importance: float = 5.0  # 1-10 scale
    strategic_value: float = 5.0  # 1-10 scale
    
    # Dependencies
    blocks_goals: Optional[Set[str]] = None  # Goals this blocks
    blocked_by_goals: Optional[Set[str]] = None  # Goals blocking this
    
    # Resources
    required_resources: Optional[Dict[str, float]] = None  # resource_id -> amount
    available_resources: Optional[Dict[str, float]] = None  # resource_id -> available
    
    # Time
    estimated_duration: Optional[timedelta] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.blocks_goals is None:
            self.blocks_goals = set()
        if self.blocked_by_goals is None:
            self.blocked_by_goals = set()
        if self.required_resources is None:
            self.required_resources = {}
        if self.available_resources is None:
            self.available_resources = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PriorityScore:
    """Detailed priority score breakdown."""
    final_score: float  # 0-10 scale
    urgency_score: float
    importance_score: float
    dependency_score: float
    resource_score: float
    
    factors: Dict[str, float]  # Additional factor details
    timestamp: datetime
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GoalPriorityCalculator:
    """
    Calculates dynamic goal priorities based on multiple factors.
    
    Uses weighted scoring model that combines:
    - Urgency: How soon the goal needs to be completed
    - Importance: User and system-defined importance
    - Dependencies: Impact on other goals
    - Resources: Availability and constraints
    
    Example:
        calculator = GoalPriorityCalculator()
        context = GoalContext(
            goal_id="research_ai",
            deadline=datetime.now() + timedelta(days=2),
            user_importance=9.0,
            blocks_goals={"write_paper", "present_findings"}
        )
        score = calculator.calculate_priority(context)
        print(f"Priority: {score.final_score:.2f}")
    """
    
    def __init__(self, weights: Optional[PriorityWeights] = None):
        """
        Initialize priority calculator.
        
        Args:
            weights: Custom weights for priority factors. Defaults to balanced weights.
        """
        self.weights = weights or PriorityWeights()
    
    def calculate_priority(self, context: GoalContext) -> PriorityScore:
        """
        Calculate comprehensive priority score for a goal.
        
        Args:
            context: Goal context with all relevant information
            
        Returns:
            PriorityScore with breakdown of all factors
        """
        # Calculate individual factor scores
        urgency = self._calculate_urgency_score(context)
        importance = self._calculate_importance_score(context)
        dependencies = self._calculate_dependency_score(context)
        resources = self._calculate_resource_score(context)
        
        # Combine using weights
        final_score = (
            urgency * self.weights.urgency +
            importance * self.weights.importance +
            dependencies * self.weights.dependencies +
            resources * self.weights.resources
        )
        
        # Clamp to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        # Build detailed factors dict
        factors = {
            "deadline_pressure": self._calculate_deadline_pressure(context),
            "age_factor": self._calculate_age_factor(context),
            "blocking_impact": len(context.blocks_goals) if context.blocks_goals else 0,
            "blocked_penalty": len(context.blocked_by_goals) if context.blocked_by_goals else 0,
            "resource_availability": self._calculate_resource_availability(context)
        }
        
        return PriorityScore(
            final_score=final_score,
            urgency_score=urgency,
            importance_score=importance,
            dependency_score=dependencies,
            resource_score=resources,
            factors=factors,
            timestamp=datetime.now()
        )
    
    def _calculate_urgency_score(self, context: GoalContext) -> float:
        """
        Calculate urgency score based on deadline and age.
        
        Returns score from 0 (not urgent) to 10 (extremely urgent).
        """
        deadline_pressure = self._calculate_deadline_pressure(context)
        age_factor = self._calculate_age_factor(context)
        
        # Combine deadline pressure (80%) and age (20%)
        # Give more weight to deadline for truly urgent situations
        urgency = deadline_pressure * 0.8 + age_factor * 0.2
        
        return max(0.0, min(10.0, urgency))
    
    def _calculate_deadline_pressure(self, context: GoalContext) -> float:
        """
        Calculate pressure from approaching deadline.
        
        Returns 0 (far away) to 10 (overdue/imminent).
        """
        if context.deadline is None:
            return 5.0  # Neutral - no deadline info
        
        now = datetime.now()
        time_until = (context.deadline - now).total_seconds()
        
        # Overdue or immediate
        if time_until <= 0:
            return 10.0
        
        # Use exponential decay for deadline pressure
        # High pressure when deadline is close
        hours_until = time_until / 3600
        
        if hours_until < 1:
            return 10.0
        elif hours_until < 24:
            return 8.0 + (24 - hours_until) / 24 * 2
        elif hours_until < 72:
            return 6.0 + (72 - hours_until) / 72 * 2
        elif hours_until < 168:  # 1 week
            return 4.0 + (168 - hours_until) / 168 * 2
        else:
            # Long term - low pressure
            return max(1.0, 4.0 - math.log10(hours_until / 168))
    
    def _calculate_age_factor(self, context: GoalContext) -> float:
        """
        Calculate factor based on how long goal has existed.
        
        Older goals get slight priority boost to avoid starvation.
        Returns 0-3 (contribution to urgency).
        """
        if context.created_at is None:
            return 0.0
        
        age = (datetime.now() - context.created_at).total_seconds()
        hours_old = age / 3600
        
        # Gradual increase: 0 at creation, max 3 after 30 days
        if hours_old < 24:
            return 0.0
        elif hours_old < 168:  # 1 week
            return 0.5
        elif hours_old < 720:  # 30 days
            return 1.0 + (hours_old - 168) / 552 * 2
        else:
            return 3.0
    
    def _calculate_importance_score(self, context: GoalContext) -> float:
        """
        Calculate importance based on user, system, and strategic weights.
        
        Returns score from 0 (not important) to 10 (critical).
        """
        # Combine three importance sources
        # User importance: 50%, System: 30%, Strategic: 20%
        importance = (
            context.user_importance * 0.5 +
            context.system_importance * 0.3 +
            context.strategic_value * 0.2
        )
        
        return max(0.0, min(10.0, importance))
    
    def _calculate_dependency_score(self, context: GoalContext) -> float:
        """
        Calculate priority based on dependency relationships.
        
        Goals that block others get higher priority.
        Goals that are blocked get lower priority.
        
        Returns score from 0-10.
        """
        # Base score
        score = 5.0
        
        # Boost for blocking other goals (enables parallelism)
        blocking_boost = min(5.0, len(context.blocks_goals) * 1.0) if context.blocks_goals else 0.0
        
        # Penalty for being blocked (can't execute yet)
        blocked_penalty = min(5.0, len(context.blocked_by_goals) * 1.5) if context.blocked_by_goals else 0.0
        
        score = score + blocking_boost - blocked_penalty
        
        return max(0.0, min(10.0, score))
    
    def _calculate_resource_score(self, context: GoalContext) -> float:
        """
        Calculate priority based on resource availability.
        
        Goals with available resources get higher priority.
        Goals with scarce resources get lower priority.
        
        Returns score from 0-10.
        """
        if not context.required_resources:
            return 5.0  # Neutral - no resource requirements
        
        availability_ratio = self._calculate_resource_availability(context)
        
        # Convert 0-1 availability to 0-10 priority
        # High availability -> high priority (can execute now)
        score = availability_ratio * 10
        
        return max(0.0, min(10.0, score))
    
    def _calculate_resource_availability(self, context: GoalContext) -> float:
        """
        Calculate ratio of available to required resources.
        
        Returns 0.0 (none available) to 1.0 (all available).
        """
        if not context.required_resources:
            return 1.0
        
        availability_scores = []
        
        for resource_id, required in context.required_resources.items():
            available = context.available_resources.get(resource_id, 0.0) if context.available_resources else 0.0
            
            if required <= 0:
                availability_scores.append(1.0)
            else:
                ratio = min(1.0, available / required)
                availability_scores.append(ratio)
        
        # Return average availability across all resources
        return sum(availability_scores) / len(availability_scores) if availability_scores else 1.0
    
    def recalculate_batch(
        self,
        contexts: List[GoalContext]
    ) -> Dict[str, PriorityScore]:
        """
        Calculate priorities for multiple goals efficiently.
        
        Args:
            contexts: List of goal contexts
            
        Returns:
            Dictionary mapping goal_id to PriorityScore
        """
        return {
            context.goal_id: self.calculate_priority(context)
            for context in contexts
        }
    
    def compare_priorities(
        self,
        context1: GoalContext,
        context2: GoalContext
    ) -> int:
        """
        Compare priorities of two goals.
        
        Returns:
            -1 if goal1 has lower priority than goal2
             0 if equal priority
             1 if goal1 has higher priority than goal2
        """
        score1 = self.calculate_priority(context1).final_score
        score2 = self.calculate_priority(context2).final_score
        
        if math.isclose(score1, score2, rel_tol=1e-2):
            return 0
        return 1 if score1 > score2 else -1
