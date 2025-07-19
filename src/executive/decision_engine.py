"""
Decision Engine - Multi-criteria decision making

This module provides sophisticated decision-making capabilities using
multiple criteria, weighted scoring, and uncertainty handling. It supports
both analytical and heuristic decision-making approaches.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from abc import ABC, abstractmethod

class DecisionType(Enum):
    """Types of decisions"""
    CHOICE = "choice"  # Choose one option from many
    RANKING = "ranking"  # Rank multiple options
    BINARY = "binary"  # Yes/No decision
    RESOURCE_ALLOCATION = "resource_allocation"  # Distribute resources
    SCHEDULING = "scheduling"  # Time-based decisions
    STRATEGIC = "strategic"  # Long-term strategic decisions

class CriterionType(Enum):
    """Types of decision criteria"""
    BENEFIT = "benefit"  # Higher is better
    COST = "cost"  # Lower is better
    CONSTRAINT = "constraint"  # Must meet requirement

@dataclass
class DecisionCriterion:
    """
    Individual criterion for decision making
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        description: Detailed description
        criterion_type: Type of criterion
        weight: Importance weight (0.0 to 1.0)
        evaluator: Function to evaluate options against this criterion
        threshold: Minimum/maximum threshold for constraints
        uncertainty_factor: Factor accounting for uncertainty (0.0 to 1.0)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    criterion_type: CriterionType = CriterionType.BENEFIT
    weight: float = 1.0
    evaluator: Optional[Callable[[Any], float]] = None
    threshold: Optional[float] = None
    uncertainty_factor: float = 0.0
    
    def __post_init__(self):
        """Validate criterion parameters"""
        if not self.name:
            raise ValueError("Criterion name cannot be empty")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0")
        if not (0.0 <= self.uncertainty_factor <= 1.0):
            raise ValueError("Uncertainty factor must be between 0.0 and 1.0")
    
    def evaluate(self, option: Any) -> float:
        """Evaluate an option against this criterion"""
        if not self.evaluator:
            raise ValueError(f"No evaluator set for criterion {self.name}")
        
        raw_score = self.evaluator(option)
        
        # Apply uncertainty factor
        if self.uncertainty_factor > 0:
            # Reduce score based on uncertainty
            raw_score *= (1.0 - self.uncertainty_factor)
        
        return raw_score
    
    def meets_constraint(self, option: Any) -> bool:
        """Check if option meets constraint threshold"""
        if self.criterion_type != CriterionType.CONSTRAINT:
            return True
        
        if self.threshold is None:
            return True
        
        score = self.evaluate(option)
        return score >= self.threshold

@dataclass
class DecisionOption:
    """
    Individual option in a decision
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        description: Detailed description
        data: Option-specific data
        constraints: List of constraints this option must satisfy
        metadata: Additional metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate option parameters"""
        if not self.name:
            raise ValueError("Option name cannot be empty")

@dataclass
class DecisionResult:
    """
    Result of a decision-making process
    
    Attributes:
        decision_id: ID of the decision
        recommended_option: Top recommended option
        option_scores: Scores for all options
        criterion_weights: Final weights used
        rationale: Explanation of the decision
        confidence: Confidence level (0.0 to 1.0)
        alternatives: Alternative options considered
        timestamp: When decision was made
        metadata: Additional result metadata
    """
    decision_id: str
    recommended_option: Optional[DecisionOption] = None
    option_scores: Dict[str, float] = field(default_factory=dict)
    criterion_weights: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0
    alternatives: List[DecisionOption] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'decision_id': self.decision_id,
            'recommended_option': self.recommended_option.name if self.recommended_option else None,
            'option_scores': self.option_scores,
            'criterion_weights': self.criterion_weights,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'alternatives': [opt.name for opt in self.alternatives],
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class DecisionStrategy(ABC):
    """Abstract base class for decision strategies"""
    
    @abstractmethod
    def decide(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any]
    ) -> DecisionResult:
        """Make a decision using this strategy"""
        pass

class WeightedScoringStrategy(DecisionStrategy):
    """Simple weighted scoring decision strategy"""
    
    def decide(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any]
    ) -> DecisionResult:
        """Make decision using weighted scoring"""
        if not options:
            return DecisionResult(
                decision_id=str(uuid.uuid4()),
                rationale="No options provided"
            )
        
        # Calculate scores for each option
        option_scores = {}
        detailed_scores = {}
        
        for option in options:
            total_score = 0.0
            max_possible = 0.0
            option_detail = {}
            
            # Check constraints first
            constraint_met = True
            for criterion in criteria:
                if criterion.criterion_type == CriterionType.CONSTRAINT:
                    if not criterion.meets_constraint(option):
                        constraint_met = False
                        break
            
            if not constraint_met:
                option_scores[option.id] = 0.0
                continue
            
            # Calculate weighted score
            for criterion in criteria:
                if criterion.criterion_type == CriterionType.CONSTRAINT:
                    continue
                
                raw_score = criterion.evaluate(option)
                
                # Invert score for cost criteria
                if criterion.criterion_type == CriterionType.COST:
                    raw_score = 1.0 - raw_score
                
                weighted_score = raw_score * criterion.weight
                total_score += weighted_score
                max_possible += criterion.weight
                
                option_detail[criterion.name] = {
                    'raw_score': raw_score,
                    'weighted_score': weighted_score,
                    'weight': criterion.weight
                }
            
            # Normalize score
            final_score = total_score / max_possible if max_possible > 0 else 0.0
            option_scores[option.id] = final_score
            detailed_scores[option.id] = option_detail
        
        # Find best option
        if not option_scores:
            return DecisionResult(
                decision_id=str(uuid.uuid4()),
                rationale="No viable options found"
            )
        
        best_option_id = max(option_scores.keys(), key=lambda k: option_scores[k])
        best_option = next(opt for opt in options if opt.id == best_option_id)
        best_score = option_scores[best_option_id]
        
        # Calculate confidence based on score separation
        sorted_scores = sorted(option_scores.values(), reverse=True)
        confidence = 0.5  # Base confidence
        if len(sorted_scores) > 1:
            score_gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, 0.5 + score_gap)
        
        # Generate rationale
        rationale_parts = [
            f"Selected '{best_option.name}' with score {best_score:.3f}",
            f"Evaluated {len(options)} options against {len(criteria)} criteria"
        ]
        
        if best_option_id in detailed_scores:
            top_criteria = sorted(
                detailed_scores[best_option_id].items(),
                key=lambda x: x[1]['weighted_score'],
                reverse=True
            )[:3]
            top_factors = []
            for name, data in top_criteria:
                top_factors.append(f"{name} ({data['weighted_score']:.3f})")
            rationale_parts.append(
                f"Top contributing factors: {', '.join(top_factors)}"
            )
        
        # Get alternatives
        alternatives = [
            opt for opt in options
            if opt.id != best_option_id and option_scores.get(opt.id, 0) > 0
        ]
        alternatives.sort(key=lambda opt: option_scores.get(opt.id, 0), reverse=True)
        
        return DecisionResult(
            decision_id=str(uuid.uuid4()),
            recommended_option=best_option,
            option_scores=option_scores,
            criterion_weights={c.name: c.weight for c in criteria},
            rationale=". ".join(rationale_parts),
            confidence=confidence,
            alternatives=alternatives[:3],  # Top 3 alternatives
            metadata={
                'strategy': 'weighted_scoring',
                'detailed_scores': detailed_scores,
                'context': context
            }
        )

class AHPStrategy(DecisionStrategy):
    """Analytic Hierarchy Process decision strategy"""
    
    def decide(
        self,
        options: List[DecisionOption],
        criteria: List[DecisionCriterion],
        context: Dict[str, Any]
    ) -> DecisionResult:
        """Make decision using simplified AHP"""
        # For now, fall back to weighted scoring
        # TODO: Implement full AHP with pairwise comparisons
        weighted_strategy = WeightedScoringStrategy()
        result = weighted_strategy.decide(options, criteria, context)
        result.metadata['strategy'] = 'ahp_simplified'
        return result

class DecisionEngine:
    """
    Advanced decision-making engine with multiple strategies
    """
    
    def __init__(self):
        """Initialize decision engine"""
        self.strategies: Dict[str, DecisionStrategy] = {
            'weighted_scoring': WeightedScoringStrategy(),
            'ahp': AHPStrategy()
        }
        self.decision_history: List[DecisionResult] = []
        self.criterion_templates: Dict[str, List[DecisionCriterion]] = {}
        self._load_default_criteria()
    
    def _load_default_criteria(self) -> None:
        """Load default criterion templates"""
        self.criterion_templates = {
            'task_selection': [
                DecisionCriterion(
                    name="Priority",
                    description="Task priority level",
                    criterion_type=CriterionType.BENEFIT,
                    weight=0.4,
                    evaluator=lambda task: getattr(task, 'priority_score', 0.5)
                ),
                DecisionCriterion(
                    name="Effort",
                    description="Required cognitive effort",
                    criterion_type=CriterionType.COST,
                    weight=0.3,
                    evaluator=lambda task: getattr(task, 'cognitive_load', 0.5)
                ),
                DecisionCriterion(
                    name="Duration",
                    description="Time to complete",
                    criterion_type=CriterionType.COST,
                    weight=0.2,
                    evaluator=lambda task: min(1.0, getattr(task, 'estimated_duration', 60) / 120.0)
                ),
                DecisionCriterion(
                    name="Urgency",
                    description="How urgent the task is",
                    criterion_type=CriterionType.BENEFIT,
                    weight=0.1,
                    evaluator=lambda task: 1.0 if getattr(task, 'is_overdue', lambda: False)() else 0.5
                )
            ],
            'resource_allocation': [
                DecisionCriterion(
                    name="Impact",
                    description="Expected impact of allocation",
                    criterion_type=CriterionType.BENEFIT,
                    weight=0.5,
                    evaluator=lambda option: option.data.get('impact', 0.5)
                ),
                DecisionCriterion(
                    name="Cost",
                    description="Resource cost",
                    criterion_type=CriterionType.COST,
                    weight=0.3,
                    evaluator=lambda option: option.data.get('cost', 0.5)
                ),
                DecisionCriterion(
                    name="Feasibility",
                    description="How feasible the option is",
                    criterion_type=CriterionType.CONSTRAINT,
                    weight=0.2,
                    evaluator=lambda option: option.data.get('feasibility', 0.5),
                    threshold=0.6
                )
            ]
        }
    
    def make_decision(
        self,
        options: List[DecisionOption],
        criteria: Optional[List[DecisionCriterion]] = None,
        strategy: str = 'weighted_scoring',
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """
        Make a decision given options and criteria
        
        Args:
            options: List of options to choose from
            criteria: Decision criteria (uses template if None)
            strategy: Decision strategy to use
            context: Additional context for decision
            
        Returns:
            Decision result with recommendation
        """
        if not options:
            return DecisionResult(
                decision_id=str(uuid.uuid4()),
                rationale="No options provided"
            )
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Use default criteria if none provided
        if not criteria:
            criteria = self.criterion_templates.get('task_selection', [])
        
        context = context or {}
        
        # Make decision using selected strategy
        result = self.strategies[strategy].decide(options, criteria, context)
        
        # Store in history
        self.decision_history.append(result)
        
        return result
    
    def recommend_task(
        self,
        tasks: List[Any],
        available_time: int = 60,
        max_cognitive_load: float = 0.8,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Recommend the best task to work on
        
        Args:
            tasks: List of available tasks
            available_time: Available time in minutes
            max_cognitive_load: Maximum cognitive load
            context: Additional context
            
        Returns:
            Recommended task or None
        """
        if not tasks:
            return None
        
        # Convert tasks to decision options
        options = []
        for task in tasks:
            # Filter by constraints
            if hasattr(task, 'estimated_duration') and task.estimated_duration > available_time:
                continue
            if hasattr(task, 'cognitive_load') and task.cognitive_load > max_cognitive_load:
                continue
            
            option = DecisionOption(
                name=getattr(task, 'title', str(task)),
                description=getattr(task, 'description', ''),
                data={'task': task}
            )
            options.append(option)
        
        if not options:
            return None
        
        # Make decision
        result = self.make_decision(
            options=options,
            criteria=self.criterion_templates['task_selection'],
            context=context or {}
        )
        
        if result.recommended_option:
            return result.recommended_option.data['task']
        
        return None
    
    def evaluate_goal_priority(
        self,
        goals: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, float]]:
        """
        Evaluate and rank goals by priority
        
        Args:
            goals: List of goals to evaluate
            context: Additional context
            
        Returns:
            List of (goal, priority_score) tuples sorted by priority
        """
        if not goals:
            return []
        
        # Define goal evaluation criteria
        criteria = [
            DecisionCriterion(
                name="Priority Level",
                description="Assigned priority level",
                criterion_type=CriterionType.BENEFIT,
                weight=0.3,
                evaluator=lambda goal: getattr(goal, 'priority', 2) / 5.0
            ),
            DecisionCriterion(
                name="Progress",
                description="Current progress towards goal",
                criterion_type=CriterionType.BENEFIT,
                weight=0.2,
                evaluator=lambda goal: getattr(goal, 'progress', 0.0)
            ),
            DecisionCriterion(
                name="Urgency",
                description="Time sensitivity",
                criterion_type=CriterionType.BENEFIT,
                weight=0.3,
                evaluator=lambda goal: 1.0 if getattr(goal, 'is_overdue', lambda: False)() else 0.5
            ),
            DecisionCriterion(
                name="Dependencies",
                description="Number of other goals depending on this",
                criterion_type=CriterionType.BENEFIT,
                weight=0.2,
                evaluator=lambda goal: min(1.0, len(getattr(goal, 'dependents', [])) / 5.0)
            )
        ]
        
        # Convert goals to options
        options = [
            DecisionOption(
                name=getattr(goal, 'title', str(goal)),
                description=getattr(goal, 'description', ''),
                data={'goal': goal}
            )
            for goal in goals
        ]
        
        # Evaluate each option
        results = []
        for option in options:
            total_score = 0.0
            max_possible = 0.0
            
            for criterion in criteria:
                raw_score = criterion.evaluate(option.data['goal'])
                weighted_score = raw_score * criterion.weight
                total_score += weighted_score
                max_possible += criterion.weight
            
            final_score = total_score / max_possible if max_possible > 0 else 0.0
            results.append((option.data['goal'], final_score))
        
        # Sort by score
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_decision_history(self, limit: int = 10) -> List[DecisionResult]:
        """Get recent decision history"""
        return self.decision_history[-limit:]
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision-making statistics"""
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'average_confidence': 0.0,
                'strategy_usage': {},
                'most_recent': None
            }
        
        strategy_counts = {}
        total_confidence = 0.0
        
        for result in self.decision_history:
            strategy = result.metadata.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            total_confidence += result.confidence
        
        return {
            'total_decisions': len(self.decision_history),
            'average_confidence': total_confidence / len(self.decision_history),
            'strategy_usage': strategy_counts,
            'most_recent': self.decision_history[-1].timestamp.isoformat()
        }
