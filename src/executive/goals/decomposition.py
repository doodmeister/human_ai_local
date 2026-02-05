"""
Decomposition Methods for HTN Planning.

Methods define how compound goals are decomposed into subtasks.

Key Concepts:
- Method: A rule for decomposing a goal type
- Preconditions: When the method is applicable
- Subtask template: Pattern for generating subtasks
- Ordering constraints: Sequential, parallel, or partial order

Example:
    Goal: "Write report"
    Method: "StandardReportMethod"
    Preconditions: {has_data: True}
    Subtasks: ["Research topic", "Draft content", "Review draft", "Finalize"]
    Ordering: Sequential
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .goal_taxonomy import Goal, GoalType


class OrderingConstraint(Enum):
    """Ordering relationships between subtasks."""
    SEQUENTIAL = "sequential"  # Must execute in order
    PARALLEL = "parallel"      # Can execute simultaneously
    PARTIAL = "partial"        # Some ordering constraints


@dataclass
class SubtaskTemplate:
    """
    Template for creating a subtask during decomposition.
    
    Attributes:
        description: Subtask description
        goal_type: Primitive or compound
        priority: Subtask priority (relative to parent)
        preconditions: Required preconditions
        postconditions: Expected postconditions
        metadata: Additional data
    """
    
    description: str
    goal_type: GoalType = GoalType.PRIMITIVE
    priority: Optional[int] = None  # Inherit from parent if None
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Method:
    """
    A method for decomposing a compound goal into subtasks.
    
    Methods are like "recipes" that define how to break down
    high-level goals into concrete subtasks.
    
    Attributes:
        name: Method name
        applicable_goal_patterns: Goal descriptions/types this method applies to
        preconditions: When this method can be used
        subtask_templates: Templates for generating subtasks
        ordering: Ordering constraint between subtasks
        priority: Method priority (if multiple methods match)
    """
    
    name: str
    applicable_goal_patterns: List[str]
    preconditions: Dict[str, Any] = field(default_factory=dict)
    subtask_templates: List[SubtaskTemplate] = field(default_factory=list)
    ordering: OrderingConstraint = OrderingConstraint.SEQUENTIAL
    priority: int = 5
    
    def is_applicable(
        self,
        goal: Goal,
        current_state: Dict[str, Any]
    ) -> bool:
        """
        Check if method is applicable to a goal.
        
        Args:
            goal: Goal to decompose
            current_state: Current world state
            
        Returns:
            True if method can be applied
        """
        # Check if goal matches any pattern
        matches_pattern = False
        for pattern in self.applicable_goal_patterns:
            if pattern.lower() in goal.description.lower():
                matches_pattern = True
                break
        
        if not matches_pattern:
            return False
        
        # Check preconditions
        for key, required_value in self.preconditions.items():
            if key not in current_state:
                return False
            if current_state[key] != required_value:
                return False
        
        return True
    
    def decompose(
        self,
        goal: Goal,
        current_state: Dict[str, Any],
        goal_id_generator: Callable[[], str]
    ) -> List[Goal]:
        """
        Decompose goal into subtasks using this method.
        
        Args:
            goal: Compound goal to decompose
            current_state: Current world state
            goal_id_generator: Function to generate unique goal IDs
            
        Returns:
            List of subtask goals
        """
        subtasks = []
        
        for i, template in enumerate(self.subtask_templates):
            # Generate unique ID
            subtask_id = goal_id_generator()
            
            # Inherit priority if not specified
            priority = template.priority
            if priority is None:
                priority = goal.priority
            
            # Create subtask
            subtask = Goal(
                id=subtask_id,
                description=template.description,
                goal_type=template.goal_type,
                priority=priority,
                parent_id=goal.id,
                preconditions=template.preconditions.copy(),
                postconditions=template.postconditions.copy(),
                metadata=template.metadata.copy()
            )
            
            # Add ordering dependencies for sequential execution
            if self.ordering == OrderingConstraint.SEQUENTIAL and i > 0:
                # Depends on previous subtask
                subtask.add_dependency(subtasks[i-1].id)
            
            subtasks.append(subtask)
        
        return subtasks


@dataclass
class DecompositionRule:
    """
    A rule that selects which method to use for a goal.
    
    Multiple methods may be applicable; rules help choose the best one.
    """
    
    name: str
    condition: Callable[[Goal, Dict[str, Any]], bool]
    method: Method
    
    def matches(self, goal: Goal, current_state: Dict[str, Any]) -> bool:
        """Check if rule matches the goal and state."""
        return self.condition(goal, current_state)


# ============================================================================
# Default Method Library
# ============================================================================

def create_default_methods() -> List[Method]:
    """
    Create a library of default decomposition methods.
    
    These methods cover common goal patterns:
    - Research tasks
    - Document creation
    - Data analysis
    - Communication
    - Learning
    
    Returns:
        List of predefined methods
    """
    
    methods = []
    
    # Method 1: Research Task
    methods.append(Method(
        name="ResearchMethod",
        applicable_goal_patterns=["research", "investigate", "explore", "study"],
        preconditions={},
        subtask_templates=[
            SubtaskTemplate(
                description="Define research questions",
                goal_type=GoalType.PRIMITIVE,
                postconditions={"has_questions": True}
            ),
            SubtaskTemplate(
                description="Gather information",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_questions": True},
                postconditions={"has_data": True}
            ),
            SubtaskTemplate(
                description="Analyze findings",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_data": True},
                postconditions={"has_analysis": True}
            ),
            SubtaskTemplate(
                description="Summarize results",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_analysis": True},
                postconditions={"has_summary": True}
            ),
        ],
        ordering=OrderingConstraint.SEQUENTIAL,
        priority=8
    ))
    
    # Method 2: Document Creation
    methods.append(Method(
        name="DocumentMethod",
        applicable_goal_patterns=["write", "create document", "draft", "report"],
        preconditions={},
        subtask_templates=[
            SubtaskTemplate(
                description="Outline structure",
                goal_type=GoalType.PRIMITIVE,
                postconditions={"has_outline": True}
            ),
            SubtaskTemplate(
                description="Write content",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_outline": True},
                postconditions={"has_draft": True}
            ),
            SubtaskTemplate(
                description="Review and edit",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_draft": True},
                postconditions={"has_final": True}
            ),
        ],
        ordering=OrderingConstraint.SEQUENTIAL,
        priority=7
    ))
    
    # Method 3: Data Analysis
    methods.append(Method(
        name="AnalysisMethod",
        applicable_goal_patterns=["analyze", "process", "evaluate"],
        preconditions={"has_data": True},
        subtask_templates=[
            SubtaskTemplate(
                description="Clean and prepare data",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_data": True},
                postconditions={"data_prepared": True}
            ),
            SubtaskTemplate(
                description="Perform analysis",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"data_prepared": True},
                postconditions={"has_analysis": True}
            ),
            SubtaskTemplate(
                description="Interpret results",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_analysis": True},
                postconditions={"has_interpretation": True}
            ),
        ],
        ordering=OrderingConstraint.SEQUENTIAL,
        priority=7
    ))
    
    # Method 4: Communication Task
    methods.append(Method(
        name="CommunicationMethod",
        applicable_goal_patterns=["communicate", "inform", "notify", "message"],
        preconditions={},
        subtask_templates=[
            SubtaskTemplate(
                description="Prepare message content",
                goal_type=GoalType.PRIMITIVE,
                postconditions={"has_message": True}
            ),
            SubtaskTemplate(
                description="Send message",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_message": True},
                postconditions={"message_sent": True}
            ),
        ],
        ordering=OrderingConstraint.SEQUENTIAL,
        priority=6
    ))
    
    # Method 5: Learning Task
    methods.append(Method(
        name="LearningMethod",
        applicable_goal_patterns=["learn", "understand", "master", "practice"],
        preconditions={},
        subtask_templates=[
            SubtaskTemplate(
                description="Acquire knowledge",
                goal_type=GoalType.PRIMITIVE,
                postconditions={"has_knowledge": True}
            ),
            SubtaskTemplate(
                description="Practice skills",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_knowledge": True},
                postconditions={"has_practice": True}
            ),
            SubtaskTemplate(
                description="Validate understanding",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_practice": True},
                postconditions={"skill_validated": True}
            ),
        ],
        ordering=OrderingConstraint.SEQUENTIAL,
        priority=6
    ))
    
    # Method 6: Problem Solving
    methods.append(Method(
        name="ProblemSolvingMethod",
        applicable_goal_patterns=["solve", "fix", "resolve", "debug"],
        preconditions={},
        subtask_templates=[
            SubtaskTemplate(
                description="Identify problem",
                goal_type=GoalType.PRIMITIVE,
                postconditions={"problem_identified": True}
            ),
            SubtaskTemplate(
                description="Generate solutions",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"problem_identified": True},
                postconditions={"has_solutions": True}
            ),
            SubtaskTemplate(
                description="Implement solution",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"has_solutions": True},
                postconditions={"solution_implemented": True}
            ),
            SubtaskTemplate(
                description="Verify solution",
                goal_type=GoalType.PRIMITIVE,
                preconditions={"solution_implemented": True},
                postconditions={"problem_solved": True}
            ),
        ],
        ordering=OrderingConstraint.SEQUENTIAL,
        priority=8
    ))
    
    return methods
