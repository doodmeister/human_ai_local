"""
GOAP (Goal-Oriented Action Planning) Planner using A* search.

Implements A* search over the state space to find optimal action sequences
that transform an initial world state into a goal state.

Based on: Orkin, J. (2006). "Three States and a Plan: The A.I. of F.E.A.R."
Game Developers Conference.

Key concepts:
- World states are nodes in the search graph
- Actions are edges that transform states
- A* finds optimal path from initial to goal state
- Heuristic guides search toward goal (admissible for optimality)
"""

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .action_library import Action, ActionLibrary
from .world_state import WorldState

logger = logging.getLogger(__name__)


def get_metrics_registry():
    """
    Get metrics registry with graceful fallback.
    
    Attempts to import from chat metrics system. If unavailable,
    returns a dummy registry that silently ignores metrics.
    """
    try:
        from ...chat.metrics import metrics_registry
        return metrics_registry
    except ImportError:
        # Dummy registry for when chat system not available
        class DummyRegistry:
            def inc(self, *args, **kwargs):
                pass
            def observe(self, *args, **kwargs):
                pass
            def observe_hist(self, *args, **kwargs):
                pass
            def mark_event(self, *args, **kwargs):
                pass
        return DummyRegistry()


@dataclass
class PlanStep:
    """A single step in a plan (one action to execute)."""

    action: Action
    state_before: WorldState
    state_after: WorldState
    cost: float
    step_number: int

    def __repr__(self) -> str:
        return (
            f"PlanStep({self.step_number}: {self.action.name}, "
            f"cost={self.cost:.2f})"
        )


@dataclass
class Plan:
    """A complete plan from initial state to goal state."""

    steps: list[PlanStep]
    initial_state: WorldState
    goal_state: WorldState
    total_cost: float
    nodes_expanded: int
    planning_time_ms: float

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"Plan(steps={len(self.steps)}, "
            f"cost={self.total_cost:.2f}, "
            f"nodes={self.nodes_expanded}, "
            f"time={self.planning_time_ms:.1f}ms)"
        )

    def is_empty(self) -> bool:
        """Check if plan has no steps."""
        return len(self.steps) == 0

    def get_action_sequence(self) -> list[str]:
        """Get list of action names in execution order."""
        return [step.action.name for step in self.steps]


@dataclass(order=True)
class _SearchNode:
    """Internal node for A* search priority queue."""

    f_score: float  # g + h (total estimated cost)
    g_score: float = field(compare=False)  # Cost from start
    h_score: float = field(compare=False)  # Heuristic to goal
    state: WorldState = field(compare=False)
    parent: Optional["_SearchNode"] = field(default=None, compare=False)
    action: Optional[Action] = field(default=None, compare=False)


class GOAPPlanner:
    """
    Goal-Oriented Action Planner using A* search.

    Finds optimal action sequences to achieve goals by:
    1. Starting from initial world state
    2. Exploring applicable actions (satisfying preconditions)
    3. Using heuristic to guide search toward goal
    4. Returning optimal path when goal is satisfied
    """

    def __init__(
        self,
        action_library: ActionLibrary,
        heuristic: Optional[Callable[[WorldState, WorldState], float]] = None,
        constraints: Optional[list] = None,
    ):
        """
        Initialize GOAP planner.

        Args:
            action_library: Library of available actions
            heuristic: Heuristic function (state, goal) -> estimated_cost
                      If None, uses simple goal distance heuristic
            constraints: Optional list of Constraint objects to check during planning
        """
        self.action_library = action_library
        self.heuristic = heuristic or self._default_heuristic
        self.constraints = constraints or []
        self._nodes_expanded = 0
        self._metrics = get_metrics_registry()

    def plan(
        self,
        initial_state: WorldState,
        goal_state: WorldState,
        max_iterations: int = 1000,
        plan_context: Optional[dict] = None,
    ) -> Optional[Plan]:
        """
        Find optimal action sequence from initial to goal state.
        
        Args:
            initial_state: Starting world state
            goal_state: Desired world state (only specified keys must match)
            max_iterations: Maximum search iterations (prevents infinite loops)
            plan_context: Optional planning context (for constraint checking)

        Returns:
            Plan object with action sequence, or None if no plan found
        """
        start_time = time.perf_counter()
        self._nodes_expanded = 0
        
        # Track planning attempt
        self._metrics.inc("goap_planning_attempts_total")

        # Check if already at goal
        if initial_state.satisfies(goal_state):
            logger.info("Already at goal state, no planning needed")
            self._metrics.inc("goap_plans_found_total")
            self._metrics.observe("goap_plan_length", 0)
            self._metrics.observe("goap_plan_cost", 0.0)
            return Plan(
                steps=[],
                initial_state=initial_state,
                goal_state=goal_state,
                total_cost=0.0,
                nodes_expanded=0,
                planning_time_ms=0.0,
            )

        # Initialize A* search
        open_set: list[_SearchNode] = []
        closed_set: set[WorldState] = set()

        # Start node
        start_node = _SearchNode(
            f_score=self.heuristic(initial_state, goal_state),
            g_score=0.0,
            h_score=self.heuristic(initial_state, goal_state),
            state=initial_state,
            parent=None,
            action=None,
        )
        heapq.heappush(open_set, start_node)

        # A* search loop
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_score
            current = heapq.heappop(open_set)

            # Skip if already visited
            if current.state in closed_set:
                continue

            # Mark as visited
            closed_set.add(current.state)
            self._nodes_expanded += 1

            # Check if goal reached
            if current.state.satisfies(goal_state):
                planning_time_ms = (time.perf_counter() - start_time) * 1000
                plan = self._reconstruct_plan(
                    current, initial_state, goal_state, planning_time_ms
                )
                
                # Track successful planning metrics
                self._metrics.inc("goap_plans_found_total")
                self._metrics.observe("goap_plan_length", len(plan.steps))
                self._metrics.observe("goap_plan_cost", plan.total_cost)
                self._metrics.observe("goap_nodes_expanded", plan.nodes_expanded)
                self._metrics.observe_hist("goap_planning_latency_ms", plan.planning_time_ms)
                
                logger.info(
                    f"Plan found: {len(plan.steps)} steps, "
                    f"cost={plan.total_cost:.2f}, "
                    f"nodes={plan.nodes_expanded}, "
                    f"time={plan.planning_time_ms:.1f}ms"
                )
                return plan

            # Expand neighbors (applicable actions)
            applicable_actions = self.action_library.get_applicable_actions(
                current.state
            )

            for action in applicable_actions:
                # Check constraints before applying action
                if self.constraints:
                    constraint_satisfied = True
                    for constraint in self.constraints:
                        if not constraint.is_satisfied(current.state, action, plan_context):
                            constraint_satisfied = False
                            break
                    if not constraint_satisfied:
                        continue  # Skip this action, violates constraints
                
                # Apply action to get next state
                next_state = action.apply(current.state)

                # Skip if already visited
                if next_state in closed_set:
                    continue

                # Calculate costs
                action_cost = action.get_cost(current.state, next_state)
                tentative_g_score = current.g_score + action_cost
                h_score = self.heuristic(next_state, goal_state)
                f_score = tentative_g_score + h_score

                # Create neighbor node
                neighbor = _SearchNode(
                    f_score=f_score,
                    g_score=tentative_g_score,
                    h_score=h_score,
                    state=next_state,
                    parent=current,
                    action=action,
                )

                heapq.heappush(open_set, neighbor)

        # No plan found
        planning_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Track planning failure metrics
        self._metrics.inc("goap_plans_not_found_total")
        self._metrics.observe("goap_failed_iterations", iterations)
        self._metrics.observe("goap_failed_nodes_expanded", self._nodes_expanded)
        self._metrics.observe_hist("goap_failed_planning_latency_ms", planning_time_ms)
        
        logger.warning(
            f"No plan found after {iterations} iterations, "
            f"expanded {self._nodes_expanded} nodes, "
            f"time={planning_time_ms:.1f}ms"
        )
        return None

    def _reconstruct_plan(
        self,
        goal_node: _SearchNode,
        initial_state: WorldState,
        goal_state: WorldState,
        planning_time_ms: float,
    ) -> Plan:
        """
        Reconstruct plan by following parent pointers from goal to start.

        Args:
            goal_node: Final node that satisfied goal
            initial_state: Original starting state
            goal_state: Original goal state
            planning_time_ms: Time spent planning

        Returns:
            Complete Plan object
        """
        # Build path from goal to start
        path: list[_SearchNode] = []
        current = goal_node
        while current.parent is not None:
            path.append(current)
            current = current.parent

        # Reverse to get start to goal
        path.reverse()

        # Build plan steps
        steps: list[PlanStep] = []
        for i, node in enumerate(path):
            step = PlanStep(
                action=node.action,  # type: ignore  # Already filtered out None
                state_before=node.parent.state,  # type: ignore  # Already filtered
                state_after=node.state,
                cost=node.action.get_cost(node.parent.state, node.state),  # type: ignore
                step_number=i + 1,
            )
            steps.append(step)

        return Plan(
            steps=steps,
            initial_state=initial_state,
            goal_state=goal_state,
            total_cost=goal_node.g_score,
            nodes_expanded=self._nodes_expanded,
            planning_time_ms=planning_time_ms,
        )

    @staticmethod
    def _default_heuristic(current: WorldState, goal: WorldState) -> float:
        """
        Default heuristic: count number of goal keys not yet satisfied.

        This is admissible (never overestimates) if:
        - Each action can satisfy at most one goal key (cost >= 1 per key)
        - In practice, actions may satisfy multiple keys, so this underestimates

        Args:
            current: Current world state
            goal: Goal world state

        Returns:
            Estimated cost to reach goal (number of unsatisfied keys)
        """
        unsatisfied_count = 0
        for key, goal_value in goal.state.items():
            if current.get(key) != goal_value:
                unsatisfied_count += 1
        return float(unsatisfied_count)


def create_planner(
    action_library: Optional[ActionLibrary] = None,
    heuristic: Optional[Callable[[WorldState, WorldState], float]] = None,
) -> GOAPPlanner:
    """
    Convenience factory for creating a GOAP planner.

    Args:
        action_library: Action library (creates default if None)
        heuristic: Heuristic function (uses default if None)

    Returns:
        Configured GOAPPlanner instance
    """
    if action_library is None:
        from .action_library import create_default_action_library

        action_library = create_default_action_library()

    return GOAPPlanner(action_library=action_library, heuristic=heuristic)
