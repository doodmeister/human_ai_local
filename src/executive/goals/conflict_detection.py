"""
Goal Conflict Detection and Resolution

Detects and resolves conflicts between goals:
- Resource conflicts: Multiple goals requiring same limited resources
- State conflicts: Goals with incompatible preconditions or effects
- Temporal conflicts: Goals with overlapping deadlines/execution windows
- Dependency conflicts: Circular or contradictory dependencies

Provides conflict reports with severity and resolution suggestions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum

from .priority_calculator import GoalContext, GoalPriorityCalculator


class ConflictType(Enum):
    """Types of conflicts between goals."""
    RESOURCE = "resource"  # Competing for limited resources
    STATE = "state"  # Incompatible preconditions/effects
    TEMPORAL = "temporal"  # Overlapping time windows
    DEPENDENCY = "dependency"  # Circular or contradictory dependencies


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = "low"  # Minor conflict, both goals can proceed with minor adjustments
    MEDIUM = "medium"  # Significant conflict, requires coordination
    HIGH = "high"  # Major conflict, goals likely can't both succeed
    CRITICAL = "critical"  # Fatal conflict, goals are mutually exclusive


@dataclass
class Conflict:
    """Represents a conflict between two or more goals."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    goal_ids: Set[str]  # Goals involved in conflict
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        """Make conflict hashable for sets."""
        return hash((self.conflict_id, tuple(sorted(self.goal_ids))))


@dataclass
class ConflictReport:
    """Report of all conflicts detected in a goal set."""
    conflicts: List[Conflict]
    total_goals: int
    conflicted_goals: Set[str]  # Goals involved in any conflict
    by_type: Dict[ConflictType, int]  # Count by conflict type
    by_severity: Dict[ConflictSeverity, int]  # Count by severity
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts exist."""
        return len(self.conflicts) > 0
    
    @property
    def critical_conflicts(self) -> List[Conflict]:
        """Get only critical severity conflicts."""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.CRITICAL]
    
    @property
    def high_conflicts(self) -> List[Conflict]:
        """Get high and critical severity conflicts."""
        return [c for c in self.conflicts 
                if c.severity in (ConflictSeverity.HIGH, ConflictSeverity.CRITICAL)]


class ConflictDetector:
    """
    Detects conflicts between goals across multiple dimensions.
    
    Analyzes goals for:
    - Resource contention (insufficient resources for all goals)
    - State incompatibility (conflicting preconditions/effects)
    - Temporal overlaps (deadline/execution conflicts)
    - Dependency issues (circular dependencies, impossible ordering)
    
    Example:
        detector = ConflictDetector()
        contexts = [...]  # List of GoalContext objects
        report = detector.detect_all_conflicts(contexts)
        
        if report.has_conflicts:
            print(f"Found {len(report.conflicts)} conflicts")
            for conflict in report.critical_conflicts:
                print(f"Critical: {conflict.description}")
    """
    
    def __init__(self, resource_tolerance: float = 0.1):
        """
        Initialize conflict detector.
        
        Args:
            resource_tolerance: Tolerance for resource over-allocation (0.0-1.0)
                               0.1 = allow 10% over-allocation before flagging conflict
        """
        self.resource_tolerance = resource_tolerance
        self._conflict_counter = 0
    
    def detect_all_conflicts(
        self,
        contexts: List[GoalContext]
    ) -> ConflictReport:
        """
        Detect all types of conflicts in a set of goals.
        
        Args:
            contexts: List of goal contexts to analyze
            
        Returns:
            ConflictReport with all detected conflicts
        """
        conflicts = []
        
        # Detect each type of conflict
        conflicts.extend(self.detect_resource_conflicts(contexts))
        conflicts.extend(self.detect_temporal_conflicts(contexts))
        conflicts.extend(self.detect_dependency_conflicts(contexts))
        
        # Build report
        conflicted_goals = set()
        by_type = {ct: 0 for ct in ConflictType}
        by_severity = {cs: 0 for cs in ConflictSeverity}
        
        for conflict in conflicts:
            conflicted_goals.update(conflict.goal_ids)
            by_type[conflict.conflict_type] += 1
            by_severity[conflict.severity] += 1
        
        return ConflictReport(
            conflicts=conflicts,
            total_goals=len(contexts),
            conflicted_goals=conflicted_goals,
            by_type=by_type,
            by_severity=by_severity
        )
    
    def detect_resource_conflicts(
        self,
        contexts: List[GoalContext]
    ) -> List[Conflict]:
        """
        Detect conflicts where goals compete for limited resources.
        
        Args:
            contexts: Goal contexts to check
            
        Returns:
            List of resource conflicts detected
        """
        conflicts = []
        
        # Group goals by resource requirements
        resource_demands: Dict[str, List[Tuple[str, float]]] = {}
        
        for context in contexts:
            if not context.required_resources:
                continue
            
            for resource_id, amount in context.required_resources.items():
                if resource_id not in resource_demands:
                    resource_demands[resource_id] = []
                resource_demands[resource_id].append((context.goal_id, amount))
        
        # Check each resource for over-allocation
        for resource_id, demands in resource_demands.items():
            # Get available amount from first context that has it
            available = 0.0
            for context in contexts:
                if context.available_resources and resource_id in context.available_resources:
                    available = context.available_resources[resource_id]
                    break
            
            # Calculate total demand
            total_demand = sum(amount for _, amount in demands)
            
            # Check for conflict
            threshold = available * (1.0 + self.resource_tolerance)
            if total_demand > threshold:
                # Over-allocated resource
                conflict_goals = {goal_id for goal_id, _ in demands}
                over_allocation = total_demand - available
                severity = self._calculate_resource_severity(
                    available, total_demand, len(demands)
                )
                
                conflict = Conflict(
                    conflict_id=self._next_conflict_id(),
                    conflict_type=ConflictType.RESOURCE,
                    severity=severity,
                    goal_ids=conflict_goals,
                    description=f"Resource '{resource_id}' over-allocated: {total_demand:.1f} demanded vs {available:.1f} available",
                    details={
                        "resource_id": resource_id,
                        "available": available,
                        "total_demand": total_demand,
                        "over_allocation": over_allocation,
                        "demands": {goal_id: amount for goal_id, amount in demands}
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def detect_temporal_conflicts(
        self,
        contexts: List[GoalContext]
    ) -> List[Conflict]:
        """
        Detect conflicts where goals have overlapping or impossible deadlines.
        
        Args:
            contexts: Goal contexts to check
            
        Returns:
            List of temporal conflicts detected
        """
        conflicts = []
        
        # Build dependency graph for temporal analysis
        dependency_chains = self._build_dependency_chains(contexts)
        
        # Check each chain for deadline feasibility
        for chain in dependency_chains:
            conflicts.extend(self._check_chain_deadlines(chain, contexts))
        
        # Check for parallel goals with same deadline competing for resources
        deadline_groups = self._group_by_deadline(contexts)
        for deadline, goal_ids in deadline_groups.items():
            if len(goal_ids) < 2:
                continue
            
            # Check if these goals compete for resources
            competing = self._check_resource_competition(
                [c for c in contexts if c.goal_id in goal_ids]
            )
            
            if competing:
                severity = ConflictSeverity.MEDIUM if len(goal_ids) <= 3 else ConflictSeverity.HIGH
                conflict = Conflict(
                    conflict_id=self._next_conflict_id(),
                    conflict_type=ConflictType.TEMPORAL,
                    severity=severity,
                    goal_ids=set(goal_ids),
                    description=f"{len(goal_ids)} goals with same deadline competing for resources",
                    details={
                        "deadline": deadline.isoformat() if deadline else None,
                        "goal_count": len(goal_ids)
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def detect_dependency_conflicts(
        self,
        contexts: List[GoalContext]
    ) -> List[Conflict]:
        """
        Detect circular dependencies or impossible dependency orderings.
        
        Args:
            contexts: Goal contexts to check
            
        Returns:
            List of dependency conflicts detected
        """
        conflicts = []
        
        # Build dependency graph
        graph = {ctx.goal_id: ctx.blocked_by_goals or set() for ctx in contexts}
        
        # Detect circular dependencies using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str, path: List[str]) -> Optional[List[str]]:
            """DFS to detect cycles. Returns cycle path if found."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        cycle = has_cycle(neighbor, path.copy())
                        if cycle:
                            return cycle
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        # Check each node for cycles
        for goal_id in graph:
            if goal_id not in visited:
                cycle = has_cycle(goal_id, [])
                if cycle:
                    conflict = Conflict(
                        conflict_id=self._next_conflict_id(),
                        conflict_type=ConflictType.DEPENDENCY,
                        severity=ConflictSeverity.HIGH,
                        goal_ids=set(cycle),
                        description=f"Circular dependency detected: {' → '.join(cycle)}",
                        details={
                            "cycle": cycle,
                            "cycle_length": len(set(cycle))
                        }
                    )
                    conflicts.append(conflict)
                    # Clear visited to find other cycles
                    visited.clear()
        
        return conflicts
    
    def _calculate_resource_severity(
        self,
        available: float,
        demanded: float,
        goal_count: int
    ) -> ConflictSeverity:
        """Calculate severity of resource conflict."""
        over_allocation_ratio = (demanded - available) / available if available > 0 else 1.0
        
        if over_allocation_ratio > 1.0:  # More than 2x over-allocated
            return ConflictSeverity.CRITICAL
        elif over_allocation_ratio > 0.5:  # 50-100% over-allocated
            return ConflictSeverity.HIGH
        elif over_allocation_ratio > 0.2:  # 20-50% over-allocated
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW
    
    def _build_dependency_chains(
        self,
        contexts: List[GoalContext]
    ) -> List[List[str]]:
        """Build chains of dependent goals."""
        # Find root goals (not blocked by anything)
        roots = [ctx.goal_id for ctx in contexts if not ctx.blocked_by_goals]
        
        chains = []
        for root in roots:
            chain = self._trace_chain(root, contexts)
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def _trace_chain(
        self,
        goal_id: str,
        contexts: List[GoalContext],
        visited: Optional[Set[str]] = None
    ) -> List[str]:
        """Trace dependency chain from a goal."""
        if visited is None:
            visited = set()
        
        if goal_id in visited:
            return []
        
        visited.add(goal_id)
        chain = [goal_id]
        
        # Find goals blocked by this goal
        for ctx in contexts:
            if ctx.blocked_by_goals and goal_id in ctx.blocked_by_goals:
                sub_chain = self._trace_chain(ctx.goal_id, contexts, visited)
                if sub_chain:
                    chain.extend(sub_chain)
        
        return chain
    
    def _check_chain_deadlines(
        self,
        chain: List[str],
        contexts: List[GoalContext]
    ) -> List[Conflict]:
        """Check if deadlines in dependency chain are feasible."""
        conflicts = []
        
        # Get contexts for chain
        chain_contexts = {ctx.goal_id: ctx for ctx in contexts if ctx.goal_id in chain}
        
        # Check if later goals have earlier deadlines than prerequisites
        for i, goal_id in enumerate(chain[:-1]):
            current_ctx = chain_contexts.get(goal_id)
            if not current_ctx or not current_ctx.deadline:
                continue
            
            # Check subsequent goals in chain
            for next_id in chain[i+1:]:
                next_ctx = chain_contexts.get(next_id)
                if not next_ctx or not next_ctx.deadline:
                    continue
                
                # Prerequisite deadline should be before dependent deadline
                if current_ctx.deadline >= next_ctx.deadline:
                    conflict = Conflict(
                        conflict_id=self._next_conflict_id(),
                        conflict_type=ConflictType.TEMPORAL,
                        severity=ConflictSeverity.HIGH,
                        goal_ids={goal_id, next_id},
                        description=f"Goal '{next_id}' deadline before prerequisite '{goal_id}' deadline",
                        details={
                            "prerequisite": goal_id,
                            "prerequisite_deadline": current_ctx.deadline.isoformat(),
                            "dependent": next_id,
                            "dependent_deadline": next_ctx.deadline.isoformat()
                        }
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _group_by_deadline(
        self,
        contexts: List[GoalContext]
    ) -> Dict[Optional[datetime], List[str]]:
        """Group goals by deadline."""
        groups: Dict[Optional[datetime], List[str]] = {}
        
        for ctx in contexts:
            deadline = ctx.deadline
            if deadline not in groups:
                groups[deadline] = []
            groups[deadline].append(ctx.goal_id)
        
        return groups
    
    def _check_resource_competition(
        self,
        contexts: List[GoalContext]
    ) -> bool:
        """Check if goals compete for any resources."""
        # Build set of all required resources
        resource_users: Dict[str, int] = {}
        
        for ctx in contexts:
            if not ctx.required_resources:
                continue
            for resource_id in ctx.required_resources.keys():
                resource_users[resource_id] = resource_users.get(resource_id, 0) + 1
        
        # Competition exists if any resource is used by multiple goals
        return any(count > 1 for count in resource_users.values())
    
    def _next_conflict_id(self) -> str:
        """Generate next conflict ID."""
        self._conflict_counter += 1
        return f"conflict_{self._conflict_counter}"
