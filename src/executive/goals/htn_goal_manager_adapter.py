"""
HTN Goal Manager Adapter.

Bridges HTN planning with the existing legacy GoalManager system.

The adapter provides:
1. Goal conversion between HTN and Legacy formats
2. Unified API for creating/managing goals
3. Automatic decomposition for compound goals
4. Backward compatibility with existing code

Architecture:
    User/System
        ↓
    HTNGoalManagerAdapter
        ↓
    ├─> HTNManager (decomposition)
    └─> GoalManager (storage/CRUD)

Usage:
    adapter = HTNGoalManagerAdapter(goal_manager, htn_manager)
    
    # Create and decompose compound goal
    result = adapter.create_compound_goal(
        description="Research and write report on AI agents",
        priority=8
    )
    
    # Get primitive goals ready to execute
    ready_goals = adapter.get_ready_primitive_goals()
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import uuid

# HTN imports
from .goal_taxonomy import Goal as HTNGoal, GoalType, GoalStatus as HTNGoalStatus, GoalPriority as HTNGoalPriority
from .htn_manager import HTNManager, DecompositionResult

# Legacy imports
from ..goal_manager import GoalManager, Goal as LegacyGoal, GoalStatus as LegacyGoalStatus, GoalPriority as LegacyGoalPriority

# Goal intelligence imports
from .priority_calculator import GoalPriorityCalculator, GoalContext, PriorityScore
from .conflict_detection import ConflictDetector, ConflictReport


class HTNGoalManagerAdapter:
    """
    Adapter bridging HTN planning with legacy GoalManager.
    
    Responsibilities:
    - Convert goals between HTN and Legacy formats
    - Store all goals (compound + primitive) in GoalManager
    - Maintain HTN decomposition metadata
    - Provide unified API for both systems
    """
    
    def __init__(
        self,
        goal_manager: GoalManager,
        htn_manager: Optional[HTNManager] = None,
        enable_priority_calculation: bool = True,
        enable_conflict_detection: bool = True
    ):
        """
        Initialize adapter.
        
        Args:
            goal_manager: Legacy goal manager for storage/CRUD
            htn_manager: HTN manager for decomposition (creates default if None)
            enable_priority_calculation: Enable dynamic priority calculation
            enable_conflict_detection: Enable conflict detection
        """
        self.goal_manager = goal_manager
        self.htn_manager = htn_manager or HTNManager()
        
        # Goal intelligence features (Week 10)
        self.enable_priority_calculation = enable_priority_calculation
        self.enable_conflict_detection = enable_conflict_detection
        
        if enable_priority_calculation:
            self.priority_calculator = GoalPriorityCalculator()
        else:
            self.priority_calculator = None
        
        if enable_conflict_detection:
            self.conflict_detector = ConflictDetector()
        else:
            self.conflict_detector = None
        
        # Track HTN-specific metadata (not stored in legacy system)
        # Maps legacy goal_id -> HTN metadata
        self._htn_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Track priority scores
        self._priority_scores: Dict[str, PriorityScore] = {}
        
        # Track resource availability (for conflict detection)
        self._available_resources: Dict[str, float] = {}
    
    # ====================
    # Goal Creation
    # ====================
    
    def create_compound_goal(
        self,
        description: str,
        priority: int = 5,
        preconditions: Optional[Dict[str, Any]] = None,
        postconditions: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        current_state: Optional[Dict[str, Any]] = None
    ) -> DecompositionResult:
        """
        Create a compound goal and decompose it into subtasks.
        
        Workflow:
        1. Create HTN compound goal
        2. Decompose using HTN manager
        3. Convert all goals to legacy format
        4. Store in legacy GoalManager
        5. Return decomposition result
        
        Args:
            description: Goal description
            priority: Priority (1-10, HTN scale)
            preconditions: Required conditions to start
            postconditions: Expected effects when complete
            deadline: Target completion date
            dependencies: Goal IDs this depends on
            current_state: Current world state for decomposition
            
        Returns:
            DecompositionResult with all generated goals
        """
        # Create HTN goal
        htn_goal = HTNGoal(
            id=str(uuid.uuid4()),
            description=description,
            goal_type=GoalType.COMPOUND,
            status=HTNGoalStatus.PENDING,
            priority=priority,
            preconditions=preconditions or {},
            postconditions=postconditions or {},
            deadline=deadline,
            dependencies=dependencies or []
        )
        
        # Decompose using HTN
        result = self.htn_manager.decompose(
            goal=htn_goal,
            current_state=current_state or {}
        )
        
        if not result.success:
            # Store the failed compound goal anyway
            self._store_htn_goal(htn_goal)
            return result
        
        # Store all generated goals in legacy system
        for goal in result.goals:
            self._store_htn_goal(goal)
        
        return result
    
    def create_primitive_goal(
        self,
        description: str,
        priority: int = 5,
        preconditions: Optional[Dict[str, Any]] = None,
        postconditions: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """
        Create a primitive (executable) goal.
        
        Args:
            description: Goal description
            priority: Priority (1-10, HTN scale)
            preconditions: Required conditions to start
            postconditions: Expected effects when complete
            deadline: Target completion date
            dependencies: Goal IDs this depends on
            parent_id: Parent goal ID
            
        Returns:
            Goal ID
        """
        htn_goal = HTNGoal(
            id=str(uuid.uuid4()),
            description=description,
            goal_type=GoalType.PRIMITIVE,
            status=HTNGoalStatus.PENDING,
            priority=priority,
            preconditions=preconditions or {},
            postconditions=postconditions or {},
            deadline=deadline,
            dependencies=dependencies or [],
            parent_id=parent_id
        )
        
        self._store_htn_goal(htn_goal)
        return htn_goal.id
    
    # ====================
    # Goal Retrieval
    # ====================
    
    def get_goal(self, goal_id: str) -> Optional[LegacyGoal]:
        """Get goal from legacy system."""
        return self.goal_manager.get_goal(goal_id)
    
    def get_htn_metadata(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get HTN-specific metadata for a goal."""
        return self._htn_metadata.get(goal_id)
    
    def get_ready_primitive_goals(
        self,
        completed_goal_ids: Optional[Set[str]] = None
    ) -> List[LegacyGoal]:
        """
        Get primitive goals ready to execute.
        
        A goal is ready if:
        - It's primitive (executable)
        - All dependencies are satisfied
        - Status is PENDING or ACTIVE
        
        Args:
            completed_goal_ids: Set of completed goal IDs
            
        Returns:
            List of ready goals (sorted by priority)
        """
        if completed_goal_ids is None:
            # Get completed goals from legacy system
            completed_goal_ids = {
                g.id for g in self.goal_manager.goals.values()
                if g.status in [LegacyGoalStatus.COMPLETED, LegacyGoalStatus.CANCELLED]
            }
        
        ready_goals = []
        for goal in self.goal_manager.goals.values():
            # Check if primitive
            metadata = self._htn_metadata.get(goal.id)
            if not metadata or metadata.get('goal_type') != 'primitive':
                continue
            
            # Check status
            if goal.status not in [LegacyGoalStatus.CREATED, LegacyGoalStatus.ACTIVE]:
                continue
            
            # Check dependencies
            dependencies = set(metadata.get('dependencies', []))
            if not dependencies.issubset(completed_goal_ids):
                continue
            
            ready_goals.append(goal)
        
        # Sort by priority (higher first)
        ready_goals.sort(key=lambda g: g.priority.value, reverse=True)
        return ready_goals
    
    def get_all_primitive_goals(self) -> List[LegacyGoal]:
        """Get all primitive goals."""
        return [
            goal for goal in self.goal_manager.goals.values()
            if self._htn_metadata.get(goal.id, {}).get('goal_type') == 'primitive'
        ]
    
    def get_all_compound_goals(self) -> List[LegacyGoal]:
        """Get all compound goals."""
        return [
            goal for goal in self.goal_manager.goals.values()
            if self._htn_metadata.get(goal.id, {}).get('goal_type') == 'compound'
        ]
    
    # ====================
    # Goal Updates
    # ====================
    
    def update_goal_status(
        self,
        goal_id: str,
        status: LegacyGoalStatus,
        reason: str = ""
    ) -> bool:
        """
        Update goal status.
        
        Args:
            goal_id: Goal ID
            status: New status
            reason: Reason for status change
            
        Returns:
            True if successful
        """
        goal = self.goal_manager.get_goal(goal_id)
        if not goal:
            return False
        
        goal.status = status
        goal.updated_at = datetime.now()
        
        # Log reason
        if reason:
            goal.context.setdefault('status_history', []).append({
                'timestamp': goal.updated_at.isoformat(),
                'status': status.value,
                'reason': reason
            })
        
        return True
    
    def mark_goal_completed(self, goal_id: str, reason: str = "") -> bool:
        """Mark goal as completed."""
        return self.update_goal_status(goal_id, LegacyGoalStatus.COMPLETED, reason)
    
    def mark_goal_failed(self, goal_id: str, reason: str = "") -> bool:
        """Mark goal as failed."""
        return self.update_goal_status(goal_id, LegacyGoalStatus.FAILED, reason)
    
    # ====================
    # Conversion Helpers
    # ====================
    
    def _store_htn_goal(self, htn_goal: HTNGoal) -> str:
        """
        Convert HTN goal to legacy format and store.
        
        Args:
            htn_goal: HTN goal to store
            
        Returns:
            Goal ID
        """
        # Convert HTN goal to legacy goal
        legacy_goal = self._htn_to_legacy(htn_goal)
        
        # Store in legacy system
        self.goal_manager.goals[legacy_goal.id] = legacy_goal
        
        # Store HTN metadata separately
        self._htn_metadata[legacy_goal.id] = {
            'goal_type': htn_goal.goal_type.value,
            'preconditions': htn_goal.preconditions,
            'postconditions': htn_goal.postconditions,
            'dependencies': htn_goal.dependencies,
            'subtask_ids': htn_goal.subtask_ids,
            'created_at': htn_goal.created_at
        }
        
        return legacy_goal.id
    
    def _htn_to_legacy(self, htn_goal: HTNGoal) -> LegacyGoal:
        """
        Convert HTN goal to legacy goal format.
        
        Conversions:
        - description → title (first 100 chars) + description (full)
        - HTN status → Legacy status
        - HTN priority (1-10) → Legacy priority (1-5)
        - preconditions/postconditions → context
        """
        # Create title from description (first 100 chars)
        title = htn_goal.description[:100]
        if len(htn_goal.description) > 100:
            title += "..."
        
        # Convert status
        legacy_status = self._convert_htn_status_to_legacy(htn_goal.status)
        
        # Convert priority (HTN: 1-10 → Legacy: 1-5)
        legacy_priority = self._convert_htn_priority_to_legacy(htn_goal.priority)
        
        # Build context with HTN metadata
        context = {
            'htn_managed': True,
            'goal_type': htn_goal.goal_type.value,
            'preconditions': htn_goal.preconditions,
            'postconditions': htn_goal.postconditions
        }
        if htn_goal.metadata:
            context['htn_metadata'] = htn_goal.metadata
        
        # Create legacy goal
        legacy_goal = LegacyGoal(
            id=htn_goal.id,
            title=title,
            description=htn_goal.description,
            priority=legacy_priority,
            status=legacy_status,
            parent_id=htn_goal.parent_id,
            dependencies=set(htn_goal.dependencies),
            target_date=htn_goal.deadline,
            context=context
        )
        
        return legacy_goal
    
    def _convert_htn_status_to_legacy(self, htn_status: HTNGoalStatus) -> LegacyGoalStatus:
        """Convert HTN status to legacy status."""
        mapping = {
            HTNGoalStatus.PENDING: LegacyGoalStatus.CREATED,
            HTNGoalStatus.ACTIVE: LegacyGoalStatus.ACTIVE,
            HTNGoalStatus.COMPLETED: LegacyGoalStatus.COMPLETED,
            HTNGoalStatus.FAILED: LegacyGoalStatus.FAILED,
            HTNGoalStatus.BLOCKED: LegacyGoalStatus.PAUSED
        }
        return mapping.get(htn_status, LegacyGoalStatus.CREATED)
    
    def _convert_htn_priority_to_legacy(self, htn_priority: int) -> LegacyGoalPriority:
        """
        Convert HTN priority (1-10) to legacy priority (1-5).
        
        Mapping:
        - 10 (CRITICAL) → 5 (CRITICAL)
        - 8 (HIGH) → 4 (URGENT)
        - 5 (MEDIUM) → 3 (HIGH)
        - 3 (LOW) → 2 (MEDIUM)
        - 1 (OPTIONAL) → 1 (LOW)
        """
        if htn_priority >= 10:
            return LegacyGoalPriority.CRITICAL
        elif htn_priority >= 7:
            return LegacyGoalPriority.URGENT
        elif htn_priority >= 5:
            return LegacyGoalPriority.HIGH
        elif htn_priority >= 3:
            return LegacyGoalPriority.MEDIUM
        else:
            return LegacyGoalPriority.LOW
    
    # ====================
    # Statistics
    # ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed goals."""
        all_goals = list(self.goal_manager.goals.values())
        htn_goals = [g for g in all_goals if g.id in self._htn_metadata]
        
        primitive_goals = self.get_all_primitive_goals()
        compound_goals = self.get_all_compound_goals()
        
        return {
            'total_goals': len(all_goals),
            'htn_managed_goals': len(htn_goals),
            'primitive_goals': len(primitive_goals),
            'compound_goals': len(compound_goals),
            'ready_primitive_goals': len(self.get_ready_primitive_goals()),
            'completed_goals': len([
                g for g in htn_goals
                if g.status == LegacyGoalStatus.COMPLETED
            ]),
            'active_goals': len([
                g for g in htn_goals
                if g.status == LegacyGoalStatus.ACTIVE
            ]),
            'failed_goals': len([
                g for g in htn_goals
                if g.status == LegacyGoalStatus.FAILED
            ])
        }
    
    # ====================
    # Goal Intelligence (Week 10)
    # ====================
    
    def calculate_goal_priority(
        self,
        goal_id: str,
        user_importance: float = 5.0,
        system_importance: float = 5.0,
        strategic_value: float = 5.0
    ) -> Optional[PriorityScore]:
        """
        Calculate dynamic priority for a goal.
        
        Args:
            goal_id: Goal to calculate priority for
            user_importance: User-assigned importance (1-10)
            system_importance: System-assigned importance (1-10)
            strategic_value: Strategic value (1-10)
            
        Returns:
            PriorityScore with detailed breakdown, or None if not enabled
        """
        if not self.enable_priority_calculation or not self.priority_calculator:
            return None
        
        goal = self.goal_manager.get_goal(goal_id)
        if not goal:
            return None
        
        metadata = self._htn_metadata.get(goal_id, {})
        
        # Build goal context
        context = GoalContext(
            goal_id=goal_id,
            deadline=goal.target_date,
            user_importance=user_importance,
            system_importance=system_importance,
            strategic_value=strategic_value,
            blocks_goals=self._get_blocked_by_goal(goal_id),
            blocked_by_goals=goal.dependencies,
            required_resources=metadata.get('required_resources'),
            available_resources=self._available_resources,
            created_at=goal.created_at
        )
        
        # Calculate priority
        score = self.priority_calculator.calculate_priority(context)
        
        # Cache the score
        self._priority_scores[goal_id] = score
        
        return score
    
    def recalculate_all_priorities(
        self,
        importance_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, PriorityScore]:
        """
        Recalculate priorities for all goals.
        
        Args:
            importance_overrides: Map of goal_id -> user_importance overrides
            
        Returns:
            Dictionary mapping goal_id to PriorityScore
        """
        if not self.enable_priority_calculation:
            return {}
        
        importance_overrides = importance_overrides or {}
        scores = {}
        
        for goal_id in self._htn_metadata.keys():
            user_importance = importance_overrides.get(goal_id, 5.0)
            score = self.calculate_goal_priority(goal_id, user_importance=user_importance)
            if score:
                scores[goal_id] = score
        
        return scores
    
    def detect_conflicts(self) -> Optional[ConflictReport]:
        """
        Detect conflicts between all managed goals.
        
        Returns:
            ConflictReport with all detected conflicts, or None if not enabled
        """
        if not self.enable_conflict_detection or not self.conflict_detector:
            return None
        
        # Build contexts for all goals
        contexts = []
        for goal_id in self._htn_metadata.keys():
            goal = self.goal_manager.get_goal(goal_id)
            if not goal:
                continue
            
            metadata = self._htn_metadata.get(goal_id, {})
            
            context = GoalContext(
                goal_id=goal_id,
                deadline=goal.target_date,
                blocks_goals=self._get_blocked_by_goal(goal_id),
                blocked_by_goals=goal.dependencies,
                required_resources=metadata.get('required_resources'),
                available_resources=self._available_resources,
                created_at=goal.created_at
            )
            contexts.append(context)
        
        # Detect conflicts
        report = self.conflict_detector.detect_all_conflicts(contexts)
        return report
    
    def set_resource_availability(self, resource_id: str, amount: float):
        """
        Set available amount for a resource.
        
        Args:
            resource_id: Resource identifier
            amount: Available amount
        """
        self._available_resources[resource_id] = amount
    
    def update_resource_availability(self, resources: Dict[str, float]):
        """
        Update multiple resource availabilities.
        
        Args:
            resources: Dictionary mapping resource_id to available amount
        """
        self._available_resources.update(resources)
    
    def get_prioritized_goals(
        self,
        filter_ready: bool = False
    ) -> List[tuple[LegacyGoal, PriorityScore]]:
        """
        Get goals sorted by priority (highest first).
        
        Args:
            filter_ready: Only return ready-to-execute primitive goals
            
        Returns:
            List of (goal, priority_score) tuples sorted by priority
        """
        if not self.enable_priority_calculation:
            return []
        
        # Recalculate priorities
        self.recalculate_all_priorities()
        
        # Get goals
        if filter_ready:
            goals = self.get_ready_primitive_goals()
        else:
            goals = [self.goal_manager.get_goal(gid) for gid in self._htn_metadata.keys()]
            goals = [g for g in goals if g is not None]
        
        # Pair with scores
        goal_score_pairs = []
        for goal in goals:
            score = self._priority_scores.get(goal.id)
            if score:
                goal_score_pairs.append((goal, score))
        
        # Sort by priority (descending)
        goal_score_pairs.sort(key=lambda x: x[1].final_score, reverse=True)
        
        return goal_score_pairs
    
    def _get_blocked_by_goal(self, goal_id: str) -> Set[str]:
        """
        Get set of goals that are blocked by this goal.
        
        Args:
            goal_id: Goal that may block others
            
        Returns:
            Set of goal IDs that depend on this goal
        """
        blocked = set()
        for gid, metadata in self._htn_metadata.items():
            goal = self.goal_manager.get_goal(gid)
            if goal and goal_id in goal.dependencies:
                blocked.add(gid)
        return blocked
