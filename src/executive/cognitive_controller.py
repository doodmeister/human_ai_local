"""Cognitive Controller (Phase 2: Advisor-Only)

This module provides resource and mode *recommendations* to the executive.

Phase 2 constraint:
    - No module except the executive core may commit actions.

Accordingly, this controller:
    - Does not autonomously run background control loops by default.
    - Does not mutate external subsystems (e.g., attention capacity).
    - Produces advisory recommendations and status for the executive to consume.
"""

from typing import Deque, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import logging
from collections import deque

from ..cognition.attention.attention_mechanism import AttentionMechanism
from ..memory import MemorySystem
from .goal_manager import GoalManager
from .task_planner import TaskPlanner
from .decision_engine import DecisionEngine
from ..core.cognitive_tick import CognitiveStep, CognitiveTick

# Initialize logger
logger = logging.getLogger(__name__)

class CognitiveMode(Enum):
    """Different modes of cognitive operation"""
    FOCUSED = "focused"  # Deep focus on single task
    MULTI_TASK = "multi_task"  # Managing multiple tasks
    EXPLORATION = "exploration"  # Exploring new information
    REFLECTION = "reflection"  # Metacognitive reflection
    RECOVERY = "recovery"  # Rest and recovery
    EMERGENCY = "emergency"  # High-priority interrupt handling

class ResourceType(Enum):
    """Types of cognitive resources"""
    ATTENTION = "attention"
    MEMORY = "memory"
    PROCESSING = "processing"
    ENERGY = "energy"
    TIME = "time"

@dataclass
class CognitiveResource:
    """
    Represents a cognitive resource with current state and limits
    
    Attributes:
        resource_type: Type of resource
        current_level: Current resource level (0.0 to 1.0)
        max_level: Maximum resource level
        depletion_rate: Rate at which resource depletes
        recovery_rate: Rate at which resource recovers
        critical_threshold: Threshold below which resource is critically low
        reserved_amount: Amount reserved for critical operations
    """
    resource_type: ResourceType
    current_level: float = 1.0
    max_level: float = 1.0
    depletion_rate: float = 0.01
    recovery_rate: float = 0.02
    critical_threshold: float = 0.2
    reserved_amount: float = 0.1
    
    def __post_init__(self):
        """Validate resource parameters"""
        if not (0.0 <= self.current_level <= 1.0):
            raise ValueError("Current level must be between 0.0 and 1.0")
        if not (0.0 <= self.critical_threshold <= 1.0):
            raise ValueError("Critical threshold must be between 0.0 and 1.0")
    
    def is_available(self, amount: float) -> bool:
        """Check if requested amount of resource is available"""
        available = self.current_level - self.reserved_amount
        return available >= amount
    
    def is_critical(self) -> bool:
        """Check if resource is at critical level"""
        return self.current_level <= self.critical_threshold
    
    def consume(self, amount: float) -> bool:
        """
        Consume resource if available
        
        Args:
            amount: Amount to consume
            
        Returns:
            True if successfully consumed, False otherwise
        """
        if not self.is_available(amount):
            return False
        
        self.current_level = max(0.0, self.current_level - amount)
        return True
    
    def recover(self, amount: float) -> None:
        """Recover resource"""
        self.current_level = min(self.max_level, self.current_level + amount)
    
    def update(self, time_delta: float) -> None:
        """Update resource levels based on time passage"""
        if self.current_level < self.max_level:
            recovery_amount = self.recovery_rate * time_delta
            self.recover(recovery_amount)

@dataclass
class CognitiveState:
    """
    Current state of the cognitive system
    
    Attributes:
        mode: Current cognitive mode
        resources: Current resource levels
        active_processes: Currently active processes
        performance_metrics: Performance tracking
        last_update: Last update timestamp
        mode_duration: How long in current mode
        transition_count: Number of mode transitions
    """
    mode: CognitiveMode = CognitiveMode.MULTI_TASK
    resources: Dict[ResourceType, CognitiveResource] = field(default_factory=dict)
    active_processes: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    mode_duration: timedelta = field(default_factory=timedelta)
    transition_count: int = 0
    
    def __post_init__(self):
        """Initialize default resources if not provided"""
        if not self.resources:
            self.resources = {
                ResourceType.ATTENTION: CognitiveResource(ResourceType.ATTENTION),
                ResourceType.MEMORY: CognitiveResource(ResourceType.MEMORY),
                ResourceType.PROCESSING: CognitiveResource(ResourceType.PROCESSING),
                ResourceType.ENERGY: CognitiveResource(ResourceType.ENERGY),
                ResourceType.TIME: CognitiveResource(ResourceType.TIME)
            }
    
    def get_resource_summary(self) -> Dict[str, float]:
        """Get summary of all resource levels"""
        return {
            resource_type.value: resource.current_level
            for resource_type, resource in self.resources.items()
        }
    
    def get_critical_resources(self) -> List[ResourceType]:
        """Get list of resources at critical levels"""
        return [
            resource_type for resource_type, resource in self.resources.items()
            if resource.is_critical()
        ]

class CognitiveController:
    """
    Manages cognitive resources and coordinates cognitive processes
    """
    
    def __init__(
        self,
        attention_mechanism: AttentionMechanism,
        memory_system: MemorySystem,
        goal_manager: GoalManager,
        task_planner: TaskPlanner,
        decision_engine: DecisionEngine,
        *,
        enable_autonomous_monitoring: bool = False,
    ):
        """
        Initialize cognitive controller
        
        Args:
            attention_mechanism: Attention management system
            memory_system: Memory management system
            goal_manager: Goal management system
            task_planner: Task planning system
            decision_engine: Decision making system
        """
        self.attention = attention_mechanism
        self.memory = memory_system
        self.goals = goal_manager
        self.tasks = task_planner
        self.decisions = decision_engine

        # Advisor-only configuration
        self._enable_autonomous_monitoring = enable_autonomous_monitoring
        self.recommended_mode: Optional[CognitiveMode] = None
        self.advisor_notes: Deque[str] = deque(maxlen=200)
        
        # Cognitive state
        self.state = CognitiveState()
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.mode_transitions: List[Tuple[datetime, CognitiveMode, CognitiveMode]] = []
        
        # Resource allocation
        self.resource_allocations: Dict[str, Dict[ResourceType, float]] = {}
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Coordination lock
        self._coordination_lock = threading.RLock()
    
    def start_monitoring(self) -> None:
        """Start background monitoring of cognitive state"""
        if not self._enable_autonomous_monitoring:
            logger.warning(
                "CognitiveController.start_monitoring() ignored (advisor-only; enable_autonomous_monitoring=False)"
            )
            return
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                tick = CognitiveTick(owner="cognitive_controller_advisor", kind="monitoring_tick")

                # Perceive
                tick.assert_step(CognitiveStep.PERCEIVE)
                tick.state["timestamp"] = datetime.now().isoformat()
                tick.advance(CognitiveStep.PERCEIVE)

                # Update STM (update internal state / resources)
                tick.assert_step(CognitiveStep.UPDATE_STM)
                self._update_cognitive_state()
                tick.state["mode"] = self.state.mode.value
                tick.advance(CognitiveStep.UPDATE_STM)

                # Retrieve (resource checks)
                tick.assert_step(CognitiveStep.RETRIEVE)
                self._check_resource_levels()
                tick.advance(CognitiveStep.RETRIEVE)

                # Decide
                tick.assert_step(CognitiveStep.DECIDE)
                tick.mark_decided({"recommended_mode": (self.recommended_mode.value if self.recommended_mode else None)})
                tick.advance(CognitiveStep.DECIDE)

                # Act (advisor-only: update recommendations, do not commit changes)
                tick.assert_step(CognitiveStep.ACT)
                self._update_resource_recommendations()
                tick.advance(CognitiveStep.ACT)

                # Reflect
                tick.assert_step(CognitiveStep.REFLECT)
                tick.advance(CognitiveStep.REFLECT)

                # Consolidate
                tick.assert_step(CognitiveStep.CONSOLIDATE)
                tick.finish()
                time.sleep(1.0)  # Update every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def _update_cognitive_state(self) -> None:
        """Update cognitive state based on current conditions"""
        with self._coordination_lock:
            current_time = datetime.now()
            time_delta = (current_time - self.state.last_update).total_seconds()
            
            # Update resources
            for resource in self.state.resources.values():
                resource.update(time_delta)
            
            # Update mode duration
            self.state.mode_duration += timedelta(seconds=time_delta)
            
            # Check for mode transitions
            self._check_mode_transitions()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            self.state.last_update = current_time
    
    def _check_mode_transitions(self) -> None:
        """Check if cognitive mode should transition"""
        current_mode = self.state.mode
        new_mode = self._determine_optimal_mode()
        
        if new_mode != current_mode:
            # Advisor-only: recommend a transition rather than committing it.
            self.recommended_mode = new_mode
            self.advisor_notes.append(
                f"Recommend mode transition: {current_mode.value} -> {new_mode.value}"
            )
    
    def _determine_optimal_mode(self) -> CognitiveMode:
        """Determine the optimal cognitive mode based on current state"""
        # Check for emergency conditions
        critical_resources = self.state.get_critical_resources()
        if critical_resources:
            return CognitiveMode.EMERGENCY
        
        # Check energy levels for recovery
        energy_resource = self.state.resources.get(ResourceType.ENERGY)
        if energy_resource and energy_resource.current_level < 0.3:
            return CognitiveMode.RECOVERY
        
        # Check active task load
        active_tasks = len([
            task for task in self.tasks.tasks.values()
            if task.status.value == "in_progress"
        ])
        
        if active_tasks == 0:
            return CognitiveMode.EXPLORATION
        elif active_tasks == 1:
            return CognitiveMode.FOCUSED
        else:
            return CognitiveMode.MULTI_TASK
    
    def _transition_mode(self, new_mode: CognitiveMode) -> None:
        """Transition to a new cognitive mode"""
        old_mode = self.state.mode
        self.state.mode = new_mode
        self.state.mode_duration = timedelta()
        self.state.transition_count += 1
        
        # Record transition
        self.mode_transitions.append((
            datetime.now(),
            old_mode,
            new_mode
        ))
        
        # Adjust resource allocation for new mode
        self._adjust_resources_for_mode(new_mode)
        
        logger.info(f"Cognitive mode transition: {old_mode.value} -> {new_mode.value}")
    
    def _adjust_resources_for_mode(self, mode: CognitiveMode) -> None:
        """Adjust resource allocation based on cognitive mode"""
        mode_configs = {
            CognitiveMode.FOCUSED: {
                ResourceType.ATTENTION: 0.8,
                ResourceType.MEMORY: 0.6,
                ResourceType.PROCESSING: 0.7,
                ResourceType.ENERGY: 0.8
            },
            CognitiveMode.MULTI_TASK: {
                ResourceType.ATTENTION: 0.6,
                ResourceType.MEMORY: 0.7,
                ResourceType.PROCESSING: 0.8,
                ResourceType.ENERGY: 0.6
            },
            CognitiveMode.EXPLORATION: {
                ResourceType.ATTENTION: 0.5,
                ResourceType.MEMORY: 0.8,
                ResourceType.PROCESSING: 0.6,
                ResourceType.ENERGY: 0.5
            },
            CognitiveMode.REFLECTION: {
                ResourceType.ATTENTION: 0.4,
                ResourceType.MEMORY: 0.9,
                ResourceType.PROCESSING: 0.5,
                ResourceType.ENERGY: 0.4
            },
            CognitiveMode.RECOVERY: {
                ResourceType.ATTENTION: 0.2,
                ResourceType.MEMORY: 0.3,
                ResourceType.PROCESSING: 0.2,
                ResourceType.ENERGY: 0.1
            },
            CognitiveMode.EMERGENCY: {
                ResourceType.ATTENTION: 0.9,
                ResourceType.MEMORY: 0.8,
                ResourceType.PROCESSING: 0.9,
                ResourceType.ENERGY: 0.9
            }
        }
        
        config = mode_configs.get(mode, {})
        for resource_type, allocation in config.items():
            if resource_type in self.state.resources:
                self.state.resources[resource_type].reserved_amount = 1.0 - allocation
    
    def _check_resource_levels(self) -> None:
        """Check resource levels and take action if needed"""
        critical_resources = self.state.get_critical_resources()
        
        if critical_resources:
            # Advisor-only: recommend recovery mode if multiple critical resources
            if len(critical_resources) > 1:
                self.recommended_mode = CognitiveMode.RECOVERY
                self.advisor_notes.append("Recommend RECOVERY mode due to multiple critical resources")

            # Advisor-only: record recommendations instead of mutating subsystems
            self._reduce_resource_consumption()
    
    def _reduce_resource_consumption(self) -> None:
        """Reduce resource consumption to preserve critical resources"""
        # Advisor-only: do not mutate attention capacity here.
        if ResourceType.ATTENTION in self.state.get_critical_resources():
            self.advisor_notes.append("Recommend reducing attention load (critical ATTENTION)")
        
        # Trigger memory consolidation
        if ResourceType.MEMORY in self.state.get_critical_resources():
            # Could trigger dream state or memory cleanup
            pass
    
    def _update_resource_recommendations(self) -> None:
        """Update resource allocation recommendations based on current needs."""
        # Get current tasks and their resource requirements
        active_tasks = [
            task for task in self.tasks.tasks.values()
            if task.status.value in ["ready", "in_progress"]
        ]
        
        if not active_tasks:
            return
        
        # Calculate optimal allocation
        total_demand = {resource_type: 0.0 for resource_type in ResourceType}
        
        for task in active_tasks:
            # Estimate resource demands
            total_demand[ResourceType.ATTENTION] += task.cognitive_load
            total_demand[ResourceType.PROCESSING] += task.cognitive_load
            total_demand[ResourceType.ENERGY] += task.cognitive_load * 0.5
        
        # Adjust allocations based on availability
        for resource_type, demand in total_demand.items():
            if resource_type in self.state.resources:
                resource = self.state.resources[resource_type]
                if demand > resource.current_level:
                    # Need to scale back or prioritize
                    self._prioritize_resource_allocation(resource_type, demand)
    
    def _prioritize_resource_allocation(self, resource_type: ResourceType, demand: float) -> None:
        """Prioritize resource allocation when demand exceeds supply"""
        # Get task priorities
        ready_tasks = self.tasks.get_ready_tasks()
        
        if not ready_tasks:
            return
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority_score, reverse=True)
        
        # Allocate resources to highest priority tasks first
        available_resource = self.state.resources[resource_type].current_level
        
        for task in ready_tasks:
            task_demand = task.cognitive_load  # Simplified
            if available_resource >= task_demand:
                # Allocate to this task
                self.resource_allocations[task.id] = {
                    resource_type: task_demand
                }
                available_resource -= task_demand
            else:
                # Pause or defer lower priority tasks
                if task.status.value == "in_progress":
                    # Could pause the task
                    pass
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        resource_efficiency = sum(
            resource.current_level for resource in self.state.resources.values()
        ) / len(self.state.resources)
        
        task_completion_rate = len([
            task for task in self.tasks.tasks.values()
            if task.status.value == "completed"
        ]) / max(1, len(self.tasks.tasks))
        
        attention_utilization = 1.0 - (
            self.attention.attention_capacity - self.attention.total_cognitive_load
        )
        
        self.state.performance_metrics = {
            'resource_efficiency': resource_efficiency,
            'task_completion_rate': task_completion_rate,
            'attention_utilization': attention_utilization,
            'mode_stability': 1.0 / max(1, self.state.transition_count),
            'overall_performance': (
                resource_efficiency + task_completion_rate + attention_utilization
            ) / 3.0
        }
        
        # Add to history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': self.state.performance_metrics.copy()
        })
    
    def allocate_resources(
        self,
        process_id: str,
        resource_requirements: Dict[ResourceType, float]
    ) -> bool:
        """
        Allocate resources to a process.

        Phase 2 advisor-only semantics: this records a recommendation rather than
        consuming resources or mutating external subsystems.
        
        Args:
            process_id: ID of the process requesting resources
            resource_requirements: Required resources
            
        Returns:
            True if allocation successful, False otherwise
        """
        with self._coordination_lock:
            # Check if resources are available
            for resource_type, amount in resource_requirements.items():
                if resource_type not in self.state.resources:
                    return False
                if not self.state.resources[resource_type].is_available(amount):
                    return False

            # Record recommendation (do not consume)
            self.resource_allocations[process_id] = resource_requirements
            return True
    
    def release_resources(self, process_id: str) -> None:
        """Release resources from a process"""
        if process_id not in self.resource_allocations:
            return
        
        with self._coordination_lock:
            allocations = self.resource_allocations[process_id]
            
            for resource_type, amount in allocations.items():
                if resource_type in self.state.resources:
                    self.state.resources[resource_type].recover(amount)
            
            del self.resource_allocations[process_id]
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get comprehensive cognitive status"""
        with self._coordination_lock:
            return {
                'mode': self.state.mode.value,
                'mode_duration': self.state.mode_duration.total_seconds(),
                'resources': self.state.get_resource_summary(),
                'critical_resources': [r.value for r in self.state.get_critical_resources()],
                'performance_metrics': self.state.performance_metrics,
                'active_processes': len(self.resource_allocations),
                'transition_count': self.state.transition_count,
                'last_update': self.state.last_update.isoformat()
            }
    
    def suggest_optimization(self) -> List[str]:
        """Suggest cognitive optimizations"""
        suggestions = []
        
        # Check for resource issues
        critical_resources = self.state.get_critical_resources()
        if critical_resources:
            suggestions.append(f"Critical resources detected: {[r.value for r in critical_resources]}")
            suggestions.append("Consider taking a break or reducing task load")
        
        # Check for mode instability
        if self.state.transition_count > 10:
            suggestions.append("High mode transition count - consider stabilizing workflow")
        
        # Check performance metrics
        if self.state.performance_metrics.get('overall_performance', 0) < 0.5:
            suggestions.append("Overall performance below optimal - review task priorities")
        
        # Check attention utilization
        if self.state.performance_metrics.get('attention_utilization', 0) > 0.9:
            suggestions.append("Attention heavily utilized - consider breaks or delegation")
        
        return suggestions
    
    def force_mode_transition(self, mode: CognitiveMode) -> None:
        """Advisor-only: recommend a transition to a specific cognitive mode."""
        self.recommended_mode = mode
        self.advisor_notes.append(f"Recommend mode transition to: {mode.value}")
    
    def get_performance_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance history for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            entry for entry in self.performance_history
            if entry['timestamp'] >= cutoff_time
        ]
