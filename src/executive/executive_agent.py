"""
Executive Agent - Main orchestrator for cognitive processes

This module provides the main executive agent that coordinates and orchestrates
all cognitive processes, integrating goal management, task planning, decision making,
and cognitive control into a unified system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..attention.attention_mechanism import AttentionMechanism
from ..memory.memory_system import MemorySystem
from .goal_manager import GoalManager, Goal, GoalPriority
from .task_planner import TaskPlanner, Task, TaskStatus
from .decision_engine import DecisionEngine, DecisionCriterion, DecisionOption
from .cognitive_controller import CognitiveController, ResourceType

class ExecutiveState(Enum):
    """States of the executive agent"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    REFLECTING = "reflecting"
    ADAPTING = "adapting"
    IDLE = "idle"
    ERROR = "error"

@dataclass
class ExecutiveContext:
    """
    Context information for executive decision making
    
    Attributes:
        current_input: Current input being processed
        recent_inputs: Recent inputs for context
        active_goals: Currently active goals
        urgent_tasks: Tasks requiring immediate attention
        resource_constraints: Current resource limitations
        environmental_factors: External factors affecting performance
    """
    current_input: Optional[str] = None
    recent_inputs: List[str] = field(default_factory=list)
    active_goals: List[Goal] = field(default_factory=list)
    urgent_tasks: List[Task] = field(default_factory=list)
    resource_constraints: Dict[ResourceType, float] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    
    def update_context(self, new_input: str) -> None:
        """Update context with new input"""
        self.current_input = new_input
        self.recent_inputs.append(new_input)
        if len(self.recent_inputs) > 10:
            self.recent_inputs.pop(0)

@dataclass
class ExecutiveDecision:
    """
    Represents a decision made by the executive agent
    
    Attributes:
        decision_id: Unique identifier for the decision
        decision_type: Type of decision (goal, task, resource, etc.)
        context: Context in which decision was made
        options_considered: Options that were considered
        chosen_option: The option that was selected
        rationale: Reasoning behind the decision
        expected_outcomes: Expected outcomes of the decision
        confidence: Confidence level in the decision
        timestamp: When the decision was made
    """
    decision_id: str
    decision_type: str
    context: ExecutiveContext
    options_considered: List[str]
    chosen_option: str
    rationale: str
    expected_outcomes: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

class ExecutiveAgent:
    """
    Main executive agent that orchestrates all cognitive processes
    
    This agent serves as the central coordinator for the cognitive system,
    managing goals, tasks, decisions, and resources in an integrated manner.
    """
    
    def __init__(
        self,
        attention_mechanism: AttentionMechanism,
        memory_system: MemorySystem,
        goal_manager: Optional[GoalManager] = None,
        task_planner: Optional[TaskPlanner] = None,
        decision_engine: Optional[DecisionEngine] = None,
        cognitive_controller: Optional[CognitiveController] = None
    ):
        """
        Initialize the executive agent
        
        Args:
            attention_mechanism: Attention management system
            memory_system: Memory management system
            goal_manager: Goal management system (optional, will create if not provided)
            task_planner: Task planning system (optional, will create if not provided)
            decision_engine: Decision making system (optional, will create if not provided)
            cognitive_controller: Cognitive control system (optional, will create if not provided)
        """
        self.attention = attention_mechanism
        self.memory = memory_system
        
        # Initialize or use provided components
        self.goals = goal_manager or GoalManager()
        self.tasks = task_planner or TaskPlanner(self.goals)
        self.decisions = decision_engine or DecisionEngine()
        self.controller = cognitive_controller or CognitiveController(
            self.attention,
            self.memory,
            self.goals,
            self.tasks,
            self.decisions
        )
        
        # Executive state
        self.state = ExecutiveState.INITIALIZING
        self.context = ExecutiveContext()
        
        # Decision history
        self.decision_history: List[ExecutiveDecision] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            'max_concurrent_goals': 5,
            'max_concurrent_tasks': 10,
            'reflection_interval': 300,  # seconds
            'adaptation_threshold': 0.7,
            'decision_confidence_threshold': 0.6
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Start cognitive controller monitoring
        self.controller.start_monitoring()
        
        # Set state to idle
        self.state = ExecutiveState.IDLE
    
    async def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process input through the executive system
        
        Args:
            input_text: Input text to process
            
        Returns:
            Dictionary containing response and executive actions
        """
        try:
            self.state = ExecutiveState.PLANNING
            
            # Update context
            self.context.update_context(input_text)
            
            # Analyze input and determine actions
            analysis = await self._analyze_input(input_text)
            
            # Generate response plan
            response_plan = await self._generate_response_plan(analysis)
            
            # Execute plan
            self.state = ExecutiveState.EXECUTING
            result = await self._execute_plan(response_plan)
            
            # Monitor execution
            self.state = ExecutiveState.MONITORING
            self._monitor_execution(result)
            
            self.state = ExecutiveState.IDLE
            
            return {
                'response': result.get('response', ''),
                'executive_actions': result.get('actions', []),
                'goals_updated': result.get('goals_updated', []),
                'tasks_created': result.get('tasks_created', []),
                'decisions_made': result.get('decisions_made', []),
                'cognitive_state': self.controller.get_cognitive_status()
            }
            
        except Exception as e:
            self.state = ExecutiveState.ERROR
            self.logger.error(f"Error processing input: {e}")
            return {
                'response': "I encountered an error processing your request.",
                'error': str(e),
                'executive_actions': [],
                'cognitive_state': self.controller.get_cognitive_status()
            }
    
    async def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        """
        Analyze input to determine required actions
        
        Args:
            input_text: Input text to analyze
            
        Returns:
            Analysis results
        """
        analysis = {
            'input_type': 'unknown',
            'intent': 'unknown',
            'requires_goals': False,
            'requires_tasks': False,
            'requires_decision': False,
            'urgency': 'normal',
            'complexity': 'medium',
            'resource_requirements': {}
        }
        
        # Simple intent analysis (in real implementation, would use NLP)
        input_lower = input_text.lower()
        
        # Check for goal-related keywords
        if any(word in input_lower for word in ['goal', 'objective', 'target', 'achieve']):
            analysis['requires_goals'] = True
            analysis['intent'] = 'goal_management'
        
        # Check for task-related keywords
        if any(word in input_lower for word in ['task', 'do', 'action', 'plan', 'execute']):
            analysis['requires_tasks'] = True
            analysis['intent'] = 'task_management'
        
        # Check for decision-related keywords
        if any(word in input_lower for word in ['decide', 'choose', 'option', 'should']):
            analysis['requires_decision'] = True
            analysis['intent'] = 'decision_making'
        
        # Check for urgency
        if any(word in input_lower for word in ['urgent', 'immediate', 'asap', 'now']):
            analysis['urgency'] = 'high'
        
        # Check for complexity
        if any(word in input_lower for word in ['complex', 'difficult', 'multi-step', 'analyze']):
            analysis['complexity'] = 'high'
        
        # Estimate resource requirements
        if analysis['complexity'] == 'high':
            analysis['resource_requirements'] = {
                ResourceType.ATTENTION: 0.8,
                ResourceType.PROCESSING: 0.7,
                ResourceType.MEMORY: 0.6
            }
        else:
            analysis['resource_requirements'] = {
                ResourceType.ATTENTION: 0.4,
                ResourceType.PROCESSING: 0.3,
                ResourceType.MEMORY: 0.3
            }
        
        return analysis
    
    async def _generate_response_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a plan for responding to the input
        
        Args:
            analysis: Input analysis results
            
        Returns:
            Response plan
        """
        plan = {
            'primary_action': 'respond',
            'secondary_actions': [],
            'resource_allocation': analysis.get('resource_requirements', {}),
            'expected_duration': 30,  # seconds
            'success_criteria': []
        }
        
        # Determine primary action based on intent
        if analysis['intent'] == 'goal_management':
            plan['primary_action'] = 'manage_goals'
            plan['secondary_actions'].append('update_tasks')
        elif analysis['intent'] == 'task_management':
            plan['primary_action'] = 'manage_tasks'
            plan['secondary_actions'].append('check_goals')
        elif analysis['intent'] == 'decision_making':
            plan['primary_action'] = 'make_decision'
            plan['secondary_actions'].append('update_context')
        
        # Adjust plan based on urgency
        if analysis['urgency'] == 'high':
            plan['expected_duration'] = 10
            plan['resource_allocation'] = {
                ResourceType.ATTENTION: 0.9,
                ResourceType.PROCESSING: 0.8,
                ResourceType.MEMORY: 0.7
            }
        
        # Add success criteria
        plan['success_criteria'] = [
            'meaningful_response_generated',
            'user_intent_addressed',
            'cognitive_state_maintained'
        ]
        
        return plan
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the response plan
        
        Args:
            plan: Response plan to execute
            
        Returns:
            Execution results
        """
        results = {
            'response': '',
            'actions': [],
            'goals_updated': [],
            'tasks_created': [],
            'decisions_made': []
        }
        
        # Allocate resources
        resource_allocated = self.controller.allocate_resources(
            'executive_processing',
            plan['resource_allocation']
        )
        
        if not resource_allocated:
            results['response'] = "I'm currently at capacity. Please wait a moment and try again."
            return results
        
        try:
            # Execute primary action
            primary_result = await self._execute_action(plan['primary_action'])
            results.update(primary_result)
            
            # Execute secondary actions
            for action in plan['secondary_actions']:
                secondary_result = await self._execute_action(action)
                # Merge results
                for key, value in secondary_result.items():
                    if key in results:
                        if isinstance(results[key], list):
                            results[key].extend(value)
                        else:
                            results[key] = value
                    else:
                        results[key] = value
            
            # Generate final response if not already set
            if not results['response']:
                results['response'] = await self._generate_response(results)
            
        finally:
            # Release resources
            self.controller.release_resources('executive_processing')
        
        return results
    
    async def _execute_action(self, action: str) -> Dict[str, Any]:
        """
        Execute a specific action
        
        Args:
            action: Action to execute
            
        Returns:
            Action results
        """
        results = {
            'response': '',
            'actions': [action],
            'goals_updated': [],
            'tasks_created': [],
            'decisions_made': []
        }
        
        if action == 'respond':
            results['response'] = await self._generate_basic_response()
        
        elif action == 'manage_goals':
            goal_results = await self._manage_goals()
            results.update(goal_results)
        
        elif action == 'manage_tasks':
            task_results = await self._manage_tasks()
            results.update(task_results)
        
        elif action == 'make_decision':
            decision_results = await self._make_decision()
            results.update(decision_results)
        
        elif action == 'update_tasks':
            # Update tasks based on current goals
            # TaskPlanner doesn't have update_task_priorities method
            # Instead, we'll just mark that we've updated tasks
            results['actions'].append('tasks_updated')
        
        elif action == 'check_goals':
            # Check goal status and update if needed
            active_goals = self.goals.get_active_goals()
            results['goals_updated'] = [goal.id for goal in active_goals]
        
        elif action == 'update_context':
            # Update executive context
            self.context.active_goals = self.goals.get_active_goals()
            self.context.urgent_tasks = [
                task for task in self.tasks.get_ready_tasks()
                if task.priority_score > 0.8
            ]
            results['actions'].append('context_updated')
        
        return results
    
    async def _generate_basic_response(self) -> str:
        """Generate a basic response to the current input"""
        if self.context.current_input:
            return f"I understand you're saying: {self.context.current_input}. Let me help you with that."
        return "I'm ready to help. What would you like me to do?"
    
    async def _manage_goals(self) -> Dict[str, Any]:
        """Manage goals based on current context"""
        results = {
            'goals_updated': [],
            'tasks_created': [],
            'response': ''
        }
        
        # Check if we should create a new goal
        if self.context.current_input and 'goal' in self.context.current_input.lower():
            # Extract goal from input (simplified)
            goal_description = self.context.current_input
            
            goal_id = self.goals.create_goal(
                title=goal_description,
                description=goal_description,
                priority=GoalPriority.MEDIUM,
                target_date=datetime.now() + timedelta(days=7)
            )
            results['goals_updated'].append(goal_id)
            results['response'] = f"I've created a new goal: {goal_description}"
        
        return results
    
    async def _manage_tasks(self) -> Dict[str, Any]:
        """Manage tasks based on current context"""
        results = {
            'tasks_created': [],
            'response': ''
        }
        
        # Check if we should create a new task
        if self.context.current_input and any(word in self.context.current_input.lower() for word in ['task', 'do', 'action']):
            # Get ready tasks
            ready_tasks = self.tasks.get_ready_tasks()
            
            if ready_tasks:
                task = ready_tasks[0]  # Get highest priority task
                results['response'] = f"I'll focus on: {task.description}"
            else:
                results['response'] = "I don't have any ready tasks at the moment."
        
        return results
    
    async def _make_decision(self) -> Dict[str, Any]:
        """Make a decision based on current context"""
        results = {
            'decisions_made': [],
            'response': ''
        }
        
        # Simple decision making (in real implementation, would be more sophisticated)
        if self.context.current_input and 'choose' in self.context.current_input.lower():
            # Create decision criteria
            criteria = [
                DecisionCriterion(name="importance", description="How important is this option?", weight=0.4),
                DecisionCriterion(name="feasibility", description="How feasible is this option?", weight=0.3),
                DecisionCriterion(name="risk", description="What are the risks?", weight=0.3)
            ]
            
            # Create options (simplified)
            options = [
                DecisionOption(name="option_a", description="First option", data={"importance": 0.8, "feasibility": 0.7, "risk": 0.3}),
                DecisionOption(name="option_b", description="Second option", data={"importance": 0.6, "feasibility": 0.9, "risk": 0.2})
            ]
            
            # Make decision
            decision_result = self.decisions.make_decision(
                options=options,
                criteria=criteria
            )
            
            results['decisions_made'].append(decision_result.decision_id)
            results['response'] = f"I recommend: {decision_result.recommended_option.name if decision_result.recommended_option else 'no option'}"
        
        return results
    
    async def _generate_response(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive response based on execution results"""
        response_parts = []
        
        if results.get('goals_updated'):
            response_parts.append(f"Updated {len(results['goals_updated'])} goals.")
        
        if results.get('tasks_created'):
            response_parts.append(f"Created {len(results['tasks_created'])} tasks.")
        
        if results.get('decisions_made'):
            response_parts.append(f"Made {len(results['decisions_made'])} decisions.")
        
        if results.get('actions'):
            response_parts.append(f"Executed {len(results['actions'])} actions.")
        
        if response_parts:
            return " ".join(response_parts)
        
        return "I've processed your request."
    
    def _monitor_execution(self, results: Dict[str, Any]) -> None:
        """Monitor execution results and update performance metrics"""
        # Record execution
        self.execution_log.append({
            'timestamp': datetime.now(),
            'results': results,
            'cognitive_state': self.controller.get_cognitive_status()
        })
        
        # Update performance metrics
        self.performance_metrics['total_executions'] = len(self.execution_log)
        self.performance_metrics['successful_executions'] = len([
            entry for entry in self.execution_log
            if 'error' not in entry['results']
        ])
        self.performance_metrics['success_rate'] = (
            self.performance_metrics['successful_executions'] / 
            max(1, self.performance_metrics['total_executions'])
        )
    
    def reflect(self) -> Dict[str, Any]:
        """
        Perform metacognitive reflection on executive performance
        
        Returns:
            Reflection results
        """
        self.state = ExecutiveState.REFLECTING
        
        reflection = {
            'timestamp': datetime.now(),
            'executive_state': self.state.value,
            'performance_metrics': self.performance_metrics,
            'cognitive_status': self.controller.get_cognitive_status(),
            'goal_status': self.goals.get_statistics(),
            'task_status': {
                'total_tasks': len(self.tasks.tasks),
                'ready_tasks': len(self.tasks.get_ready_tasks()),
                'completed_tasks': len([t for t in self.tasks.tasks.values() if t.status == TaskStatus.COMPLETED])
            },
            'decision_history': len(self.decision_history),
            'recent_decisions': [
                {
                    'decision_id': d.decision_id,
                    'decision_type': d.decision_type,
                    'chosen_option': d.chosen_option,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in self.decision_history[-5:]  # Last 5 decisions
            ],
            'insights': [],
            'recommendations': []
        }
        
        # Generate insights
        if self.performance_metrics.get('success_rate', 0) < 0.8:
            reflection['insights'].append("Executive success rate is below optimal")
            reflection['recommendations'].append("Review decision criteria and execution strategies")
        
        if len(self.goals.get_active_goals()) > self.config['max_concurrent_goals']:
            reflection['insights'].append("Too many active goals may be causing resource contention")
            reflection['recommendations'].append("Consider prioritizing or deferring some goals")
        
        # Check cognitive controller suggestions
        cognitive_suggestions = self.controller.suggest_optimization()
        reflection['recommendations'].extend(cognitive_suggestions)
        
        self.state = ExecutiveState.IDLE
        return reflection
    
    def adapt(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt executive behavior based on feedback
        
        Args:
            feedback: Feedback for adaptation
            
        Returns:
            Adaptation results
        """
        self.state = ExecutiveState.ADAPTING
        
        adaptation_results = {
            'adaptations_made': [],
            'config_changes': {},
            'performance_impact': {}
        }
        
        # Adapt based on performance metrics
        if self.performance_metrics.get('success_rate', 0) < self.config['adaptation_threshold']:
            # Reduce concurrent goals
            if self.config['max_concurrent_goals'] > 3:
                self.config['max_concurrent_goals'] -= 1
                adaptation_results['adaptations_made'].append('reduced_concurrent_goals')
                adaptation_results['config_changes']['max_concurrent_goals'] = self.config['max_concurrent_goals']
        
        # Adapt based on cognitive controller feedback
        cognitive_status = self.controller.get_cognitive_status()
        if cognitive_status.get('transition_count', 0) > 20:
            # Increase reflection interval to stabilize
            self.config['reflection_interval'] *= 1.2
            adaptation_results['adaptations_made'].append('increased_reflection_interval')
            adaptation_results['config_changes']['reflection_interval'] = self.config['reflection_interval']
        
        # Adapt based on user feedback
        if feedback.get('response_quality', 0) < 0.7:
            # Lower decision confidence threshold
            self.config['decision_confidence_threshold'] = max(0.3, self.config['decision_confidence_threshold'] - 0.1)
            adaptation_results['adaptations_made'].append('lowered_confidence_threshold')
            adaptation_results['config_changes']['decision_confidence_threshold'] = self.config['decision_confidence_threshold']
        
        self.state = ExecutiveState.IDLE
        return adaptation_results
    
    def get_executive_status(self) -> Dict[str, Any]:
        """Get comprehensive executive status"""
        return {
            'state': self.state.value,
            'context': {
                'current_input': self.context.current_input,
                'recent_inputs_count': len(self.context.recent_inputs),
                'active_goals_count': len(self.context.active_goals),
                'urgent_tasks_count': len(self.context.urgent_tasks)
            },
            'performance_metrics': self.performance_metrics,
            'cognitive_status': self.controller.get_cognitive_status(),
            'component_status': {
                'goals': self.goals.get_statistics(),
                'tasks': {
                    'total_tasks': len(self.tasks.tasks),
                    'ready_tasks': len(self.tasks.get_ready_tasks()),
                    'completed_tasks': len([t for t in self.tasks.tasks.values() if t.status == TaskStatus.COMPLETED])
                },
                'decisions': {
                    'total_decisions': len(self.decision_history),
                    'recent_decisions': len([
                        d for d in self.decision_history
                        if (datetime.now() - d.timestamp).total_seconds() < 3600
                    ])
                }
            },
            'configuration': self.config
        }
    
    def shutdown(self) -> None:
        """Shutdown the executive agent"""
        self.controller.stop_monitoring()
        self.state = ExecutiveState.IDLE
        self.logger.info("Executive agent shut down")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except Exception:
            pass
