"""
Experiment Manager - Week 16 Phase 4

A/B testing framework for comparing decision strategy performance.
Enables systematic experimentation with randomized strategy assignment,
outcome tracking, and statistical analysis.

Key responsibilities:
- Manage strategy experiments
- Randomized strategy assignment (random, epsilon-greedy, Thompson sampling)
- Track strategy outcomes
- Aggregate performance metrics
- Statistical comparison of strategies
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid

import numpy as np

from .experiment_analyzer import (
    StrategyPerformance,
    calculate_proportion_confidence_interval,
    recommend_strategy
)

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"  # Created but not started
    ACTIVE = "active"  # Currently running
    PAUSED = "paused"  # Temporarily stopped
    COMPLETED = "completed"  # Finished
    CANCELLED = "cancelled"  # Stopped early


class AssignmentMethod(Enum):
    """Method for assigning strategies."""
    RANDOM = "random"  # Pure random (uniform)
    EPSILON_GREEDY = "epsilon_greedy"  # Explore (ε) vs exploit (1-ε)
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian approach


@dataclass
class StrategyExperiment:
    """
    Represents a strategy comparison experiment.
    
    An experiment compares multiple decision strategies to determine
    which performs best based on empirical outcomes.
    """
    experiment_id: str
    name: str
    strategies: List[str]  # List of strategy names to compare
    assignment_method: AssignmentMethod
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    # Configuration
    description: str = ""
    epsilon: float = 0.1  # For epsilon-greedy (10% exploration)
    confidence_level: float = 0.95  # For statistical tests
    min_sample_size: int = 30  # Minimum samples before analysis
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'strategies': self.strategies,
            'assignment_method': self.assignment_method.value,
            'status': self.status.value,
            'description': self.description,
            'epsilon': self.epsilon,
            'confidence_level': self.confidence_level,
            'min_sample_size': self.min_sample_size,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'StrategyExperiment':
        """Create from dictionary."""
        return StrategyExperiment(
            experiment_id=data['experiment_id'],
            name=data['name'],
            strategies=data['strategies'],
            assignment_method=AssignmentMethod(data['assignment_method']),
            status=ExperimentStatus(data['status']),
            description=data.get('description', ''),
            epsilon=data.get('epsilon', 0.1),
            confidence_level=data.get('confidence_level', 0.95),
            min_sample_size=data.get('min_sample_size', 30),
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class ExperimentAssignment:
    """
    Records which strategy was assigned for a decision.
    
    Links a decision to an experimental strategy assignment.
    """
    assignment_id: str
    experiment_id: str
    decision_id: str
    goal_id: str
    assigned_strategy: str
    assignment_timestamp: datetime = field(default_factory=datetime.now)
    assignment_method: AssignmentMethod = AssignmentMethod.RANDOM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'assignment_id': self.assignment_id,
            'experiment_id': self.experiment_id,
            'decision_id': self.decision_id,
            'goal_id': self.goal_id,
            'assigned_strategy': self.assigned_strategy,
            'assignment_timestamp': self.assignment_timestamp.isoformat(),
            'assignment_method': self.assignment_method.value,
            'metadata': self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ExperimentAssignment':
        """Create from dictionary."""
        return ExperimentAssignment(
            assignment_id=data['assignment_id'],
            experiment_id=data['experiment_id'],
            decision_id=data['decision_id'],
            goal_id=data['goal_id'],
            assigned_strategy=data['assigned_strategy'],
            assignment_timestamp=datetime.fromisoformat(data['assignment_timestamp']),
            assignment_method=AssignmentMethod(data['assignment_method']),
            metadata=data.get('metadata', {})
        )


@dataclass
class StrategyOutcome:
    """
    Records the outcome for an experimental strategy assignment.
    
    Links an assignment to its actual execution outcome.
    """
    assignment_id: str
    success: bool
    outcome_score: float  # 0-1
    
    # Performance metrics
    execution_time_seconds: Optional[float] = None
    decision_confidence: float = 0.0
    
    # Timing
    completion_timestamp: datetime = field(default_factory=datetime.now)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'assignment_id': self.assignment_id,
            'success': self.success,
            'outcome_score': self.outcome_score,
            'execution_time_seconds': self.execution_time_seconds,
            'decision_confidence': self.decision_confidence,
            'completion_timestamp': self.completion_timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'StrategyOutcome':
        """Create from dictionary."""
        return StrategyOutcome(
            assignment_id=data['assignment_id'],
            success=data['success'],
            outcome_score=data['outcome_score'],
            execution_time_seconds=data.get('execution_time_seconds'),
            decision_confidence=data.get('decision_confidence', 0.0),
            completion_timestamp=datetime.fromisoformat(data['completion_timestamp']),
            metadata=data.get('metadata', {})
        )


class ExperimentManager:
    """
    Manages strategy experiments for A/B testing.
    
    Provides functionality to:
    - Create and manage experiments
    - Assign strategies using different methods (random, epsilon-greedy, Thompson)
    - Track outcomes
    - Aggregate performance metrics
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize experiment manager.
        
        Args:
            storage_dir: Directory for storing experiment data
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/experiments")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.experiments: Dict[str, StrategyExperiment] = {}
        self.assignments: Dict[str, List[ExperimentAssignment]] = {}  # exp_id -> assignments
        self.outcomes: Dict[str, StrategyOutcome] = {}  # assignment_id -> outcome
        
        # Load existing experiments
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """Load experiments from disk."""
        if not self.storage_dir.exists():
            return
        
        for exp_file in self.storage_dir.glob("experiment_*.json"):
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                exp = StrategyExperiment.from_dict(data)
                self.experiments[exp.experiment_id] = exp
                
                # Load assignments
                self._load_assignments(exp.experiment_id)
                
                # Load outcomes
                self._load_outcomes(exp.experiment_id)
                
            except Exception as e:
                logger.error(f"Failed to load experiment {exp_file}: {e}")
    
    def _load_assignments(self, experiment_id: str) -> None:
        """Load assignments for an experiment."""
        assignments_file = self.storage_dir / f"assignments_{experiment_id}.json"
        if not assignments_file.exists():
            self.assignments[experiment_id] = []
            return
        
        try:
            with open(assignments_file, 'r') as f:
                data = json.load(f)
            
            assignments = [ExperimentAssignment.from_dict(a) for a in data]
            self.assignments[experiment_id] = assignments
            
        except Exception as e:
            logger.error(f"Failed to load assignments for {experiment_id}: {e}")
            self.assignments[experiment_id] = []
    
    def _load_outcomes(self, experiment_id: str) -> None:
        """Load outcomes for an experiment."""
        outcomes_file = self.storage_dir / f"outcomes_{experiment_id}.json"
        if not outcomes_file.exists():
            return
        
        try:
            with open(outcomes_file, 'r') as f:
                data = json.load(f)
            
            for outcome_data in data:
                outcome = StrategyOutcome.from_dict(outcome_data)
                self.outcomes[outcome.assignment_id] = outcome
                
        except Exception as e:
            logger.error(f"Failed to load outcomes for {experiment_id}: {e}")
    
    def _save_experiment(self, experiment: StrategyExperiment) -> None:
        """Save experiment to disk."""
        exp_file = self.storage_dir / f"experiment_{experiment.experiment_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)
    
    def _save_assignments(self, experiment_id: str) -> None:
        """Save assignments for an experiment."""
        assignments_file = self.storage_dir / f"assignments_{experiment_id}.json"
        assignments = self.assignments.get(experiment_id, [])
        with open(assignments_file, 'w') as f:
            json.dump([a.to_dict() for a in assignments], f, indent=2)
    
    def _save_outcomes(self, experiment_id: str) -> None:
        """Save outcomes for an experiment."""
        outcomes_file = self.storage_dir / f"outcomes_{experiment_id}.json"
        
        # Get all outcomes for this experiment's assignments
        assignment_ids = {a.assignment_id for a in self.assignments.get(experiment_id, [])}
        outcomes = [
            self.outcomes[aid].to_dict()
            for aid in assignment_ids
            if aid in self.outcomes
        ]
        
        with open(outcomes_file, 'w') as f:
            json.dump(outcomes, f, indent=2)
    
    def create_experiment(
        self,
        name: str,
        strategies: List[str],
        assignment_method: AssignmentMethod = AssignmentMethod.RANDOM,
        description: str = "",
        epsilon: float = 0.1,
        confidence_level: float = 0.95,
        min_sample_size: int = 30,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StrategyExperiment:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            strategies: List of strategy names to compare
            assignment_method: How to assign strategies
            description: Experiment description
            epsilon: Exploration rate for epsilon-greedy (0.0-1.0)
            confidence_level: Statistical confidence level (e.g., 0.95)
            min_sample_size: Minimum samples before analysis
            metadata: Additional metadata
            
        Returns:
            Created experiment
        """
        if len(strategies) < 2:
            raise ValueError("Experiment requires at least 2 strategies")
        
        experiment = StrategyExperiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            strategies=strategies,
            assignment_method=assignment_method,
            description=description,
            epsilon=epsilon,
            confidence_level=confidence_level,
            min_sample_size=min_sample_size,
            metadata=metadata or {}
        )
        
        self.experiments[experiment.experiment_id] = experiment
        self.assignments[experiment.experiment_id] = []
        self._save_experiment(experiment)
        
        logger.info(
            f"Created experiment '{name}' ({experiment.experiment_id}) "
            f"comparing {len(strategies)} strategies"
        )
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment is {experiment.status.value}, cannot start")
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.started_at = datetime.now()
        self._save_experiment(experiment)
        
        logger.info(f"Started experiment '{experiment.name}' ({experiment_id})")
    
    def pause_experiment(self, experiment_id: str) -> None:
        """Pause an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            raise ValueError(f"Experiment is {experiment.status.value}, cannot pause")
        
        experiment.status = ExperimentStatus.PAUSED
        self._save_experiment(experiment)
        
        logger.info(f"Paused experiment '{experiment.name}' ({experiment_id})")
    
    def complete_experiment(self, experiment_id: str) -> None:
        """Complete an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status not in (ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED):
            raise ValueError(f"Experiment is {experiment.status.value}, cannot complete")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now()
        self._save_experiment(experiment)
        
        logger.info(f"Completed experiment '{experiment.name}' ({experiment_id})")
    
    def assign_strategy(
        self,
        experiment_id: str,
        decision_id: str,
        goal_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExperimentAssignment:
        """
        Assign a strategy for a decision.
        
        Args:
            experiment_id: ID of the experiment
            decision_id: ID of the decision
            goal_id: ID of the goal
            context: Additional context for assignment
            
        Returns:
            ExperimentAssignment with assigned strategy
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            raise ValueError(f"Experiment is {experiment.status.value}, not active")
        
        # Select strategy based on assignment method
        if experiment.assignment_method == AssignmentMethod.RANDOM:
            strategy = self._assign_random(experiment)
        elif experiment.assignment_method == AssignmentMethod.EPSILON_GREEDY:
            strategy = self._assign_epsilon_greedy(experiment)
        elif experiment.assignment_method == AssignmentMethod.THOMPSON_SAMPLING:
            strategy = self._assign_thompson_sampling(experiment)
        else:
            raise ValueError(f"Unknown assignment method: {experiment.assignment_method}")
        
        # Create assignment
        assignment = ExperimentAssignment(
            assignment_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            decision_id=decision_id,
            goal_id=goal_id,
            assigned_strategy=strategy,
            assignment_method=experiment.assignment_method,
            metadata=context or {}
        )
        
        # Store assignment
        self.assignments[experiment_id].append(assignment)
        self._save_assignments(experiment_id)
        
        logger.debug(
            f"Assigned strategy '{strategy}' for decision {decision_id} "
            f"in experiment {experiment_id}"
        )
        
        return assignment
    
    def _assign_random(self, experiment: StrategyExperiment) -> str:
        """Assign strategy uniformly at random."""
        return random.choice(experiment.strategies)
    
    def _assign_epsilon_greedy(self, experiment: StrategyExperiment) -> str:
        """Assign strategy using epsilon-greedy exploration."""
        # Explore with probability epsilon
        if random.random() < experiment.epsilon:
            return random.choice(experiment.strategies)
        
        # Exploit: choose best performing strategy
        performance = self._get_strategy_performance(experiment.experiment_id)
        
        if not performance:
            # No data yet, random choice
            return random.choice(experiment.strategies)
        
        # Find best strategy by success rate
        best_strategy = max(performance.items(), key=lambda x: x[1]['success_rate'])[0]
        return best_strategy
    
    def _assign_thompson_sampling(self, experiment: StrategyExperiment) -> str:
        """Assign strategy using Thompson sampling (Bayesian approach)."""
        # Get performance for each strategy
        performance = self._get_strategy_performance(experiment.experiment_id)
        
        if not performance:
            # No data yet, random choice
            return random.choice(experiment.strategies)
        
        # Sample from Beta distribution for each strategy
        samples = {}
        for strategy in experiment.strategies:
            perf = performance.get(strategy, {'success_count': 0, 'failure_count': 0})
            successes = perf.get('success_count', 0)
            failures = perf.get('failure_count', 0)
            
            # Beta(α=successes+1, β=failures+1) - Jeffrey's prior
            sample = np.random.beta(successes + 1, failures + 1)
            samples[strategy] = sample
        
        # Choose strategy with highest sample
        best_strategy = max(samples.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    def record_outcome(
        self,
        assignment_id: str,
        success: bool,
        outcome_score: float,
        execution_time_seconds: Optional[float] = None,
        decision_confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StrategyOutcome:
        """
        Record outcome for an assignment.
        
        Args:
            assignment_id: ID of the assignment
            success: Whether execution succeeded
            outcome_score: Outcome quality (0-1)
            execution_time_seconds: Execution time
            decision_confidence: Decision confidence (0-1)
            metadata: Additional metadata
            
        Returns:
            Created outcome record
        """
        # Find assignment
        assignment = None
        for assignments in self.assignments.values():
            for a in assignments:
                if a.assignment_id == assignment_id:
                    assignment = a
                    break
            if assignment:
                break
        
        if not assignment:
            raise ValueError(f"Assignment {assignment_id} not found")
        
        # Create outcome
        outcome = StrategyOutcome(
            assignment_id=assignment_id,
            success=success,
            outcome_score=outcome_score,
            execution_time_seconds=execution_time_seconds,
            decision_confidence=decision_confidence,
            metadata=metadata or {}
        )
        
        # Store outcome
        self.outcomes[assignment_id] = outcome
        self._save_outcomes(assignment.experiment_id)
        
        logger.debug(
            f"Recorded outcome for assignment {assignment_id}: "
            f"success={success}, score={outcome_score:.2f}"
        )
        
        return outcome
    
    def _get_strategy_performance(
        self,
        experiment_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get aggregated performance metrics per strategy."""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        assignments = self.assignments.get(experiment_id, [])
        
        # Group by strategy
        strategy_data: Dict[str, Dict[str, List]] = {
            strategy: {
                'successes': [],
                'scores': [],
                'times': [],
                'confidences': []
            }
            for strategy in experiment.strategies
        }
        
        # Aggregate outcomes
        for assignment in assignments:
            if assignment.assignment_id not in self.outcomes:
                continue
            
            outcome = self.outcomes[assignment.assignment_id]
            strategy = assignment.assigned_strategy
            
            if strategy not in strategy_data:
                continue
            
            strategy_data[strategy]['successes'].append(1 if outcome.success else 0)
            strategy_data[strategy]['scores'].append(outcome.outcome_score)
            if outcome.execution_time_seconds is not None:
                strategy_data[strategy]['times'].append(outcome.execution_time_seconds)
            strategy_data[strategy]['confidences'].append(outcome.decision_confidence)
        
        # Calculate metrics
        performance = {}
        for strategy, data in strategy_data.items():
            n = len(data['successes'])
            if n == 0:
                continue
            
            success_count = sum(data['successes'])
            failure_count = n - success_count
            
            performance[strategy] = {
                'total_assignments': n,
                'success_count': success_count,
                'failure_count': failure_count,
                'success_rate': success_count / n if n > 0 else 0.0,
                'avg_outcome_score': np.mean(data['scores']) if data['scores'] else 0.0,
                'std_outcome_score': np.std(data['scores']) if data['scores'] else 0.0,
                'avg_execution_time': np.mean(data['times']) if data['times'] else 0.0,
                'avg_confidence': np.mean(data['confidences']) if data['confidences'] else 0.0
            }
        
        return performance
    
    def get_experiment(self, experiment_id: str) -> Optional[StrategyExperiment]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[StrategyExperiment]:
        """List experiments, optionally filtered by status."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary statistics for an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        assignments = self.assignments.get(experiment_id, [])
        performance = self._get_strategy_performance(experiment_id)
        
        # Calculate total outcomes
        total_outcomes = sum(
            1 for a in assignments
            if a.assignment_id in self.outcomes
        )
        
        return {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'strategies': experiment.strategies,
            'assignment_method': experiment.assignment_method.value,
            'total_assignments': len(assignments),
            'total_outcomes': total_outcomes,
            'completion_rate': total_outcomes / len(assignments) if assignments else 0.0,
            'performance': performance,
            'started_at': experiment.started_at.isoformat() if experiment.started_at else None,
            'duration_hours': (
                (experiment.completed_at or datetime.now()) - experiment.started_at
            ).total_seconds() / 3600 if experiment.started_at else 0.0
        }
    
    def get_strategy_performances(
        self,
        experiment_id: str
    ) -> Dict[str, StrategyPerformance]:
        """
        Get detailed performance metrics for each strategy.
        
        Returns StrategyPerformance objects with confidence intervals.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        assignments = self.assignments.get(experiment_id, [])
        
        # Group by strategy
        strategy_data: Dict[str, Dict[str, Any]] = {
            strategy: {
                'successes': [],
                'scores': [],
                'times': [],
                'confidences': []
            }
            for strategy in experiment.strategies
        }
        
        # Aggregate outcomes
        for assignment in assignments:
            if assignment.assignment_id not in self.outcomes:
                continue
            
            outcome = self.outcomes[assignment.assignment_id]
            strategy = assignment.assigned_strategy
            
            if strategy not in strategy_data:
                continue
            
            strategy_data[strategy]['successes'].append(1 if outcome.success else 0)
            strategy_data[strategy]['scores'].append(outcome.outcome_score)
            if outcome.execution_time_seconds is not None:
                strategy_data[strategy]['times'].append(outcome.execution_time_seconds)
            strategy_data[strategy]['confidences'].append(outcome.decision_confidence)
        
        # Build StrategyPerformance objects
        performances = {}
        for strategy, data in strategy_data.items():
            n = len(data['successes'])
            if n == 0:
                continue
            
            success_count = sum(data['successes'])
            failure_count = n - success_count
            success_rate = success_count / n
            
            # Calculate confidence interval for success rate
            ci_lower, ci_upper = calculate_proportion_confidence_interval(
                success_count,
                n,
                experiment.confidence_level
            )
            
            performances[strategy] = StrategyPerformance(
                strategy_name=strategy,
                total_assignments=n,
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_rate,
                success_rate_ci_lower=ci_lower,
                success_rate_ci_upper=ci_upper,
                avg_outcome_score=float(np.mean(data['scores'])) if data['scores'] else 0.0,
                std_outcome_score=float(np.std(data['scores'])) if data['scores'] else 0.0,
                outcome_scores=data['scores'],
                avg_execution_time=float(np.mean(data['times'])) if data['times'] else 0.0,
                std_execution_time=float(np.std(data['times'])) if data['times'] else 0.0,
                avg_confidence=float(np.mean(data['confidences'])) if data['confidences'] else 0.0
            )
        
        return performances
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Perform statistical analysis and recommend best strategy.
        
        Returns recommendation with statistical tests and confidence.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Get detailed performance metrics
        performances = self.get_strategy_performances(experiment_id)
        
        if not performances:
            return {
                'experiment_id': experiment_id,
                'recommendation': None,
                'reason': 'No performance data available'
            }
        
        # Get recommendation
        recommendation = recommend_strategy(
            performances,
            alpha=1.0 - experiment.confidence_level,
            min_sample_size=experiment.min_sample_size
        )
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'status': experiment.status.value,
            **recommendation
        }


def create_experiment_manager(storage_dir: Optional[Path] = None) -> ExperimentManager:
    """
    Factory function to create ExperimentManager.
    
    Args:
        storage_dir: Directory for storing experiment data
        
    Returns:
        ExperimentManager instance
    """
    return ExperimentManager(storage_dir=storage_dir)
