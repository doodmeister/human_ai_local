"""
Outcome Tracker - Week 16 Phase 1

Tracks execution outcomes for learning and continuous improvement.
Records actual results, compares with predictions, persists history.

Key responsibilities:
- Record goal outcomes with detailed metrics
- Analyze decision/planning/scheduling accuracy
- Persist outcome history for training data
- Calculate improvement metrics over time
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .feature_extractor import FeatureExtractor, FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """
    Metrics comparing predicted vs actual execution.
    
    All ratios are actual/predicted (1.0 = perfect, <1.0 = faster/better, >1.0 = slower/worse).
    """
    # Time accuracy
    time_accuracy_ratio: float = 0.0  # actual_time / predicted_time
    
    # Resource accuracy
    resource_accuracy: Dict[str, float] = field(default_factory=dict)  # resource -> actual/predicted
    
    # Quality metrics
    plan_adherence_score: float = 0.0  # % of plan executed as designed (0-1)
    goal_achievement_score: float = 0.0  # How well goal was achieved (0-1)
    
    # Deviation counts
    major_deviations: int = 0
    minor_deviations: int = 0
    
    def overall_accuracy(self) -> float:
        """Calculate overall accuracy score (0-1)."""
        scores = [
            self.plan_adherence_score,
            self.goal_achievement_score,
        ]
        
        # Time accuracy: penalize if off by >20%
        if 0.8 <= self.time_accuracy_ratio <= 1.2:
            scores.append(1.0)
        elif 0.6 <= self.time_accuracy_ratio <= 1.4:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # Deviation penalty
        deviation_score = max(0.0, 1.0 - (self.major_deviations * 0.2) - (self.minor_deviations * 0.05))
        scores.append(deviation_score)
        
        return sum(scores) / len(scores)


@dataclass
class OutcomeRecord:
    """
    Complete record of a goal execution outcome.
    
    Links execution context with actual results for learning.
    """
    # Identification
    record_id: str
    goal_id: str
    goal_title: str
    
    # Timing
    start_time: datetime
    predicted_completion_time: Optional[datetime]
    actual_completion_time: datetime
    
    # Results
    success: bool
    outcome_score: float  # Quality 0-1
    
    # Strategy used
    decision_strategy: str
    decision_confidence: float
    selected_option: str
    
    # Planning
    plan_length: int  # Number of actions
    plan_cost: float
    actions_completed: int
    
    # Scheduling  
    predicted_makespan_minutes: float
    actual_makespan_minutes: float
    
    # Accuracy
    accuracy_metrics: AccuracyMetrics
    
    # Issues
    deviations: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        # Convert datetime to ISO strings
        data['start_time'] = self.start_time.isoformat()
        data['actual_completion_time'] = self.actual_completion_time.isoformat()
        data['timestamp'] = self.timestamp.isoformat()
        if self.predicted_completion_time:
            data['predicted_completion_time'] = self.predicted_completion_time.isoformat()
        return data
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'OutcomeRecord':
        """Reconstruct from dict."""
        # Convert ISO strings back to datetime
        data = data.copy()
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        data['actual_completion_time'] = datetime.fromisoformat(data['actual_completion_time'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('predicted_completion_time'):
            data['predicted_completion_time'] = datetime.fromisoformat(data['predicted_completion_time'])
        
        # Reconstruct AccuracyMetrics
        if 'accuracy_metrics' in data and isinstance(data['accuracy_metrics'], dict):
            data['accuracy_metrics'] = AccuracyMetrics(**data['accuracy_metrics'])
        
        return OutcomeRecord(**data)


class OutcomeTracker:
    """
    Service for tracking and analyzing execution outcomes.
    
    Provides:
    - Outcome recording from ExecutionContext
    - Historical outcome retrieval
    - Accuracy analysis (decision/planning/scheduling)
    - Persistent storage (JSON files)
    
    Usage:
        tracker = OutcomeTracker()
        tracker.record_outcome(execution_context)
        accuracy = tracker.analyze_decision_accuracy("strategy_name")
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize outcome tracker.
        
        Args:
            storage_dir: Directory for outcome files (default: data/outcomes/)
        """
        self.storage_dir = storage_dir or Path("data/outcomes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._outcomes: List[OutcomeRecord] = []
        self._feature_extractor: Optional['FeatureExtractor'] = None
        self._load_outcomes()
        
        logger.info(f"OutcomeTracker initialized with {len(self._outcomes)} historical outcomes")
    
    @property
    def feature_extractor(self) -> 'FeatureExtractor':
        """Lazy initialization of feature extractor."""
        if self._feature_extractor is None:
            from .feature_extractor import create_feature_extractor
            self._feature_extractor = create_feature_extractor()
        return self._feature_extractor
    
    def record_outcome(
        self,
        execution_context: Any,  # ExecutionContext
        outcome_score: Optional[float] = None,
        deviations: Optional[List[str]] = None,
    ) -> OutcomeRecord:
        """
        Record an execution outcome.
        
        Args:
            execution_context: ExecutionContext from ExecutiveSystem
            outcome_score: Quality score 0-1 (optional, auto-calculated if not provided)
            deviations: List of deviations from plan (optional)
        
        Returns:
            OutcomeRecord for this execution
        """
        from src.executive.integration import ExecutionContext
        
        context: ExecutionContext = execution_context
        
        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy_metrics(context)
        
        # Auto-calculate outcome score if not provided
        if outcome_score is None:
            outcome_score = accuracy.overall_accuracy()
        
        # Extract timing
        actual_completion = context.actual_completion_time or context.end_time or datetime.now()
        actual_makespan = (actual_completion - context.start_time).total_seconds() / 60
        
        # Extract predicted makespan from schedule
        predicted_makespan = 0.0
        if context.schedule:
            if isinstance(context.schedule.makespan, timedelta):
                predicted_makespan = context.schedule.makespan.total_seconds() / 60
            else:
                predicted_makespan = float(context.schedule.makespan)
        
        # Build outcome record
        record = OutcomeRecord(
            record_id=f"{context.goal_id}_{context.start_time.strftime('%Y%m%d_%H%M%S')}",
            goal_id=context.goal_id,
            goal_title=context.goal_title,
            start_time=context.start_time,
            predicted_completion_time=None,  # Could calculate from schedule
            actual_completion_time=actual_completion,
            success=context.actual_success if context.actual_success is not None else context.success,
            outcome_score=outcome_score,
            decision_strategy="unknown",  # DecisionResult doesn't expose strategy
            decision_confidence=context.decision_result.confidence if context.decision_result else 0.0,
            selected_option=(
                context.decision_result.recommended_option.name 
                if context.decision_result and context.decision_result.recommended_option 
                else "unknown"
            ),
            plan_length=len(context.plan.steps) if context.plan else 0,
            plan_cost=context.plan.total_cost if context.plan else 0.0,
            actions_completed=context.actions_completed,
            predicted_makespan_minutes=float(predicted_makespan) if isinstance(predicted_makespan, (int, float)) else predicted_makespan.total_seconds() / 60,
            actual_makespan_minutes=actual_makespan,
            accuracy_metrics=accuracy,
            deviations=deviations or context.deviations,
            failure_reason=context.failure_reason,
            lessons_learned=context.lessons_learned,
        )
        
        # Store
        self._outcomes.append(record)
        self._save_outcome(record)
        
        logger.info(
            f"Recorded outcome for goal '{context.goal_title}': "
            f"success={record.success}, score={record.outcome_score:.2f}, "
            f"accuracy={accuracy.overall_accuracy():.2f}"
        )
        
        return record
    
    def get_outcome_history(
        self,
        limit: Optional[int] = None,
        strategy: Optional[str] = None,
        success_only: bool = False,
    ) -> List[OutcomeRecord]:
        """
        Retrieve outcome history with optional filters.
        
        Args:
            limit: Max number of outcomes to return (most recent first)
            strategy: Filter by decision strategy
            success_only: Only return successful outcomes
        
        Returns:
            List of OutcomeRecord objects
        """
        outcomes = self._outcomes.copy()
        
        # Apply filters
        if strategy:
            outcomes = [o for o in outcomes if o.decision_strategy == strategy]
        if success_only:
            outcomes = [o for o in outcomes if o.success]
        
        # Sort by timestamp (most recent first)
        outcomes.sort(key=lambda o: o.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            outcomes = outcomes[:limit]
        
        return outcomes
    
    def analyze_decision_accuracy(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze decision-making accuracy.
        
        Args:
            strategy: Analyze specific strategy (None = all strategies)
        
        Returns:
            Dict with accuracy metrics:
            - success_rate: % of successful outcomes
            - avg_confidence: Average decision confidence
            - avg_outcome_score: Average outcome quality
            - confidence_calibration: correlation between confidence and success
        """
        outcomes = self.get_outcome_history(strategy=strategy)
        
        if not outcomes:
            return {
                "strategy": strategy or "all",
                "sample_size": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_outcome_score": 0.0,
                "confidence_calibration": 0.0,
            }
        
        successes = sum(1 for o in outcomes if o.success)
        total_confidence = sum(o.decision_confidence for o in outcomes)
        total_score = sum(o.outcome_score for o in outcomes)
        
        # Calibration: high confidence should correlate with success
        high_conf_outcomes = [o for o in outcomes if o.decision_confidence > 0.7]
        high_conf_success = sum(1 for o in high_conf_outcomes if o.success) if high_conf_outcomes else 0
        calibration = high_conf_success / len(high_conf_outcomes) if high_conf_outcomes else 0.0
        
        return {
            "strategy": strategy or "all",
            "sample_size": len(outcomes),
            "success_rate": successes / len(outcomes),
            "avg_confidence": total_confidence / len(outcomes),
            "avg_outcome_score": total_score / len(outcomes),
            "confidence_calibration": calibration,
        }
    
    def analyze_planning_accuracy(self) -> Dict[str, Any]:
        """
        Analyze planning accuracy.
        
        Returns:
            Dict with planning metrics:
            - avg_plan_adherence: How well plans were followed
            - avg_completion_rate: % of planned actions completed
            - deviation_rate: Average deviations per execution
        """
        outcomes = self._outcomes
        
        if not outcomes:
            return {
                "sample_size": 0,
                "avg_plan_adherence": 0.0,
                "avg_completion_rate": 0.0,
                "deviation_rate": 0.0,
            }
        
        total_adherence = sum(o.accuracy_metrics.plan_adherence_score for o in outcomes)
        
        completion_rates = []
        for o in outcomes:
            if o.plan_length > 0:
                completion_rates.append(o.actions_completed / o.plan_length)
        
        total_deviations = sum(
            o.accuracy_metrics.major_deviations + o.accuracy_metrics.minor_deviations
            for o in outcomes
        )
        
        return {
            "sample_size": len(outcomes),
            "avg_plan_adherence": total_adherence / len(outcomes),
            "avg_completion_rate": sum(completion_rates) / len(completion_rates) if completion_rates else 0.0,
            "deviation_rate": total_deviations / len(outcomes),
        }
    
    def analyze_scheduling_accuracy(self) -> Dict[str, Any]:
        """
        Analyze scheduling accuracy.
        
        Returns:
            Dict with scheduling metrics:
            - avg_time_accuracy_ratio: Actual/predicted time (1.0 = perfect)
            - underestimate_rate: % of times we underestimated
            - overestimate_rate: % of times we overestimated
            - avg_time_error_pct: Average % error in time estimates
        """
        outcomes = [o for o in self._outcomes if o.predicted_makespan_minutes > 0]
        
        if not outcomes:
            return {
                "sample_size": 0,
                "avg_time_accuracy_ratio": 0.0,
                "underestimate_rate": 0.0,
                "overestimate_rate": 0.0,
                "avg_time_error_pct": 0.0,
            }
        
        ratios = [o.accuracy_metrics.time_accuracy_ratio for o in outcomes]
        avg_ratio = sum(ratios) / len(ratios)
        
        underestimates = sum(1 for r in ratios if r > 1.0)
        overestimates = sum(1 for r in ratios if r < 1.0)
        
        errors = [abs(r - 1.0) * 100 for r in ratios]
        avg_error = sum(errors) / len(errors)
        
        return {
            "sample_size": len(outcomes),
            "avg_time_accuracy_ratio": avg_ratio,
            "underestimate_rate": underestimates / len(outcomes),
            "overestimate_rate": overestimates / len(outcomes),
            "avg_time_error_pct": avg_error,
        }
    
    def get_improvement_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Calculate improvement trends over time.
        
        Args:
            window_size: Number of recent outcomes to compare
        
        Returns:
            Dict comparing recent vs historical performance
        """
        if len(self._outcomes) < window_size * 2:
            return {
                "insufficient_data": True,
                "sample_size": len(self._outcomes),
                "required_size": window_size * 2,
            }
        
        # Split into recent and historical
        sorted_outcomes = sorted(self._outcomes, key=lambda o: o.timestamp)
        recent = sorted_outcomes[-window_size:]
        historical = sorted_outcomes[:-window_size]
        
        def calc_stats(outcomes: List[OutcomeRecord]) -> Dict[str, float]:
            return {
                "success_rate": sum(1 for o in outcomes if o.success) / len(outcomes),
                "avg_score": sum(o.outcome_score for o in outcomes) / len(outcomes),
                "avg_accuracy": sum(o.accuracy_metrics.overall_accuracy() for o in outcomes) / len(outcomes),
            }
        
        recent_stats = calc_stats(recent)
        historical_stats = calc_stats(historical)
        
        return {
            "window_size": window_size,
            "recent": recent_stats,
            "historical": historical_stats,
            "improvements": {
                "success_rate": recent_stats["success_rate"] - historical_stats["success_rate"],
                "avg_score": recent_stats["avg_score"] - historical_stats["avg_score"],
                "avg_accuracy": recent_stats["avg_accuracy"] - historical_stats["avg_accuracy"],
            },
        }
    
    def _calculate_accuracy_metrics(self, context: Any) -> AccuracyMetrics:
        """
        Calculate accuracy metrics from execution context.
        
        Args:
            context: ExecutionContext
        
        Returns:
            AccuracyMetrics object
        """
        metrics = AccuracyMetrics()
        
        # Time accuracy
        if context.schedule and context.schedule.makespan:
            makespan_minutes = context.schedule.makespan.total_seconds() / 60
            if makespan_minutes > 0:
                actual_time = (
                    (context.actual_completion_time or context.end_time or datetime.now()) 
                    - context.start_time
                ).total_seconds() / 60
                metrics.time_accuracy_ratio = actual_time / makespan_minutes
        
        # Plan adherence
        if context.plan and len(context.plan.steps) > 0:
            metrics.plan_adherence_score = context.actions_completed / len(context.plan.steps)
        
        # Goal achievement
        metrics.goal_achievement_score = 1.0 if context.actual_success else 0.0
        
        # Deviations
        for deviation in (context.deviations or []):
            if "major" in deviation.lower() or "critical" in deviation.lower():
                metrics.major_deviations += 1
            else:
                metrics.minor_deviations += 1
        
        return metrics
    
    def _load_outcomes(self) -> None:
        """Load all outcome records from storage."""
        outcome_files = list(self.storage_dir.glob("outcome_*.json"))
        
        for file_path in outcome_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    record = OutcomeRecord.from_dict(data)
                    self._outcomes.append(record)
            except Exception as e:
                logger.warning(f"Failed to load outcome from {file_path}: {e}")
        
        logger.debug(f"Loaded {len(self._outcomes)} outcomes from {self.storage_dir}")
    
    def _save_outcome(self, record: OutcomeRecord) -> None:
        """Save outcome record to storage."""
        file_path = self.storage_dir / f"outcome_{record.record_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(record.to_dict(), f, indent=2)
            logger.debug(f"Saved outcome to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save outcome to {file_path}: {e}")
    
    def clear_history(self) -> int:
        """
        Clear all outcome history (destructive!).
        
        Returns:
            Number of outcomes cleared
        """
        count = len(self._outcomes)
        self._outcomes.clear()
        
        # Delete all outcome files
        for file_path in self.storage_dir.glob("outcome_*.json"):
            file_path.unlink()
        
        logger.warning(f"Cleared {count} outcomes from history")
        return count
    
    def get_training_dataset(
        self,
        limit: Optional[int] = None,
        strategy: Optional[str] = None,
        success_only: bool = False,
        export_path: Optional[Path] = None,
        export_format: str = 'csv',
    ) -> List['FeatureVector']:
        """
        Get training dataset as feature vectors.
        
        Args:
            limit: Max number of outcomes to include
            strategy: Filter by decision strategy
            success_only: Only include successful outcomes
            export_path: Optional path to export dataset
            export_format: Export format ('csv', 'json', 'parquet')
        
        Returns:
            List of FeatureVector objects
        """
        # Get filtered outcomes
        outcomes = self.get_outcome_history(
            limit=limit,
            strategy=strategy,
            success_only=success_only,
        )
        
        # Extract features
        features = self.feature_extractor.extract_dataset(outcomes)
        
        # Export if requested
        if export_path:
            if export_format == 'csv':
                self.feature_extractor.export_csv(features, export_path)
            elif export_format == 'json':
                self.feature_extractor.export_json(features, export_path)
            elif export_format == 'parquet':
                self.feature_extractor.export_parquet(features, export_path)
            else:
                logger.warning(f"Unknown export format: {export_format}")
        
        logger.info(f"Generated training dataset with {len(features)} feature vectors")
        return features
