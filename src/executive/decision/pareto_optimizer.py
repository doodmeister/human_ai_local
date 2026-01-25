"""
Pareto Optimization for Multi-Objective Decision Making

Implements Pareto frontier analysis for decisions with competing objectives:
- Identify non-dominated solutions
- Calculate Pareto frontier
- Trade-off analysis
- Distance to ideal point

Useful when objectives conflict (e.g., speed vs. quality, cost vs. performance)
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
import time

from .base import (
    ParetoSolution,
    EnhancedDecisionContext,
    EnhancedDecisionResult,
    DecisionStrategy,
)

# Initialize logger
logger = logging.getLogger(__name__)

# Metrics tracking (lazy import to avoid circular dependency)
_metrics_registry = None

def get_metrics_registry():
    """Lazy import of metrics registry from chat system"""
    global _metrics_registry
    if _metrics_registry is None:
        try:
            from src.memory.metrics import metrics_registry
            _metrics_registry = metrics_registry
        except ImportError:
            # Fallback to dummy metrics if chat system unavailable
            class DummyMetrics:
                def inc(self, name, value=1): pass
                def observe(self, name, ms): pass
                def observe_hist(self, name, value, max_len=500): pass
            _metrics_registry = DummyMetrics()
    return _metrics_registry


@dataclass
class ParetoFrontier:
    """
    Complete Pareto frontier analysis
    
    Attributes:
        solutions: All solutions analyzed
        frontier: Non-dominated solutions (Pareto optimal)
        ideal_point: Ideal objective values
        nadir_point: Worst objective values
        hypervolume: Hypervolume indicator (quality metric)
    """
    solutions: List[ParetoSolution]
    frontier: List[ParetoSolution]
    ideal_point: Dict[str, float]
    nadir_point: Dict[str, float]
    hypervolume: float = 0.0


class ParetoOptimizer:
    """
    Pareto optimization for multi-objective decision making
    
    Identifies the Pareto frontier - set of solutions where improving
    one objective requires sacrificing another.
    """
    
    def __init__(self):
        """Initialize Pareto optimizer"""
        self.last_frontier: Optional[ParetoFrontier] = None
    
    def find_pareto_frontier(
        self,
        options: List[Dict[str, float]],
        objectives: List[str],
        maximize: Optional[Dict[str, bool]] = None
    ) -> ParetoFrontier:
        """
        Find Pareto frontier among options
        
        Args:
            options: List of options with objective scores
            objectives: List of objective names
            maximize: Dict mapping objective -> whether to maximize (default all True)
            
        Returns:
            Pareto frontier with non-dominated solutions
        """
        if maximize is None:
            maximize = {obj: True for obj in objectives}
        
        # Create ParetoSolution objects
        solutions = []
        for idx, option in enumerate(options):
            # Extract objective values, flip sign if minimizing
            obj_values = {}
            for obj in objectives:
                value = option.get(obj, 0.0)
                obj_values[obj] = value if maximize.get(obj, True) else -value
            
            solutions.append(ParetoSolution(
                option_id=f"option_{idx}",
                objectives=obj_values
            ))
        
        # Find dominated solutions
        for i, sol_i in enumerate(solutions):
            for j, sol_j in enumerate(solutions):
                if i != j and sol_j.dominates(sol_i):
                    sol_i.is_dominated = True
                if i != j and sol_i.dominates(sol_j):
                    sol_i.domination_count += 1
        
        # Extract Pareto frontier (non-dominated solutions)
        frontier = [sol for sol in solutions if not sol.is_dominated]
        
        # Calculate ideal and nadir points
        ideal_point = {}
        nadir_point = {}
        for obj in objectives:
            all_values = [sol.objectives[obj] for sol in solutions]
            ideal_point[obj] = max(all_values)
            nadir_point[obj] = min(all_values)
        
        # Calculate distance to ideal point for each solution
        for sol in solutions:
            sol.distance_to_ideal = self._euclidean_distance(
                sol.objectives,
                ideal_point
            )
        
        # Calculate hypervolume (quality indicator)
        hypervolume = self._calculate_hypervolume(frontier, nadir_point)
        
        result = ParetoFrontier(
            solutions=solutions,
            frontier=frontier,
            ideal_point=ideal_point,
            nadir_point=nadir_point,
            hypervolume=hypervolume
        )
        
        self.last_frontier = result
        return result
    
    def _euclidean_distance(
        self,
        point1: Dict[str, float],
        point2: Dict[str, float]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        squared_diffs = []
        for key in point1:
            diff = point1[key] - point2.get(key, 0.0)
            squared_diffs.append(diff ** 2)
        return np.sqrt(sum(squared_diffs))
    
    def _calculate_hypervolume(
        self,
        frontier: List[ParetoSolution],
        reference_point: Dict[str, float]
    ) -> float:
        """
        Calculate hypervolume indicator
        
        Simplified implementation for 2D case.
        For production, consider using pymoo or pygmo libraries.
        """
        if not frontier:
            return 0.0
        
        # Get objective names
        obj_names = list(reference_point.keys())
        
        if len(obj_names) == 2:
            # 2D case: calculate area
            # Sort by first objective
            sorted_frontier = sorted(
                frontier,
                key=lambda s: s.objectives[obj_names[0]],
                reverse=True
            )
            
            area = 0.0
            prev_x = reference_point[obj_names[0]]
            
            for sol in sorted_frontier:
                x = sol.objectives[obj_names[0]]
                y = sol.objectives[obj_names[1]]
                width = abs(x - prev_x)
                height = abs(y - reference_point[obj_names[1]])
                area += width * height
                prev_x = x
            
            return area
        else:
            # For higher dimensions, use approximation
            # Sum of volumes of hypercubes
            total_volume = 0.0
            for sol in frontier:
                volume = 1.0
                for obj in obj_names:
                    diff = abs(sol.objectives[obj] - reference_point[obj])
                    volume *= diff
                total_volume += volume
            return total_volume
    
    def select_from_frontier(
        self,
        frontier: ParetoFrontier,
        preference_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Select single solution from Pareto frontier
        
        If no preferences given, selects solution closest to ideal point.
        Otherwise, uses weighted sum with preferences.
        
        Args:
            frontier: Pareto frontier
            preference_weights: Optional weights for objectives
            
        Returns:
            ID of selected solution
        """
        if not frontier.frontier:
            return "no_solution"
        
        if preference_weights:
            # Weighted sum approach
            best_score = -float('inf')
            best_id = "no_solution"
            
            for sol in frontier.frontier:
                score = sum(
                    sol.objectives[obj] * preference_weights.get(obj, 1.0)
                    for obj in sol.objectives
                )
                if score > best_score:
                    best_score = score
                    best_id = sol.option_id
            
            return best_id
        else:
            # Select closest to ideal point
            best_solution = min(frontier.frontier, key=lambda s: s.distance_to_ideal)
            return best_solution.option_id
    
    def analyze_trade_offs(
        self,
        frontier: ParetoFrontier,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Analyze trade-offs between solutions on Pareto frontier
        
        Args:
            frontier: Pareto frontier
            top_n: Number of solutions to analyze
            
        Returns:
            List of trade-off analyses
        """
        trade_offs = []
        
        # Sort frontier by distance to ideal
        sorted_frontier = sorted(
            frontier.frontier,
            key=lambda s: s.distance_to_ideal
        )[:top_n]
        
        for i, sol in enumerate(sorted_frontier):
            # Find what this solution is good/bad at
            strengths = []
            weaknesses = []
            
            for obj, value in sol.objectives.items():
                ideal_val = frontier.ideal_point[obj]
                nadir_val = frontier.nadir_point[obj]
                
                # Normalize to 0-1 scale
                if ideal_val != nadir_val:
                    normalized = (value - nadir_val) / (ideal_val - nadir_val)
                else:
                    normalized = 1.0
                
                if normalized > 0.8:
                    strengths.append((obj, value, normalized))
                elif normalized < 0.4:
                    weaknesses.append((obj, value, normalized))
            
            trade_offs.append({
                'option_id': sol.option_id,
                'rank': i + 1,
                'distance_to_ideal': sol.distance_to_ideal,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'objectives': sol.objectives,
            })
        
        return trade_offs


class ParetoStrategy(DecisionStrategy):
    """
    Decision strategy using Pareto optimization
    
    Finds Pareto frontier and selects based on preferences or
    proximity to ideal point.
    """
    
    def __init__(self):
        """Initialize Pareto strategy"""
        self.optimizer = ParetoOptimizer()
    
    def decide(
        self,
        options: List[Dict[str, float]],
        criteria: List[str],  # List of objective names
        context: EnhancedDecisionContext
    ) -> EnhancedDecisionResult:
        """
        Make decision using Pareto optimization
        
        Args:
            options: List of options with objective scores
            criteria: List of objective names to consider
            context: Decision context (may include preference weights)
            
        Returns:
            Enhanced decision result with Pareto frontier
        """
        start_time = time.time()
        metrics = get_metrics_registry()
        
        try:
            # Find Pareto frontier
            frontier_start = time.time()
            frontier = self.optimizer.find_pareto_frontier(options, criteria)
            frontier_duration = (time.time() - frontier_start) * 1000.0
            
            # Track frontier metrics
            metrics.inc('pareto_decisions_total')
            metrics.observe_hist('pareto_frontier_size', len(frontier.frontier))
            metrics.observe_hist('pareto_frontier_ratio', len(frontier.frontier) / max(len(frontier.solutions), 1))
            metrics.observe_hist('pareto_hypervolume', frontier.hypervolume)
            
            # Extract preference weights from context
            preference_weights = context.user_preferences.get('objective_weights', None)
            
            # Select best solution from frontier
            selection_start = time.time()
            best_option_id = self.optimizer.select_from_frontier(
                frontier,
                preference_weights
            )
            selection_duration = (time.time() - selection_start) * 1000.0
            
            # Calculate option scores (distance to ideal, inverted)
            option_scores = {}
            for sol in frontier.solutions:
                # Lower distance is better, so invert
                max_dist = max(s.distance_to_ideal for s in frontier.solutions)
                if max_dist > 0:
                    score = 1.0 - (sol.distance_to_ideal / max_dist)
                else:
                    score = 1.0
                option_scores[sol.option_id] = score
            
            # Analyze trade-offs
            tradeoff_start = time.time()
            trade_offs = self.optimizer.analyze_trade_offs(frontier)
            tradeoff_duration = (time.time() - tradeoff_start) * 1000.0
            
            # Calculate confidence
            confidence = self._calculate_confidence(frontier, best_option_id)
            metrics.observe_hist('pareto_confidence', confidence)
            
            # Generate rationale
            rationale = self._generate_rationale(frontier, best_option_id, trade_offs)
            
            # Track performance metrics
            total_duration = (time.time() - start_time) * 1000.0
            metrics.observe('pareto_decision_latency_ms', total_duration)
            metrics.observe('pareto_frontier_calculation_ms', frontier_duration)
            metrics.observe('pareto_selection_ms', selection_duration)
            metrics.observe('pareto_tradeoff_analysis_ms', tradeoff_duration)
            
            # Track counts
            metrics.inc('pareto_alternatives_processed_total', len(options))
            metrics.observe_hist('pareto_alternatives_count', len(options))
            metrics.observe_hist('pareto_objectives_count', len(criteria))
            
            logger.info(
                f"Pareto decision complete: {len(options)} alternatives, "
                f"{len(criteria)} objectives, "
                f"frontier={len(frontier.frontier)}, "
                f"confidence={confidence:.2f}, "
                f"latency={total_duration:.1f}ms"
            )
            
            return EnhancedDecisionResult(
                recommended_option_id=best_option_id,
                option_scores=option_scores,
                criterion_weights={},  # Pareto doesn't use explicit weights
                pareto_frontier=[sol for sol in frontier.frontier],
                rationale=rationale,
                confidence=confidence,
                trade_offs=trade_offs,
                context=context,
                metadata={
                    'frontier_size': len(frontier.frontier),
                    'total_solutions': len(frontier.solutions),
                    'hypervolume': frontier.hypervolume,
                    'ideal_point': frontier.ideal_point,
                    'nadir_point': frontier.nadir_point,
                    'pareto_latency_ms': total_duration,
                    'pareto_frontier_calculation_ms': frontier_duration,
                    'pareto_selection_ms': selection_duration,
                    'pareto_tradeoff_analysis_ms': tradeoff_duration,
                }
            )
            
        except Exception as e:
            metrics.inc('pareto_errors_total')
            logger.error(f"Pareto decision failed: {e}", exc_info=True)
            raise
    
    def explain(self, result: EnhancedDecisionResult) -> str:
        """
        Generate natural language explanation
        
        Args:
            result: Decision result
            
        Returns:
            Human-readable explanation
        """
        explanation = [f"Recommended option: {result.recommended_option_id}"]
        explanation.append(f"Confidence: {result.confidence:.2%}")
        
        frontier_size = result.metadata.get('frontier_size', 0)
        total_solutions = result.metadata.get('total_solutions', 0)
        
        explanation.append(
            f"\nPareto analysis: {frontier_size} optimal solutions "
            f"out of {total_solutions} total."
        )
        
        if result.trade_offs:
            explanation.append("\nTop solutions and their trade-offs:")
            for trade_off in result.trade_offs[:3]:
                explanation.append(f"\n  {trade_off['option_id']} (rank {trade_off['rank']}):")
                
                if trade_off['strengths']:
                    strengths_str = ", ".join([
                        f"{obj} ({val:.2f})"
                        for obj, val, _ in trade_off['strengths']
                    ])
                    explanation.append(f"    ✓ Strong in: {strengths_str}")
                
                if trade_off['weaknesses']:
                    weaknesses_str = ", ".join([
                        f"{obj} ({val:.2f})"
                        for obj, val, _ in trade_off['weaknesses']
                    ])
                    explanation.append(f"    ✗ Weak in: {weaknesses_str}")
        
        explanation.append(f"\n{result.rationale}")
        
        return "\n".join(explanation)
    
    def _calculate_confidence(
        self,
        frontier: ParetoFrontier,
        best_option_id: str
    ) -> float:
        """Calculate confidence based on frontier characteristics"""
        # Confidence is higher when:
        # 1. Recommended solution dominates many others
        # 2. Frontier is small (clear winners)
        # 3. Solution is close to ideal point
        
        best_solution = next(
            (s for s in frontier.solutions if s.option_id == best_option_id),
            None
        )
        
        if not best_solution:
            return 0.5
        
        # Factor 1: Domination count (normalized)
        max_dom = max(s.domination_count for s in frontier.solutions)
        dom_confidence = best_solution.domination_count / max_dom if max_dom > 0 else 0.5
        
        # Factor 2: Frontier size (smaller = clearer)
        frontier_ratio = len(frontier.frontier) / len(frontier.solutions)
        frontier_confidence = 1.0 - frontier_ratio
        
        # Factor 3: Distance to ideal (normalized)
        max_dist = max(s.distance_to_ideal for s in frontier.solutions)
        dist_confidence = 1.0 - (best_solution.distance_to_ideal / max_dist) if max_dist > 0 else 1.0
        
        # Combine factors
        confidence = (0.4 * dom_confidence + 
                     0.3 * frontier_confidence + 
                     0.3 * dist_confidence)
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_rationale(
        self,
        frontier: ParetoFrontier,
        best_option_id: str,
        trade_offs: List[Dict[str, Any]]
    ) -> str:
        """Generate rationale for the decision"""
        best_trade_off = next(
            (t for t in trade_offs if t['option_id'] == best_option_id),
            None
        )
        
        rationale = (
            f"Pareto analysis identified {len(frontier.frontier)} optimal solutions. "
            f"{best_option_id} was selected as closest to the ideal point."
        )
        
        if best_trade_off and best_trade_off['strengths']:
            strengths_str = ", ".join([obj for obj, _, _ in best_trade_off['strengths']])
            rationale += f" It excels in: {strengths_str}."
        
        return rationale
