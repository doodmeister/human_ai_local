"""
Constraint Programming Scheduler using Google OR-Tools CP-SAT

Implements a sophisticated task scheduler that:
- Models scheduling as a Constraint Satisfaction Problem (CSP)
- Uses CP-SAT solver for efficient solution finding
- Handles multiple constraints (precedence, resources, deadlines)
- Supports multi-objective optimization
"""

from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any, cast
from dataclasses import dataclass
import logging

from ortools.sat.python import cp_model

from .models import (
    Task, Resource, Schedule, SchedulingProblem,
    SchedulingConstraint, OptimizationObjective, TaskStatus
)

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for the CP-SAT scheduler."""
    max_solve_time_seconds: float = 30.0  # Maximum time to search for solution
    num_workers: int = 4  # Parallel search threads
    log_search_progress: bool = False
    find_all_solutions: bool = False  # Find all solutions vs first feasible
    optimize: bool = True  # Optimize vs just find feasible


class CPScheduler:
    """
    Constraint Programming Scheduler using OR-Tools CP-SAT.
    
    Features:
    - Precedence constraints (task ordering)
    - Resource capacity constraints
    - Deadline constraints
    - Cognitive load limits
    - Multi-objective optimization
    
    Usage:
        problem = SchedulingProblem(tasks=tasks, resources=resources)
        scheduler = CPScheduler()
        schedule = scheduler.schedule(problem)
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize scheduler with configuration."""
        self.config = config or SchedulerConfig()
        self.model: Optional[cp_model.CpModel] = None
        self.solver: Optional[cp_model.CpSolver] = None
        
    def schedule(self, problem: SchedulingProblem) -> Schedule:
        """
        Create optimal schedule for the given problem.
        
        Args:
            problem: Scheduling problem with tasks, resources, constraints
            
        Returns:
            Schedule with task assignments and timings
            
        Raises:
            ValueError: If problem is invalid or infeasible
        """
        logger.info(f"Scheduling {len(problem.tasks)} tasks with {len(problem.resources)} resources")
        
        # Validate problem
        self._validate_problem(problem)
        
        # Create CP model
        self.model = cp_model.CpModel()
        assert self.model is not None  # Type guard
        
        # Convert times to discrete time steps
        time_steps = self._calculate_time_steps(problem)
        
        # Create decision variables
        task_vars = self._create_task_variables(problem, time_steps)
        resource_vars = self._create_resource_variables(problem, time_steps, task_vars)
        
        # Add constraints
        self._add_precedence_constraints(problem, task_vars)
        self._add_resource_constraints(problem, task_vars, resource_vars, time_steps)
        self._add_deadline_constraints(problem, task_vars, time_steps)
        self._add_cognitive_load_constraints(problem, task_vars, time_steps)
        
        # Add optimization objectives
        if self.config.optimize:
            self._add_objectives(problem, task_vars, time_steps)
        
        # Solve
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.config.max_solve_time_seconds
        self.solver.parameters.num_workers = self.config.num_workers
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        
        status = self.solver.Solve(self.model)
        
        # Extract solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = self._extract_schedule(problem, task_vars, time_steps, status)
            logger.info(f"Found {'optimal' if status == cp_model.OPTIMAL else 'feasible'} schedule")
            return schedule
        elif status == cp_model.INFEASIBLE:
            return self._create_infeasible_schedule(problem, "No feasible schedule exists")
        else:
            return self._create_infeasible_schedule(problem, f"Solver status: {status}")
    
    def _validate_problem(self, problem: SchedulingProblem):
        """Validate scheduling problem inputs."""
        if not problem.tasks:
            raise ValueError("No tasks to schedule")
        
        # Check for cyclic dependencies
        self._check_cyclic_dependencies(problem.tasks)
        
        # Validate time windows
        for task in problem.tasks:
            if task.time_window and task.time_window.duration() < task.duration:
                raise ValueError(f"Task {task.id} time window too short for duration")
    
    def _check_cyclic_dependencies(self, tasks: List[Task]):
        """Check for cycles in task dependencies."""
        task_map = {t.id: t for t in tasks}
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValueError(f"Cyclic dependency detected involving task {task.id}")
    
    def _calculate_time_steps(self, problem: SchedulingProblem) -> Dict[str, Any]:
        """
        Calculate time discretization for CP model.
        
        Returns dict with:
        - horizon: Number of time steps
        - step_size: Duration of each step
        - start_time: Schedule start time
        """
        # Use earliest time window start as schedule start
        start_time = datetime.now()
        for task in problem.tasks:
            if task.time_window:
                start_time = min(start_time, task.time_window.earliest_start)
        
        # Calculate horizon in time steps
        horizon_steps = int(problem.horizon.total_seconds() / problem.time_resolution.total_seconds())
        
        return {
            "horizon": horizon_steps,
            "step_size": problem.time_resolution,
            "start_time": start_time
        }
    
    def _create_task_variables(self, problem: SchedulingProblem, time_steps: Dict) -> Dict[str, Dict]:
        """
        Create CP variables for task scheduling.
        
        For each task:
        - start: Start time step
        - end: End time step
        - interval: Interval variable (start, duration, end)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        model = cast(cp_model.CpModel, self.model)  # Type guard for Pylance
            
        task_vars = {}
        horizon = time_steps["horizon"]
        step_size = time_steps["step_size"]
        
        for task in problem.tasks:
            duration_steps = int(task.duration.total_seconds() / step_size.total_seconds())
            duration_steps = max(1, duration_steps)  # At least 1 step
            
            # Create start variable
            start = model.NewIntVar(0, horizon - duration_steps, f"{task.id}_start")
            
            # Create end variable
            end = model.NewIntVar(duration_steps, horizon, f"{task.id}_end")
            
            # Create interval variable (enforces end = start + duration)
            interval = model.NewIntervalVar(start, duration_steps, end, f"{task.id}_interval")
            
            task_vars[task.id] = {
                "task": task,
                "start": start,
                "end": end,
                "interval": interval,
                "duration_steps": duration_steps
            }
        
        return task_vars
    
    def _create_resource_variables(self, problem: SchedulingProblem, 
                                   time_steps: Dict, task_vars: Dict) -> Dict:
        """Create variables for resource allocation."""
        # For now, we'll track resource usage but not create explicit variables
        # This can be extended for more complex resource optimization
        return {}
    
    def _add_precedence_constraints(self, problem: SchedulingProblem, task_vars: Dict):
        """Add precedence constraints (task A before task B)."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        model = cast(cp_model.CpModel, self.model)  # Type guard for Pylance
            
        for constraint in problem.constraints:
            if constraint.type == "precedence":
                before_id = constraint.parameters["before"]
                after_id = constraint.parameters["after"]
                
                if before_id in task_vars and after_id in task_vars:
                    # Task A must end before Task B starts
                    model.Add(
                        task_vars[before_id]["end"] <= task_vars[after_id]["start"]
                    )
                    logger.debug(f"Added precedence: {before_id} -> {after_id}")
        
        # Also add dependencies from task metadata
        for task in problem.tasks:
            for dep_id in task.dependencies:
                if dep_id in task_vars:
                    model.Add(
                        task_vars[dep_id]["end"] <= task_vars[task.id]["start"]
                    )
    
    def _add_resource_constraints(self, problem: SchedulingProblem, 
                                  task_vars: Dict, resource_vars: Dict, time_steps: Dict):
        """Add resource capacity constraints."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        model = cast(cp_model.CpModel, self.model)  # Type guard for Pylance
            
        for resource in problem.resources:
            # Find tasks that need this resource
            demanding_tasks = [
                (task, task_vars[task.id]) 
                for task in problem.tasks 
                if resource in task.resource_requirements and task.id in task_vars
            ]
            
            if not demanding_tasks:
                continue
            
            # Create cumulative constraint for resource capacity
            intervals = [tv["interval"] for _, tv in demanding_tasks]
            demands = [int(task.resource_requirements[resource]) for task, _ in demanding_tasks]
            
            model.AddCumulative(intervals, demands, int(resource.capacity))
            logger.debug(f"Added resource constraint for {resource.name} (capacity={resource.capacity})")
    
    def _add_deadline_constraints(self, problem: SchedulingProblem, 
                                  task_vars: Dict, time_steps: Dict):
        """Add deadline constraints."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        model = cast(cp_model.CpModel, self.model)  # Type guard for Pylance
        
        start_time = time_steps["start_time"]
        step_size = time_steps["step_size"]
        
        for constraint in problem.constraints:
            if constraint.type == "deadline":
                task_id = constraint.parameters["task"]
                deadline = constraint.parameters["deadline"]
                
                if task_id in task_vars:
                    # Convert deadline to time steps
                    deadline_steps = int((deadline - start_time).total_seconds() / step_size.total_seconds())
                    
                    # Task must end before deadline
                    model.Add(task_vars[task_id]["end"] <= deadline_steps)
                    logger.debug(f"Added deadline for {task_id}: step {deadline_steps}")
        
        # Also check time windows
        for task in problem.tasks:
            if task.time_window and task.id in task_vars:
                # Convert time window to steps
                earliest_step = int((task.time_window.earliest_start - start_time).total_seconds() / step_size.total_seconds())
                latest_step = int((task.time_window.latest_end - start_time).total_seconds() / step_size.total_seconds())
                
                model.Add(task_vars[task.id]["start"] >= earliest_step)
                model.Add(task_vars[task.id]["end"] <= latest_step)
    
    def _add_cognitive_load_constraints(self, problem: SchedulingProblem, 
                                       task_vars: Dict, time_steps: Dict):
        """Add cognitive load limit constraints."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        model = cast(cp_model.CpModel, self.model)  # Type guard for Pylance
        
        for constraint in problem.constraints:
            if constraint.type == "cognitive_load_limit":
                max_load = constraint.parameters["max_load"]
                
                # For each time step, sum cognitive load of active tasks must not exceed limit
                # This is complex in CP-SAT, so we use cumulative constraint
                intervals = [tv["interval"] for tv in task_vars.values()]
                loads = [int(tv["task"].cognitive_load * 100) for tv in task_vars.values()]
                
                model.AddCumulative(intervals, loads, int(max_load * 100))
                logger.debug(f"Added cognitive load limit: {max_load}")
    
    def _add_objectives(self, problem: SchedulingProblem, task_vars: Dict, time_steps: Dict):
        """Add optimization objectives."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        model = cast(cp_model.CpModel, self.model)  # Type guard for Pylance
        
        objective_terms = []
        
        for obj in problem.objectives:
            if obj.name == "minimize_makespan":
                # Minimize maximum end time
                max_end = model.NewIntVar(0, time_steps["horizon"], "makespan")
                for tv in task_vars.values():
                    model.Add(max_end >= tv["end"])
                objective_terms.append(int(obj.weight * 1000) * max_end)
            
            elif obj.name == "maximize_priority":
                # Maximize weighted early completion
                for tv in task_vars.values():
                    priority = int(tv["task"].priority * 100)
                    # Reward early completion (negative cost)
                    objective_terms.append(-priority * (time_steps["horizon"] - tv["end"]))
        
        if objective_terms:
            model.Minimize(sum(objective_terms))
    
    def _extract_schedule(self, problem: SchedulingProblem, task_vars: Dict, 
                         time_steps: Dict, status) -> Schedule:
        """Extract schedule from solved CP model."""
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        solver = cast(cp_model.CpSolver, self.solver)  # Type guard for Pylance
        
        start_time = time_steps["start_time"]
        step_size = time_steps["step_size"]
        
        # Update tasks with scheduled times
        scheduled_tasks = []
        max_end_step = 0
        
        for task in problem.tasks:
            if task.id not in task_vars:
                continue
            
            tv = task_vars[task.id]
            start_step = solver.Value(tv["start"])
            end_step = solver.Value(tv["end"])
            
            task.scheduled_start = start_time + timedelta(seconds=start_step * step_size.total_seconds())
            task.scheduled_end = start_time + timedelta(seconds=end_step * step_size.total_seconds())
            task.status = TaskStatus.SCHEDULED
            
            scheduled_tasks.append(task)
            max_end_step = max(max_end_step, end_step)
        
        # Calculate makespan
        makespan = timedelta(seconds=max_end_step * step_size.total_seconds())
        end_time = start_time + makespan
        
        # Create schedule
        schedule = Schedule(
            tasks=scheduled_tasks,
            makespan=makespan,
            start_time=start_time,
            end_time=end_time,
            is_feasible=True
        )
        
        # Calculate basic metrics
        schedule.metrics = {
            "makespan_hours": makespan.total_seconds() / 3600,
            "num_tasks": len(scheduled_tasks),
            "solve_time": solver.WallTime(),
            "optimal": status == cp_model.OPTIMAL
        }
        
        # Calculate quality metrics
        schedule.update_quality_metrics()
        
        return schedule
    
    def _create_infeasible_schedule(self, problem: SchedulingProblem, reason: str) -> Schedule:
        """Create schedule object for infeasible problem."""
        return Schedule(
            tasks=problem.tasks,
            makespan=timedelta(0),
            start_time=datetime.now(),
            end_time=datetime.now(),
            is_feasible=False,
            infeasibility_reasons=[reason]
        )
