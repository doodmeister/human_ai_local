"""
REST API for Human-AI Cognition Framework: Executive Functions (Simplified)

This module provides basic RESTful API endpoints for interacting with the executive
functioning system. This is a simplified version that works with the actual
executive system implementation.

Endpoints:
- POST /executive/goals: Create a new goal
- GET /executive/goals: List all goals  
- GET /executive/goals/{goal_id}: Get specific goal details
- GET /executive/status: Get executive system status
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from ...executive.executive_agent import ExecutiveAgent
from ...executive.goal_manager import GoalPriority
from ...executive.integration import ExecutiveSystem
from ...executive.planning.world_state import WorldState

router = APIRouter()

# Simplified Pydantic models for API requests/responses

class CreateGoalRequest(BaseModel):
    title: str = Field(..., description="Goal title")
    description: str = Field(default="", description="Goal description")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Goal priority (0.0-1.0)")
    deadline: Optional[str] = Field(None, description="Goal deadline (YYYY-MM-DD or YYYY-MM-DD HH:MM)")

# Helper function to get executive agent from cognitive agent
def get_executive_agent(request: Request) -> ExecutiveAgent:
    """Get or create executive agent from the main cognitive agent"""
    cognitive_agent = request.app.state.agent
    
    # Check if cognitive agent has executive capabilities
    if not hasattr(cognitive_agent, 'executive_agent'):
        # Create executive agent with required dependencies
        cognitive_agent.executive_agent = ExecutiveAgent(
            attention_mechanism=cognitive_agent.attention,
            memory_system=cognitive_agent.memory
        )
    
    return cognitive_agent.executive_agent


def get_executive_system(request: Request) -> ExecutiveSystem:
    """Get or create ExecutiveSystem for integrated pipeline operations"""
    cognitive_agent = request.app.state.agent
    
    # Check if cognitive agent has integrated executive system
    if not hasattr(cognitive_agent, 'executive_system'):
        cognitive_agent.executive_system = ExecutiveSystem()
    
    return cognitive_agent.executive_system

# Goal Management Endpoints

@router.post("/goals")
async def create_goal(goal_request: CreateGoalRequest, request: Request):
    """Create a new goal"""
    try:
        executive = get_executive_agent(request)
        
        # Parse deadline if provided
        target_date = None
        if goal_request.deadline:
            try:
                if len(goal_request.deadline) == 10:  # YYYY-MM-DD
                    date_obj = datetime.strptime(goal_request.deadline, "%Y-%m-%d")
                    target_date = datetime.combine(date_obj.date(), datetime.min.time())
                else:  # YYYY-MM-DD HH:MM
                    target_date = datetime.strptime(goal_request.deadline, "%Y-%m-%d %H:%M")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid deadline format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM")
        
        # Convert priority float to GoalPriority enum
        if goal_request.priority > 0.7:
            priority = GoalPriority.HIGH
        elif goal_request.priority < 0.3:
            priority = GoalPriority.LOW
        else:
            priority = GoalPriority.MEDIUM
        
        goal_id = executive.goals.create_goal(
            title=goal_request.title,
            description=goal_request.description,
            priority=priority,
            target_date=target_date
        )
        
        return {"status": "success", "goal_id": goal_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create goal: {str(e)}")

@router.get("/goals")
async def list_goals(request: Request, active_only: bool = False):
    """List all goals"""
    try:
        executive = get_executive_agent(request)
        
        if active_only:
            goals = executive.goals.get_active_goals()
        else:
            # Get all goals from the goals dictionary
            goals = list(executive.goals.goals.values())
        
        # Convert goals to dictionaries for JSON response
        goals_data = []
        for goal in goals:
            goal_dict = {
                "id": goal.id,
                "title": goal.title,
                "description": goal.description,
                "priority": goal.priority.value,
                "progress": goal.progress,
                "status": goal.status.value,
                "created_at": goal.created_at.isoformat(),
                "target_date": goal.target_date.isoformat() if goal.target_date else None
            }
            goals_data.append(goal_dict)
        
        return {
            "status": "success",
            "goals": goals_data,
            "count": len(goals_data)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list goals: {str(e)}")

@router.get("/goals/{goal_id}")
async def get_goal(goal_id: str, request: Request):
    """Get specific goal details"""
    try:
        executive = get_executive_agent(request)
        goal = executive.goals.get_goal(goal_id)
        
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        goal_dict = {
            "id": goal.id,
            "title": goal.title,
            "description": goal.description,
            "priority": goal.priority.value,
            "progress": goal.progress,
            "status": goal.status.value,
            "created_at": goal.created_at.isoformat(),
            "target_date": goal.target_date.isoformat() if goal.target_date else None
        }
        
        return {
            "status": "success",
            "goal": goal_dict
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get goal: {str(e)}")

@router.get("/status")
async def get_executive_status(request: Request):
    """Get comprehensive executive system status matching UI expectations"""
    try:
        executive = get_executive_agent(request)
        
        # Get all goals
        all_goals = list(executive.goals.goals.values())
        active_goals = executive.goals.get_active_goals()
        completed_goals = [g for g in all_goals if g.status.value == "completed"]
        
        # Convert goals to dictionaries for JSON response
        active_goals_data = []
        for goal in active_goals:
            goal_dict = {
                "id": goal.id,
                "title": getattr(goal, 'title', goal.description[:50]),
                "description": goal.description,
                "priority": goal.priority.value if hasattr(goal.priority, 'value') else str(goal.priority),
                "progress": goal.progress,
                "status": goal.status.value,
                "created_at": goal.created_at.isoformat(),
                "target_date": goal.target_date.isoformat() if goal.target_date else None
            }
            active_goals_data.append(goal_dict)
        
        completed_goals_data = []
        for goal in completed_goals:
            goal_dict = {
                "id": goal.id,
                "title": getattr(goal, 'title', goal.description[:50]),
                "description": goal.description,
                "priority": goal.priority.value if hasattr(goal.priority, 'value') else str(goal.priority),
                "progress": goal.progress,
                "status": goal.status.value,
                "created_at": goal.created_at.isoformat(),
                "target_date": goal.target_date.isoformat() if goal.target_date else None
            }
            completed_goals_data.append(goal_dict)
        
        # Recent decisions (simulated for now)
        recent_decisions = [
            {
                "context": "Sample decision context",
                "selected_option": "Option A",
                "confidence": 0.75,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Resource allocation (simulated)
        resources = {
            "attention": 0.7,
            "memory": 0.6,
            "processing": 0.8,
            "energy": 0.5
        }
        
        status = {
            "goals": {
                "active_goals": active_goals_data,
                "completed_goals": completed_goals_data,
                "total_goals": len(all_goals)
            },
            "recent_decisions": recent_decisions,
            "resources": resources,
            "executive_state": executive.state.value if hasattr(executive, 'state') else "active",
            "performance_metrics": {
                "goal_completion_rate": len(completed_goals) / max(len(all_goals), 1),
                "average_decision_confidence": 0.75,
                "resource_efficiency": 0.68
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    except Exception as e:
        # Return error in expected format
        return {
            "goals": {"active_goals": [], "completed_goals": [], "total_goals": 0},
            "recent_decisions": [],
            "resources": {},
            "executive_state": "error",
            "error": f"Failed to get executive status: {str(e)}"
        }


# Integrated Pipeline Endpoints (ExecutiveSystem)

@router.post("/goals/{goal_id}/execute")
async def execute_goal(goal_id: str, request: Request):
    """Execute integrated pipeline for a goal: Decision → Plan → Schedule"""
    try:
        system = get_executive_system(request)
        
        # Execute goal through integrated pipeline
        context = system.execute_goal(goal_id, initial_state=WorldState({}))
        
        return {
            "status": "success",
            "execution_context": {
                "goal_id": context.goal_id,
                "status": context.status.value,
                "decision_time_ms": context.decision_time_ms,
                "planning_time_ms": context.planning_time_ms,
                "scheduling_time_ms": context.scheduling_time_ms,
                "total_actions": context.total_actions,
                "scheduled_tasks": context.scheduled_tasks,
                "makespan": context.makespan,
                "has_plan": context.plan is not None,
                "has_schedule": context.schedule is not None,
                "failure_reason": context.failure_reason
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute goal: {str(e)}")


@router.get("/goals/{goal_id}/plan")
async def get_goal_plan(goal_id: str, request: Request):
    """Get GOAP plan for a goal"""
    try:
        system = get_executive_system(request)
        
        # Check if execution context exists
        context = system.execution_contexts.get(goal_id)
        if not context or not context.plan:
            return {
                "status": "no_plan",
                "message": "No plan generated yet. Execute the goal first."
            }
        
        plan = context.plan
        
        # Convert plan to JSON-serializable format
        plan_data = {
            "steps": [
                {
                    "name": step.name,
                    "preconditions": step.preconditions.state if hasattr(step.preconditions, 'state') else {},
                    "effects": step.effects.state if hasattr(step.effects, 'state') else {},
                    "cost": step.cost
                }
                for step in plan.steps
            ],
            "total_cost": plan.total_cost,
            "length": len(plan.steps),
            "planning_time_ms": context.planning_time_ms,
            "heuristic": plan.metadata.get("heuristic", "unknown") if hasattr(plan, 'metadata') else "unknown"
        }
        
        return {
            "status": "success",
            "plan": plan_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plan: {str(e)}")


@router.get("/goals/{goal_id}/schedule")
async def get_goal_schedule(goal_id: str, request: Request):
    """Get CP-SAT schedule for a goal"""
    try:
        system = get_executive_system(request)
        
        # Check if execution context exists
        context = system.execution_contexts.get(goal_id)
        if not context or not context.schedule:
            return {
                "status": "no_schedule",
                "message": "No schedule generated yet. Execute the goal first."
            }
        
        schedule = context.schedule
        
        # Convert schedule to JSON-serializable format
        tasks_data = []
        for task in schedule.tasks:
            task_data = {
                "id": task.id,
                "name": task.name if hasattr(task, 'name') else task.id,
                "scheduled_start": task.scheduled_start.isoformat() if task.scheduled_start else None,
                "scheduled_end": task.scheduled_end.isoformat() if task.scheduled_end else None,
                "duration": task.duration,
                "priority": task.priority,
                "resources": [r.name if hasattr(r, 'name') else str(r) for r in task.resource_requirements],
                "dependencies": task.dependencies
            }
            tasks_data.append(task_data)
        
        schedule_data = {
            "tasks": tasks_data,
            "makespan": schedule.makespan,
            "robustness_score": schedule.quality_metrics.get("robustness_score", 0.0) if hasattr(schedule, 'quality_metrics') else 0.0,
            "cognitive_smoothness": schedule.quality_metrics.get("cognitive_smoothness", 0.0) if hasattr(schedule, 'quality_metrics') else 0.0,
            "resource_utilization": {},  # Simplified for now
            "scheduling_time_ms": context.scheduling_time_ms
        }
        
        return {
            "status": "success",
            "schedule": schedule_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schedule: {str(e)}")


@router.get("/goals/{goal_id}/status")
async def get_execution_status(goal_id: str, request: Request):
    """Get execution context and pipeline status for a goal"""
    try:
        system = get_executive_system(request)
        
        context = system.execution_contexts.get(goal_id)
        if not context:
            return {
                "status": "not_executed",
                "message": "Goal has not been executed yet"
            }
        
        # Build comprehensive context
        context_data = {
            "goal_id": context.goal_id,
            "goal_title": context.goal_title,
            "status": context.status.value,
            "decision_time_ms": context.decision_time_ms,
            "planning_time_ms": context.planning_time_ms,
            "scheduling_time_ms": context.scheduling_time_ms,
            "total_actions": context.total_actions,
            "scheduled_tasks": context.scheduled_tasks,
            "makespan": context.makespan,
            "failure_reason": context.failure_reason,
            "actual_success": context.actual_success,
            "outcome_score": context.outcome_score,
            "decision_result": {
                "strategy": context.decision_result.strategy if context.decision_result else "unknown",
                "confidence": context.decision_result.confidence if context.decision_result else 0.0,
                "selected_option": str(context.decision_result.selected_option) if context.decision_result else None
            } if context.decision_result else None,
            "accuracy_metrics": context.accuracy_metrics if hasattr(context, 'accuracy_metrics') else None
        }
        
        return {
            "status": "success",
            "context": context_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")


@router.get("/system/health")
async def get_system_health(request: Request):
    """Get comprehensive executive system health metrics"""
    try:
        system = get_executive_system(request)
        health = system.get_system_health()
        
        # Convert to JSON-serializable format
        health_data = {
            "active_goals": health.get("active_goals", 0),
            "total_executions": health.get("total_executions", 0),
            "success_rate": health.get("success_rate", 0.0),
            "avg_execution_time_ms": health.get("avg_execution_time_ms", 0.0),
            "subsystems": health.get("subsystems", {}),
            "recent_activity": health.get("recent_activity", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "health": health_data
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to get system health: {str(e)}"
        }

