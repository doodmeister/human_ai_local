"""
REST API for Human-AI Cognition Framework: Executive Functions

This module provides RESTful API endpoints for interacting with the executive
functioning system, including goal management, task planning, decision making,
and resource allocation.

Endpoints:
- POST /executive/goals: Create a new goal
- GET /executive/goals: List all goals
- GET /executive/goals/{goal_id}: Get specific goal details
- PUT /executive/goals/{goal_id}: Update a goal
- DELETE /executive/goals/{goal_id}: Delete a goal
- POST /executive/tasks: Create tasks for a goal
- GET /executive/tasks: List all tasks
- GET /executive/tasks/{task_id}: Get specific task details
- PUT /executive/tasks/{task_id}: Update task status
- POST /executive/decisions: Make a decision
- GET /executive/decisions/{decision_id}: Get decision details
- GET /executive/resources: Get resource allocation status
- POST /executive/resources/allocate: Allocate resources
- GET /executive/status: Get executive system status
- POST /executive/reflect: Trigger executive reflection
- GET /executive/performance: Get performance metrics
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.orchestration.executive_facade import ExecutiveAgent
from ..executive.goals import GoalPriority
from src.orchestration.executive_facade import TaskStatus

router = APIRouter()

# Pydantic models for API requests/responses

class CreateGoalRequest(BaseModel):
    description: str = Field(..., description="Goal description")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Goal priority (0.0-1.0)")
    deadline: Optional[str] = Field(None, description="Goal deadline (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parent_goal_id: Optional[str] = Field(None, description="Parent goal ID for hierarchical goals")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UpdateGoalRequest(BaseModel):
    description: Optional[str] = None
    priority: Optional[float] = Field(None, ge=0.0, le=1.0)
    deadline: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class CreateTaskRequest(BaseModel):
    goal_id: str = Field(..., description="Goal ID this task belongs to")
    description: str = Field(..., description="Task description")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_effort: float = Field(default=1.0, gt=0.0, description="Estimated effort (hours)")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Task IDs this depends on")

class UpdateTaskRequest(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    actual_effort: Optional[float] = Field(None, gt=0.0)

class MakeDecisionRequest(BaseModel):
    context: str = Field(..., description="Decision context/question")
    options: List[str] = Field(..., description="Available options")
    criteria: Dict[str, float] = Field(..., description="Decision criteria with weights")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AllocateResourcesRequest(BaseModel):
    resource_demands: Dict[str, float] = Field(..., description="Resource demands by type")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_minutes: Optional[float] = Field(None, gt=0.0)

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
        
        goal_id = executive.goals.create_goal(
            title=goal_request.description,  # Using description as title for now
            description=goal_request.description,
            priority=GoalPriority.HIGH if goal_request.priority > 0.7 else 
                     GoalPriority.LOW if goal_request.priority < 0.3 else GoalPriority.MEDIUM,
            parent_id=goal_request.parent_goal_id,
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
        
        return {
            "status": "success",
            "goals": [goal.to_dict() for goal in goals],
            "count": len(goals)
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
        
        return {
            "status": "success",
            "goal": goal.to_dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get goal: {str(e)}")

@router.put("/goals/{goal_id}")
async def update_goal(goal_id: str, update_request: UpdateGoalRequest, request: Request):
    """Update a goal"""
    try:
        executive = get_executive_agent(request)
        
        # Parse deadline if provided
        deadline = None
        if update_request.deadline:
            try:
                if len(update_request.deadline) == 10:  # YYYY-MM-DD
                    deadline = datetime.strptime(update_request.deadline, "%Y-%m-%d").date()
                else:  # YYYY-MM-DD HH:MM
                    deadline = datetime.strptime(update_request.deadline, "%Y-%m-%d %H:%M")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid deadline format")
        
        success = executive.goals.update_goal(
            goal_id=goal_id,
            description=update_request.description,
            priority=update_request.priority,
            deadline=deadline,
            metadata=update_request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        return {"status": "success", "updated": True}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update goal: {str(e)}")

@router.delete("/goals/{goal_id}")
async def delete_goal(goal_id: str, request: Request):
    """Delete a goal"""
    try:
        executive = get_executive_agent(request)
        success = executive.goals.delete_goal(goal_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        return {"status": "success", "deleted": True}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete goal: {str(e)}")

# Task Management Endpoints

@router.post("/tasks")
async def create_tasks(task_request: CreateTaskRequest, request: Request):
    """Create tasks for a goal"""
    try:
        executive = get_executive_agent(request)
        
        tasks = executive.tasks.create_tasks_for_goal(
            goal_id=task_request.goal_id,
            custom_description=task_request.description,
            priority_override=task_request.priority,
            estimated_effort=task_request.estimated_effort,
            dependencies=task_request.dependencies
        )
        
        return {
            "status": "success",
            "tasks": [task.to_dict() for task in tasks],
            "count": len(tasks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tasks: {str(e)}")

@router.get("/tasks")
async def list_tasks(request: Request, goal_id: Optional[str] = None, status: Optional[str] = None):
    """List all tasks, optionally filtered by goal or status"""
    try:
        executive = get_executive_agent(request)
        
        if goal_id:
            tasks = executive.tasks.get_tasks_for_goal(goal_id)
        else:
            tasks = executive.tasks.get_all_tasks()
        
        # Filter by status if provided
        if status:
            try:
                status_enum = TaskStatus(status)
                tasks = [task for task in tasks if task.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        return {
            "status": "success",
            "tasks": [task.to_dict() for task in tasks],
            "count": len(tasks)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    """Get specific task details"""
    try:
        executive = get_executive_agent(request)
        task = executive.tasks.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "status": "success",
            "task": task.to_dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task: {str(e)}")

@router.put("/tasks/{task_id}")
async def update_task(task_id: str, update_request: UpdateTaskRequest, request: Request):
    """Update task status and progress"""
    try:
        executive = get_executive_agent(request)
        
        # Parse status if provided
        task_status = None
        if update_request.status:
            try:
                task_status = TaskStatus(update_request.status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {update_request.status}")
        
        success = executive.tasks.update_task(
            task_id=task_id,
            status=task_status,
            progress=update_request.progress,
            actual_effort=update_request.actual_effort
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"status": "success", "updated": True}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")

# Decision Making Endpoints

@router.post("/decisions")
async def make_decision(decision_request: MakeDecisionRequest, request: Request):
    """Make a decision using the decision engine"""
    try:
        executive = get_executive_agent(request)
        
        decision = executive.decisions.make_decision(
            context=decision_request.context,
            options=decision_request.options,
            criteria=decision_request.criteria,
            metadata=decision_request.metadata
        )
        
        return {
            "status": "success",
            "decision": decision.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to make decision: {str(e)}")

@router.get("/decisions/{decision_id}")
async def get_decision(decision_id: str, request: Request):
    """Get decision details"""
    try:
        executive = get_executive_agent(request)
        decision = executive.decisions.get_decision(decision_id)
        
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        
        return {
            "status": "success",
            "decision": decision.to_dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get decision: {str(e)}")

# Resource Management Endpoints

@router.get("/resources")
async def get_resource_status(request: Request):
    """Get current resource allocation status"""
    try:
        executive = get_executive_agent(request)
        allocation = executive.controller.get_resource_allocation()
        
        return {
            "status": "success",
            "resources": allocation,
            "cognitive_mode": executive.controller.current_mode.value
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource status: {str(e)}")

@router.post("/resources/allocate")
async def allocate_resources(allocation_request: AllocateResourcesRequest, request: Request):
    """Allocate cognitive resources"""
    try:
        executive = get_executive_agent(request)
        
        allocation = executive.controller.allocate_resources(
            resource_demands=allocation_request.resource_demands,
            priority=allocation_request.priority,
            duration_minutes=allocation_request.duration_minutes
        )
        
        return {
            "status": "success",
            "allocation": allocation
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to allocate resources: {str(e)}")

# Executive Status and Performance Endpoints

@router.get("/status")
async def get_executive_status(request: Request):
    """Get comprehensive executive system status"""
    try:
        executive = get_executive_agent(request)
        
        status = {
            "executive_state": executive.state.value,
            "current_context": executive.current_context.__dict__ if executive.current_context else None,
            "goal_summary": {
                "total_goals": len(executive.goals.goals),
                "active_goals": len([g for g in executive.goals.goals.values() if not g.completed]),
                "completed_goals": len([g for g in executive.goals.goals.values() if g.completed])
            },
            "task_summary": {
                "total_tasks": len(executive.tasks.tasks),
                "pending_tasks": len([t for t in executive.tasks.tasks.values() if t.status == TaskStatus.PENDING]),
                "in_progress_tasks": len([t for t in executive.tasks.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                "completed_tasks": len([t for t in executive.tasks.tasks.values() if t.status == TaskStatus.COMPLETED])
            },
            "decision_summary": {
                "total_decisions": len(executive.decisions.decisions),
                "recent_decisions": len([d for d in executive.decisions.decisions.values() 
                                       if (datetime.now() - d.timestamp).days <= 1])
            },
            "resource_status": executive.controller.get_resource_allocation(),
            "cognitive_mode": executive.controller.current_mode.value,
            "performance_metrics": executive.get_performance_metrics()
        }
        
        return {"status": "success", "executive_status": status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get executive status: {str(e)}")

@router.post("/reflect")
async def executive_reflect(request: Request):
    """Trigger executive reflection and optimization"""
    try:
        executive = get_executive_agent(request)
        
        reflection = executive.reflect()
        
        return {
            "status": "success",
            "reflection": reflection
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform executive reflection: {str(e)}")

@router.get("/performance")
async def get_performance_metrics(request: Request):
    """Get detailed performance metrics"""
    try:
        executive = get_executive_agent(request)
        metrics = executive.get_performance_metrics()
        
        return {
            "status": "success",
            "performance_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")
