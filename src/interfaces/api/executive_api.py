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
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...executive.executive_agent import ExecutiveAgent
from ...executive.goal_manager import GoalPriority

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
