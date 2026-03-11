from __future__ import annotations

import logging
import traceback
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.resilience import CircuitBreaker, get_telemetry
from src.orchestration.runtime.app_container import get_runtime

logger = logging.getLogger(__name__)


def create_api_app() -> FastAPI:
    """Create the canonical FastAPI app for the project."""
    app = FastAPI(
        title="George Cognitive API",
        version="1.0.0",
        description="Human-AI Cognition Framework API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        error_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        logger.error(
            "Unhandled exception [%s] on %s %s: %s: %s\n%s",
            error_id,
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "error_id": error_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.middleware("http")
    async def ensure_agent_state(request: Request, call_next):
        path = request.url.path
        needs_agent = path.startswith(
            (
                "/memory",
                "/semantic",
                "/executive",
                "/procedure",
                "/prospective",
                "/agent",
                "/reflect",
            )
        )
        if needs_agent and getattr(request.app.state, "agent", None) is None:
            request.app.state.agent = get_runtime().get_agent()
        return await call_next(request)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = datetime.now()
        response = await call_next(request)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        if request.url.path not in {"/health", "/metrics"}:
            logger.info(
                "%s %s - Status: %s - Duration: %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
        return response

    from src.interfaces.api.chat_endpoints import router as chat_router
    from src.interfaces.api.executive_api import router as executive_router
    from src.interfaces.api.memory_api import router as memory_router
    from src.interfaces.api.semantic_api import router as semantic_router
    from src.interfaces.api.prospective_api import router as prospective_router
    from src.interfaces.api.procedural_api import router as procedural_router

    app.include_router(chat_router)
    app.include_router(executive_router, prefix="/executive", tags=["executive"])
    app.include_router(memory_router, tags=["memory"])
    app.include_router(semantic_router, tags=["semantic"])
    app.include_router(prospective_router, tags=["prospective"])
    app.include_router(procedural_router, tags=["procedural"])

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "George Cognitive API"}

    @app.get("/health/detailed")
    async def detailed_health_check(request: Request):
        runtime = get_runtime()
        agent = getattr(request.app.state, "agent", None)

        components = {
            "api": {"healthy": True, "message": "API server running"},
            "agent": {
                "healthy": agent is not None or runtime.has_agent(),
                "message": "Agent ready" if agent is not None or runtime.has_agent() else "Not initialized",
            },
        }

        if agent is not None:
            try:
                if hasattr(agent, "memory"):
                    components["memory"] = {"healthy": True, "message": "Memory system available"}
            except Exception as exc:
                components["memory"] = {"healthy": False, "message": str(exc)}

            try:
                if hasattr(agent, "executive_agent") or hasattr(agent, "executive_system"):
                    components["executive"] = {"healthy": True, "message": "Executive system available"}
            except Exception as exc:
                components["executive"] = {"healthy": False, "message": str(exc)}

        overall_healthy = all(component.get("healthy", False) for component in components.values())
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": components,
        }

    @app.get("/agent/init-status")
    async def initialization_status(request: Request):
        agent = getattr(request.app.state, "agent", None)
        runtime = get_runtime()
        if agent is not None or runtime.has_agent():
            return {"status": "ready", "message": "Agent is fully initialized"}
        return {"status": "not_started", "message": "Agent initialization has not started"}

    @app.get("/agent/status")
    async def agent_status(request: Request):
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            agent = get_runtime().get_agent()
            request.app.state.agent = agent
        return {"status": "ok", "cognitive_status": agent.get_cognitive_status()}

    @app.get("/telemetry")
    async def get_telemetry_metrics():
        return get_telemetry().get_metrics()

    @app.get("/circuit-breakers")
    async def get_circuit_breakers():
        return {
            "circuit_breakers": {
                name: breaker.get_status()
                for name, breaker in CircuitBreaker._instances.items()
            }
        }

    return app