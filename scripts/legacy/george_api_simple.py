"""Simplified George API for testing - delayed agent initialization.

P1 Observability:
- Optional Sentry error reporting via `SENTRY_DSN`
- Optional Prometheus `/metrics` via `PROMETHEUS_ENABLED=1`
"""
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, Optional
import os
import sys
import logging
import traceback
from pathlib import Path
import threading
from datetime import datetime

# Ensure repo root + src/ are importable.
# This file lives under scripts/legacy/, so repo root is two levels up.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app with production settings
app = FastAPI(
    title="George Cognitive API",
    version="1.0.0",
    description="Human-AI Cognition Framework API"
)


def _mark_deprecated(response: Response, successor_path: str) -> None:
    """Mark an endpoint as deprecated using HTTP headers.

    This is intentionally lightweight (no warnings emitted in-process) so it
    doesn't interfere with tests while still guiding API consumers.
    """
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = f'<{successor_path}>; rel="successor-version"'

# Mount shared API routers (canonical paths)
try:
    from src.interfaces.api.procedural_api import router as procedural_router

    app.include_router(procedural_router)
    logger.info("Mounted procedural API router at /")
except Exception as exc:
    logger.warning(f"Procedural API router not mounted: {type(exc).__name__}: {exc}")

try:
    from src.interfaces.api.prospective_api import router as prospective_router

    app.include_router(prospective_router)
    logger.info("Mounted prospective API router at /")
except Exception as exc:
    logger.warning(f"Prospective API router not mounted: {type(exc).__name__}: {exc}")

try:
    from src.interfaces.api.memory_api import router as memory_router

    # The Streamlit UI expects /memory/* (no /api prefix).
    app.include_router(memory_router)
    logger.info("Mounted memory API router at /")
except Exception as exc:
    logger.warning(f"Memory API router not mounted: {type(exc).__name__}: {exc}")

try:
    from src.interfaces.api.semantic_api import router as semantic_router

    app.include_router(semantic_router)
    logger.info("Mounted semantic API router at /")
except Exception as exc:
    logger.warning(f"Semantic API router not mounted: {type(exc).__name__}: {exc}")

try:
    from src.interfaces.api.executive_api import router as executive_router

    # Only mount at root to avoid clashing with other executive routes.
    app.include_router(executive_router)
    logger.info("Mounted executive API router at /")
except Exception as exc:
    logger.warning(f"Executive API router not mounted: {type(exc).__name__}: {exc}")

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _setup_sentry_if_configured(target_app: FastAPI) -> None:
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0"))
        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
            release=os.getenv("SENTRY_RELEASE"),
            traces_sample_rate=traces_sample_rate,
        )
        target_app.add_middleware(SentryAsgiMiddleware)
        logger.info("Sentry enabled")
    except Exception as exc:
        logger.warning(f"Sentry disabled (init failed): {type(exc).__name__}: {exc}")


def _setup_prometheus_if_configured(target_app: FastAPI) -> None:
    if not _get_bool_env("PROMETHEUS_ENABLED", default=False):
        return
    try:
        from prometheus_client import Counter, Histogram, generate_latest
        from prometheus_client import CONTENT_TYPE_LATEST

        request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "path", "status"],
        )
        request_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration (seconds)",
            ["method", "path"],
        )

        @target_app.middleware("http")
        async def prometheus_middleware(request: Request, call_next):
            start_time = datetime.now()
            response = await call_next(request)

            # Prefer route template to avoid high-cardinality labels.
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            duration_s = (datetime.now() - start_time).total_seconds()

            request_count.labels(request.method, path, str(response.status_code)).inc()
            request_latency.labels(request.method, path).observe(duration_s)
            return response

        @target_app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        logger.info("Prometheus /metrics enabled")
    except Exception as exc:
        logger.warning(
            f"Prometheus disabled (init failed): {type(exc).__name__}: {exc}"
        )


# Optional observability integrations (env-driven)
_setup_sentry_if_configured(app)
_setup_prometheus_if_configured(app)


# Global error handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions gracefully."""
    error_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Report to Sentry if enabled.
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
    
    # Log the full traceback
    logger.error(
        f"Unhandled exception [{error_id}] on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    
    # Return sanitized error to client
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
            "error_id": error_id,
            "timestamp": datetime.now().isoformat()
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = datetime.now()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    # Log request (skip health/metrics checks to reduce noise)
    if request.url.path not in {"/health", "/metrics"}:
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration_ms:.1f}ms"
        )
    
    return response




@app.middleware("http")
async def ensure_agent_state(request: Request, call_next):
    """Ensure `app.state.agent` is populated for routers that depend on it."""
    path = request.url.path
    needs_agent = path.startswith(
        (
            "/memory",
            "/semantic",
            "/executive",
            "/procedure",
            "/prospective",
            "/agent",
        )
    )
    if needs_agent and getattr(app.state, "agent", None) is None:
        get_agent()
    return await call_next(request)


# Global agent variable
_agent = None
_agent_initializing = False
_initialization_error = None

# Reflection state
reflection_lock = threading.Lock()
last_reflection_report = None

def get_agent():
    """Lazy agent initialization"""
    global _agent, _agent_initializing, _initialization_error
    
    if _agent is not None:
        return _agent
    
    if _agent_initializing:
        raise HTTPException(status_code=503, detail="Agent is currently initializing, please wait...")
    
    if _initialization_error:
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {_initialization_error}")
    
    try:
        _agent_initializing = True
        print("[INIT] Initializing George cognitive agent...")
        from src.orchestration.agent_singleton import create_agent
        _agent = create_agent()
        # Expose agent via FastAPI app state for shared routers.
        app.state.agent = _agent
        print("[OK] Agent initialized")
        _agent_initializing = False
        return _agent
    except Exception as e:
        _agent_initializing = False
        _initialization_error = str(e)
        print(f"[ERROR] Agent initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for verifying server is running."""
    return {"status": "healthy", "service": "George Cognitive API"}


# Enhanced health check with component status
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    global _agent, _agent_initializing, _initialization_error
    
    components = {
        "api": {"healthy": True, "message": "API server running"},
        "agent": {"healthy": False, "message": "Not initialized"}
    }
    
    # Check agent status
    if _agent is not None:
        components["agent"] = {"healthy": True, "message": "Agent ready"}
        
        # Check memory systems
        try:
            if hasattr(_agent, 'memory'):
                components["memory"] = {"healthy": True, "message": "Memory system available"}
        except Exception as e:
            components["memory"] = {"healthy": False, "message": str(e)}
        
        # Check executive system
        try:
            if hasattr(_agent, 'executive_agent'):
                components["executive"] = {"healthy": True, "message": "Executive system available"}
        except Exception as e:
            components["executive"] = {"healthy": False, "message": str(e)}
    
    elif _agent_initializing:
        components["agent"] = {"healthy": False, "message": "Initializing..."}
    elif _initialization_error:
        components["agent"] = {"healthy": False, "message": f"Init failed: {_initialization_error}"}
    
    overall_healthy = all(c.get("healthy", False) for c in components.values())
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": components
    }


# Initialization status endpoint
@app.get("/agent/init-status")
async def initialization_status(request: Request, response: Response):
    """Check agent initialization status"""
    global _agent, _agent_initializing, _initialization_error

    if _agent is not None:
        return {"status": "ready", "message": "Agent is fully initialized"}
    elif _agent_initializing:
        return {"status": "initializing", "message": "Agent is currently initializing..."}
    elif _initialization_error:
        return {"status": "error", "message": f"Initialization failed: {_initialization_error}"}
    else:
        return {"status": "not_started", "message": "Agent initialization has not started"}

# Agent status endpoint
@app.get("/agent/status")
async def agent_status(request: Request, response: Response):
    """Get agent status"""
    try:
        agent = get_agent()
        status = agent.get_cognitive_status()
        return {"status": "ok", "cognitive_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

class ProcessInputRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str


class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    flags: Optional[Dict[str, bool]] = None
    stream: bool = False
    consolidation_salience_threshold: Optional[float] = None

class ProactiveRecallRequest(BaseModel):
    query: str
    max_results: int = 5
    min_relevance: float = 0.7
    use_ai_summary: bool = False

@app.post("/agent/process")
async def process_input(request: ProcessInputRequest, response: Response):
    """Process user input through the cognitive agent"""
    try:
        agent = get_agent()
        response = await agent.process_input(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/agent/chat")
async def agent_chat(req: AgentChatRequest, request: Request, response: Response):
    """Chat endpoint compatible with the main API's /agent/chat."""
    headers: dict[str, str] | None = None
    from src.orchestration.chat.api_runtime import get_chat_service

    chat_svc = get_chat_service()

    original_threshold = None
    if req.consolidation_salience_threshold is not None:
        try:
            adaptive_cfg = chat_svc.context_builder.cfg
            original_threshold = adaptive_cfg.get("consolidation_salience_threshold")
            adaptive_cfg["consolidation_salience_threshold"] = req.consolidation_salience_threshold
        except Exception:
            original_threshold = None

    try:
        if req.stream:
            async def _gen() -> AsyncGenerator[bytes, None]:
                async for chunk in chat_svc.process_user_message_stream(
                    req.message,
                    session_id=req.session_id,
                    flags=req.flags,
                ):
                    yield (f"{chunk}\n").encode("utf-8")

            return StreamingResponse(_gen(), media_type="text/plain", headers=headers)

        result = await asyncio.to_thread(
            chat_svc.process_user_message,
            req.message,
            req.session_id,
            req.flags,
        )
        return JSONResponse(result, headers=headers)
    finally:
        if original_threshold is not None:
            try:
                adaptive_cfg = chat_svc.context_builder.cfg
                adaptive_cfg["consolidation_salience_threshold"] = original_threshold
            except Exception:
                pass


@app.get("/agent/chat/preview")
async def agent_chat_preview(
    request: Request,
    response: Response,
    message: str,
    session_id: Optional[str] = None,
):
    from src.orchestration.chat.api_runtime import get_chat_service

    chat_svc = get_chat_service()
    return chat_svc.get_context_preview(message=message, session_id=session_id)


@app.get("/agent/chat/metrics")
async def agent_chat_metrics(
    request: Request,
    response: Response,
    light: bool = True,
):
    from src.orchestration.chat.api_runtime import metrics_registry

    return metrics_registry.snapshot_light() if light else metrics_registry.snapshot()


@app.get("/agent/chat/performance")
async def agent_chat_performance_status(
    request: Request,
    response: Response,
):
    from src.orchestration.chat.api_runtime import get_chat_service

    chat_svc = get_chat_service()
    return chat_svc.performance_status()


@app.get("/agent/chat/metacog/status")
async def agent_chat_metacog_status(
    request: Request,
    response: Response,
):
    from src.orchestration.chat.api_runtime import get_chat_service

    chat_svc = get_chat_service()
    snap = getattr(chat_svc, "_last_metacog_snapshot", None)
    if not snap:
        return {"available": False}
    history = []
    try:
        hist = getattr(chat_svc, "_metacog_history", None)
        if hist is not None:
            history = list(hist)[-10:]
    except Exception:
        history = []
    return {"available": True, "snapshot": snap, "history_tail": history}


@app.get("/agent/chat/consolidation/status")
async def agent_chat_consolidation_status(
    request: Request,
    response: Response,
):
    from src.orchestration.chat.api_runtime import get_chat_service

    chat_svc = get_chat_service()
    cons = getattr(chat_svc, "consolidator", None)
    if not cons:
        return {"active": False, "reason": "consolidator_not_configured"}
    status = cons.status()
    events = cons.events_tail(10)
    return {"active": True, "status": status, "recent_events": events}


class DreamRequest(BaseModel):
    cycle_type: str = "light"


@app.post("/agent/dream/start")
async def agent_dream_start(dream_req: DreamRequest, request: Request, response: Response):
    from src.orchestration.chat.api_runtime import get_agent, get_chat_service

    agent = get_agent()
    if agent is None:
        return {"error": "agent_not_initialized", "dream_results": None}

    try:
        dream_proc = getattr(agent, "dream_processor", None)
        if dream_proc is not None:
            results = await dream_proc.enter_dream_cycle(dream_req.cycle_type)
            return {"dream_results": results}

        chat_svc = get_chat_service()
        cons = getattr(chat_svc, "consolidator", None)
        if cons is None:
            return {"error": "consolidator_not_available", "dream_results": None}

        promoted_count = 0
        for turn_id, stats in list(cons._turn_stats.items()):
            if stats.get("stm_id") and not stats.get("ltm_id"):
                cons._maybe_promote(turn_id, stats)
                if stats.get("ltm_id"):
                    promoted_count += 1

        return {
            "dream_results": {
                "cycle_type": dream_req.cycle_type,
                "promoted_count": promoted_count,
                "method": "consolidator_fallback",
            }
        }
    except Exception as exc:
        return {"error": str(exc), "dream_results": None}


class LLMConfigUpdate(BaseModel):
    provider: str
    openai_model: Optional[str] = "gpt-4.1-nano"
    ollama_base_url: Optional[str] = "http://localhost:11434"
    ollama_model: Optional[str] = "llama3.2"


@app.post("/agent/config/llm")
async def agent_update_llm_config(config: LLMConfigUpdate, request: Request, response: Response):
    """Update LLM provider configuration and reinitialize the agent provider."""
    try:
        agent = get_agent()
        if agent is None:
            return {"status": "error", "message": "Agent not available"}

        agent.config.llm.provider = config.provider
        if config.openai_model:
            agent.config.llm.openai_model = config.openai_model
        if config.ollama_base_url:
            agent.config.llm.ollama_base_url = config.ollama_base_url
        if config.ollama_model:
            agent.config.llm.ollama_model = config.ollama_model

        from src.orchestration.llm_provider import LLMProviderFactory

        new_provider = LLMProviderFactory.create_from_config(agent.config.llm)
        if not new_provider.is_available():
            return {"status": "error", "message": "LLM provider not available"}
        agent.llm_provider = new_provider
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

# Proactive recall endpoint for Streamlit UI
@app.post("/agent/memory/proactive-recall")
async def agent_proactive_recall(req: ProactiveRecallRequest, response: Response):
    """Perform proactive recall using the agent's memory system."""
    _mark_deprecated(response, "/memory/proactive-recall")
    try:
        agent = get_agent()
        result = agent.memory.proactive_recall(
            query=req.query,
            max_results=req.max_results,
            min_relevance=req.min_relevance,
            use_ai_summary=req.use_ai_summary,
            openai_client=getattr(agent, 'openai_client', None)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proactive recall failed: {str(e)}")

def _build_neural_status(agent: Any) -> dict[str, Any]:
    neural_status: dict[str, Any] = {
        "dpad_available": False,
        "lshn_available": False,
        "neural_activity": {
            "attention_enhancement": 0.0,
            "pattern_completion": 0.0,
            "consolidation_activity": 0.0,
        },
        "performance_metrics": {
            "response_time": 0.0,
            "accuracy": 0.0,
            "efficiency": 0.0,
        },
    }

    if hasattr(agent, "neural_integration"):
        neural_status["dpad_available"] = True
        neural_status["lshn_available"] = True

    return neural_status


@app.get("/neural/status")
async def neural_status_root():
    """Get neural network status and activity."""
    try:
        agent = get_agent()
        return _build_neural_status(agent)
    except Exception as e:
        return {"error": str(e), "neural_activity": {}, "performance_metrics": {}}



def _build_performance_analytics(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "cognitive_efficiency": {
            "overall": status.get("cognitive_integration", {}).get(
                "overall_efficiency", 0.0
            ),
            "memory": status.get("memory_status", {}).get("system_active", False),
            "attention": status.get("attention_status", {}).get(
                "capacity_utilization", 0.0
            ),
            "processing": status.get("cognitive_integration", {}).get(
                "processing_capacity", 0.0
            ),
        },
        "usage_statistics": {
            "session_duration": status.get("memory_status", {}).get("uptime_seconds", 0),
            "interactions": status.get("conversation_length", 0),
            "memory_operations": status.get("memory_status", {}).get(
                "operation_counts", {}
            ),
            "error_rate": 0.0,
        },
        "trends": {
            "cognitive_load_trend": "stable",
            "memory_usage_trend": "increasing",
            "performance_trend": "improving",
        },
    }


@app.get("/analytics/performance")
async def performance_analytics_root():
    """Get comprehensive performance analytics."""
    try:
        agent = get_agent()
        status = agent.get_cognitive_status()
        return _build_performance_analytics(status)
    except Exception as e:
        return {
            "error": str(e),
            "cognitive_efficiency": {},
            "usage_statistics": {},
            "trends": {},
        }


# Canonical agent reminder endpoints (preferred over /memory/prospective/*)
class AgentReminderCreate(BaseModel):
    content: str
    due_in_seconds: float
    metadata: Optional[Dict[str, Any]] = None


@app.post("/agent/reminders")
async def agent_create_reminder(rem: AgentReminderCreate, request: Request, response: Response):
    """Create a reminder using the agent's prospective memory system."""
    try:
        agent = get_agent()
        memsys = getattr(agent.memory, "prospective", None)
        if memsys is None:
            raise HTTPException(status_code=503, detail="prospective_memory_not_available")
        reminder = memsys.add_reminder(rem.content, rem.due_in_seconds, metadata=rem.metadata)
        as_dict = memsys.to_dict(reminder) if hasattr(memsys, "to_dict") else jsonable_encoder(reminder)
        return {"reminder": as_dict}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reminder creation error: {str(e)}")


@app.get("/agent/reminders")
async def agent_list_reminders(request: Request, response: Response, include_triggered: bool = True):
    """List reminders from the agent's prospective memory system."""
    try:
        agent = get_agent()
        memsys = getattr(agent.memory, "prospective", None)
        if memsys is None:
            return {"reminders": []}

        # Map legacy include_triggered -> new include_completed
        include_completed = bool(include_triggered)
        try:
            reminders = memsys.list_reminders(include_completed=include_completed)
        except TypeError:
            reminders = memsys.list_reminders(include_triggered=include_triggered)

        if hasattr(memsys, "to_dict"):
            data = [memsys.to_dict(r) for r in reminders]
        else:
            data = jsonable_encoder(reminders)
        return {"reminders": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reminder listing error: {str(e)}")


@app.get("/agent/reminders/due")
async def agent_due_reminders(request: Request, response: Response):
    """Return due reminders."""
    try:
        agent = get_agent()
        memsys = getattr(agent.memory, "prospective", None)
        if memsys is None:
            return {"due": []}
        if hasattr(memsys, "get_due_reminders"):
            due = memsys.get_due_reminders()
        else:
            due = memsys.check_due()
        if hasattr(memsys, "to_dict"):
            return {"due": [memsys.to_dict(r) for r in due]}
        return {"due": jsonable_encoder(due)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reminder due error: {str(e)}")


@app.delete("/agent/reminders/triggered")
async def agent_purge_triggered_reminders(request: Request, response: Response):
    """Delete all triggered/completed reminders."""
    try:
        agent = get_agent()
        memsys = getattr(agent.memory, "prospective", None)
        if memsys is None:
            return {"purged": 0}
        if hasattr(memsys, "purge_triggered"):
            removed = memsys.purge_triggered()
        else:
            removed = memsys.purge_completed()
        return {"purged": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reminder purge error: {str(e)}")


@app.post("/agent/reminders/{reminder_id}/complete")
async def agent_complete_reminder(reminder_id: str, request: Request, response: Response):
    try:
        agent = get_agent()
        memsys = getattr(agent.memory, "prospective", None)
        if memsys is None:
            raise HTTPException(status_code=503, detail="prospective_memory_not_available")
        success = memsys.complete_reminder(reminder_id)
        if not success:
            raise HTTPException(status_code=404, detail="reminder_not_found")
        return {"status": "completed", "reminder_id": reminder_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reminder complete error: {str(e)}")


@app.delete("/agent/reminders/{reminder_id}")
async def agent_delete_reminder(reminder_id: str, request: Request, response: Response):
    try:
        agent = get_agent()
        memsys = getattr(agent.memory, "prospective", None)
        if memsys is None:
            raise HTTPException(status_code=503, detail="prospective_memory_not_available")
        success = memsys.delete_reminder(reminder_id)
        if not success:
            raise HTTPException(status_code=404, detail="reminder_not_found")
        return {"status": "deleted", "reminder_id": reminder_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reminder delete error: {str(e)}")

# Reflection endpoints
class ReflectionStartRequest(BaseModel):
    interval: Optional[int] = None  # minutes

@app.post("/reflect")
async def reflect():
    """Trigger agent-level metacognitive reflection and return the report."""
    global last_reflection_report
    try:
        agent = get_agent()
        with reflection_lock:
            report = agent.reflect()
            last_reflection_report = report
        return {"status": "ok", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reflection failed: {str(e)}")

@app.get("/reflection/status")
async def reflection_status():
    """Get current reflection scheduler status"""
    try:
        agent = get_agent()
        status = agent.get_reflection_status()
        return {"status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reflection status: {str(e)}")

@app.get("/reflection/report")
async def reflection_report():
    """Get the most recent reflection report"""
    if last_reflection_report is None or last_reflection_report == {}:
        raise HTTPException(status_code=404, detail="No reflection report available.")
    return {"report": last_reflection_report}

@app.post("/reflection/start")
async def reflection_start(req: ReflectionStartRequest):
    """Start periodic reflection scheduler"""
    try:
        agent = get_agent()
        interval = req.interval if req.interval is not None else 10
        agent.start_reflection_scheduler(interval_minutes=interval)
        return {"status": "started", "interval_minutes": interval}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start reflection: {str(e)}")

@app.post("/reflection/stop")
async def reflection_stop():
    """Stop periodic reflection scheduler"""
    try:
        agent = get_agent()
        agent.stop_reflection_scheduler()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop reflection: {str(e)}")


# Telemetry endpoint
@app.get("/telemetry")
async def get_telemetry_metrics():
    """Get telemetry metrics for monitoring."""
    try:
        from src.core.resilience import get_telemetry
        telemetry = get_telemetry()
        return {"status": "ok", "metrics": telemetry.get_metrics()}
    except ImportError:
        return {"status": "ok", "metrics": {}, "message": "Telemetry not available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Circuit breaker status endpoint
@app.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers."""
    try:
        from src.core.resilience import CircuitBreaker
        statuses = {
            name: breaker.get_status()
            for name, breaker in CircuitBreaker._instances.items()
        }
        return {"status": "ok", "circuit_breakers": statuses}
    except ImportError:
        return {"status": "ok", "circuit_breakers": {}, "message": "Resilience module not available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _purge_legacy_api_routes() -> None:
    """Remove any /api/* routes from the app router."""
    app.router.routes = [
        route
        for route in app.router.routes
        if not getattr(route, "path", "").startswith("/api/")
    ]


_purge_legacy_api_routes()


if __name__ == "__main__":
    raise SystemExit(
        "george_api_simple.py is not an entrypoint anymore. "
        "Start the API with `python main.py api`."
    )
