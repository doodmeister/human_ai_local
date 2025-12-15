#!/usr/bin/env python3
"""
Startup script for Executive API server

Production-ready server with:
- Configuration validation
- Error handling
- Health checks
- Telemetry endpoints
"""

import uvicorn
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from george_api_simple import app
    from src.interfaces.api.chat_endpoints import router as chat_router
    from src.interfaces.api.executive_api import router as executive_router
    from src.interfaces.api.memory_api import router as memory_router
    from src.interfaces.api.semantic_api import router as semantic_router
    from src.interfaces.api.prospective_api import router as prospective_router
    from src.interfaces.api.procedural_api import router as procedural_router

    # Mount the advanced chat router
    app.include_router(chat_router)
    
    # Mount the executive functions router
    app.include_router(executive_router, prefix="/executive", tags=["executive"])
    
    # Mount memory system routers (routes already have /memory, /semantic, etc. prefixes)
    app.include_router(memory_router, tags=["memory"])
    app.include_router(semantic_router, tags=["semantic"])
    app.include_router(prospective_router, tags=["prospective"])
    app.include_router(procedural_router, tags=["procedural"])

except ImportError as e:
    logger.error(f"Import error: {e}")
    if __name__ == "__main__":
        sys.exit(1)
    raise
except Exception as e:
    logger.error(f"Server startup error: {e}")
    if __name__ == "__main__":
        sys.exit(1)
    raise


def validate_startup() -> bool:
    """Validate configuration and dependencies at startup."""
    print("🔍 Validating configuration...")
    
    warnings = []
    
    # Validate configuration
    try:
        from src.core.config import validate_config
        is_valid, config_warnings = validate_config()
        warnings.extend(config_warnings)
    except ImportError:
        warnings.append("Could not import config validation")
    except Exception as e:
        warnings.append(f"Config validation error: {e}")
    
    # Check required directories
    data_dir = project_root / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"  📁 Created data directory: {data_dir}")
    
    # Check for .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        warnings.append("No .env file found - using defaults")
    
    # Print warnings
    if warnings:
        print("⚠️  Configuration warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  ✅ Configuration valid")
    
    return True


def main() -> None:
    """Start the George Cognitive API server via uvicorn."""
    print("=" * 60)
    print("🧠 George Cognitive API Server")
    print("=" * 60)
    
    # Run startup validation
    validate_startup()
    
    print()
    print("✅ API app imported successfully")
    print("🚀 Starting server on http://localhost:8000...")
    print()
    print("📋 Core Endpoints:")
    print("   GET  /health           - Basic health check")
    print("   GET  /health/detailed  - Detailed component health")
    print("   GET  /telemetry        - Performance metrics")
    print("   GET  /circuit-breakers - Resilience status")
    print()
    print("🎯 Executive Functions:")
    print("   POST /executive/goals        - Create goal")
    print("   GET  /executive/goals        - List goals")
    print("   GET  /executive/learning/metrics - Learning metrics")
    print("   GET  /executive/experiments  - A/B experiments")
    print()
    print("🧠 Memory Systems:")
    print("   GET  /memory/{system}/list  - List memories")
    print("   POST /memory/{system}/search - Search memories")
    print()
    print("💬 Chat:")
    print("   POST /agent/chat - Chat with agent")
    print()
    print("=" * 60)

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
