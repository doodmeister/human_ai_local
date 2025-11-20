#!/usr/bin/env python3
"""
Startup script for Executive API server
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from george_api_simple import app
    from src.interfaces.api.chat_endpoints import router as chat_router
    from src.interfaces.api.executive_api import router as executive_router

    # Mount the advanced chat router
    app.include_router(chat_router)
    
    # Mount the executive functions router
    app.include_router(executive_router, prefix="/executive", tags=["executive"])

except ImportError as e:
    print(f"❌ Import error: {e}")
    if __name__ == "__main__":
        sys.exit(1)
    raise
except Exception as e:
    print(f"❌ Server startup error: {e}")
    if __name__ == "__main__":
        sys.exit(1)
    raise


def main() -> None:
    """Start the George Cognitive API server via uvicorn."""
    print("✅ API app imported successfully")
    print("🚀 Starting George Cognitive API server on http://localhost:8000...")
    print("📋 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /api/agent/status - Get cognitive status")
    print("   POST /api/agent/process - Process user input")
    print("   GET  /api/agent/memory/list/{system} - List STM/LTM memories")
    print("   POST /reflect - Trigger metacognitive reflection")
    print("   GET  /reflection/status - Get reflection scheduler status")
    print("   GET  /reflection/report - Get last reflection report")
    print()
    print("🎯 Executive Functions:")
    print("   POST /executive/goals - Create new goal")
    print("   GET  /executive/goals - List all goals")
    print("   GET  /executive/goals/{id} - Get goal details")
    print("   GET  /executive/status - Executive system status")
    print()

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
