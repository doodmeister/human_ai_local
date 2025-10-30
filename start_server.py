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

    # Mount the advanced chat router
    app.include_router(chat_router)

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
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Server startup error: {e}")
    sys.exit(1)
