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
    from src.interfaces.api.reflection_api import app
    print("✅ API app imported successfully")
    print("🚀 Starting Executive API server on http://localhost:8000...")
    print("📋 Available endpoints:")
    print("   POST /api/executive/goals - Create a goal")  
    print("   GET  /api/executive/goals - List all goals")
    print("   GET  /api/executive/goals/{id} - Get specific goal")
    print("   GET  /api/executive/status - Get executive status")
    print("   + 10 more endpoints for tasks, decisions, resources...")
    print()
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Server startup error: {e}")
    sys.exit(1)
