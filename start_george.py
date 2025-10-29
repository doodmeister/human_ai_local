#!/usr/bin/env python3
"""
Simple George Startup Script - Start both backend and frontend
"""

import subprocess
import time
import sys
import requests
import webbrowser
from pathlib import Path

# Configuration
API_PORT = 8000
STREAMLIT_PORT = 8501

def print_banner():
    print("🧠" + "="*60)
    print("   GEORGE - Human-AI Cognitive Architecture")
    print("   Simple Startup Script")
    print("="*62)

def check_api_health():
    """Check if API is responding"""
    try:
        response = requests.get(f"http://localhost:{API_PORT}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_for_old_server():
    """Check if an old server without init-status endpoint is running"""
    try:
        # First check if server is running
        if not check_api_health():
            return False
        
        # Check if it has the new endpoint
        response = requests.get(f"http://localhost:{API_PORT}/api/agent/init-status", timeout=5)
        return response.status_code == 404  # 404 means old server
    except:
        return False

def stop_old_server():
    """Stop old server running on port 8000"""
    try:
        import subprocess
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
        for line in result.stdout.split('\n'):
            if f':{API_PORT}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[4]
                    print(f"   Stopping old server (PID: {pid})...")
                    subprocess.run([f'taskkill /PID {pid} /F'], shell=True, capture_output=True)
                    time.sleep(2)
                    return True
    except Exception as e:
        print(f"   Could not stop old server: {e}")
    return False

def start_backend():
    """Start the backend API server"""
    # Check for old server first
    if check_for_old_server():
        print("🔄 Detected old server running, stopping it first...")
        stop_old_server()
    
    print("🔧 Starting George API Server...")
    print(f"   • API will be available at: http://localhost:{API_PORT}")
    print(f"   • API Documentation: http://localhost:{API_PORT}/docs")
    
    try:
        # Use virtual environment Python if available
        venv_python = Path("venv/Scripts/python.exe")
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable
        
        # Start the API server using start_server.py
        process = subprocess.Popen([
            python_cmd, "start_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"   • API Server started (PID: {process.pid})")
        
        # Wait for server to be ready
        print("⏳ Waiting for API server to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            if check_api_health():
                print("✅ API Server is ready!")
                return process
            time.sleep(1)
            if i % 5 == 4:
                print(f"   Still waiting... ({i+1}s)")
        
        print("⚠️ API Server may still be starting (proceeding anyway)")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🖥️ Starting Streamlit Interface...")
    print(f"   • Interface will be available at: http://localhost:{STREAMLIT_PORT}")
    print("   • Opening browser automatically...")
    
    try:
        # Use virtual environment Python if available
        venv_python = Path("venv/Scripts/python.exe")
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable
        
        # Start Streamlit
        subprocess.run([
            python_cmd,
            "-m",
            "streamlit",
            "run",
            "scripts/george_streamlit_chat.py",
            "--server.port",
            str(STREAMLIT_PORT),
            "--server.address",
            "localhost",
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

def main():
    """Main startup function"""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("start_server.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (The directory containing start_server.py)")
        sys.exit(1)
    
    # Start backend
    api_process = start_backend()
    if not api_process:
        print("❌ Failed to start backend. Exiting.")
        sys.exit(1)
    
    # Wait a moment before starting frontend
    time.sleep(2)
    
    # Open browser to the interface
    try:
        webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")
    except:
        pass  # Browser opening is optional
    
    print("\n🎉 George is ready!")
    print("   • Use Ctrl+C to stop both services")
    print("   • Close this window to stop the backend")
    print()
    
    # Start frontend (this will block)
    try:
        start_frontend()
    finally:
        # Clean up backend process
        if api_process:
            print("🛑 Stopping API server...")
            api_process.terminate()
            api_process.wait()

if __name__ == "__main__":
    main()
