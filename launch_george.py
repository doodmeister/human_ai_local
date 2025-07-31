#!/usr/bin/env python3
"""
George Production Launcher
Unified startup script for George Human-AI Cognitive Architecture

Starts both:
1. Backend API Server (port 8000)
2. Streamlit Interface (port 8501)

Author: GitHub Copilot
Date: July 2025
"""

import subprocess
import sys
import time
import signal
import requests
from pathlib import Path
import threading
import os

# Configuration
API_PORT = 8000
STREAMLIT_PORT = 8501
STARTUP_DELAY = 5

class GeorgeLauncher:
    def __init__(self):
        self.api_process = None
        self.streamlit_process = None
        self.project_root = Path(__file__).parent
        self.venv_python = self.project_root / "venv" / "Scripts" / "python.exe"
        self.venv_streamlit = self.project_root / "venv" / "Scripts" / "streamlit.exe"
        
    def print_banner(self):
        print("🧠" + "="*60)
        print("   GEORGE - Human-AI Cognitive Architecture")
        print("   Production Launcher v1.0")
        print("="*63)
        print()
    
    def check_environment(self):
        """Check if the environment is properly set up"""
        print("🔍 Checking environment...")
        
        if not self.venv_python.exists():
            print(f"❌ Virtual environment not found: {self.venv_python}")
            print("   Please run: python -m venv venv")
            return False
            
        if not self.venv_streamlit.exists():
            print(f"❌ Streamlit not found: {self.venv_streamlit}")
            print("   Please run: pip install streamlit")
            return False
            
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("⚠️  requirements.txt not found")
            
        print("✅ Environment check passed")
        return True
    
    def start_api_server(self):
        """Start the API server in background"""
        print("🔧 Starting George API Server...")
        print(f"   • API will be available at: http://localhost:{API_PORT}")
        print(f"   • API Documentation: http://localhost:{API_PORT}/docs")
        
        start_server_path = self.project_root / "start_server.py"
        
        try:
            self.api_process = subprocess.Popen(
                [str(self.venv_python), str(start_server_path)],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            print(f"   • API Server started (PID: {self.api_process.pid})")
            return True
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False
    
    def wait_for_api(self):
        """Wait for API server to be ready"""
        print(f"⏳ Waiting for API server to initialize...")
        
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"http://localhost:{API_PORT}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API Server is ready!")
                    return True
            except:
                pass
            
            time.sleep(1)
            if i % 5 == 4:  # Print progress every 5 seconds
                print(f"   Still waiting... ({i+1}s)")
        
        print("⚠️  API Server may still be starting (proceeding anyway)")
        return False
    
    def start_streamlit(self):
        """Start the Streamlit interface"""
        print("🖥️  Starting Streamlit Interface...")
        print(f"   • Interface will be available at: http://localhost:{STREAMLIT_PORT}")
        print("   • Use Ctrl+C to stop both services")
        print()
        
        streamlit_script = self.project_root / "scripts" / "george_streamlit_production.py"
        
        try:
            # Start Streamlit in foreground (blocking)
            self.streamlit_process = subprocess.run([
                str(self.venv_streamlit), "run", str(streamlit_script),
                "--server.port", str(STREAMLIT_PORT),
                "--server.address", "localhost"
            ], cwd=str(self.project_root / "scripts"))
            
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
        except Exception as e:
            print(f"❌ Streamlit error: {e}")
    
    def cleanup(self):
        """Clean up background processes"""
        print("🧹 Shutting down services...")
        
        if self.api_process and self.api_process.poll() is None:
            print("   • Stopping API server...")
            try:
                if os.name == 'nt':
                    # Windows
                    self.api_process.terminate()
                else:
                    # Unix/Linux
                    self.api_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.api_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("   • Force killing API server...")
                    self.api_process.kill()
                    
            except Exception as e:
                print(f"   • Error stopping API server: {e}")
        
        print("✅ George services stopped")
    
    def run(self):
        """Main launcher routine"""
        try:
            self.print_banner()
            
            if not self.check_environment():
                return 1
            
            print()
            
            # Start API server
            if not self.start_api_server():
                return 1
            
            # Wait for API to be ready
            self.wait_for_api()
            
            print()
            
            # Start Streamlit (blocking)
            self.start_streamlit()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
        
        return 0

def main():
    """Entry point"""
    launcher = GeorgeLauncher()
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())
