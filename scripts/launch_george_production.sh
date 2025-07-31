#!/bin/bash

# George Production Interface Launcher
# Starts both API server and Streamlit interface

echo "🚀 Starting George Production Interface..."
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🔧 Starting George API Server..."
echo "   • API will be available at: http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/docs"

# Start API server in background using the root start_server.py
cd ..
C:/dev/human_ai_local/venv/Scripts/python.exe start_server.py &
API_PID=$!
cd scripts

echo "   • API Server PID: $API_PID"

# Wait for API server to start
echo "⏳ Waiting for API server to initialize..."
sleep 5

# Check if API server is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API Server is running"
else
    echo "⚠️  API Server may still be starting..."
fi

echo ""
echo "🖥️  Starting Streamlit Interface..."
echo "   • Interface will be available at: http://localhost:8501"
echo "   • Use Ctrl+C to stop both services"

# Start Streamlit interface with proper virtual environment
C:/dev/human_ai_local/venv/Scripts/streamlit.exe run george_streamlit_production.py --server.port 8501 --server.address localhost

# Cleanup: Kill API server when Streamlit exits
echo ""
echo "🛑 Shutting down services..."
kill $API_PID 2>/dev/null
echo "✅ George services stopped"
