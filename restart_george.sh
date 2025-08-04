#!/bin/bash

# Quick Fix Script - Restart George with Updated Code
# Use this when you need to restart with new changes

echo "🔄 Restarting George with Updated Code"
echo "====================================="

# Kill existing processes
echo "🛑 Stopping existing services..."

# Find and kill processes using ports 8000 and 8501
for port in 8000 8501; do
    pid=$(netstat -ano | grep ":$port " | grep LISTENING | awk '{print $5}' | head -1)
    if [ ! -z "$pid" ]; then
        echo "   Stopping process $pid using port $port"
        cmd //c "taskkill /PID $pid /F" 2>/dev/null || true
    fi
done

# Wait for ports to be free
sleep 2

# Determine Python command
if [ -f "venv/Scripts/python.exe" ]; then
    PYTHON_CMD="venv/Scripts/python.exe"
else
    PYTHON_CMD="python"
fi

echo "🚀 Starting updated services..."

# Start API server
echo "   Starting API server on port 8000..."
$PYTHON_CMD start_server.py &
API_PID=$!

# Wait for API to be ready
sleep 5

# Start Streamlit
echo "   Starting Streamlit on port 8501..."
echo "   Visit: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services"

# Cleanup function
cleanup() {
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start Streamlit (blocking)
$PYTHON_CMD -m streamlit run scripts/george_streamlit_production.py --server.port 8501 --server.address localhost
