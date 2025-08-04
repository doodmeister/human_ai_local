#!/bin/bash

# George Startup Script for Git Bash/Linux/Mac
# Starts both backend and frontend services

echo "🧠 Starting George - Human-AI Cognitive Architecture"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "start_server.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   (The directory containing start_server.py)"
    exit 1
fi

echo "📦 Checking Python environment..."

# Determine Python command
if [ -f "venv/Scripts/python.exe" ]; then
    PYTHON_CMD="venv/Scripts/python.exe"
    echo "✅ Using virtual environment: $PYTHON_CMD"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    echo "✅ Using virtual environment: $PYTHON_CMD"
else
    PYTHON_CMD="python"
    echo "⚠️ Using system Python (consider using a virtual environment)"
fi

# Function to check if API is responding
check_api() {
    curl -s http://localhost:8000/health > /dev/null 2>&1
    return $?
}

echo ""
echo "🔧 Starting George API Server..."
echo "   • API will be available at: http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/docs"

# Start API server in background
$PYTHON_CMD start_server.py &
API_PID=$!

echo "   • API Server started (PID: $API_PID)"

# Wait for API to be ready
echo "⏳ Waiting for API server to be ready..."
for i in {1..30}; do
    if check_api; then
        echo "✅ API Server is ready!"
        break
    fi
    sleep 1
    if [ $((i % 5)) -eq 0 ]; then
        echo "   Still waiting... (${i}s)"
    fi
done

if ! check_api; then
    echo "⚠️ API Server may still be starting (proceeding anyway)"
fi

# Wait a moment before starting frontend
sleep 2

echo ""
echo "🖥️ Starting Streamlit Interface..."
echo "   • Interface will be available at: http://localhost:8501"
echo "   • Opening browser automatically..."
echo "   • Use Ctrl+C to stop both services"
echo ""

# Try to open browser (optional)
if command -v start > /dev/null 2>&1; then
    # Windows
    start http://localhost:8501 2>/dev/null || true
elif command -v open > /dev/null 2>&1; then
    # macOS
    open http://localhost:8501 2>/dev/null || true
elif command -v xdg-open > /dev/null 2>&1; then
    # Linux
    xdg-open http://localhost:8501 2>/dev/null || true
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down George..."
    if [ ! -z "$API_PID" ]; then
        echo "   Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
    fi
    echo "   Goodbye!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Streamlit (this will block until user presses Ctrl+C)
$PYTHON_CMD -m streamlit run scripts/george_streamlit_production.py --server.port 8501 --server.address localhost

# Cleanup will be called by trap when Streamlit exits
