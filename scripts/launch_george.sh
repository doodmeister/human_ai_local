#!/bin/bash

# George Streamlit Interface Launcher
# ===================================

echo "🧠 George: Human-AI Cognition Interface Launcher"
echo "================================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly pandas
fi

# Install additional requirements if needed
if [ -f "streamlit_requirements.txt" ]; then
    echo "📦 Installing Streamlit requirements..."
    pip install -r streamlit_requirements.txt
fi

echo ""
echo "🚀 Starting George interfaces..."
echo ""
echo "Choose an interface:"
echo "1) 🎯 Standard George Interface (Recommended)"
echo "2) 🌟 Enhanced George Interface (Full-featured)"
echo "3) 🔧 Both interfaces (side-by-side comparison)"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🎯 Starting Standard George Interface..."
        echo "📍 URL: http://localhost:8501"
        streamlit run george_streamlit.py --server.port 8501
        ;;
    2)
        echo "🌟 Starting Enhanced George Interface..."
        echo "📍 URL: http://localhost:8502"
        streamlit run george_streamlit_enhanced.py --server.port 8502
        ;;
    3)
        echo "🔧 Starting both interfaces..."
        echo "📍 Standard: http://localhost:8501"
        echo "📍 Enhanced: http://localhost:8502"
        echo ""
        echo "Starting Standard interface in background..."
        streamlit run george_streamlit.py --server.port 8501 &
        sleep 2
        echo "Starting Enhanced interface..."
        streamlit run george_streamlit_enhanced.py --server.port 8502
        ;;
    *)
        echo "❌ Invalid choice. Starting Standard interface..."
        streamlit run george_streamlit.py --server.port 8501
        ;;
esac
