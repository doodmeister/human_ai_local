@echo off
REM George Streamlit Interface Launcher (Windows)
REM =============================================

echo.🧠 George: Human-AI Cognition Interface Launcher
echo.================================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.❌ Streamlit not found. Installing...
    pip install streamlit plotly pandas
)

REM Install additional requirements if needed
if exist "streamlit_requirements.txt" (
    echo.📦 Installing Streamlit requirements...
    pip install -r streamlit_requirements.txt
)

echo.
echo.🚀 Starting George interfaces...
echo.
echo.Choose an interface:
echo.1^) 🎯 Standard George Interface ^(Recommended^)
echo.2^) 🌟 Enhanced George Interface ^(Full-featured^)
echo.3^) 🔧 Both interfaces ^(side-by-side comparison^)
echo.

set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" (
    echo.🎯 Starting Standard George Interface...
    echo.📍 URL: http://localhost:8501
    streamlit run george_streamlit.py --server.port 8501
) else if "%choice%"=="2" (
    echo.🌟 Starting Enhanced George Interface...
    echo.📍 URL: http://localhost:8502
    streamlit run george_streamlit_enhanced.py --server.port 8502
) else if "%choice%"=="3" (
    echo.🔧 Starting both interfaces...
    echo.📍 Standard: http://localhost:8501
    echo.📍 Enhanced: http://localhost:8502
    echo.
    echo.Starting Standard interface in background...
    start /B streamlit run george_streamlit.py --server.port 8501
    timeout /t 3 /nobreak >nul
    echo.Starting Enhanced interface...
    streamlit run george_streamlit_enhanced.py --server.port 8502
) else (
    echo.❌ Invalid choice. Starting Standard interface...
    streamlit run george_streamlit.py --server.port 8501
)
