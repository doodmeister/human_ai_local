@echo off
echo 🚀 Starting George Production Interface...
echo ===============================================

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ❌ Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Install/update dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo.
echo 🔧 Starting George API Server...
echo    • API will be available at: http://localhost:8000
echo    • API Documentation: http://localhost:8000/docs

REM Start API server in background using the root start_server.py
cd ..
start /B C:/dev/human_ai_local/venv/Scripts/python.exe start_server.py
cd scripts

echo    • API Server started in background

REM Wait for API server to start
echo ⏳ Waiting for API server to initialize...
timeout /t 5 /nobreak >nul

REM Check if API server is running (simple check)
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ API Server is running
) else (
    echo ⚠️  API Server may still be starting...
)

echo.
echo 🖥️  Starting Streamlit Interface...
echo    • Interface will be available at: http://localhost:8501
echo    • Close this window to stop both services

REM Start Streamlit interface with proper virtual environment
C:/dev/human_ai_local/venv/Scripts/streamlit.exe run george_streamlit_production.py --server.port 8501 --server.address localhost

echo.
echo 🛑 Services stopped when window closed
pause
