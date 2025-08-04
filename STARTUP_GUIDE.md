# George Startup Guide for Git Bash Users

## 🚀 Quick Start Options

### Option 1: Full-Featured Startup (Recommended)
```bash
./start_george.sh
```
Features:
- Automatic environment detection
- Health checks and progress feedback
- Browser auto-opening
- Proper cleanup on exit

### Option 2: Simple Startup
```bash
# Use the Python script instead
python start_george.py
```
Features:
- Clean output
- Automatic environment detection
- Error handling

### Option 3: Manual Startup (if scripts don't work)
```bash
# Terminal 1 (or background): Start API server
venv/Scripts/python.exe start_server.py &

# Wait a few seconds, then start frontend
venv/Scripts/python.exe -m streamlit run scripts/george_streamlit_production.py --server.port 8501
```

## 🛠️ Troubleshooting

### If you get "permission denied" errors:
```bash
chmod +x start_george.sh
```

### If Python commands don't work:
```bash
# Try these alternatives:
python start_server.py
python.exe start_server.py
py start_server.py
```

### If virtual environment isn't found:
```bash
# Activate manually first:
source venv/Scripts/activate  # Git Bash
# or
venv/Scripts/activate.bat     # Command Prompt
```

### If ports are busy:
```bash
# Kill existing processes:
pkill -f "start_server.py"
pkill -f "streamlit"

# Or use different ports:
python -c "import uvicorn; from george_api_simple import app; uvicorn.run(app, port=8001)"
```

## 📋 What Should Happen

1. **API Server starts** on http://localhost:8000
   - Health check: http://localhost:8000/health
   - Documentation: http://localhost:8000/docs

2. **Streamlit Interface starts** on http://localhost:8501
   - Chat interface with George
   - Memory and attention monitoring
   - Executive function dashboard

3. **Initialization takes 10-60 seconds**
   - First time: Downloads models, creates databases
   - Subsequent times: Loads existing data

## 🔧 Services Overview

- **Backend (API Server)**: George's cognitive architecture
- **Frontend (Streamlit)**: Interactive web interface
- **Both needed**: Frontend talks to backend via REST API

## 💡 Pro Tips

- Keep both services running simultaneously
- Use Ctrl+C to stop both services cleanly
- Check terminal output for error messages
- Visit http://localhost:8000/docs for API documentation
