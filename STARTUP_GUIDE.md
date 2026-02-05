# George Startup Guide for Git Bash Users

## 🚀 Quick Start Options

### Option 1: Single Entrypoint (Recommended)
```bash
# Starts API + Streamlit UI
python main.py ui
```

### Option 2: API Only
```bash
python main.py api
```
API will be available at: http://localhost:8000

### Option 3: Manual Startup (if scripts don't work)
```bash
# Terminal 1 (or background): Start API server
python main.py api

# Wait a few seconds, then start frontend
venv/Scripts/python.exe -m streamlit run scripts/george_streamlit_chat.py --server.port 8501
```

## 🛠️ Troubleshooting

### If you get "permission denied" errors:
```bash
# (No longer applicable; startup scripts were consolidated into main.py)
```

### If Python commands don't work:
```bash
# Try these alternatives:
python main.py api
python -m streamlit run scripts/george_streamlit_chat.py --server.port 8501
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
pkill -f "uvicorn"
pkill -f "streamlit"

# Or use different ports:
python -c "import uvicorn; from scripts.legacy.george_api_simple import app; uvicorn.run(app, port=8001)"
```

## 📋 What Should Happen

1. **API Server starts** on http://localhost:8000
   - Health check: http://localhost:8000/health
   - (Compat) Health check: http://localhost:8000/api/health
   - Documentation: http://localhost:8000/docs

2. **Streamlit Interface starts** on http://localhost:8501
   - Minimal chat interface with memory context visibility

## 🔌 API base URL tip

When running Streamlit, set the API base to the server root (no `/api` prefix), e.g. `http://localhost:8000`.

The simple dev server (`scripts/legacy/george_api_simple.py`) also supports both unprefixed endpoints (like `/agent/chat`) and `/api/*` aliases.

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
