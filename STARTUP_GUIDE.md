# George Startup Guide

## 🚀 Quick Start Options

### Option 1: Single Entrypoint (Recommended)
```bash
# Starts API + Chainlit UI
python main.py chainlit --with-backend
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

# Terminal 2: Start frontend
python main.py chainlit
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
python main.py chainlit
python main.py ui
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
pkill -f "chainlit"

# Or use different ports:
python main.py api --port 8001
python main.py chainlit --port 8502
```

## 📋 What Should Happen

1. **API Server starts** on http://localhost:8000
   - Health check: http://localhost:8000/health
   - Documentation: http://localhost:8000/docs

2. **Chainlit Interface starts** on http://localhost:8501
   - Conversational UI backed by the canonical FastAPI runtime

## 🔌 API base URL tip

When running Streamlit, set the API base to the server root (no `/api` prefix), e.g. `http://localhost:8000`.

The canonical backend is `python main.py api`.

3. **Initialization takes 10-60 seconds**
   - First time: Downloads models, creates databases
   - Subsequent times: Loads existing data

## 🔧 Services Overview

- **Backend (API Server)**: George's cognitive architecture
- **Frontend (Chainlit)**: Current interactive chat interface
- **Frontend (Streamlit)**: Legacy interface
- **Both needed**: Frontend talks to backend via REST API

## 💡 Pro Tips

- Keep both services running simultaneously
- Use Ctrl+C to stop both services cleanly
- Check terminal output for error messages
- Visit http://localhost:8000/docs for API documentation
