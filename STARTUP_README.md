# George Startup Scripts

This directory contains the streamlined startup scripts for George's Human-AI Cognitive Architecture.

## 🚀 How to Start George

### Option 1: Git Bash / Linux / Mac
```bash
./start_george.sh
```

### Option 2: Any Terminal (Python)
```bash
python start_george.py
```

### Option 3: Manual (if you prefer control)
```bash
# Start backend API server
python start_server.py

# In another terminal, start frontend
python -m streamlit run scripts/george_streamlit_chat.py --server.port 8501
```

## 📁 Current Scripts

### Core Scripts (Keep These)
- **`start_server.py`** - Starts the George API server on port 8000
- **`start_george.py`** - Python launcher that starts both backend and frontend
- **`start_george.sh`** - Bash script for Git Bash/Linux/Mac users  
- **`scripts/george_streamlit_chat.py`** - Streamlit chat interface (minimal)

### Configuration Files
- **`STARTUP_GUIDE.md`** - Detailed startup instructions
- **`george_api_simple.py`** - FastAPI application
- **`scripts/streamlit_requirements.txt`** - Frontend dependencies

## ✅ Features

- **Automatic Environment Detection** - Finds your virtual environment
- **Port Conflict Resolution** - Stops old servers automatically  
- **Initialization Progress** - Real-time feedback during startup
- **Timeout Fixes** - No more 10-second timeout errors
- **Cross-Platform** - Works on Windows, Linux, Mac

## 🎯 What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `start_george.sh` | Full-featured bash launcher | Git Bash, Linux, Mac |
| `start_george.py` | Python launcher with auto-cleanup | Any Python environment |
| `start_server.py` | API server only | When you only need the backend |

## 🔧 Troubleshooting

If you get port conflicts:
```bash
# Kill existing processes
tasklist | findstr python
taskkill /PID [process_id] /F
```

If you get permission errors:
```bash
chmod +x start_george.sh
```

## 🌐 Access Points

- **Streamlit Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
