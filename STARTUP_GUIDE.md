# Startup Guide

## Quick Start

### Option 1: Chainlit with backend

```bash
python main.py chainlit --with-backend
```

This is the shortest path for local interactive use. It starts the FastAPI backend on `http://127.0.0.1:8000` and the Chainlit UI on `http://localhost:8501`.

### Option 2: API only

```bash
python main.py api
```

Use this when you only need the backend or when another frontend is already configured to talk to it.

### Option 3: Split terminals

```bash
# Terminal 1
python main.py api

# Terminal 2
python main.py chainlit
```

Use this when you want the backend and UI logs separated.

### Option 4: Interactive CLI

```bash
python main.py chat
```

Use this for direct terminal interaction without a web UI.

### Option 5: Legacy Streamlit UI

```bash
python main.py ui
```

If you want the legacy UI and backend started together:

```bash
python main.py ui --with-backend
```

## What Should Come Up

When the API is running on the default port, these endpoints should respond:

- `http://127.0.0.1:8000/health` for a basic health check
- `http://127.0.0.1:8000/health/detailed` for component-level health
- `http://127.0.0.1:8000/docs` for FastAPI docs
- `http://127.0.0.1:8000/agent/init-status` for initialization state
- `http://127.0.0.1:8000/agent/status` for the basic cognitive status snapshot
- `http://127.0.0.1:8000/agent/metacog/dashboard` for the combined metacognition status, background, and scorecard view
- `http://127.0.0.1:8000/agent/metacog/reflect` for a manual reflection trigger
- `http://127.0.0.1:8000/status` for memory-system status

If you start Chainlit, the UI should be available at `http://localhost:8501`.

## API Base URL

The canonical backend root is:

```text
http://127.0.0.1:8000
```

When configuring a frontend, set `GEORGE_API_BASE_URL` to the server root without an `/api` suffix.

## Common Commands

```bash
# Activate the venv in Git Bash on Windows
source venv/Scripts/activate

# Start API with a different port
python main.py api --port 8001

# Start Chainlit on a different port
python main.py chainlit --port 8502

# Start Chainlit and point its helper backend to a non-default API port
python main.py chainlit --with-backend --api-port 8001 --port 8502

# Run the API smoke check
python scripts/smoke_api_compat.py --base http://localhost:8000
```

## Troubleshooting

### Virtual environment not active

```bash
source venv/Scripts/activate
```

If you are not using Git Bash, activate the environment with the shell-appropriate script for your terminal.

### Backend starts but UI cannot connect

Check that the API is healthy first:

```bash
python main.py api
```

Then verify:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

If the backend is running on a non-default port, update the frontend configuration to use that port.

### Ports are already in use

Start the services on alternate ports:

```bash
python main.py api --port 8001
python main.py chainlit --port 8502
```

If you use `--with-backend`, keep `--api-port` aligned with the backend port you expect to start.

### Startup feels slow

Initial startup can take longer while the runtime loads models, initializes memory stores, and opens persistent data directories. Later launches should be faster.

### Need deeper runtime diagnostics

Use these endpoints after the API is up:

- `/health/detailed` for component readiness
- `/telemetry` for resilience telemetry
- `/circuit-breakers` for circuit breaker state
- `/agent/chat/performance` for chat latency status
- `/agent/metacog/status`, `/agent/metacog/dashboard`, or `/agent/metacog/reflect` for metacognition state and manual reflection

## Services Overview

- Backend: canonical FastAPI runtime created from `python main.py api`
- Chainlit: current interactive chat UI (conversation-first, backend-driven)
- Streamlit: legacy UI path
- CLI: direct terminal interaction path via `python main.py chat`

---

## Chainlit UI Features

The Chainlit UI at `http://localhost:8501` is conversation-first. Normal messages go directly to `/agent/chat` and the backend handles intent classification, memory retrieval, reminders, and goal detection. Slash commands are explicit shortcuts.

### Available Commands

Open with the `/` icon in the message bar or type the command directly:

| Command | Description |
|---------|-------------|
| `/memory <query>` | Search across STM, LTM, episodic, and semantic memory |
| `/memory <system> [query]` | Browse a specific system: `stm`, `ltm`, `episodic`, `semantic`, `prospective`, `procedural` |
| `/reminders` | List active reminders with complete/snooze/delete actions |
| `/remind <minutes> <text>` | Create a reminder |
| `/goals` | List active goals with execute actions |
| `/goal <title>` | Create a goal |
| `/dream` | Run a dream consolidation cycle |
| `/reflect` | Run a manual reflection report |
| `/learning` | Show learning dashboard metrics |
| `/metacog` | Narrative self-report of internal state |
| `/metacog --raw` | Detailed diagnostic dashboard dump |

### Settings Panel

Click the gear icon to access:

- **LLM Provider / Model** — switch between OpenAI and Ollama and select a model
- **Memory capture sensitivity** — `adaptive` defers to the backend default; `capture_more` or `capture_less` override the salience threshold
- **Default Snooze** — minutes used when the Snooze button is clicked on a reminder
- **Include memory retrieval / attention signals** — enable or disable those retrieval layers per turn
- **Include trace details** — exposes per-turn latency, intent confidence, context items, and capture details in the UI

### Proactive Suggestions

The UI surfaces non-blocking suggestions automatically when backend signals warrant it:

- A `/dream` suggestion when STM utilization reaches 85% or after a heavy session
- A `/reflect` suggestion when unresolved contradictions, due background tasks, or a high follow-up rate are detected
- Reminder nudges when upcoming reminders are present

## Related Docs

- `README.md` for the full project overview and API summary
- `docs/UI_DEVELOPER_API_QUICKSTART.md` for frontend integration details
- `cognition.md` for the metacognition rollout and architecture
- `phase3.md` for runtime refactor status
- `phase4.md` for Chainlit UI redesign plan and completion status
