# Chainlit UI

Chainlit is the primary interactive UI for this repository. It sits on top of the canonical FastAPI backend and exposes chat, memory browsing, reminders, goals, learning snapshots, and metacognition dashboards through a chat-first workflow.

## Start Here

Recommended local startup:

```bash
python main.py chainlit --with-backend
```

Split-terminal startup:

```bash
# Terminal 1
python main.py api

# Terminal 2
python main.py chainlit
```

Default endpoints:

- Chainlit UI: `http://localhost:8501`
- FastAPI backend: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

The UI client reads `GEORGE_API_BASE_URL` and defaults to `http://localhost:8000` if the variable is unset.

## What The UI Exposes

- Normal conversational chat through `POST /agent/chat`
- Memory browsing across STM, LTM, episodic, semantic, prospective, and procedural stores
- Reminder listing, creation, completion, snoozing, and deletion
- Goal creation, listing, and execution through the executive pipeline
- Dream-cycle triggering for consolidation
- Learning summaries backed by executive learning, experiments, and outcomes endpoints
- Metacognition dashboards backed by `/agent/metacog/dashboard`, `/agent/metacog/tasks`, and `/agent/metacog/reflections` for internal diagnostics

## Slash Commands

The current Chainlit app registers these commands in the input bar:

- `/memory <system> [query]` to browse or search `stm`, `ltm`, `episodic`, `semantic`, `prospective`, or `procedural`
- `/reminders` to list active reminders
- `/remind <minutes> <text>` to create a reminder
- `/goals` to list active goals
- `/goal <title>` to create a goal
- `/dream` to run a dream consolidation cycle
- `/learning` to show learning metrics, experiments, and recent outcomes
- `/metacog` to show the metacognition diagnostics dashboard for the current session
- `/reflect` to trigger an immediate reflection report through the canonical metacognition API

Use normal chat or `/memory <system> [query]` when you want remembered content rather than internal diagnostics.

## Session Settings

The Chainlit settings panel currently exposes:

- LLM provider selection
- OpenAI model selection
- STM consolidation salience threshold
- Default reminder snooze duration
- Toggles for memory retrieval, attention signals, and trace details

The UI pushes provider/model changes to the backend config endpoint when available and sends the selected salience threshold with chat requests.

## Operational Notes

- `python main.py chainlit --with-backend` is the shortest supported startup path
- `python main.py ui` still launches the legacy Streamlit interface
- The backend contract documented in `README.md` is the source of truth for the mounted API surface
- For runtime diagnostics, prefer `/agent/metacog/*`, `/agent/chat/performance`, `/health`, and `/health/detailed`
- If the backend is unreachable, start `python main.py api` and refresh the Chainlit page

## Related Docs

- `README.md` for the full runtime and API summary
- `STARTUP_GUIDE.md` for startup and troubleshooting
- `docs/UI_DEVELOPER_API_QUICKSTART.md` for frontend-facing API patterns
- `cognition.md` for the metacognition rollout and architecture
- `phase3.md` for runtime refactor status
