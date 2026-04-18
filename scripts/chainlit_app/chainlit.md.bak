# Chainlit App Notes

This Chainlit UI is the primary interactive frontend for the repository.

## Startup

- Start both services with `python main.py chainlit --with-backend`
- Or start the backend separately with `python main.py api` and then run `python main.py chainlit`
- Default UI URL: `http://localhost:8501`
- Default backend URL: `http://127.0.0.1:8000`

## Supported Commands

| Command | Description |
|---------|-------------|
| `/memory <system> [query]` | Browse or search `stm`, `ltm`, `episodic`, `semantic`, `prospective`, or `procedural` |
| `/reminders` | List active reminders |
| `/remind 30 Review PR` | Create a 30-minute reminder |
| `/goals` | List active goals |
| `/goal Finish the report` | Create a goal |
| `/dream` | Run a dream consolidation cycle |
| `/learning` | Show learning metrics, experiments, and recent outcomes |
| `/metacog` | Show the metacognition diagnostics dashboard for the current session |
| `/reflect` | Trigger an immediate reflection report through the canonical metacognition API |

## Settings

Use the settings panel to change the LLM provider, model, salience threshold, default snooze duration, and memory/attention/trace toggles.

## Notes

- The UI talks to the backend root configured by `GEORGE_API_BASE_URL`
- The main supported diagnostics path is `/agent/metacog/*`
- For remembered user content, use normal chat or `/memory <system> [query]` rather than `/metacog`
- See `../../chainlit.md` and `../../README.md` for the current high-level runtime contract
