# George

George is a cognitive AI workspace built around persistent memory, attention, executive planning, and reflective runtime behavior.

## Start Here

- Run the backend with `python main.py api`
- Or launch the UI and backend together with `python main.py chainlit --with-backend`
- The default API base is `http://127.0.0.1:8000`

## What This UI Exposes

- Chat grounded in working, long-term, semantic, and prospective memory
- Goal and reminder workflows backed by the canonical FastAPI runtime
- Reflection, consolidation, and learning-oriented control surfaces
- Runtime-configurable LLM provider settings

## Notes

- Chainlit is the primary UI
- `python main.py ui` still launches the legacy Streamlit interface
- Current architecture and verification guidance live in `README.md` and `phase3.md`
