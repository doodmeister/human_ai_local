# Proactive Reminder UI Surfacing Plan

## Objectives
- Highlight upcoming reminders surfaced by `ChatService` so users never miss critical follow-ups.
- Reuse the existing FastAPI contract that now returns `session_context` and `proactive_reminders` payloads.
- Deliver a UX that works today in Streamlit and can later inform the React/Next.js migration.

## Streamlit Enhancements (Near-Term)
1. **Data plumbing**
   - Update `scripts/george_streamlit_chat.py` to capture the `proactive_reminders` list returned with each chat turn.
   - Store reminders in session state (e.g., `st.session_state["proactive_reminders"]`).
2. **Announce-on-turn panel**
   - Add a lightweight callout above the assistant response when fresh reminders arrive.
   - Format: title ("Coming up soon"), reminder title, due time, relative countdown, and quick action buttons (mark complete, snooze).
3. **Sidebar timeline**
   - Persistent component showing all upcoming reminders sorted by due time.
   - Use color/emojis or simple ASCII icons to denote urgency tiers (<15 min, <1 hr, >1 hr).
4. **Reminder actions**
   - Wire buttons to the existing reminder endpoints (`/agent/reminders/complete`, `/agent/reminders/delete`).
   - Optimistically update the panel; reconcile on next turn to avoid stale state.
5. **Accessibility & resilience**
   - Ensure reminders degrade gracefully if payload missing (show placeholder message).
   - Add telemetry hook to log reminder display/interaction counts for future UX metrics.

## React/Next Alignment (Future)
- Mirror the data contract (same JSON structures) so the Streamlit widgets act as reference implementations.
- Component concepts to port later:
  - `ReminderToast` for inline announcements.
  - `ReminderTimeline` for sidebar/overlay listing.
  - Shared hooks for polling reminder endpoints when chat idle.
- Document styles and UX flows during Streamlit work to shorten future React build time.

## Validation Checklist
- Manual test: create reminders via chat, verify they appear in both announcement panel and sidebar.
- Regression: run `pytest tests/test_chat_service_integration.py -k reminder -q` to ensure backend contract stays stable.
- UX feedback: capture screenshots/gifs once implemented and link them in `docs/ui_showcase.md` (to be created).
