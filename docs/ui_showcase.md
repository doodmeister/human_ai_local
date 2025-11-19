# George UI Showcase

This document tracks the Streamlit interface upgrades that accompany the proactive reminder and session context work. It also acts as the specs hand-off for the future React/Next.js client.

## Session Context Snapshot
- Collapsible panel under the main banner shows `session_context` metrics (`captured_memory_count`, `prospective_due_count`, `active_goal_ids`, etc.).
- Highlights the next upcoming reminder with a human-friendly due phrase.
- Includes optional classifier context JSON for debugging intent classification.

## Proactive Reminder Surfacing
- Top-of-feed "⏰ Coming up soon" banner appears whenever the backend signals new due reminders.
- Each reminder row now supports **Mark done**, **Snooze** (default duration configurable in the sidebar), and **Dismiss** actions.
- Dismissals and snoozes update the in-memory caches immediately to keep the UI consistent without extra API calls.

## Reminder Timeline & Creation
- Sidebar timeline lists due + upcoming reminders sorted by due time, with the same quick actions (Complete, Snooze, Delete).
- Inline "Create reminder" form POSTs to `/agent/reminders`, auto-injects the new reminder into the timeline, and persists form defaults (`due_in_minutes`, `snooze_minutes`).
- Metadata textarea lets users attach lightweight notes; these are stored in the reminder metadata payload.

## Telemetry Panel
- Sidebar expander summarizes reminder interaction counts (surfaced, snoozed, completed, etc.) and shows the last few events with timestamps and content snippets.
- Telemetry data is stored in `st.session_state.reminder_metrics` so it survives rerenders within the Streamlit session.

## React/Next Alignment Notes
- Component parity targets: `ReminderToast`, `ReminderTimeline`, `ReminderComposer`, and `ReminderTelemetry` should map directly to their Streamlit counterparts.
- The API contracts (`proactive_reminders`, `session_context`, `/agent/reminders/*`) stay identical for both UIs.
- Capture screenshots/GIFs of each panel once visual polish is complete and add them to this document (placeholder sections can be added near each heading).
