import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from src.core.cognitive_agent import CognitiveAgent
from datetime import datetime

def format_time(ts):
    if not ts:
        return "unknown"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            return str(ts)
    return ts.strftime('%Y-%m-%d %H:%M:%S')

async def main():
    agent = CognitiveAgent()
    print("[George] Hello! I am George, your virtual cognitive agent. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            cmd = user_input.strip().lower()
            if cmd in {"exit", "quit"}:
                print("[George] Goodbye!")
                break
            if cmd == "/dream":
                print("[George] Initiating dream (consolidation) cycle...")
                await agent.enter_dream_state()
                print("[George] Dream cycle complete. STM memories have been consolidated to LTM (if eligible).")
                continue
            if cmd == "/memories":
                print("[George] Recent STM memories:")
                stm_items = getattr(agent.memory.stm, 'items', {})
                for m in list(stm_items.values())[-5:]:
                    print(f"  [{format_time(getattr(m, 'encoding_time', None))}] {getattr(m, 'content', '')}")
                print("[George] Recent LTM memories:")
                ltm_memories = getattr(agent.memory.ltm, 'memories', {})
                for m in list(ltm_memories.values())[-5:]:
                    print(f"  [{format_time(getattr(m, 'encoding_time', None))}] {getattr(m, 'content', '')}")
                continue
            if cmd == "/episodic":
                print("[George] Recent episodic memories:")
                episodic_cache = getattr(getattr(agent.memory, 'episodic', None), '_memory_cache', {})
                for m in list(episodic_cache.values())[-5:]:
                    print(f"  [{format_time(getattr(m, 'timestamp', None))}] {getattr(m, 'summary', '')}")
                continue
            if cmd.startswith("/remember"):
                to_remember = user_input[len("/remember"):].strip()
                if not to_remember:
                    print("[George] Please provide something for me to remember, e.g. /remember your favorite color is blue.")
                    continue
                # Summarize and store in STM
                import uuid
                summary = f"User asked me to remember: {to_remember}"
                memory_id = str(uuid.uuid4())
                agent.memory.stm.store(memory_id=memory_id, content=summary)
                print(f"[George] I will remember: '{to_remember}' (STM id: {memory_id})")
                continue
            # On recall, prioritize explicit 'remember' entries
            if any(kw in cmd for kw in ["what did i ask you to remember", "what do you remember", "recall", "remind me"]):
                stm_items = getattr(agent.memory.stm, 'items', {})
                remembered = [m for m in stm_items.values() if "remember:" in str(m.content).lower() or "asked me to remember" in str(m.content).lower()]
                if remembered:
                    print("[George] Here is what you asked me to remember:")
                    for m in remembered[-5:]:
                        print(f"  [{format_time(getattr(m, 'encoding_time', None))}] {getattr(m, 'content', '')}")
                else:
                    print("[George] I don't have any explicit memories you asked me to remember in STM.")
                continue
            response = await agent.process_input(user_input)
            # Print memory context for transparency
            # (We call the agent's internal memory search directly for this printout)
            processed_input = {"raw_input": user_input, "type": "text"}
            memory_context = await agent._retrieve_memory_context(processed_input)
            if memory_context:
                print("[George] (Memory context for this response):")
                for m in memory_context:
                    ts = format_time(m.get('timestamp'))
                    print(f"  [{ts}] ({m.get('source')}) {m.get('content')}")
            print(f"[George] {response}")
        except KeyboardInterrupt:
            print("\n[George] Session ended.")
            break
        if cmd.startswith("/remind"):
            import re
            from datetime import datetime, timedelta
            # Example: /remind me to call mom at 2025-06-21 15:00
            match = re.match(r"/remind (me to )?(?P<desc>.+?) (at|on|in) (?P<time>.+)", user_input, re.IGNORECASE)
            if not match:
                print("[George] Usage: /remind me to <task> at <YYYY-MM-DD HH:MM> or /remind me to <task> in <minutes> minutes")
                continue
            desc = match.group("desc").strip()
            time_str = match.group("time").strip()
            # Try to parse time
            due_time = None
            try:
                if "minute" in time_str:
                    match_result = re.search(r"(\d+)", time_str)
                    if match_result:
                        mins = int(match_result.group(1))
                        due_time = datetime.now() + timedelta(minutes=mins)
                    else:
                        raise ValueError("No number found in time string")
                else:
                    due_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            except Exception:
                print("[George] Could not parse time. Use 'at YYYY-MM-DD HH:MM' or 'in N minutes'.")
                continue
            reminder_id = agent.memory.add_prospective_reminder(desc, due_time)
            print(f"[George] Reminder set: '{desc}' at {due_time.strftime('%Y-%m-%d %H:%M')}")
            continue
        if cmd == "/reminders":
            reminders = agent.memory.list_prospective_reminders()
            if not reminders:
                print("[George] No scheduled reminders.")
            else:
                print("[George] Scheduled reminders:")
                for r in reminders:
                    print(f"  [Due: {format_time(r.due_time)}] {r.description} (ID: {r.id})")
            continue
        # Announce due reminders at each turn
        due = agent.memory.get_due_prospective_reminders()
        if due:
            print("[George] Reminder(s) due:")
            for r in due:
                print(f"  [Due: {format_time(r.due_time)}] {r.description}")
                agent.memory.complete_prospective_reminder(r.id)

if __name__ == "__main__":
    asyncio.run(main())
