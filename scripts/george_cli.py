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
            if cmd == "/dream-batch":
                print("[George] Running dream-state batch consolidation (episodic memory)...")
                result = agent.memory.dream_state_consolidation()
                print(f"[George] Dream-state batch consolidation complete.")
                print(f"  Consolidated: {len(result['consolidated'])}")
                print(f"  Merged: {len(result['merged'])}")
                print(f"  Total candidates: {result['total_candidates']}")
                print(f"  Clusters merged: {result['clusters']}")
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
            continue
            # Procedural memory commands
            if cmd.startswith("/procedure"):
                subcmd = user_input[len("/procedure"):].strip()
                pm = agent.memory.procedural
                if subcmd.startswith("add"):
                    print("[George] Adding a new procedural memory.")
                    desc = input("  Description: ").strip()
                    steps = []
                    print("  Enter steps (blank line to finish):")
                    while True:
                        step = input(f"    Step {len(steps)+1}: ").strip()
                        if not step:
                            break
                        steps.append(step)
                    tags = input("  Tags (comma separated): ").strip().split(",")
                    tags = [t.strip() for t in tags if t.strip()]
                    memtype = input("  Store in (stm/ltm) [stm]: ").strip().lower() or "stm"
                    proc_id = pm.store(description=desc, steps=steps, tags=tags, memory_type=memtype)
                    print(f"[George] Procedure stored with ID: {proc_id}")
                    continue
                if subcmd.startswith("list"):
                    procs = pm.all_procedures()
                    if not procs:
                        print("[George] No procedural memories stored.")
                    else:
                        print("[George] Stored procedures:")
                        for p in procs:
                            print(f"  ID: {p['id']} | Desc: {p['description']} | Steps: {len(p['steps'])} | Strength: {p.get('strength', 0):.2f}")
                    continue
                if subcmd.startswith("search"):
                    query = subcmd[len("search"):].strip()
                    if not query:
                        print("[George] Usage: /procedure search <query>")
                        continue
                    results = pm.search(query)
                    if not results:
                        print("[George] No procedures found matching query.")
                    else:
                        print("[George] Search results:")
                        for p in results:
                            print(f"  ID: {p['id']} | Desc: {p['description']} | Steps: {len(p['steps'])} | Strength: {p.get('strength', 0):.2f}")
                    continue
                if subcmd.startswith("use"):
                    proc_id = subcmd[len("use"):].strip()
                    if not proc_id:
                        print("[George] Usage: /procedure use <id>")
                        continue
                    success = pm.use(proc_id)
                    if success:
                        proc = pm.retrieve(proc_id)
                        print(f"[George] Used procedure: {proc['description']}")
                        print("  Steps:")
                        for i, step in enumerate(proc['steps'], 1):
                            print(f"    {i}. {step}")
                    else:
                        print("[George] Procedure not found.")
                    continue
                if subcmd.startswith("delete"):
                    proc_id = subcmd[len("delete"):].strip()
                    if not proc_id:
                        print("[George] Usage: /procedure delete <id>")
                        continue
                    if pm.delete(proc_id):
                        print(f"[George] Procedure {proc_id} deleted.")
                    else:
                        print("[George] Procedure not found.")
                    continue
                if subcmd.startswith("clear"):
                    pm.clear()
                    print("[George] All procedural memories cleared.")
                    continue
                print("[George] Usage: /procedure [add|list|search <query>|use <id>|delete <id>|clear]")
                continue
            # Episodic memory commands
            if cmd.startswith("/episodic "):
                subcmd = user_input[len("/episodic"):].strip()
                em = getattr(agent.memory, 'episodic', None)
                if em is None:
                    print("[George] Episodic memory system not available.")
                    continue
                if subcmd.startswith("add"):
                    print("[George] Adding a new episodic memory.")
                    content = input("  Detailed content: ").strip()
                    importance = float(input("  Importance (0.0-1.0) [0.5]: ") or 0.5)
                    emotional_valence = float(input("  Emotional valence (-1.0 to 1.0) [0.0]: ") or 0.0)
                    life_period = input("  Life period (optional): ").strip() or None
                    mem_id = em.store(detailed_content=content, importance=importance, emotional_valence=emotional_valence, life_period=life_period)
                    print(f"[George] Episodic memory stored with ID: {mem_id}")
                    continue
                if subcmd.startswith("list"):
                    memories = em.search()
                    if not memories:
                        print("[George] No episodic memories stored.")
                    else:
                        print("[George] Stored episodic memories:")
                        for m in memories[-10:]:
                            print(f"  ID: {m['id']} | {m['summary']} | Importance: {m.get('importance', 0):.2f} | Valence: {m.get('emotional_valence', 0):.2f}")
                    continue
                if subcmd.startswith("search"):
                    query = subcmd[len("search"):].strip()
                    if not query:
                        print("[George] Usage: /episodic search <query>")
                        continue
                    results = em.search(query)
                    if not results:
                        print("[George] No episodic memories found matching query.")
                    else:
                        print("[George] Search results:")
                        for m in results:
                            print(f"  ID: {m['id']} | {m['summary']} | Importance: {m.get('importance', 0):.2f} | Valence: {m.get('emotional_valence', 0):.2f}")
                    continue
                if subcmd.startswith("retrieve"):
                    mem_id = subcmd[len("retrieve"):].strip()
                    if not mem_id:
                        print("[George] Usage: /episodic retrieve <id>")
                        continue
                    mem = em.retrieve(mem_id)
                    if mem:
                        print(f"[George] Episodic memory: {mem['summary']}")
                        print(f"  Content: {mem['detailed_content']}")
                        print(f"  Importance: {mem.get('importance', 0):.2f} | Valence: {mem.get('emotional_valence', 0):.2f}")
                    else:
                        print("[George] Episodic memory not found.")
                    continue
                if subcmd.startswith("delete"):
                    mem_id = subcmd[len("delete"):].strip()
                    if not mem_id:
                        print("[George] Usage: /episodic delete <id>")
                        continue
                    if em.delete(mem_id):
                        print(f"[George] Episodic memory {mem_id} deleted.")
                    else:
                        print("[George] Episodic memory not found.")
                    continue
                print("[George] Usage: /episodic [add|list|search <query>|retrieve <id>|delete <id>]")
                continue

if __name__ == "__main__":
    asyncio.run(main())
