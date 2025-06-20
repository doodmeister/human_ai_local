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

if __name__ == "__main__":
    asyncio.run(main())
