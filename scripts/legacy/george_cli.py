import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json
from datetime import datetime, timedelta
import re

BASE_URL = "http://127.0.0.1:8000/api"

def format_time(ts):
    if not ts:
        return "unknown"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            return str(ts)
    return ts.strftime('%Y-%m-%d %H:%M:%S')

def main():
    print("[George] Hello! I am George, your virtual cognitive agent. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            cmd = user_input.strip().lower()
            if cmd in {"exit", "quit"}:
                print("[George] Goodbye!")
                break

            # Reminders
            if cmd.startswith("/reminders"):
                subcmd = user_input[len("/reminders"):].strip()

                # Low-level processing endpoint (kept for compatibility with older prospective router)
                if subcmd.startswith("process"):
                    try:
                        response = requests.post(f"{BASE_URL}/prospective/process_due")
                        response.raise_for_status()
                        data = response.json()
                        print(f"[George] Processed {data.get('processed', 0)} due reminders (migrated to LTM).")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error processing due reminders: {e}")
                    continue

                if subcmd.startswith("add"):
                    print("[George] Creating a reminder.")
                    content = input("  Content: ").strip()
                    if not content:
                        print("[George] Reminder content cannot be empty.")
                        continue
                    minutes_str = input("  Due in minutes (default 15): ").strip()
                    due_minutes = 15
                    if minutes_str:
                        try:
                            due_minutes = max(1, int(minutes_str))
                        except ValueError:
                            print("[George] Invalid minutes; using default 15.")
                            due_minutes = 15
                    payload = {"content": content, "due_in_seconds": float(due_minutes * 60)}
                    try:
                        response = requests.post(f"{BASE_URL}/agent/reminders", json=payload)
                        response.raise_for_status()
                        reminder = response.json().get("reminder") or {}
                        rid = reminder.get("id") or reminder.get("reminder_id")
                        due_time = reminder.get("due_time")
                        print(f"[George] Reminder created: {rid} (due: {format_time(due_time)})")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error creating reminder: {e}")
                    continue

                if not subcmd or subcmd.startswith("list"):
                    try:
                        response = requests.get(f"{BASE_URL}/agent/reminders")
                        response.raise_for_status()
                        reminders = response.json().get("reminders", [])
                        if not reminders:
                            print("[George] No reminders.")
                            continue
                        print("[George] Reminders:")
                        for r in reminders:
                            rid = r.get("id") or r.get("reminder_id")
                            print(f"  ID: {rid} | Due: {format_time(r.get('due_time'))} | {r.get('content') or r.get('description')}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error listing reminders: {e}")
                    continue

                if subcmd.startswith("due"):
                    try:
                        response = requests.get(f"{BASE_URL}/agent/reminders/due")
                        response.raise_for_status()
                        reminders = response.json().get("due", [])
                        if not reminders:
                            print("[George] No due reminders.")
                            continue
                        print("[George] Due reminders:")
                        for r in reminders:
                            rid = r.get("id") or r.get("reminder_id")
                            print(f"  ID: {rid} | Due: {format_time(r.get('due_time'))} | {r.get('content') or r.get('description')}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error fetching due reminders: {e}")
                    continue

                if subcmd.startswith("complete"):
                    reminder_id = subcmd[len("complete"):].strip()
                    if not reminder_id:
                        print("[George] Usage: /reminders complete <id>")
                        continue
                    try:
                        response = requests.post(f"{BASE_URL}/agent/reminders/{reminder_id}/complete")
                        response.raise_for_status()
                        print(f"[George] Reminder completed: {reminder_id}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error completing reminder: {e}")
                    continue

                if subcmd.startswith("delete"):
                    reminder_id = subcmd[len("delete"):].strip()
                    if not reminder_id:
                        print("[George] Usage: /reminders delete <id>")
                        continue
                    try:
                        response = requests.delete(f"{BASE_URL}/agent/reminders/{reminder_id}")
                        response.raise_for_status()
                        print(f"[George] Reminder deleted: {reminder_id}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error deleting reminder: {e}")
                    continue

                if subcmd.startswith("purge"):
                    try:
                        response = requests.delete(f"{BASE_URL}/agent/reminders/triggered")
                        response.raise_for_status()
                        purged = response.json().get("purged", 0)
                        print(f"[George] Purged {purged} triggered/completed reminders.")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error purging reminders: {e}")
                    continue

                print("[George] Usage: /reminders [add|list|due|complete <id>|delete <id>|purge|process]")
                continue

            # Non-API commands (for now)
            if cmd in {"/dream", "/dream-batch", "/memories", "/remember"} or any(kw in cmd for kw in ["what did i ask you to remember", "what do you remember", "recall", "remind me"]):
                 print(f"[George] Command '{cmd}' is not fully supported via API yet or pending refactor.")
                 continue

            # Reflection commands
            if cmd.startswith("/reflect"):
                subcmd = user_input[len("/reflect"):].strip()
                if not subcmd:
                    try:
                        response = requests.post(f"{BASE_URL}/reflect")
                        response.raise_for_status()
                        report = response.json().get('report', {})
                        print("[George] Metacognitive reflection complete. Summary:")
                        print(f"  Timestamp: {report.get('timestamp')}")
                        print(f"  LTM memories: {report.get('ltm_status', {}).get('total_memories', 'N/A')}")
                        print(f"  STM items: {report.get('stm_status', {}).get('size', 'N/A')}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error triggering reflection: {e}")
                    continue

                if subcmd.startswith("ion"):
                    subcmd = subcmd[len("ion"):].strip()
                    if subcmd.startswith("status"):
                        try:
                            response = requests.get(f"{BASE_URL}/reflection/status")
                            response.raise_for_status()
                            status = response.json()
                            print(f"[George] Reflection scheduler status: {status.get('status')}")
                        except requests.exceptions.RequestException as e:
                            print(f"[George] Error getting reflection status: {e}")
                        continue
                    if subcmd.startswith("start"):
                        interval_str = subcmd[len("start"):].strip()
                        interval = 10
                        if interval_str and interval_str.isdigit():
                            interval = int(interval_str)
                        try:
                            response = requests.post(f"{BASE_URL}/reflection/start", json={"interval": interval})
                            response.raise_for_status()
                            print(f"[George] Reflection scheduler started (every {interval} min).")
                        except requests.exceptions.RequestException as e:
                            print(f"[George] Error starting reflection scheduler: {e}")
                        continue
                    if subcmd.startswith("stop"):
                        try:
                            response = requests.post(f"{BASE_URL}/reflection/stop")
                            response.raise_for_status()
                            print("[George] Reflection scheduler stopped.")
                        except requests.exceptions.RequestException as e:
                            print(f"[George] Error stopping reflection scheduler: {e}")
                        continue
                    print("[George] Usage: /reflection [status|start [interval]|stop]")
                    continue

            # Procedural memory commands
            if cmd.startswith("/procedure"):
                subcmd = user_input[len("/procedure"):].strip()
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
                    payload = {"description": desc, "steps": steps, "tags": tags}
                    try:
                        response = requests.post(f"{BASE_URL}/procedure/store", json=payload)
                        response.raise_for_status()
                        proc_id = response.json()
                        print(f"[George] Procedure stored with ID: {proc_id}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error storing procedure: {e}")
                    continue
                if subcmd.startswith("list"):
                    try:
                        response = requests.get(f"{BASE_URL}/procedure/list")
                        response.raise_for_status()
                        procs = response.json()
                        if not procs:
                            print("[George] No procedural memories stored.")
                        else:
                            print("[George] Stored procedures:")
                            for p in procs:
                                print(f"  ID: {p['id']} | Desc: {p['description']} | Steps: {len(p['steps'])} | Strength: {p.get('strength', 0):.2f}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error listing procedures: {e}")
                    continue
                if subcmd.startswith("search"):
                    query = subcmd[len("search"):].strip()
                    if not query:
                        print("[George] Usage: /procedure search <query>")
                        continue
                    try:
                        response = requests.get(f"{BASE_URL}/procedure/search", params={"query": query})
                        response.raise_for_status()
                        results = response.json()
                        if not results:
                            print("[George] No procedures found matching query.")
                        else:
                            print("[George] Search results:")
                            for p in results:
                                print(f"  ID: {p['id']} | Desc: {p['description']} | Steps: {len(p['steps'])} | Strength: {p.get('strength', 0):.2f}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error searching procedures: {e}")
                    continue
                if subcmd.startswith("use"):
                    proc_id = subcmd[len("use"):].strip()
                    if not proc_id:
                        print("[George] Usage: /procedure use <id>")
                        continue
                    try:
                        response = requests.get(f"{BASE_URL}/procedure/retrieve/{proc_id}")
                        response.raise_for_status()
                        proc = response.json()
                        if proc:
                            print(f"[George] Used procedure: {proc['description']}")
                            print("  Steps:")
                            for i, step in enumerate(proc['steps'], 1):
                                print(f"    {i}. {step}")
                        else:
                            print("[George] Procedure not found.")
                    except requests.exceptions.RequestException as e:
                        if e.response and e.response.status_code == 404:
                            print("[George] Procedure not found.")
                        else:
                            print(f"[George] Error using procedure: {e}")
                    continue
                if subcmd.startswith("delete"):
                    proc_id = subcmd[len("delete"):].strip()
                    if not proc_id:
                        print("[George] Usage: /procedure delete <id>")
                        continue
                    try:
                        response = requests.delete(f"{BASE_URL}/procedure/delete/{proc_id}")
                        response.raise_for_status()
                        if response.json().get("success"):
                            print(f"[George] Procedure {proc_id} deleted.")
                        else:
                            print("[George] Procedure not found or could not be deleted.")
                    except requests.exceptions.RequestException as e:
                        if e.response and e.response.status_code == 404:
                            print("[George] Procedure not found.")
                        else:
                            print(f"[George] Error deleting procedure: {e}")
                    continue
                if subcmd.startswith("clear"):
                    print("[George] /procedure clear is not supported via API yet.")
                    continue
                print("[George] Usage: /procedure [add|list|search <query>|use <id>|delete <id>|clear]")
                continue

            # Generic memory commands
            if cmd.startswith("/memory "):
                parts = user_input.strip().split(" ", 3)
                if len(parts) < 3:
                    print("[George] Usage: /memory <store|retrieve|search|delete> <system> [args...]")
                    continue

                command, system, args_str = parts[1], parts[2], parts[3] if len(parts) > 3 else ""

                if command == "list":
                    payload = {"query": ""} # Sending an empty query to get all items
                    try:
                        # The memory API uses a POST for search, we'll use it to list all
                        response = requests.post(f"{BASE_URL}/memory/{system}/search", json=payload)
                        response.raise_for_status()
                        results = response.json().get("results", [])
                        if not results:
                            print(f"[George] No memories found in {system}.")
                        else:
                            print(f"[George] All memories in {system}:")
                            for r in results:
                                print(json.dumps(r, indent=2))
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error listing memories: {e}")
                    continue

                if command == "store":
                    if not args_str:
                        print(f"[George] Usage: /memory store {system} <content>")
                        continue
                    payload = {"content": args_str}
                    try:
                        response = requests.post(f"{BASE_URL}/memory/{system}/store", json=payload)
                        response.raise_for_status()
                        mem_id = response.json().get("memory_id")
                        print(f"[George] Memory stored in {system} with ID: {mem_id}")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error storing memory: {e}")
                    continue

                if command == "retrieve":
                    if not args_str:
                        print(f"[George] Usage: /memory retrieve {system} <memory_id>")
                        continue
                    try:
                        response = requests.get(f"{BASE_URL}/memory/{system}/retrieve/{args_str}")
                        response.raise_for_status()
                        memory = response.json()
                        print(f"[George] Retrieved from {system}:")
                        print(json.dumps(memory, indent=2))
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error retrieving memory: {e}")
                    continue

                if command == "search":
                    # args_str is the query
                    payload = {"query": args_str}
                    try:
                        # The memory API uses a POST for search
                        response = requests.post(f"{BASE_URL}/memory/{system}/search", json=payload)
                        response.raise_for_status()
                        results = response.json().get("results", [])
                        if not results:
                            print(f"[George] No memories found in {system} for that query.")
                        else:
                            print(f"[George] Search results from {system}:")
                            for r in results:
                                print(json.dumps(r, indent=2))
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error searching memory: {e}")
                    continue
                
                if command == "delete":
                    if not args_str:
                        print(f"[George] Usage: /memory delete {system} <memory_id>")
                        continue
                    try:
                        response = requests.delete(f"{BASE_URL}/memory/{system}/delete/{args_str}")
                        response.raise_for_status()
                        print(f"[George] Deleted memory {args_str} from {system}.")
                    except requests.exceptions.RequestException as e:
                        print(f"[George] Error deleting memory: {e}")
                    continue
                # This was missing the final "continue", which could cause fall-through
                continue

            # Prospective memory (reminders)
            if cmd.startswith("/remind "):
                match = re.match(r"/remind (me to )?(?P<desc>.+?) (at|on|in) (?P<time>.+)", user_input, re.IGNORECASE)
                if not match:
                    print("[George] Usage: /remind me to <task> at <YYYY-MM-DD HH:MM> or /remind me to <task> in <minutes> minutes")
                    continue
                desc = match.group("desc").strip()
                time_str = match.group("time").strip()
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
                    print(f"[George] Could not parse time: {time_str}")
                    continue

                due_in_seconds = max(0.0, (due_time - datetime.now()).total_seconds())
                payload = {"content": desc, "due_in_seconds": float(due_in_seconds)}
                try:
                    response = requests.post(f"{BASE_URL}/agent/reminders", json=payload)
                    response.raise_for_status()
                    reminder = response.json().get("reminder") or {}
                    rid = reminder.get("id") or reminder.get("reminder_id")
                    print(
                        f"[George] OK, I'll remind you to '{reminder.get('content') or desc}' "
                        f"at {format_time(reminder.get('due_time'))} (id: {rid})."
                    )
                except requests.exceptions.RequestException as e:
                    print(f"[George] Error setting reminder: {e}")
                continue

            # If no command was matched, send to the agent for processing
            payload = {"text": user_input}
            try:
                response = requests.post(f"{BASE_URL}/agent/process", json=payload)
                response.raise_for_status()
                data = response.json()
                print(f"[George] {data.get('response')}")
                if data.get("memory_context"):
                    print(f"       [context: {data.get('memory_context')}]")

            except requests.exceptions.RequestException as e:
                print(f"[George] Error processing message: {e}")


        except KeyboardInterrupt:
            print("\n[George] Shutting down. Goodbye!")
            break
        except requests.exceptions.ConnectionError:
            print("[George] Cannot connect to the API server. Is it running?")
            break
        except Exception as e:
            print(f"[George] An unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    main()
