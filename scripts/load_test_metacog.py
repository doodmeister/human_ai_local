"""Simple load test script for metacog/adaptive behavior.

Usage:
  python scripts/load_test_metacog.py --turns 50 --interval 0.05

Produces summary of key counters & interval evolution.
"""
from __future__ import annotations
import argparse
import time
import random
from src.chat import ChatService, ContextBuilder, SessionManager
from src.chat.metrics import metrics_registry
from src.core.config import get_chat_config


def run(turns: int, delay: float):
    cb = ContextBuilder(chat_config=get_chat_config().to_dict())
    sm = SessionManager()
    svc = ChatService(sm, cb)
    session_id = "load"
    for i in range(turns):
        # Inject artificial degraded state occasionally to trigger adaptations
        if i % 7 == 0 and i > 0:
            metrics_registry.state["performance_degraded"] = True
        else:
            metrics_registry.state["performance_degraded"] = False
        msg = f"Turn {i} random {random.randint(0,1000)}"
        svc.process_user_message(msg, session_id=session_id, flags={"include_trace": False})
        time.sleep(delay)
    perf = svc.performance_status()
    snap = metrics_registry.export_state()
    print("=== Load Test Summary ===")
    print("Turns:", turns)
    print("Metacog interval now:", perf.get("metacog", {}).get("interval"))
    print("Counters:")
    for k in sorted(snap["counters"].keys()):
        if k.startswith("metacog") or k in ("adaptive_retrieval_applied_total",):
            print(f"  {k}: {snap['counters'][k]}")
    print("Latency p95:", perf.get("latency_p95_ms"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--interval", type=float, default=0.02, help="Sleep between turns (s)")
    args = ap.parse_args()
    run(args.turns, args.interval)

if __name__ == "__main__":
    main()
