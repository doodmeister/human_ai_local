"""Single project entrypoint.

Phase 5: Safe Deletions & Pruning

This repo previously had multiple top-level startup scripts. To reduce surface area,
all user-facing entrypoints are consolidated here.

Usage:
  - `python main.py chat` (default): interactive CLI
  - `python main.py api`: start FastAPI server on `http://127.0.0.1:8000`
  - `python main.py ui`: start Streamlit UI (optionally also starts backend)
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))


async def _run_chat() -> int:
    _ensure_src_on_path()
    from src.core import CognitiveAgent, CognitiveConfig
    from src.utils import setup_logging

    logger = setup_logging(level="INFO")
    logger.info("Starting Human-AI Cognition Framework CLI")

    config = CognitiveConfig.from_env()
    agent = CognitiveAgent(config)

    try:
        print("\n" + "=" * 60)
        print("HUMAN-AI COGNITION FRAMEWORK")
        print("=" * 60)
        print("Type 'quit' to exit, 'status' for cognitive state")
        print()

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in {"quit", "exit", "bye"}:
                print("Agent: Goodbye! Entering dream state for consolidation...")
                await agent.enter_dream_state()
                break

            if user_input.lower() == "status":
                status = agent.get_cognitive_status()
                print(f"Agent Status: {status}")
                continue

            if not user_input:
                continue

            response = await agent.process_input(user_input)
            print(f"Agent: {response}")

            status = agent.get_cognitive_status()
            print(
                f"[Fatigue: {status['fatigue_level']:.3f}, Conversations: {status['conversation_length']}]"
            )
            print()
    except KeyboardInterrupt:
        print("\nAgent: Interrupted. Shutting down...")
    finally:
        await agent.shutdown()

    return 0


def _build_api_app():
    _ensure_src_on_path()

    from scripts.legacy.george_api_simple import app
    from src.interfaces.api.chat_endpoints import router as chat_router
    from src.interfaces.api.executive_api import router as executive_router
    from src.interfaces.api.memory_api import router as memory_router
    from src.interfaces.api.semantic_api import router as semantic_router
    from src.interfaces.api.prospective_api import router as prospective_router
    from src.interfaces.api.procedural_api import router as procedural_router

    app.include_router(chat_router)
    app.include_router(executive_router, prefix="/executive", tags=["executive"])
    app.include_router(memory_router, tags=["memory"])
    app.include_router(semantic_router, tags=["semantic"])
    app.include_router(prospective_router, tags=["prospective"])
    app.include_router(procedural_router, tags=["procedural"])
    return app


def _run_api(*, host: str, port: int, reload: bool) -> int:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise SystemExit("uvicorn is required to run the API server") from e

    app = _build_api_app()
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
    return 0


def _http_ok(url: str, *, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _run_ui(*, port: int, with_backend: bool, api_host: str, api_port: int) -> int:
    project_root = Path(__file__).parent
    streamlit_script = project_root / "scripts" / "george_streamlit_chat.py"
    if not streamlit_script.exists():
        raise SystemExit(f"Missing Streamlit script: {streamlit_script}")

    backend_proc: Optional[subprocess.Popen[str]] = None
    try:
        if with_backend:
            backend_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(project_root / "main.py"),
                    "api",
                    "--host",
                    api_host,
                    "--port",
                    str(api_port),
                ]
            )

            # Best-effort wait for health endpoint.
            health_url = f"http://{api_host}:{api_port}/health"
            for _ in range(30):
                if _http_ok(health_url, timeout=0.5):
                    break
                time.sleep(0.5)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(streamlit_script),
                "--server.port",
                str(port),
                "--server.address",
                "localhost",
            ],
            check=False,
        )
        return 0
    finally:
        if backend_proc is not None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except Exception:
                backend_proc.kill()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="human-ai")
    subparsers = parser.add_subparsers(dest="cmd")

    subparsers.add_parser("chat", help="Interactive CLI (default)")

    p_api = subparsers.add_parser("api", help="Run FastAPI server")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--reload", action="store_true")

    p_ui = subparsers.add_parser("ui", help="Run Streamlit UI")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.add_argument("--with-backend", action="store_true")
    p_ui.add_argument("--api-host", default="127.0.0.1")
    p_ui.add_argument("--api-port", type=int, default=8000)

    args = parser.parse_args(argv)
    cmd = args.cmd or "chat"

    if cmd == "chat":
        return asyncio.run(_run_chat())
    if cmd == "api":
        return _run_api(host=args.host, port=args.port, reload=args.reload)
    if cmd == "ui":
        return _run_ui(port=args.port, with_backend=args.with_backend, api_host=args.api_host, api_port=args.api_port)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
