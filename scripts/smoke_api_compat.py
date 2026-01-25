"""Smoke-check key API compatibility endpoints.

Usage:
  python scripts/smoke_api_compat.py --base http://localhost:8000

This is intentionally lightweight and safe to run against either:
- main server (python main.py api)
- simple server (uvicorn scripts.legacy.george_api_simple:app)

Exit code:
- 0 on success
- 1 if any required checks fail
"""

from __future__ import annotations

import argparse
from typing import Any

import requests


def _check_ok(resp: requests.Response, label: str) -> None:
    if resp.status_code >= 400:
        raise RuntimeError(f"{label} failed: HTTP {resp.status_code} {resp.text[:500]}")


def _get_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000", help="API base URL (no trailing slash)")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Also hit agent-initializing endpoints (may be slower).",
    )
    parser.add_argument(
        "--show-deprecations",
        action="store_true",
        help="Print any Deprecation headers + successor Link values observed.",
    )
    args = parser.parse_args()

    base = (args.base or "").rstrip("/")
    failures: list[str] = []
    deprecations: list[str] = []

    def try_req(method: str, path: str, *, json: Any | None = None) -> None:
        url = f"{base}{path}"
        try:
            resp = requests.request(method, url, json=json, timeout=10)
            _check_ok(resp, f"{method} {path}")

            if (resp.headers.get("Deprecation") or "").lower() == "true":
                link = resp.headers.get("Link")
                if link:
                    deprecations.append(f"{method} {path}: {link}")
                else:
                    deprecations.append(f"{method} {path}: (no successor Link header)")
        except Exception as exc:
            failures.append(f"{method} {path}: {exc}")

    # Always-required checks (should be fast, no agent init required)
    try_req("GET", "/health")
    try_req("GET", "/api/health")

    if args.agent:
        # These may trigger lazy agent init.
        try_req("GET", "/agent/init-status")
        try_req("GET", "/agent/status")
        try_req("POST", "/agent/process", json={"text": "smoke"})
        try_req("POST", "/agent/chat", json={"message": "hello", "session_id": "smoke", "stream": False})
        try_req("GET", "/agent/reminders")
        try_req("GET", "/executive/status")
        try_req("GET", "/procedure/list")
        try_req("GET", "/neural/status")
        try_req("GET", "/analytics/performance")

    if failures:
        print("[smoke] FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("[smoke] OK")
    if args.agent:
        print("[smoke] Agent endpoints also OK")

    if args.show_deprecations and deprecations:
        print("[smoke] Deprecations observed (follow successor Link):")
        for item in deprecations:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
