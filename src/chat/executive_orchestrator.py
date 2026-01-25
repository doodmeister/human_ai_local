# ruff: noqa
"""Deprecated shim for ``src.chat.executive_orchestrator``.

Use ``src.orchestration.chat.executive_orchestrator``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.executive_orchestrator` is deprecated; use `src.orchestration.chat.executive_orchestrator`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.executive_orchestrator import *  # type: ignore
