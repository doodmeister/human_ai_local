# ruff: noqa
"""Deprecated shim for ``src.chat.memory_capture``.

Use ``src.orchestration.chat.memory_capture``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.memory_capture` is deprecated; use `src.orchestration.chat.memory_capture`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.memory_capture import *  # type: ignore
