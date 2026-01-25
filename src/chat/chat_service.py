# ruff: noqa
"""Deprecated shim for ``src.chat.chat_service``.

Use ``src.orchestration.chat.chat_service``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.chat_service` is deprecated; use `src.orchestration.chat.chat_service`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.chat_service import *  # type: ignore
