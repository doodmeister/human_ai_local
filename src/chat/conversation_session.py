# ruff: noqa
"""Deprecated shim for ``src.chat.conversation_session``.

Use ``src.orchestration.chat.conversation_session``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.conversation_session` is deprecated; use `src.orchestration.chat.conversation_session`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.conversation_session import *  # type: ignore
