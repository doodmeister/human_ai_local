# ruff: noqa
"""Deprecated shim for ``src.chat.memory_query_parser``.

Use ``src.orchestration.chat.memory_query_parser``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.memory_query_parser` is deprecated; use `src.orchestration.chat.memory_query_parser`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.memory_query_parser import *  # type: ignore
