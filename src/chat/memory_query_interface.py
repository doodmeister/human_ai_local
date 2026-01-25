# ruff: noqa
"""Deprecated shim for ``src.chat.memory_query_interface``.

Use ``src.orchestration.chat.memory_query_interface``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.memory_query_interface` is deprecated; use `src.orchestration.chat.memory_query_interface`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.memory_query_interface import *  # type: ignore
