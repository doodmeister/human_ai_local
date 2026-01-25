# ruff: noqa
"""Deprecated shim for ``src.chat.context_builder``.

Use ``src.orchestration.chat.context_builder``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.context_builder` is deprecated; use `src.orchestration.chat.context_builder`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.context_builder import *  # type: ignore
