# ruff: noqa
"""Deprecated shim for ``src.chat.api_runtime``.

Use ``src.orchestration.chat.api_runtime``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.api_runtime` is deprecated; use `src.orchestration.chat.api_runtime`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.api_runtime import *  # type: ignore
