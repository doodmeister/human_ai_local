# ruff: noqa
"""Deprecated shim for ``src.chat.constants``.

Use ``src.orchestration.chat.constants``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.constants` is deprecated; use `src.orchestration.chat.constants`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.constants import *  # type: ignore
