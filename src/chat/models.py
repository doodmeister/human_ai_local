# ruff: noqa
"""Deprecated shim for ``src.chat.models``.

Use ``src.orchestration.chat.models``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.models` is deprecated; use `src.orchestration.chat.models`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.models import *  # type: ignore
