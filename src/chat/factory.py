# ruff: noqa
"""Deprecated shim for ``src.chat.factory``.

Use ``src.orchestration.chat.factory``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.factory` is deprecated; use `src.orchestration.chat.factory`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.factory import *  # type: ignore
