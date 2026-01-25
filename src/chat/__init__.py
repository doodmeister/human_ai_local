# ruff: noqa
"""Deprecated compatibility package.

Legacy import path: ``src.chat``

The canonical location is now ``src.orchestration.chat``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat` is deprecated; use `src.orchestration.chat` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat import *  # type: ignore
