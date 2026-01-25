# ruff: noqa
"""Deprecated shim for ``src.chat.goal_detector``.

Use ``src.orchestration.chat.goal_detector``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.goal_detector` is deprecated; use `src.orchestration.chat.goal_detector`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.goal_detector import *  # type: ignore
