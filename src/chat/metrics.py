# ruff: noqa
"""Deprecated shim for ``src.chat.metrics``.

Metrics are now sourced from ``src.memory.metrics`` (and re-exported via
``src.orchestration.chat.metrics``).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.metrics` is deprecated; use `src.memory.metrics` (or `src.orchestration.chat.metrics`).",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.metrics import *  # type: ignore
