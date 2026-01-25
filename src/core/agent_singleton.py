# ruff: noqa
"""Deprecated shim for ``src.core.agent_singleton``.

Use ``src.orchestration.agent_singleton``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.core.agent_singleton` is deprecated; use `src.orchestration.agent_singleton`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.agent_singleton import *  # type: ignore
