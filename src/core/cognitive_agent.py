# ruff: noqa
"""Deprecated shim for ``src.core.cognitive_agent``.

Use ``src.orchestration.cognitive_agent``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.core.cognitive_agent` is deprecated; use `src.orchestration.cognitive_agent`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.cognitive_agent import *  # type: ignore
