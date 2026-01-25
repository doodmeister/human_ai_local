# ruff: noqa
"""Deprecated shim for ``src.chat.scoring``.

Use ``src.orchestration.chat.scoring``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.scoring` is deprecated; use `src.orchestration.chat.scoring`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.scoring import *  # type: ignore
