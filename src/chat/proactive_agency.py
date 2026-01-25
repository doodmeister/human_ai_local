# ruff: noqa
"""Deprecated shim for ``src.chat.proactive_agency``.

Use ``src.orchestration.chat.proactive_agency``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.proactive_agency` is deprecated; use `src.orchestration.chat.proactive_agency`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.proactive_agency import *  # type: ignore
