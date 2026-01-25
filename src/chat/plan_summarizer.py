# ruff: noqa
"""Deprecated shim for ``src.chat.plan_summarizer``.

Use ``src.orchestration.chat.plan_summarizer``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.plan_summarizer` is deprecated; use `src.orchestration.chat.plan_summarizer`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.plan_summarizer import *  # type: ignore
