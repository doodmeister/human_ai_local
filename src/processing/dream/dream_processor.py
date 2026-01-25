# ruff: noqa
"""Deprecated shim for ``src.processing.dream.dream_processor``.

Use ``src.cognition.processing.dream.dream_processor``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.dream.dream_processor` is deprecated; use `src.cognition.processing.dream.dream_processor`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.dream.dream_processor import *  # type: ignore
