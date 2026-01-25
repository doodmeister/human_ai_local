# ruff: noqa
"""Deprecated compatibility package.

Legacy import path: ``src.processing``

The canonical location is now ``src.cognition.processing``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing` is deprecated; use `src.cognition.processing` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing import *  # type: ignore
