# ruff: noqa
"""Deprecated shim package for ``src.processing.neural``.

Use ``src.cognition.processing.neural``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.neural` is deprecated; use `src.cognition.processing.neural`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.neural import *  # type: ignore
