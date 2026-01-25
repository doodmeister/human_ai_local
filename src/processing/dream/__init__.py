# ruff: noqa
"""Deprecated shim package for ``src.processing.dream``.

Use ``src.cognition.processing.dream``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.dream` is deprecated; use `src.cognition.processing.dream`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.dream import *  # type: ignore
