# ruff: noqa
"""Deprecated shim package for ``src.processing.sensory``.

Use ``src.cognition.processing.sensory``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.sensory` is deprecated; use `src.cognition.processing.sensory`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.sensory import *  # type: ignore
