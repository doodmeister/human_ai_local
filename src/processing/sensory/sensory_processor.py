# ruff: noqa
"""Deprecated shim for ``src.processing.sensory.sensory_processor``.

Use ``src.cognition.processing.sensory.sensory_processor``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.sensory.sensory_processor` is deprecated; use `src.cognition.processing.sensory.sensory_processor`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.sensory.sensory_processor import *  # type: ignore
