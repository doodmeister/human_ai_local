# ruff: noqa
"""Deprecated shim for ``src.processing.sensory.sensory_interface``.

Use ``src.cognition.processing.sensory.sensory_interface``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.sensory.sensory_interface` is deprecated; use `src.cognition.processing.sensory.sensory_interface`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.sensory.sensory_interface import *  # type: ignore
