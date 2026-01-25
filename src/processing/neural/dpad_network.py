# ruff: noqa
"""Deprecated shim for ``src.processing.neural.dpad_network``.

Use ``src.cognition.processing.neural.dpad_network``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.neural.dpad_network` is deprecated; use `src.cognition.processing.neural.dpad_network`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.neural.dpad_network import *  # type: ignore
