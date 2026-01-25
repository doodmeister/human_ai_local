# ruff: noqa
"""Deprecated shim for ``src.processing.neural.neural_integration``.

Use ``src.cognition.processing.neural.neural_integration``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.neural.neural_integration` is deprecated; use `src.cognition.processing.neural.neural_integration`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.neural.neural_integration import *  # type: ignore
