# ruff: noqa
"""Deprecated shim for ``src.chat.provenance``.

Use ``src.orchestration.chat.provenance``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.chat.provenance` is deprecated; use `src.orchestration.chat.provenance`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.orchestration.chat.provenance import *  # type: ignore
