# ruff: noqa
"""Deprecated shim for ``src.core.cognitive_agent``.

Use ``src.orchestration.cognitive_agent``.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

warnings.warn(
    "`src.core.cognitive_agent` is deprecated; use `src.orchestration.cognitive_agent`.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazy attribute access to keep `core` free of static layer imports."""

    mod = importlib.import_module("src.orchestration.cognitive_agent")
    return getattr(mod, name)


def __dir__() -> list[str]:  # pragma: no cover
    mod = importlib.import_module("src.orchestration.cognitive_agent")
    return sorted(set(globals().keys()) | set(dir(mod)))
