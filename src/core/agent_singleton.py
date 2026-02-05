# ruff: noqa
"""Deprecated shim for ``src.core.agent_singleton``.

Use ``src.orchestration.agent_singleton``.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

warnings.warn(
    "`src.core.agent_singleton` is deprecated; use `src.orchestration.agent_singleton`.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazy attribute access to keep `core` free of static layer imports."""

    mod = importlib.import_module("src.orchestration.agent_singleton")
    return getattr(mod, name)


def __dir__() -> list[str]:  # pragma: no cover
    mod = importlib.import_module("src.orchestration.agent_singleton")
    return sorted(set(globals().keys()) | set(dir(mod)))
