# ruff: noqa
"""Deprecated shim for ``src.orchestration.chat.intent_classifier``.

Use ``src.orchestration.chat.intent_classifier_v2``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.orchestration.chat.intent_classifier` is deprecated; "
    "use `src.orchestration.chat.intent_classifier_v2`.",
    DeprecationWarning,
    stacklevel=2,
)

from .intent_classifier_v2 import *  # type: ignore
